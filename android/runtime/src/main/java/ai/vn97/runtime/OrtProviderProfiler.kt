package ai.vn97.runtime

import ai.onnxruntime.OrtSession
import java.io.File
import java.security.MessageDigest
import org.json.JSONArray
import org.json.JSONObject

data class OrtProfileGraph(
    val filename: String,
    val kind: String,
    val sequenceLength: Int,
    val modelBytes: ByteArray,
) {
    init {
        require(filename.isNotBlank())
        require(kind == "step" || kind == "chunk")
        require(sequenceLength > 0)
        require(modelBytes.isNotEmpty())
    }
}

interface OrtPreparedInvocation : AutoCloseable {
    fun run(session: OrtSession)
    override fun close() = Unit
}

data class OrtProfileFailure(
    val graphFilename: String,
    val requestedProvider: OrtProviderKind,
    val errorClass: String,
)

data class OrtProfileMeasurement(
    val graphFilename: String,
    val kind: String,
    val sequenceLength: Int,
    val requestedProvider: OrtProviderKind,
    val actualProvider: OrtProviderKind,
    val warmupIterations: Int,
    val steadyIterations: Int,
    val sessionCreateNanos: Long,
    val steadyLatenciesNanos: List<Long>,
    val memoryBeforeBytes: Long,
    val memoryAfterBytes: Long,
    val thermalBefore: Int,
    val thermalAfter: Int,
    val fallbackUsed: Boolean,
) {
    init {
        require(kind == "step" || kind == "chunk")
        require(sequenceLength > 0)
        require(warmupIterations >= 1)
        require(steadyIterations >= 3)
        require(steadyLatenciesNanos.size == steadyIterations)
        require(sessionCreateNanos > 0L)
        require(steadyLatenciesNanos.all { it > 0L })
        require(memoryBeforeBytes > 0L && memoryAfterBytes > 0L)
        require(thermalBefore in 0..6 && thermalAfter in 0..6)
        require(requestedProvider == actualProvider || fallbackUsed)
    }

    val latencyP50Nanos: Long
        get() = percentileNearestRank(
            steadyLatenciesNanos,
            0.50,
        )
    val latencyP95Nanos: Long
        get() = percentileNearestRank(
            steadyLatenciesNanos,
            0.95,
        )
    val tokensPerSecondMilliP50: Long
        get() = (
            sequenceLength.toLong() * 1_000_000_000_000L
        ) / latencyP50Nanos
}

data class OrtProviderProfileReport(
    val bundleId: String,
    val architectureFingerprint: String,
    val profile: String,
    val device: OrtDeviceCapabilities,
    val measurements: List<OrtProfileMeasurement>,
    val failures: List<OrtProfileFailure>,
) {
    init {
        requireSha256(bundleId, "bundleId")
        requireSha256(
            architectureFingerprint,
            "architectureFingerprint",
        )
        require(
            profile == "fast" ||
                profile == "deep" ||
                (
                    profile.startsWith("layers-") &&
                        profile.removePrefix("layers-").toIntOrNull() != null
                    )
        )
        require(measurements.isNotEmpty())
    }

    fun toJson(): JSONObject {
        val graphResults = JSONArray()
        for ((filename, group) in measurements.groupBy {
            it.graphFilename
        }.toSortedMap()) {
            val ranked = group.sortedWith(
                compareBy<OrtProfileMeasurement>(
                    { it.fallbackUsed },
                    { it.latencyP50Nanos },
                    { it.latencyP95Nanos },
                    { it.requestedProvider.name },
                )
            )
            val providers = JSONArray()
            ranked.forEach { providers.put(it.toJson()) }
            graphResults.put(
                JSONObject()
                    .put("graph_filename", filename)
                    .put("ranked_providers", providers)
                    .put(
                        "recommended_provider",
                        ranked.first().requestedProvider.name,
                    )
            )
        }

        val failureArray = JSONArray()
        failures.sortedWith(
            compareBy(
                { it.graphFilename },
                { it.requestedProvider.name },
                { it.errorClass },
            )
        ).forEach {
            failureArray.put(
                JSONObject()
                    .put("graph_filename", it.graphFilename)
                    .put("requested_provider", it.requestedProvider.name)
                    .put("error_class", it.errorClass)
            )
        }

        val body = JSONObject()
            .put("schema", RECEIPT_SCHEMA)
            .put("bundle_id", bundleId)
            .put(
                "architecture_fingerprint",
                architectureFingerprint,
            )
            .put("profile", profile)
            .put("device", device.toEvidenceJson())
            .put("graph_results", graphResults)
            .put("failures", failureArray)
            .put("device_measured", true)
            .put("synthetic", false)

        val canonical = canonicalJson(body)
        val prefix = "VN97R2E3PROFILE1" +
            String(charArrayOf(0.toChar()))
        val receiptId = sha256(
            prefix.toByteArray(Charsets.UTF_8) +
                canonical.toByteArray(Charsets.UTF_8)
        )
        return JSONObject(canonical).put("receipt_id", receiptId)
    }

    fun writeAtomic(target: File) {
        require(!target.isDirectory)
        target.parentFile?.mkdirs()
        val temp = File(target.parentFile, target.name + ".tmp")
        if (temp.exists()) {
            check(temp.delete())
        }
        temp.writeText(
            canonicalJson(toJson()) + "\n",
            Charsets.UTF_8,
        )
        if (target.exists()) {
            check(target.delete())
        }
        check(temp.renameTo(target))
    }

    companion object {
        const val RECEIPT_SCHEMA = "VN97R2E3PROFILE1"
    }
}

class VN97OrtProviderProfiler(
    private val sessionFactory: VN97OrtSessionFactory =
        VN97OrtSessionFactory(),
) {
    fun profile(
        graphs: List<OrtProfileGraph>,
        device: OrtDeviceCapabilities,
        providers: List<OrtProviderKind>,
        warmupIterations: Int = 3,
        steadyIterations: Int = 12,
        deviceSnapshot: () -> OrtDeviceCapabilities,
        prepare: (
            graph: OrtProfileGraph,
            handle: OrtSessionHandle,
        ) -> OrtPreparedInvocation,
    ): Pair<List<OrtProfileMeasurement>, List<OrtProfileFailure>> {
        require(graphs.isNotEmpty())
        require(providers.isNotEmpty())
        require(providers.distinct().size == providers.size)
        require(warmupIterations >= 1)
        require(steadyIterations >= 3)

        val measurements = mutableListOf<OrtProfileMeasurement>()
        val failures = mutableListOf<OrtProfileFailure>()

        for (graph in graphs) {
            for (provider in providers) {
                if (!providerAllowed(provider, device)) {
                    failures += OrtProfileFailure(
                        graph.filename,
                        provider,
                        "provider_not_available",
                    )
                    continue
                }

                val decision = isolatedDecision(
                    provider,
                    graph.sequenceLength,
                    device,
                )
                val before = deviceSnapshot()
                try {
                    val createStart = System.nanoTime()
                    sessionFactory.create(
                        graph.modelBytes,
                        decision,
                        device,
                    ).use { handle ->
                        val createNanos = elapsedNanos(createStart)
                        prepare(graph, handle).use { invocation ->
                            repeat(warmupIterations) {
                                invocation.run(handle.session)
                            }
                            val samples = ArrayList<Long>(
                                steadyIterations
                            )
                            repeat(steadyIterations) {
                                val start = System.nanoTime()
                                invocation.run(handle.session)
                                samples += elapsedNanos(start)
                            }
                            val after = deviceSnapshot()
                            measurements += OrtProfileMeasurement(
                                graph.filename,
                                graph.kind,
                                graph.sequenceLength,
                                provider,
                                handle.primaryProvider,
                                warmupIterations,
                                steadyIterations,
                                createNanos,
                                samples,
                                before.availableMemoryBytes,
                                after.availableMemoryBytes,
                                before.thermalStatus,
                                after.thermalStatus,
                                handle.primaryProvider != provider,
                            )
                        }
                    }
                } catch (error: Throwable) {
                    failures += OrtProfileFailure(
                        graph.filename,
                        provider,
                        error::class.java.simpleName.ifBlank {
                            "unknown_error"
                        },
                    )
                }
            }
        }
        return measurements to failures
    }

    private fun isolatedDecision(
        provider: OrtProviderKind,
        sequenceLength: Int,
        device: OrtDeviceCapabilities,
    ): OrtAdaptiveDecision {
        val chain = if (provider == OrtProviderKind.CPU) {
            listOf(OrtProviderKind.CPU)
        } else {
            listOf(provider, OrtProviderKind.CPU)
        }
        return OrtAdaptiveDecision(
            mode = if (sequenceLength == 1) {
                OrtAdaptiveMode.SEQUENTIAL
            } else {
                OrtAdaptiveMode.HYBRID
            },
            chunkSizes = intArrayOf(sequenceLength),
            providers = chain,
            xnnpackThreads = minOf(
                4,
                device.logicalCores,
            ).coerceAtLeast(1),
            reason = "e3_isolated_provider_profile",
        )
    }

    private fun providerAllowed(
        provider: OrtProviderKind,
        device: OrtDeviceCapabilities,
    ): Boolean = when (provider) {
        OrtProviderKind.QNN -> (
            device.qnnAvailable &&
                device.qnnBackendPath != null &&
                device.isQualcomm
            )
        OrtProviderKind.NNAPI -> (
            device.nnapiAvailable && device.sdkInt >= 27
            )
        OrtProviderKind.XNNPACK -> device.xnnpackAvailable
        OrtProviderKind.CPU -> true
    }
}

private fun OrtProfileMeasurement.toJson(): JSONObject =
    JSONObject()
        .put("graph_filename", graphFilename)
        .put("kind", kind)
        .put("sequence_length", sequenceLength)
        .put("requested_provider", requestedProvider.name)
        .put("actual_provider", actualProvider.name)
        .put("fallback_used", fallbackUsed)
        .put("warmup_iterations", warmupIterations)
        .put("steady_iterations", steadyIterations)
        .put("session_create_ns", sessionCreateNanos)
        .put("latency_p50_ns", latencyP50Nanos)
        .put("latency_p95_ns", latencyP95Nanos)
        .put("latency_min_ns", steadyLatenciesNanos.min())
        .put("latency_max_ns", steadyLatenciesNanos.max())
        .put(
            "tokens_per_second_milli_p50",
            tokensPerSecondMilliP50,
        )
        .put("memory_before_bytes", memoryBeforeBytes)
        .put("memory_after_bytes", memoryAfterBytes)
        .put(
            "memory_delta_bytes",
            memoryAfterBytes - memoryBeforeBytes,
        )
        .put("thermal_before", thermalBefore)
        .put("thermal_after", thermalAfter)

private fun OrtDeviceCapabilities.toEvidenceJson(): JSONObject =
    JSONObject()
        .put("sdk_int", sdkInt)
        .put("logical_cores", logicalCores)
        .put("available_memory_bytes", availableMemoryBytes)
        .put("thermal_status", thermalStatus)
        .put("hardware", hardware)
        .put("soc_manufacturer", socManufacturer)
        .put("soc_model", socModel)

private fun percentileNearestRank(
    values: List<Long>,
    q: Double,
): Long {
    require(values.isNotEmpty())
    require(q > 0.0 && q <= 1.0)
    val ordered = values.sorted()
    val rank = kotlin.math.ceil(q * ordered.size)
        .toInt()
        .coerceAtLeast(1)
    return ordered[rank - 1]
}

private fun elapsedNanos(startNanos: Long): Long =
    (System.nanoTime() - startNanos).coerceAtLeast(1L)

private fun requireSha256(value: String, label: String) {
    require(
        value.length == 64 &&
            value.all {
                it in '0'..'9' || it in 'a'..'f'
            }
    ) {
        label + " must be lowercase SHA-256"
    }
}

private fun sha256(bytes: ByteArray): String =
    MessageDigest.getInstance("SHA-256")
        .digest(bytes)
        .joinToString("") { "%02x".format(it) }

private fun canonicalJson(value: Any?): String {
    return when (value) {
        JSONObject.NULL, null -> "null"
        is JSONObject -> {
            val keys = value.keys().asSequence().toList().sorted()
            keys.joinToString(
                prefix = "{",
                postfix = "}",
                separator = ",",
            ) { key ->
                JSONObject.quote(key) + ":" +
                    canonicalJson(value.get(key))
            }
        }
        is JSONArray -> {
            (0 until value.length()).joinToString(
                prefix = "[",
                postfix = "]",
                separator = ",",
            ) { index ->
                canonicalJson(value.get(index))
            }
        }
        is String -> JSONObject.quote(value)
        is Boolean -> if (value) "true" else "false"
        is Number -> {
            val number = value.toDouble()
            require(number.isFinite())
            value.toString()
        }
        else -> error(
            "unsupported canonical JSON type: " +
                value::class.java.name
        )
    }
}
