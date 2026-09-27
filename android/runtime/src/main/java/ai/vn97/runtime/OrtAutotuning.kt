package ai.vn97.runtime

import java.io.File
import java.security.MessageDigest
import org.json.JSONArray
import org.json.JSONObject

enum class OrtThermalBand {
    NORMAL,
    HOT,
    CRITICAL,
}

enum class OrtMemoryBand {
    NORMAL,
    PRESSURE,
}

enum class OrtLatencyBand {
    NORMAL,
    SLOW,
}

data class OrtGraphTuningPolicy(
    val graphFilename: String,
    val kind: String,
    val sequenceLength: Int,
    val providerOrder: List<OrtProviderKind>,
    val preferredProvider: OrtProviderKind,
    val expectedP95Nanos: Long,
) {
    init {
        require(graphFilename.isNotBlank())
        require(kind == "step" || kind == "chunk")
        require(sequenceLength > 0)
        require(providerOrder.isNotEmpty())
        require(providerOrder.distinct().size == providerOrder.size)
        require(providerOrder.last() == OrtProviderKind.CPU)
        require(preferredProvider == providerOrder.first())
        require(expectedP95Nanos > 0L)
    }
}

data class OrtAutotuneControlPolicy(
    val thermalHotEnter: Int,
    val thermalHotExit: Int,
    val thermalCriticalEnter: Int,
    val thermalCriticalExit: Int,
    val memoryPressureEnterPpm: Int,
    val memoryPressureExitPpm: Int,
    val latencySlowEnterPpm: Int,
    val latencySlowExitPpm: Int,
    val quarantineFailureThreshold: Int,
    val quarantineCooldownDecisions: Int,
    val xnnpackThreadsNormal: Int,
    val xnnpackThreadsHot: Int,
    val xnnpackThreadsCritical: Int,
) {
    init {
        require(thermalHotExit < thermalHotEnter)
        require(thermalHotEnter < thermalCriticalEnter)
        require(thermalCriticalExit < thermalCriticalEnter)
        require(memoryPressureEnterPpm < memoryPressureExitPpm)
        require(latencySlowExitPpm < latencySlowEnterPpm)
        require(quarantineFailureThreshold > 0)
        require(quarantineCooldownDecisions > 0)
        require(xnnpackThreadsNormal > 0)
        require(xnnpackThreadsHot > 0)
        require(xnnpackThreadsCritical > 0)
    }
}

data class OrtAutotuneDeviceIdentity(
    val sdkInt: Int,
    val logicalCores: Int,
    val hardware: String,
    val socManufacturer: String,
    val socModel: String,
    val profiledAvailableMemoryBytes: Long,
) {
    init {
        require(sdkInt >= 26)
        require(logicalCores > 0)
        require(profiledAvailableMemoryBytes > 0L)
    }

    fun matches(device: OrtDeviceCapabilities): Boolean =
        sdkInt == device.sdkInt &&
            logicalCores == device.logicalCores &&
            hardware == device.hardware &&
            socManufacturer == device.socManufacturer &&
            socModel == device.socModel
}

data class OrtAutotuneProfile(
    val tuningId: String,
    val bundleId: String,
    val architectureFingerprint: String,
    val modelProfile: String,
    val e3ReceiptId: String,
    val deviceKey: String,
    val device: OrtAutotuneDeviceIdentity,
    val graphPolicies: Map<String, OrtGraphTuningPolicy>,
    val preferredChunkSizes: List<Int>,
    val control: OrtAutotuneControlPolicy,
) {
    init {
        requireSha256E4(tuningId, "tuningId")
        requireSha256E4(bundleId, "bundleId")
        requireSha256E4(
            architectureFingerprint,
            "architectureFingerprint",
        )
        requireSha256E4(e3ReceiptId, "e3ReceiptId")
        requireSha256E4(deviceKey, "deviceKey")
        require(graphPolicies.containsKey("step.onnx"))
        require(preferredChunkSizes.isNotEmpty())
        require(preferredChunkSizes.distinct().size == preferredChunkSizes.size)
        require(
            preferredChunkSizes.all {
                graphPolicies.containsKey("chunk-" + it + ".onnx")
            }
        )
    }

    fun requireCompatible(
        expectedBundleId: String,
        currentDevice: OrtDeviceCapabilities,
    ) {
        require(bundleId == expectedBundleId) {
            "E4 tuning profile belongs to another ONNX bundle"
        }
        require(device.matches(currentDevice)) {
            "E4 tuning profile belongs to another device/OS identity"
        }
    }

    companion object {
        const val SCHEMA = "VN97R2E4TUNE1"

        fun load(
            file: File,
            expectedBundleId: String,
            currentDevice: OrtDeviceCapabilities,
        ): OrtAutotuneProfile {
            require(file.isFile) {
                "E4 tuning profile must be a regular file"
            }
            val parsed = parse(
                file.readText(Charsets.UTF_8),
            )
            parsed.requireCompatible(
                expectedBundleId,
                currentDevice,
            )
            return parsed
        }

        fun parse(raw: String): OrtAutotuneProfile {
            val root = JSONObject(raw)
            require(root.getString("schema") == SCHEMA)
            require(root.getBoolean("same_weights_semantics"))
            require(!root.getBoolean("quantization_used"))

            val tuningId = root.getString("tuning_id")
            requireSha256E4(tuningId, "tuningId")
            val body = JSONObject(root.toString())
            body.remove("tuning_id")
            val canonical = canonicalJsonE4(body)
            val prefix = "VN97R2E4TUNE1" +
                String(charArrayOf(0.toChar()))
            val actual = sha256E4(
                prefix.toByteArray(Charsets.UTF_8) +
                    canonical.toByteArray(Charsets.UTF_8)
            )
            require(actual == tuningId) {
                "E4 tuning profile identity mismatch"
            }

            val deviceJson = root.getJSONObject("device")
            val device = OrtAutotuneDeviceIdentity(
                sdkInt = deviceJson.getInt("sdk_int"),
                logicalCores = deviceJson.getInt("logical_cores"),
                hardware = deviceJson.getString("hardware"),
                socManufacturer = deviceJson.getString(
                    "soc_manufacturer"
                ),
                socModel = deviceJson.getString("soc_model"),
                profiledAvailableMemoryBytes = deviceJson.getLong(
                    "profiled_available_memory_bytes"
                ),
            )

            val policies = linkedMapOf<String, OrtGraphTuningPolicy>()
            val rawPolicies = root.getJSONArray("graph_policies")
            for (index in 0 until rawPolicies.length()) {
                val item = rawPolicies.getJSONObject(index)
                val providerArray = item.getJSONArray("provider_order")
                val providerOrder = buildList {
                    for (providerIndex in 0 until providerArray.length()) {
                        add(
                            OrtProviderKind.valueOf(
                                providerArray.getString(providerIndex)
                            )
                        )
                    }
                }
                val preferred = OrtProviderKind.valueOf(
                    item.getString("preferred_provider")
                )
                val measurements = item.getJSONArray("measurements")
                var expectedP95: Long? = null
                for (measurementIndex in 0 until measurements.length()) {
                    val measurement = measurements.getJSONObject(
                        measurementIndex
                    )
                    if (
                        measurement.getString("provider") ==
                        preferred.name
                    ) {
                        expectedP95 = measurement.getLong(
                            "latency_p95_ns"
                        )
                    }
                }
                val policy = OrtGraphTuningPolicy(
                    graphFilename = item.getString("graph_filename"),
                    kind = item.getString("kind"),
                    sequenceLength = item.getInt("sequence_length"),
                    providerOrder = providerOrder,
                    preferredProvider = preferred,
                    expectedP95Nanos = requireNotNull(expectedP95) {
                        "E4 preferred provider measurement is missing"
                    },
                )
                require(!policies.containsKey(policy.graphFilename)) {
                    "E4 duplicate graph policy"
                }
                policies[policy.graphFilename] = policy
            }

            val chunkArray = root.getJSONArray("preferred_chunk_sizes")
            val preferredChunks = buildList {
                for (index in 0 until chunkArray.length()) {
                    add(chunkArray.getInt(index))
                }
            }
            val controlJson = root.getJSONObject("control_policy")
            val control = OrtAutotuneControlPolicy(
                thermalHotEnter = controlJson.getInt(
                    "thermal_hot_enter"
                ),
                thermalHotExit = controlJson.getInt(
                    "thermal_hot_exit"
                ),
                thermalCriticalEnter = controlJson.getInt(
                    "thermal_critical_enter"
                ),
                thermalCriticalExit = controlJson.getInt(
                    "thermal_critical_exit"
                ),
                memoryPressureEnterPpm = controlJson.getInt(
                    "memory_pressure_enter_ppm"
                ),
                memoryPressureExitPpm = controlJson.getInt(
                    "memory_pressure_exit_ppm"
                ),
                latencySlowEnterPpm = controlJson.getInt(
                    "latency_slow_enter_ppm"
                ),
                latencySlowExitPpm = controlJson.getInt(
                    "latency_slow_exit_ppm"
                ),
                quarantineFailureThreshold = controlJson.getInt(
                    "quarantine_failure_threshold"
                ),
                quarantineCooldownDecisions = controlJson.getInt(
                    "quarantine_cooldown_decisions"
                ),
                xnnpackThreadsNormal = controlJson.getInt(
                    "xnnpack_threads_normal"
                ),
                xnnpackThreadsHot = controlJson.getInt(
                    "xnnpack_threads_hot"
                ),
                xnnpackThreadsCritical = controlJson.getInt(
                    "xnnpack_threads_critical"
                ),
            )

            return OrtAutotuneProfile(
                tuningId = tuningId,
                bundleId = root.getString("bundle_id"),
                architectureFingerprint = root.getString(
                    "architecture_fingerprint"
                ),
                modelProfile = root.getString("profile"),
                e3ReceiptId = root.getString("e3_receipt_id"),
                deviceKey = root.getString("device_key"),
                device = device,
                graphPolicies = policies,
                preferredChunkSizes = preferredChunks,
                control = control,
            )
        }
    }
}

data class OrtTunedInvocation(
    val graphFilename: String,
    val sequenceLength: Int,
    val providers: List<OrtProviderKind>,
    val xnnpackThreads: Int,
) {
    init {
        require(graphFilename.isNotBlank())
        require(sequenceLength > 0)
        require(providers.isNotEmpty())
        require(providers.last() == OrtProviderKind.CPU)
        require(xnnpackThreads > 0)
    }
}

data class OrtAutotunedDecision(
    val mode: OrtAdaptiveMode,
    val invocations: List<OrtTunedInvocation>,
    val thermalBand: OrtThermalBand,
    val memoryBand: OrtMemoryBand,
    val latencyBand: OrtLatencyBand,
    val reason: String,
) {
    init {
        require(invocations.isNotEmpty())
    }
}

class OrtProviderQuarantine(
    private val failureThreshold: Int,
    private val cooldownDecisions: Int,
) {
    private val failures = mutableMapOf<OrtProviderKind, Int>()
    private val quarantine = mutableMapOf<OrtProviderKind, Int>()

    init {
        require(failureThreshold > 0)
        require(cooldownDecisions > 0)
    }

    fun tick() {
        val keys = quarantine.keys.toList()
        for (provider in keys) {
            val next = (quarantine[provider] ?: 0) - 1
            if (next <= 0) {
                quarantine.remove(provider)
                failures.remove(provider)
            } else {
                quarantine[provider] = next
            }
        }
    }

    fun recordFailure(provider: OrtProviderKind) {
        if (provider == OrtProviderKind.CPU) {
            return
        }
        val count = (failures[provider] ?: 0) + 1
        failures[provider] = count
        if (count >= failureThreshold) {
            // decide() decrements at entry, so +1 preserves exactly the
            // configured number of subsequent scheduler decisions.
            quarantine[provider] = cooldownDecisions + 1
        }
    }

    fun recordSuccess(provider: OrtProviderKind) {
        failures.remove(provider)
        quarantine.remove(provider)
    }

    fun isQuarantined(provider: OrtProviderKind): Boolean =
        (quarantine[provider] ?: 0) > 0

    fun snapshot(): Set<OrtProviderKind> = quarantine
        .filterValues { it > 0 }
        .keys
        .toSet()
}

class VN97OrtAutotuner(
    private val profile: OrtAutotuneProfile,
    private val device: OrtDeviceCapabilities,
    expectedBundleId: String,
) {
    private var thermalBand = OrtThermalBand.NORMAL
    private var memoryBand = OrtMemoryBand.NORMAL
    private var latencyBand = OrtLatencyBand.NORMAL
    private val quarantine = OrtProviderQuarantine(
        profile.control.quarantineFailureThreshold,
        profile.control.quarantineCooldownDecisions,
    )

    init {
        profile.requireCompatible(expectedBundleId, device)
    }

    fun recordProviderFailure(provider: OrtProviderKind) {
        quarantine.recordFailure(provider)
    }

    fun recordProviderSuccess(provider: OrtProviderKind) {
        quarantine.recordSuccess(provider)
    }

    fun decide(
        workload: OrtAdaptiveWorkload,
        telemetry: OrtAdaptiveTelemetry = OrtAdaptiveTelemetry(),
    ): OrtAutotunedDecision {
        quarantine.tick()
        val thermal = telemetry.thermalStatus ?: device.thermalStatus
        val memory = telemetry.availableMemoryBytes ?:
            device.availableMemoryBytes
        updateThermalBand(thermal)
        updateMemoryBand(memory)

        val preliminary = buildInvocationPlan(
            workload,
            applyLatencyPressure = false,
        )
        val first = preliminary.first()
        updateLatencyBand(
            movingLatencyMs = telemetry.movingLatencyMs,
            expectedP95Nanos = profile.graphPolicies.getValue(
                first.graphFilename
            ).expectedP95Nanos,
        )
        val invocations = if (latencyBand == OrtLatencyBand.SLOW) {
            buildInvocationPlan(
                workload,
                applyLatencyPressure = true,
            )
        } else {
            preliminary
        }
        val mode = when {
            invocations.all { it.sequenceLength == 1 } ->
                OrtAdaptiveMode.SEQUENTIAL
            invocations.size == 1 &&
                invocations.first().sequenceLength ==
                workload.sequenceLength ->
                OrtAdaptiveMode.PARALLEL
            else -> OrtAdaptiveMode.HYBRID
        }
        return OrtAutotunedDecision(
            mode = mode,
            invocations = invocations,
            thermalBand = thermalBand,
            memoryBand = memoryBand,
            latencyBand = latencyBand,
            reason = buildReason(),
        )
    }

    private fun updateThermalBand(status: Int) {
        val control = profile.control
        thermalBand = when (thermalBand) {
            OrtThermalBand.NORMAL -> when {
                status >= control.thermalCriticalEnter ->
                    OrtThermalBand.CRITICAL
                status >= control.thermalHotEnter ->
                    OrtThermalBand.HOT
                else -> OrtThermalBand.NORMAL
            }
            OrtThermalBand.HOT -> when {
                status >= control.thermalCriticalEnter ->
                    OrtThermalBand.CRITICAL
                status <= control.thermalHotExit ->
                    OrtThermalBand.NORMAL
                else -> OrtThermalBand.HOT
            }
            OrtThermalBand.CRITICAL -> when {
                status <= control.thermalHotExit ->
                    OrtThermalBand.NORMAL
                status <= control.thermalCriticalExit ->
                    OrtThermalBand.HOT
                else -> OrtThermalBand.CRITICAL
            }
        }
    }

    private fun updateMemoryBand(availableBytes: Long) {
        val baseline = profile.device.profiledAvailableMemoryBytes
        val ppm = (
            availableBytes.coerceAtMost(baseline).toDouble() /
                baseline.toDouble() * 1_000_000.0
            ).toInt()
        memoryBand = when (memoryBand) {
            OrtMemoryBand.NORMAL -> {
                if (
                    ppm <= profile.control.memoryPressureEnterPpm
                ) {
                    OrtMemoryBand.PRESSURE
                } else {
                    OrtMemoryBand.NORMAL
                }
            }
            OrtMemoryBand.PRESSURE -> {
                if (
                    ppm >= profile.control.memoryPressureExitPpm
                ) {
                    OrtMemoryBand.NORMAL
                } else {
                    OrtMemoryBand.PRESSURE
                }
            }
        }
    }

    private fun updateLatencyBand(
        movingLatencyMs: Double?,
        expectedP95Nanos: Long,
    ) {
        if (movingLatencyMs == null) {
            return
        }
        val observedNanos = (movingLatencyMs * 1_000_000.0)
            .toLong()
            .coerceAtLeast(1L)
        val ppm = (
            observedNanos.toDouble() /
                expectedP95Nanos.toDouble() * 1_000_000.0
            ).coerceAtMost(Int.MAX_VALUE.toDouble()).toInt()
        latencyBand = when (latencyBand) {
            OrtLatencyBand.NORMAL -> {
                if (ppm >= profile.control.latencySlowEnterPpm) {
                    OrtLatencyBand.SLOW
                } else {
                    OrtLatencyBand.NORMAL
                }
            }
            OrtLatencyBand.SLOW -> {
                if (ppm <= profile.control.latencySlowExitPpm) {
                    OrtLatencyBand.NORMAL
                } else {
                    OrtLatencyBand.SLOW
                }
            }
        }
    }

    private fun buildInvocationPlan(
        workload: OrtAdaptiveWorkload,
        applyLatencyPressure: Boolean,
    ): List<OrtTunedInvocation> {
        val forceStep = workload.realtime ||
            workload.sequenceLength <= 4 ||
            workload.stateDependency >= 0.85
        if (forceStep) {
            return List(workload.sequenceLength) {
                invocationFor("step.onnx", 1)
            }
        }

        val sortedSizes = profile.preferredChunkSizes.sorted()
        val maximumAllowed = when {
            thermalBand == OrtThermalBand.CRITICAL ||
                memoryBand == OrtMemoryBand.PRESSURE ->
                sortedSizes.first()
            thermalBand == OrtThermalBand.HOT ||
                applyLatencyPressure ->
                sortedSizes[(sortedSizes.size - 1) / 2]
            else -> sortedSizes.last()
        }
        val candidates = profile.preferredChunkSizes.filter {
            it <= maximumAllowed
        }
        require(candidates.isNotEmpty())

        var remaining = workload.sequenceLength
        val output = mutableListOf<OrtTunedInvocation>()
        while (remaining > 0) {
            val selected = candidates.firstOrNull {
                it <= remaining
            }
            if (selected == null) {
                output += invocationFor("step.onnx", 1)
                remaining -= 1
            } else {
                output += invocationFor(
                    "chunk-" + selected + ".onnx",
                    selected,
                )
                remaining -= selected
            }
        }
        return output
    }

    private fun invocationFor(
        graphFilename: String,
        sequenceLength: Int,
    ): OrtTunedInvocation {
        val policy = profile.graphPolicies.getValue(graphFilename)
        val filtered = policy.providerOrder.filter {
            !quarantine.isQuarantined(it) &&
                providerAvailable(it)
        }.toMutableList()
        if (OrtProviderKind.CPU !in filtered) {
            filtered += OrtProviderKind.CPU
        }
        val threads = when (thermalBand) {
            OrtThermalBand.NORMAL ->
                profile.control.xnnpackThreadsNormal
            OrtThermalBand.HOT ->
                profile.control.xnnpackThreadsHot
            OrtThermalBand.CRITICAL ->
                profile.control.xnnpackThreadsCritical
        }
        return OrtTunedInvocation(
            graphFilename = graphFilename,
            sequenceLength = sequenceLength,
            providers = filtered,
            xnnpackThreads = threads,
        )
    }

    private fun providerAvailable(provider: OrtProviderKind): Boolean =
        when (provider) {
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

    private fun buildReason(): String =
        "e4_measured_profile" +
            "_thermal_" + thermalBand.name.lowercase() +
            "_memory_" + memoryBand.name.lowercase() +
            "_latency_" + latencyBand.name.lowercase() +
            "_quarantine_" + quarantine.snapshot()
                .map { it.name }
                .sorted()
                .joinToString("-")
}

private fun requireSha256E4(value: String, label: String) {
    require(
        value.length == 64 &&
            value.all {
                it in '0'..'9' || it in 'a'..'f'
            }
    ) {
        label + " must be lowercase SHA-256"
    }
}

private fun sha256E4(bytes: ByteArray): String =
    MessageDigest.getInstance("SHA-256")
        .digest(bytes)
        .joinToString("") { "%02x".format(it) }

private fun canonicalJsonE4(value: Any?): String {
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
                    canonicalJsonE4(value.get(key))
            }
        }
        is JSONArray -> {
            (0 until value.length()).joinToString(
                prefix = "[",
                postfix = "]",
                separator = ",",
            ) { index ->
                canonicalJsonE4(value.get(index))
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
