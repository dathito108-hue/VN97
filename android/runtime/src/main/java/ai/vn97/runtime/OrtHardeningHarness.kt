package ai.vn97.runtime

import android.content.Context
import android.os.Build
import java.io.File
import java.security.MessageDigest
import org.json.JSONArray
import org.json.JSONObject

data class OrtHardeningExecutionResult(
    val graphFilename: String,
    val provider: OrtProviderKind,
    val elapsedNanos: Long,
    val success: Boolean,
) {
    init {
        require(graphFilename.isNotBlank())
        require(elapsedNanos > 0L)
    }
}

interface OrtHardeningExecutor {
    fun resetState()
    fun execute(
        invocation: OrtTunedInvocation,
    ): OrtHardeningExecutionResult
}

data class OrtHardeningPhaseSpec(
    val name: String,
    val kind: String,
    val workload: OrtAdaptiveWorkload,
    val iterations: Int,
    val sleepBeforeMillis: Long = 0L,
) {
    init {
        require(name.isNotBlank())
        require(
            kind in setOf(
                "cold",
                "warm_step",
                "prefill",
                "sustained",
                "recovery",
            )
        )
        require(iterations > 0)
        require(sleepBeforeMillis >= 0L)
    }
}

data class OrtHardeningPhaseResult(
    val name: String,
    val kind: String,
    val iterations: Int,
    val tokensPerIteration: Int,
    val latenciesNanos: List<Long>,
    val providerCounts: Map<String, Int>,
    val graphCounts: Map<String, Int>,
    val thermalBefore: Int,
    val thermalAfter: Int,
    val memoryBeforeBytes: Long,
    val memoryAfterBytes: Long,
    val failureCount: Int,
) {
    init {
        require(name.isNotBlank())
        require(iterations > 0)
        require(tokensPerIteration > 0)
        require(latenciesNanos.size == iterations)
        require(latenciesNanos.all { it > 0L })
        require(providerCounts.isNotEmpty())
        require(graphCounts.isNotEmpty())
        require(thermalBefore in 0..6)
        require(thermalAfter in 0..6)
        require(memoryBeforeBytes > 0L)
        require(memoryAfterBytes > 0L)
        require(failureCount >= 0)
    }

    fun toJson(): JSONObject {
        val providers = JSONObject()
        providerCounts.toSortedMap().forEach { (key, value) ->
            providers.put(key, value)
        }
        val graphs = JSONObject()
        graphCounts.toSortedMap().forEach { (key, value) ->
            graphs.put(key, value)
        }
        val latencies = JSONArray()
        latenciesNanos.forEach { latencies.put(it) }
        return JSONObject()
            .put("name", name)
            .put("kind", kind)
            .put("iterations", iterations)
            .put("tokens_per_iteration", tokensPerIteration)
            .put("latencies_ns", latencies)
            .put("provider_counts", providers)
            .put("graph_counts", graphs)
            .put("thermal_before", thermalBefore)
            .put("thermal_after", thermalAfter)
            .put("memory_before_bytes", memoryBeforeBytes)
            .put("memory_after_bytes", memoryAfterBytes)
            .put("failure_count", failureCount)
    }
}

data class OrtHardeningControlResult(
    val thermalHysteresisPassed: Boolean,
    val memoryPressurePassed: Boolean,
    val providerQuarantinePassed: Boolean,
    val providerRecoveryPassed: Boolean,
    val cpuFallbackPassed: Boolean,
) {
    fun toJson(): JSONObject =
        JSONObject()
            .put(
                "thermal_hysteresis_passed",
                thermalHysteresisPassed,
            )
            .put(
                "memory_pressure_passed",
                memoryPressurePassed,
            )
            .put(
                "provider_quarantine_passed",
                providerQuarantinePassed,
            )
            .put(
                "provider_recovery_passed",
                providerRecoveryPassed,
            )
            .put(
                "cpu_fallback_passed",
                cpuFallbackPassed,
            )
}

class VN97OrtHardeningHarness(
    private val context: Context,
    private val profile: OrtAutotuneProfile,
    private val expectedBundleId: String,
    private val executor: OrtHardeningExecutor,
    private val qnnBackendPath: String? = null,
) {
    init {
        profile.requireCompatible(
            expectedBundleId,
            OrtDeviceProbe.inspect(context, qnnBackendPath),
        )
    }

    fun runPhase(
        spec: OrtHardeningPhaseSpec,
    ): OrtHardeningPhaseResult {
        if (spec.sleepBeforeMillis > 0L) {
            Thread.sleep(spec.sleepBeforeMillis)
        }
        if (spec.kind == "cold") {
            executor.resetState()
        }

        val before = OrtDeviceProbe.inspect(
            context,
            qnnBackendPath,
        )
        val autotuner = VN97OrtAutotuner(
            profile,
            before,
            expectedBundleId,
        )
        val latencies = ArrayList<Long>(spec.iterations)
        val providers = mutableMapOf<String, Int>()
        val graphs = mutableMapOf<String, Int>()
        var failures = 0
        var movingLatencyMs: Double? = null

        repeat(spec.iterations) {
            val current = OrtDeviceProbe.inspect(
                context,
                qnnBackendPath,
            )
            val decision = autotuner.decide(
                spec.workload,
                OrtAdaptiveTelemetry(
                    movingLatencyMs = movingLatencyMs,
                    thermalStatus = current.thermalStatus,
                    availableMemoryBytes = current.availableMemoryBytes,
                ),
            )
            val iterationStart = System.nanoTime()
            var invocationFailures = 0
            for (invocation in decision.invocations) {
                val result = executor.execute(invocation)
                graphs[result.graphFilename] =
                    (graphs[result.graphFilename] ?: 0) + 1
                providers[result.provider.name] =
                    (providers[result.provider.name] ?: 0) + 1
                if (result.success) {
                    autotuner.recordProviderSuccess(result.provider)
                } else {
                    invocationFailures += 1
                    autotuner.recordProviderFailure(result.provider)
                }
            }
            val elapsed = (
                System.nanoTime() - iterationStart
            ).coerceAtLeast(1L)
            latencies += elapsed
            failures += invocationFailures
            movingLatencyMs = elapsed / 1_000_000.0
        }

        val after = OrtDeviceProbe.inspect(
            context,
            qnnBackendPath,
        )
        return OrtHardeningPhaseResult(
            name = spec.name,
            kind = spec.kind,
            iterations = spec.iterations,
            tokensPerIteration = spec.workload.sequenceLength,
            latenciesNanos = latencies,
            providerCounts = providers,
            graphCounts = graphs,
            thermalBefore = before.thermalStatus,
            thermalAfter = after.thermalStatus,
            memoryBeforeBytes = before.availableMemoryBytes,
            memoryAfterBytes = after.availableMemoryBytes,
            failureCount = failures,
        )
    }

    fun runControlTests(): OrtHardeningControlResult {
        val device = OrtDeviceProbe.inspect(
            context,
            qnnBackendPath,
        )

        val thermalAutotuner = VN97OrtAutotuner(
            profile,
            device,
            expectedBundleId,
        )
        val probeWorkload = OrtAdaptiveWorkload(
            sequenceLength = maxOf(
                8,
                profile.preferredChunkSizes.maxOrNull() ?: 8,
            ),
            realtime = false,
            stateDependency = 0.2,
        )
        val hot = thermalAutotuner.decide(
            probeWorkload,
            OrtAdaptiveTelemetry(
                thermalStatus = profile.control.thermalHotEnter,
                availableMemoryBytes = device.availableMemoryBytes,
            ),
        )
        val stickyHot = thermalAutotuner.decide(
            probeWorkload,
            OrtAdaptiveTelemetry(
                thermalStatus = (
                    profile.control.thermalHotExit + 1
                ).coerceAtMost(
                    profile.control.thermalHotEnter
                ),
                availableMemoryBytes = device.availableMemoryBytes,
            ),
        )
        val cooled = thermalAutotuner.decide(
            probeWorkload,
            OrtAdaptiveTelemetry(
                thermalStatus = profile.control.thermalHotExit,
                availableMemoryBytes = device.availableMemoryBytes,
            ),
        )
        val thermalPassed = (
            hot.thermalBand == OrtThermalBand.HOT &&
                stickyHot.thermalBand == OrtThermalBand.HOT &&
                cooled.thermalBand == OrtThermalBand.NORMAL
            )

        val memoryAutotuner = VN97OrtAutotuner(
            profile,
            device,
            expectedBundleId,
        )
        val baseline = profile.device.profiledAvailableMemoryBytes
        val pressureMemory = (
            baseline *
                profile.control.memoryPressureEnterPpm.toLong() /
                1_000_000L
            ).coerceAtLeast(1L)
        val pressured = memoryAutotuner.decide(
            probeWorkload,
            OrtAdaptiveTelemetry(
                thermalStatus = 0,
                availableMemoryBytes = pressureMemory,
            ),
        )
        val smallest = profile.preferredChunkSizes.minOrNull()
        val memoryPassed = (
            pressured.memoryBand == OrtMemoryBand.PRESSURE &&
                smallest != null &&
                pressured.invocations.any {
                    it.sequenceLength == smallest
                }
            )

        val accelerator = profile.graphPolicies.values
            .flatMap { it.providerOrder }
            .firstOrNull { it != OrtProviderKind.CPU }
        val quarantinePassed: Boolean
        val recoveryPassed: Boolean
        if (accelerator == null) {
            quarantinePassed = true
            recoveryPassed = true
        } else {
            val quarantineAutotuner = VN97OrtAutotuner(
                profile,
                device,
                expectedBundleId,
            )
            repeat(profile.control.quarantineFailureThreshold) {
                quarantineAutotuner.recordProviderFailure(accelerator)
            }
            val quarantined = quarantineAutotuner.decide(
                probeWorkload
            )
            quarantinePassed = quarantined.invocations.all {
                accelerator !in it.providers
            }
            repeat(
                profile.control.quarantineCooldownDecisions
            ) {
                quarantineAutotuner.decide(probeWorkload)
            }
            val recovered = quarantineAutotuner.decide(
                probeWorkload
            )
            recoveryPassed = recovered.invocations.any {
                accelerator in it.providers
            }
        }

        val cpuFallbackPassed = profile.graphPolicies.values.all {
            it.providerOrder.last() == OrtProviderKind.CPU
        }
        return OrtHardeningControlResult(
            thermalHysteresisPassed = thermalPassed,
            memoryPressurePassed = memoryPassed,
            providerQuarantinePassed = quarantinePassed,
            providerRecoveryPassed = recoveryPassed,
            cpuFallbackPassed = cpuFallbackPassed,
        )
    }

    fun buildRunReport(
        e3ReceiptId: String,
        phases: List<OrtHardeningPhaseResult>,
        controls: OrtHardeningControlResult,
    ): JSONObject {
        requireSha256E5(expectedBundleId, "bundleId")
        requireSha256E5(e3ReceiptId, "e3ReceiptId")
        require(phases.isNotEmpty())

        val phaseArray = JSONArray()
        phases.forEach { phaseArray.put(it.toJson()) }
        val device = OrtDeviceProbe.inspect(
            context,
            qnnBackendPath,
        )
        val body = JSONObject()
            .put("schema", RUN_SCHEMA)
            .put("bundle_id", expectedBundleId)
            .put("e3_receipt_id", e3ReceiptId)
            .put("tuning_id", profile.tuningId)
            .put(
                "device",
                JSONObject()
                    .put(
                        "manufacturer",
                        Build.MANUFACTURER.orEmpty(),
                    )
                    .put("model", Build.MODEL.orEmpty())
                    .put("sdk_int", device.sdkInt)
                    .put("logical_cores", device.logicalCores)
                    .put("hardware", device.hardware)
                    .put(
                        "soc_manufacturer",
                        device.socManufacturer,
                    )
                    .put("soc_model", device.socModel),
            )
            .put("phases", phaseArray)
            .put("control_tests", controls.toJson())
            .put("device_measured", true)
            .put("synthetic", false)

        val canonical = canonicalJsonE5(body)
        val prefix = "VN97R2E5RUN1" +
            String(charArrayOf(0.toChar()))
        val runId = sha256E5(
            prefix.toByteArray(Charsets.UTF_8) +
                canonical.toByteArray(Charsets.UTF_8)
        )
        return JSONObject(canonical).put("run_id", runId)
    }

    fun writeRunReportAtomic(
        target: File,
        e3ReceiptId: String,
        phases: List<OrtHardeningPhaseResult>,
        controls: OrtHardeningControlResult,
    ) {
        val report = buildRunReport(
            e3ReceiptId,
            phases,
            controls,
        )
        target.parentFile?.mkdirs()
        val temp = File(target.parentFile, target.name + ".tmp")
        if (temp.exists()) {
            check(temp.delete())
        }
        temp.writeText(
            canonicalJsonE5(report) + "\n",
            Charsets.UTF_8,
        )
        if (target.exists()) {
            check(target.delete())
        }
        check(temp.renameTo(target))
    }

    companion object {
        const val RUN_SCHEMA = "VN97R2E5RUN1"

        fun recommendedPhases(
            profile: OrtAutotuneProfile,
        ): List<OrtHardeningPhaseSpec> {
            val largest = profile.preferredChunkSizes.maxOrNull() ?: 8
            return listOf(
                OrtHardeningPhaseSpec(
                    name = "cold_start",
                    kind = "cold",
                    workload = OrtAdaptiveWorkload(
                        sequenceLength = 1,
                        realtime = true,
                        stateDependency = 1.0,
                    ),
                    iterations = 1,
                ),
                OrtHardeningPhaseSpec(
                    name = "warm_step",
                    kind = "warm_step",
                    workload = OrtAdaptiveWorkload(
                        sequenceLength = 1,
                        realtime = true,
                        stateDependency = 1.0,
                    ),
                    iterations = 32,
                ),
                OrtHardeningPhaseSpec(
                    name = "prefill",
                    kind = "prefill",
                    workload = OrtAdaptiveWorkload(
                        sequenceLength = maxOf(32, largest * 2),
                        realtime = false,
                        stateDependency = 0.2,
                    ),
                    iterations = 8,
                ),
                OrtHardeningPhaseSpec(
                    name = "sustained",
                    kind = "sustained",
                    workload = OrtAdaptiveWorkload(
                        sequenceLength = maxOf(64, largest * 4),
                        realtime = false,
                        stateDependency = 0.2,
                    ),
                    iterations = 32,
                ),
                OrtHardeningPhaseSpec(
                    name = "recovery",
                    kind = "recovery",
                    workload = OrtAdaptiveWorkload(
                        sequenceLength = 1,
                        realtime = true,
                        stateDependency = 1.0,
                    ),
                    iterations = 8,
                    sleepBeforeMillis = 15_000L,
                ),
            )
        }
    }
}

private fun requireSha256E5(value: String, label: String) {
    require(
        value.length == 64 &&
            value.all {
                it in '0'..'9' || it in 'a'..'f'
            }
    ) {
        label + " must be lowercase SHA-256"
    }
}

private fun sha256E5(bytes: ByteArray): String =
    MessageDigest.getInstance("SHA-256")
        .digest(bytes)
        .joinToString("") { "%02x".format(it) }

private fun canonicalJsonE5(value: Any?): String {
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
                    canonicalJsonE5(value.get(key))
            }
        }
        is JSONArray -> {
            (0 until value.length()).joinToString(
                prefix = "[",
                postfix = "]",
                separator = ",",
            ) { index ->
                canonicalJsonE5(value.get(index))
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
