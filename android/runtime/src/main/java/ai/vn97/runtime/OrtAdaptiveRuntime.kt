package ai.vn97.runtime

import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtException
import ai.onnxruntime.OrtSession
import ai.onnxruntime.providers.NNAPIFlags
import java.util.EnumSet

enum class OrtAdaptiveMode {
    SEQUENTIAL,
    HYBRID,
    PARALLEL,
}

enum class OrtProviderKind {
    QNN,
    NNAPI,
    XNNPACK,
    CPU,
}

data class OrtDeviceCapabilities(
    val sdkInt: Int,
    val logicalCores: Int,
    val availableMemoryBytes: Long,
    val thermalStatus: Int = 0,
    val hardware: String = "",
    val socManufacturer: String = "",
    val socModel: String = "",
    val nnapiAvailable: Boolean = true,
    val xnnpackAvailable: Boolean = true,
    val qnnAvailable: Boolean = false,
    val qnnBackendPath: String? = null,
) {
    init {
        require(sdkInt >= 26) { "VN97 requires Android API 26+" }
        require(logicalCores > 0) { "logicalCores must be positive" }
        require(availableMemoryBytes > 0L) {
            "availableMemoryBytes must be positive"
        }
        require(thermalStatus in 0..6) {
            "thermalStatus must be in [0, 6]"
        }
        require(qnnBackendPath == null || qnnBackendPath.isNotBlank()) {
            "qnnBackendPath must be null or non-blank"
        }
    }

    val isQualcomm: Boolean
        get() {
            val joined = "$hardware $socManufacturer $socModel".lowercase()
            return listOf("qualcomm", "snapdragon", "qcom", "sm8350").any {
                joined.contains(it)
            }
        }
}

data class OrtAdaptiveWorkload(
    val sequenceLength: Int,
    val deadlineMs: Double = 250.0,
    val realtime: Boolean = false,
    val stateDependency: Double = 0.5,
    val preferAccelerator: Boolean = true,
) {
    init {
        require(sequenceLength > 0) { "sequenceLength must be positive" }
        require(deadlineMs.isFinite() && deadlineMs > 0.0) {
            "deadlineMs must be finite and positive"
        }
        require(stateDependency in 0.0..1.0) {
            "stateDependency must be in [0, 1]"
        }
    }
}

data class OrtAdaptiveTelemetry(
    val movingLatencyMs: Double? = null,
    val failedProviders: Set<OrtProviderKind> = emptySet(),
    val thermalStatus: Int? = null,
    val availableMemoryBytes: Long? = null,
) {
    init {
        require(
            movingLatencyMs == null ||
                (movingLatencyMs.isFinite() && movingLatencyMs > 0.0)
        ) { "movingLatencyMs must be finite and positive" }
        require(thermalStatus == null || thermalStatus in 0..6) {
            "thermalStatus must be in [0, 6]"
        }
        require(
            availableMemoryBytes == null || availableMemoryBytes > 0L
        ) { "availableMemoryBytes must be positive" }
    }
}

data class OrtAdaptiveDecision(
    val mode: OrtAdaptiveMode,
    val chunkSizes: IntArray,
    val providers: List<OrtProviderKind>,
    val xnnpackThreads: Int,
    val reason: String,
) {
    init {
        require(chunkSizes.isNotEmpty()) { "chunkSizes must not be empty" }
        require(chunkSizes.all { it > 0 }) { "chunkSizes must be positive" }
        require(providers.isNotEmpty()) { "providers must not be empty" }
        require(providers.last() == OrtProviderKind.CPU) {
            "CPU must remain final fallback"
        }
        require(xnnpackThreads > 0) { "xnnpackThreads must be positive" }
    }
}

object OrtAdaptiveScheduler {
    private const val LOW_MEMORY = 512L * 1024L * 1024L
    private const val COMFORTABLE_MEMORY = 1536L * 1024L * 1024L

    fun decide(
        device: OrtDeviceCapabilities,
        workload: OrtAdaptiveWorkload,
        telemetry: OrtAdaptiveTelemetry = OrtAdaptiveTelemetry(),
    ): OrtAdaptiveDecision {
        val thermal = telemetry.thermalStatus ?: device.thermalStatus
        val memory = telemetry.availableMemoryBytes ?: device.availableMemoryBytes
        val failed = telemetry.failedProviders

        val providers = mutableListOf<OrtProviderKind>()
        if (workload.preferAccelerator) {
            if (
                device.qnnAvailable &&
                device.qnnBackendPath != null &&
                device.isQualcomm &&
                OrtProviderKind.QNN !in failed
            ) {
                providers += OrtProviderKind.QNN
            }
            if (
                device.nnapiAvailable &&
                device.sdkInt >= 27 &&
                OrtProviderKind.NNAPI !in failed
            ) {
                providers += OrtProviderKind.NNAPI
            }
            if (
                device.xnnpackAvailable &&
                OrtProviderKind.XNNPACK !in failed
            ) {
                providers += OrtProviderKind.XNNPACK
            }
        } else if (
            device.xnnpackAvailable &&
            OrtProviderKind.XNNPACK !in failed
        ) {
            providers += OrtProviderKind.XNNPACK
        }
        providers += OrtProviderKind.CPU

        var threads = minOf(4, device.logicalCores).coerceAtLeast(1)
        if (thermal >= 4) threads = minOf(2, threads)

        val latencyPressure = telemetry.movingLatencyMs?.let {
            it > workload.deadlineMs
        } ?: false
        val stronglyRecurrent = workload.stateDependency >= 0.85
        if (workload.realtime || workload.sequenceLength <= 4 || stronglyRecurrent) {
            return OrtAdaptiveDecision(
                mode = OrtAdaptiveMode.SEQUENTIAL,
                chunkSizes = IntArray(workload.sequenceLength) { 1 },
                providers = providers,
                xnnpackThreads = threads,
                reason = "realtime_or_high_state_dependency_selects_recurrent_path",
            )
        }

        var maximum: Int
        var reason: String
        when {
            memory < LOW_MEMORY || thermal >= 5 -> {
                maximum = 8
                reason = "memory_or_thermal_pressure_selects_small_hybrid_chunks"
            }
            workload.sequenceLength < 128 || thermal >= 3 -> {
                maximum = 32
                reason = "moderate_sequence_or_thermal_load_selects_hybrid"
            }
            memory >= COMFORTABLE_MEMORY &&
                workload.sequenceLength >= 256 &&
                workload.stateDependency <= 0.35 &&
                !latencyPressure -> {
                return OrtAdaptiveDecision(
                    mode = OrtAdaptiveMode.PARALLEL,
                    chunkSizes = intArrayOf(workload.sequenceLength),
                    providers = providers,
                    xnnpackThreads = threads,
                    reason = "long_low_dependency_prefill_selects_parallel_scan",
                )
            }
            else -> {
                maximum = 64
                reason = "default_variable_hybrid_scan"
            }
        }
        if (latencyPressure) {
            maximum = maxOf(8, maximum / 2)
            reason += "_latency_feedback_reduces_chunk"
        }
        return OrtAdaptiveDecision(
            mode = OrtAdaptiveMode.HYBRID,
            chunkSizes = variableChunks(
                workload.sequenceLength,
                maximum,
            ),
            providers = providers,
            xnnpackThreads = threads,
            reason = reason,
        )
    }

    internal fun variableChunks(
        sequenceLength: Int,
        maximumChunkSize: Int,
        firstChunkSize: Int = minOf(8, maximumChunkSize),
    ): IntArray {
        require(sequenceLength > 0)
        require(maximumChunkSize > 0)
        require(firstChunkSize in 1..maximumChunkSize)
        var remaining = sequenceLength
        var current = minOf(firstChunkSize, remaining)
        val chunks = mutableListOf<Int>()
        while (remaining > 0) {
            val take = minOf(current, remaining)
            chunks += take
            remaining -= take
            current = minOf(maximumChunkSize, current * 2)
        }
        return chunks.toIntArray()
    }
}

class OrtSessionHandle(
    val session: OrtSession,
    val primaryProvider: OrtProviderKind,
    private val sessionOptions: OrtSession.SessionOptions,
) : AutoCloseable {
    override fun close() {
        try {
            session.close()
        } finally {
            sessionOptions.close()
        }
    }
}

class VN97OrtSessionFactory(
    private val environment: OrtEnvironment = OrtEnvironment.getEnvironment(),
) {
    fun create(
        modelBytes: ByteArray,
        decision: OrtAdaptiveDecision,
        device: OrtDeviceCapabilities,
    ): OrtSessionHandle {
        require(modelBytes.isNotEmpty()) { "ONNX model must not be empty" }

        var lastFailure: Throwable? = null
        for (provider in decision.providers) {
            try {
                return createForProvider(
                    modelBytes = modelBytes,
                    provider = provider,
                    xnnpackThreads = decision.xnnpackThreads,
                    device = device,
                )
            } catch (error: OrtException) {
                lastFailure = error
            } catch (error: UnsatisfiedLinkError) {
                lastFailure = error
            } catch (error: IllegalArgumentException) {
                lastFailure = error
            }
        }
        throw IllegalStateException(
            "unable to create VN97 ONNX Runtime session",
            lastFailure,
        )
    }

    fun create(
        modelPath: String,
        decision: OrtAdaptiveDecision,
        device: OrtDeviceCapabilities,
    ): OrtSessionHandle {
        require(modelPath.isNotBlank()) {
            "ONNX model path must not be blank"
        }

        var lastFailure: Throwable? = null
        for (provider in decision.providers) {
            try {
                return createForProvider(
                    modelPath = modelPath,
                    provider = provider,
                    xnnpackThreads = decision.xnnpackThreads,
                    device = device,
                )
            } catch (error: OrtException) {
                lastFailure = error
            } catch (error: UnsatisfiedLinkError) {
                lastFailure = error
            } catch (error: IllegalArgumentException) {
                lastFailure = error
            }
        }
        throw IllegalStateException(
            "unable to create VN97 ONNX Runtime session",
            lastFailure,
        )
    }

    fun createForProvider(
        modelBytes: ByteArray,
        provider: OrtProviderKind,
        xnnpackThreads: Int,
        device: OrtDeviceCapabilities,
    ): OrtSessionHandle {
        require(modelBytes.isNotEmpty()) { "ONNX model must not be empty" }
        require(xnnpackThreads > 0) {
            "xnnpackThreads must be positive"
        }
        val options = sessionOptionsFor(
            provider = provider,
            xnnpackThreads = xnnpackThreads,
            device = device,
        )
        try {
            val session = environment.createSession(
                modelBytes,
                options,
            )
            return OrtSessionHandle(
                session,
                provider,
                options,
            )
        } catch (error: Throwable) {
            options.close()
            throw error
        }
    }

    fun createForProvider(
        modelPath: String,
        provider: OrtProviderKind,
        xnnpackThreads: Int,
        device: OrtDeviceCapabilities,
    ): OrtSessionHandle {
        require(modelPath.isNotBlank()) {
            "ONNX model path must not be blank"
        }
        require(xnnpackThreads > 0) {
            "xnnpackThreads must be positive"
        }
        val options = sessionOptionsFor(
            provider = provider,
            xnnpackThreads = xnnpackThreads,
            device = device,
        )
        try {
            val session = environment.createSession(
                modelPath,
                options,
            )
            return OrtSessionHandle(
                session,
                provider,
                options,
            )
        } catch (error: Throwable) {
            options.close()
            throw error
        }
    }

    private fun sessionOptionsFor(
        provider: OrtProviderKind,
        xnnpackThreads: Int,
        device: OrtDeviceCapabilities,
    ): OrtSession.SessionOptions {
        val options = OrtSession.SessionOptions()
        try {
            options.setOptimizationLevel(
                OrtSession.SessionOptions.OptLevel.ALL_OPT,
            )
            options.setIntraOpNumThreads(1)
            options.addConfigEntry(
                "session.intra_op.allow_spinning",
                "0",
            )
            when (provider) {
                OrtProviderKind.QNN -> {
                    require(device.isQualcomm) {
                        "QNN may only be selected on Qualcomm hardware"
                    }
                    val backendPath = requireNotNull(
                        device.qnnBackendPath
                    ) {
                        "QNN requires an explicitly packaged backend path"
                    }
                    options.addQnn(
                        mapOf(
                            "backend_path" to backendPath,
                            "profiling_level" to "off",
                        )
                    )
                }
                OrtProviderKind.NNAPI -> {
                    require(device.nnapiAvailable && device.sdkInt >= 27) {
                        "NNAPI is not available on this device"
                    }
                    options.addNnapi(
                        EnumSet.of(NNAPIFlags.CPU_DISABLED)
                    )
                }
                OrtProviderKind.XNNPACK -> {
                    require(device.xnnpackAvailable) {
                        "XNNPACK is not available on this device"
                    }
                    options.addXnnpack(
                        mapOf(
                            "intra_op_num_threads" to
                                xnnpackThreads.toString()
                        )
                    )
                }
                OrtProviderKind.CPU -> Unit
            }
            return options
        } catch (error: Throwable) {
            options.close()
            throw error
        }
    }
}

