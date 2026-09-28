package ai.vn97.runtime

import ai.onnxruntime.OrtEnvironment
import android.content.Context
import android.os.Debug
import java.io.File
import org.json.JSONObject

data class Mamba2K2SStepResult(
    val mode: String,
    val provider: OrtProviderKind,
    val cpuFallbackAllowed: Boolean,
    val cpuThreads: Int,
    val xnnpackThreads: Int,
    val sessionCreated: Boolean,
    val runSucceeded: Boolean,
    val sessionCreateNanos: Long,
    val runLatencyNanos: Long?,
    val memoryBeforeBytes: Long,
    val memoryAfterBytes: Long,
    val processPssBeforeBytes: Long?,
    val processPssAfterBytes: Long?,
    val processRssBeforeBytes: Long?,
    val processRssAfterBytes: Long?,
    val thermalBefore: Int,
    val thermalAfter: Int,
    val errorClass: String?,
    val errorMessage: String?,
) {
    fun toJson(): JSONObject =
        JSONObject()
            .put("schema", "VN97M2K2SSTEP1")
            .put("mode", mode)
            .put("provider", provider.name)
            .put("cpu_fallback_allowed", cpuFallbackAllowed)
            .put("cpu_threads", cpuThreads)
            .put("xnnpack_threads", xnnpackThreads)
            .put("session_created", sessionCreated)
            .put("run_succeeded", runSucceeded)
            .put("session_create_ns", sessionCreateNanos)
            .put(
                "run_latency_ns",
                runLatencyNanos ?: JSONObject.NULL,
            )
            .put("memory_before_bytes", memoryBeforeBytes)
            .put("memory_after_bytes", memoryAfterBytes)
            .put(
                "process_pss_before_bytes",
                processPssBeforeBytes ?: JSONObject.NULL,
            )
            .put(
                "process_pss_after_bytes",
                processPssAfterBytes ?: JSONObject.NULL,
            )
            .put(
                "process_rss_before_bytes",
                processRssBeforeBytes ?: JSONObject.NULL,
            )
            .put(
                "process_rss_after_bytes",
                processRssAfterBytes ?: JSONObject.NULL,
            )
            .put("thermal_before", thermalBefore)
            .put("thermal_after", thermalAfter)
            .put("error_class", errorClass ?: JSONObject.NULL)
            .put("error_message", errorMessage ?: JSONObject.NULL)
            .put("device_measured", true)
            .put("synthetic", false)
            .put("same_weights_semantics", true)
            .put("graph_changed", false)
            .put("weights_changed", false)
            .put("production_activation_authorized", false)
}

class VN97Mamba2K2SSafeDiagnostics(
    private val environment: OrtEnvironment =
        OrtEnvironment.getEnvironment(),
    private val sessionFactory: VN97OrtSessionFactory =
        VN97OrtSessionFactory(environment),
) {
    fun runMode(
        context: Context,
        runtimeRoot: File,
        mode: String,
    ): Mamba2K2SStepResult {
        val app = context.applicationContext
        val runtime = Mamba2OrtRuntimePackage.load(runtimeRoot)
        require(runtime.stateDtype == "float16") {
            "K2S requires verified FP16 runtime"
        }
        val device = OrtDeviceProbe.inspect(app)
        val maxThreads = device.logicalCores.coerceAtMost(8)
        val spec = specForMode(mode, maxThreads)

        Mamba2ProfileBuffers(
            environment,
            runtime,
        ).use { buffers ->
            buffers.prepare(1)

            val before = OrtDeviceProbe.inspect(app)
            val pssBefore = processPssBytesK2S()
            val rssBefore = processRssBytesK2S()

            var sessionCreated = false
            var runSucceeded = false
            var createNanos = 1L
            var runNanos: Long? = null
            var errorClass: String? = null
            var errorMessage: String? = null
            val createStart = System.nanoTime()

            try {
                sessionFactory.createForProvider(
                    modelPath = runtime.graphFile.absolutePath,
                    provider = spec.provider,
                    xnnpackThreads = spec.xnnpackThreads,
                    device = device,
                    allowCpuFallback = spec.cpuFallbackAllowed,
                    cpuThreads = spec.cpuThreads,
                ).use { handle ->
                    createNanos = elapsedK2S(createStart)
                    sessionCreated = true
                    val runStart = System.nanoTime()
                    buffers.run(handle.session)
                    runNanos = elapsedK2S(runStart)
                    runSucceeded = true
                }
            } catch (error: Throwable) {
                createNanos = elapsedK2S(createStart)
                errorClass = error::class.java.name.take(160)
                errorMessage = sanitizeK2SError(error.message)
            }

            val after = OrtDeviceProbe.inspect(app)
            return Mamba2K2SStepResult(
                mode = spec.mode,
                provider = spec.provider,
                cpuFallbackAllowed = spec.cpuFallbackAllowed,
                cpuThreads = spec.cpuThreads,
                xnnpackThreads = spec.xnnpackThreads,
                sessionCreated = sessionCreated,
                runSucceeded = runSucceeded,
                sessionCreateNanos = createNanos,
                runLatencyNanos = runNanos,
                memoryBeforeBytes = before.availableMemoryBytes,
                memoryAfterBytes = after.availableMemoryBytes,
                processPssBeforeBytes = pssBefore,
                processPssAfterBytes = processPssBytesK2S(),
                processRssBeforeBytes = rssBefore,
                processRssAfterBytes = processRssBytesK2S(),
                thermalBefore = before.thermalStatus,
                thermalAfter = after.thermalStatus,
                errorClass = errorClass,
                errorMessage = errorMessage,
            )
        }
    }

    companion object {
        val MODES = listOf(
            "CPU_1",
            "CPU_2",
            "CPU_4",
            "CPU_8",
            "XNNPACK_STRICT",
            "XNNPACK_HYBRID",
            "NNAPI_STRICT",
            "NNAPI_HYBRID",
        )

        private fun specForMode(
            mode: String,
            maxThreads: Int,
        ): K2SSpec {
            require(mode in MODES) {
                "unsupported K2S mode: " + mode
            }
            return when (mode) {
                "CPU_1" -> K2SSpec(
                    mode,
                    OrtProviderKind.CPU,
                    false,
                    1,
                    1,
                )
                "CPU_2" -> K2SSpec(
                    mode,
                    OrtProviderKind.CPU,
                    false,
                    minOf(2, maxThreads),
                    1,
                )
                "CPU_4" -> K2SSpec(
                    mode,
                    OrtProviderKind.CPU,
                    false,
                    minOf(4, maxThreads),
                    1,
                )
                "CPU_8" -> K2SSpec(
                    mode,
                    OrtProviderKind.CPU,
                    false,
                    maxThreads,
                    1,
                )
                "XNNPACK_STRICT" -> K2SSpec(
                    mode,
                    OrtProviderKind.XNNPACK,
                    false,
                    1,
                    minOf(4, maxThreads),
                )
                "XNNPACK_HYBRID" -> K2SSpec(
                    mode,
                    OrtProviderKind.XNNPACK,
                    true,
                    minOf(4, maxThreads),
                    minOf(4, maxThreads),
                )
                "NNAPI_STRICT" -> K2SSpec(
                    mode,
                    OrtProviderKind.NNAPI,
                    false,
                    1,
                    1,
                )
                "NNAPI_HYBRID" -> K2SSpec(
                    mode,
                    OrtProviderKind.NNAPI,
                    true,
                    minOf(4, maxThreads),
                    1,
                )
                else -> error("unreachable")
            }
        }
    }
}

private data class K2SSpec(
    val mode: String,
    val provider: OrtProviderKind,
    val cpuFallbackAllowed: Boolean,
    val cpuThreads: Int,
    val xnnpackThreads: Int,
)

private fun processPssBytesK2S(): Long? =
    runCatching {
        Debug.getPss().toLong() * 1024L
    }.getOrNull()?.takeIf { it > 0L }

private fun processRssBytesK2S(): Long? =
    runCatching {
        val line = File("/proc/self/status")
            .useLines { lines ->
                lines.firstOrNull { it.startsWith("VmRSS:") }
            } ?: return@runCatching null
        val kib = line
            .removePrefix("VmRSS:")
            .trim()
            .substringBefore(' ')
            .toLong()
        kib * 1024L
    }.getOrNull()?.takeIf { it > 0L }

private fun sanitizeK2SError(value: String?): String {
    val normalized = value
        .orEmpty()
        .replace(Regex("[\\r\\n\\t]+"), " ")
        .replace(Regex("\\s+"), " ")
        .trim()
    return normalized.take(480).ifBlank { "no_message" }
}

private fun elapsedK2S(start: Long): Long =
    (System.nanoTime() - start).coerceAtLeast(1L)
