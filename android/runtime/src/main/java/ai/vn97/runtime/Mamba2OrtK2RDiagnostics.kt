package ai.vn97.runtime

import ai.onnxruntime.OrtEnvironment
import android.content.Context
import android.os.Debug
import java.io.File
import java.security.MessageDigest
import org.json.JSONArray
import org.json.JSONObject

data class Mamba2K2RAttempt(
    val mode: String,
    val provider: OrtProviderKind,
    val cpuFallbackAllowed: Boolean,
    val cpuThreads: Int,
    val xnnpackThreads: Int,
    val sessionCreated: Boolean,
    val runSucceeded: Boolean,
    val sessionCreateNanos: Long,
    val latenciesNanos: List<Long>,
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
    init {
        require(mode.isNotBlank())
        require(cpuThreads > 0)
        require(xnnpackThreads > 0)
        require(sessionCreateNanos > 0L)
        require(memoryBeforeBytes > 0L)
        require(memoryAfterBytes > 0L)
        require(thermalBefore in 0..6)
        require(thermalAfter in 0..6)
        require(!runSucceeded || sessionCreated)
        require(!runSucceeded || latenciesNanos.isNotEmpty())
        require(latenciesNanos.all { it > 0L })
        require(
            (errorClass == null) == (errorMessage == null)
        )
    }

    fun toJson(): JSONObject {
        val latency = JSONArray()
        latenciesNanos.forEach { latency.put(it) }
        return JSONObject()
            .put("mode", mode)
            .put("provider", provider.name)
            .put("cpu_fallback_allowed", cpuFallbackAllowed)
            .put("cpu_threads", cpuThreads)
            .put("xnnpack_threads", xnnpackThreads)
            .put("session_created", sessionCreated)
            .put("run_succeeded", runSucceeded)
            .put("session_create_ns", sessionCreateNanos)
            .put("latencies_ns", latency)
            .put(
                "latency_p50_ns",
                if (latenciesNanos.isEmpty()) {
                    JSONObject.NULL
                } else {
                    percentileK2R(latenciesNanos, 0.50)
                },
            )
            .put(
                "latency_p95_ns",
                if (latenciesNanos.isEmpty()) {
                    JSONObject.NULL
                } else {
                    percentileK2R(latenciesNanos, 0.95)
                },
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
    }
}

data class Mamba2K2RReport(
    val runtimeId: String,
    val g05ManifestId: String,
    val device: OrtDeviceCapabilities,
    val attempts: List<Mamba2K2RAttempt>,
) {
    init {
        requireK2RSha(runtimeId)
        requireK2RSha(g05ManifestId)
        require(attempts.isNotEmpty())
        require(attempts.map { it.mode }.distinct().size == attempts.size)
    }

    fun toJson(): JSONObject {
        val attemptArray = JSONArray()
        attempts.forEach { attemptArray.put(it.toJson()) }
        val body = JSONObject()
            .put("schema", SCHEMA)
            .put("runtime_id", runtimeId)
            .put("g05_manifest_id", g05ManifestId)
            .put(
                "device",
                JSONObject()
                    .put("sdk_int", device.sdkInt)
                    .put("logical_cores", device.logicalCores)
                    .put("hardware", device.hardware)
                    .put("soc_manufacturer", device.socManufacturer)
                    .put("soc_model", device.socModel)
                    .put(
                        "profiled_available_memory_bytes",
                        device.availableMemoryBytes,
                    ),
            )
            .put("attempts", attemptArray)
            .put("device_measured", true)
            .put("synthetic", false)
            .put("same_weights_semantics", true)
            .put("graph_changed", false)
            .put("weights_changed", false)
            .put("production_activation_authorized", false)

        val canonical = canonicalK2R(body)
        val receipt = sha256K2R(
            ("VN97M2K2RDIAG1" + 0.toChar())
                .toByteArray(Charsets.US_ASCII) +
                canonical.toByteArray(Charsets.US_ASCII)
        )
        return JSONObject(canonical)
            .put("receipt_id", receipt)
    }

    fun writeAtomic(target: File) {
        target.parentFile?.mkdirs()
        val temp = File(target.parentFile, target.name + ".tmp")
        if (temp.exists()) check(temp.delete())
        temp.writeText(
            canonicalK2R(toJson()) + "\n",
            Charsets.US_ASCII,
        )
        if (target.exists()) check(target.delete())
        check(temp.renameTo(target))
    }

    companion object {
        const val SCHEMA = "VN97M2K2RDIAG1"
    }
}

class VN97Mamba2K2RDiagnostics(
    private val environment: OrtEnvironment =
        OrtEnvironment.getEnvironment(),
    private val sessionFactory: VN97OrtSessionFactory =
        VN97OrtSessionFactory(environment),
) {
    fun run(
        context: Context,
        runtimeRoot: File,
    ): Mamba2K2RReport {
        val app = context.applicationContext
        val runtime = Mamba2OrtRuntimePackage.load(runtimeRoot)
        require(runtime.stateDtype == "float16") {
            "K2R diagnostics require the verified FP16 G0.5 runtime"
        }
        val device = OrtDeviceProbe.inspect(app)
        val maxThreads = device.logicalCores.coerceAtMost(8)
        val cpuThreadCounts = listOf(1, 2, 4, maxThreads)
            .filter { it <= maxThreads }
            .distinct()

        val specs = buildList {
            if (device.nnapiAvailable && device.sdkInt >= 27) {
                add(
                    K2RSpec(
                        "NNAPI_STRICT",
                        OrtProviderKind.NNAPI,
                        false,
                        1,
                        1,
                        0,
                        1,
                    )
                )
                add(
                    K2RSpec(
                        "NNAPI_HYBRID",
                        OrtProviderKind.NNAPI,
                        true,
                        minOf(4, maxThreads),
                        1,
                        0,
                        1,
                    )
                )
            }
            if (device.xnnpackAvailable) {
                add(
                    K2RSpec(
                        "XNNPACK_STRICT",
                        OrtProviderKind.XNNPACK,
                        false,
                        1,
                        minOf(4, maxThreads),
                        0,
                        1,
                    )
                )
                add(
                    K2RSpec(
                        "XNNPACK_HYBRID",
                        OrtProviderKind.XNNPACK,
                        true,
                        minOf(4, maxThreads),
                        minOf(4, maxThreads),
                        0,
                        1,
                    )
                )
            }
            cpuThreadCounts.forEach { threads ->
                add(
                    K2RSpec(
                        "CPU_" + threads,
                        OrtProviderKind.CPU,
                        false,
                        threads,
                        1,
                        1,
                        3,
                    )
                )
            }
        }

        val attempts = ArrayList<Mamba2K2RAttempt>(specs.size)
        Mamba2ProfileBuffers(
            environment,
            runtime,
        ).use { buffers ->
            for (spec in specs) {
                attempts += runSpec(
                    app,
                    runtime,
                    device,
                    buffers,
                    spec,
                )
            }
        }
        return Mamba2K2RReport(
            runtimeId = runtime.runtimeId,
            g05ManifestId = runtime.g05ManifestId,
            device = device,
            attempts = attempts,
        )
    }

    private fun runSpec(
        context: Context,
        runtime: Mamba2OrtRuntimePackage,
        device: OrtDeviceCapabilities,
        buffers: Mamba2ProfileBuffers,
        spec: K2RSpec,
    ): Mamba2K2RAttempt {
        buffers.prepare(1)
        val before = OrtDeviceProbe.inspect(context)
        val pssBefore = processPssBytesK2R()
        val rssBefore = processRssBytesK2R()
        var sessionCreated = false
        var runSucceeded = false
        var errorClass: String? = null
        var errorMessage: String? = null
        val samples = mutableListOf<Long>()
        val createStart = System.nanoTime()
        var createNanos = 1L

        try {
            sessionFactory.createForProvider(
                modelPath = runtime.graphFile.absolutePath,
                provider = spec.provider,
                xnnpackThreads = spec.xnnpackThreads,
                device = device,
                allowCpuFallback = spec.cpuFallbackAllowed,
                cpuThreads = spec.cpuThreads,
            ).use { handle ->
                createNanos = elapsedK2R(createStart)
                sessionCreated = true
                repeat(spec.warmups) {
                    buffers.run(handle.session)
                }
                repeat(spec.steadyRuns) {
                    val start = System.nanoTime()
                    buffers.run(handle.session)
                    samples += elapsedK2R(start)
                }
                runSucceeded = true
            }
        } catch (error: Throwable) {
            createNanos = elapsedK2R(createStart)
            errorClass = error::class.java.name.take(160)
            errorMessage = sanitizeK2RError(error.message)
        }

        val after = OrtDeviceProbe.inspect(context)
        return Mamba2K2RAttempt(
            mode = spec.mode,
            provider = spec.provider,
            cpuFallbackAllowed = spec.cpuFallbackAllowed,
            cpuThreads = spec.cpuThreads,
            xnnpackThreads = spec.xnnpackThreads,
            sessionCreated = sessionCreated,
            runSucceeded = runSucceeded,
            sessionCreateNanos = createNanos,
            latenciesNanos = samples.toList(),
            memoryBeforeBytes = before.availableMemoryBytes,
            memoryAfterBytes = after.availableMemoryBytes,
            processPssBeforeBytes = pssBefore,
            processPssAfterBytes = processPssBytesK2R(),
            processRssBeforeBytes = rssBefore,
            processRssAfterBytes = processRssBytesK2R(),
            thermalBefore = before.thermalStatus,
            thermalAfter = after.thermalStatus,
            errorClass = errorClass,
            errorMessage = errorMessage,
        )
    }
}

private data class K2RSpec(
    val mode: String,
    val provider: OrtProviderKind,
    val cpuFallbackAllowed: Boolean,
    val cpuThreads: Int,
    val xnnpackThreads: Int,
    val warmups: Int,
    val steadyRuns: Int,
)

private fun processPssBytesK2R(): Long? =
    runCatching {
        Debug.getPss().toLong() * 1024L
    }.getOrNull()?.takeIf { it > 0L }

private fun processRssBytesK2R(): Long? =
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

private fun sanitizeK2RError(value: String?): String {
    val normalized = value
        .orEmpty()
        .replace(Regex("[\\r\\n\\t]+"), " ")
        .replace(Regex("\\s+"), " ")
        .trim()
    return normalized.take(480).ifBlank { "no_message" }
}

private fun percentileK2R(
    values: List<Long>,
    q: Double,
): Long {
    val sorted = values.sorted()
    val rank = kotlin.math.ceil(q * sorted.size)
        .toInt()
        .coerceAtLeast(1)
    return sorted[rank - 1]
}

private fun elapsedK2R(start: Long): Long =
    (System.nanoTime() - start).coerceAtLeast(1L)

private fun requireK2RSha(value: String) {
    require(
        value.length == 64 &&
            value.all { it in '0'..'9' || it in 'a'..'f' }
    )
}

private fun sha256K2R(bytes: ByteArray): String =
    MessageDigest.getInstance("SHA-256")
        .digest(bytes)
        .joinToString("") {
            "%02x".format(it.toInt() and 0xff)
        }

private fun canonicalK2R(value: Any?): String =
    when (value) {
        JSONObject.NULL, null -> "null"
        is JSONObject ->
            value.keys().asSequence().toList().sorted()
                .joinToString(
                    prefix = "{",
                    postfix = "}",
                    separator = ",",
                ) { key ->
                    JSONObject.quote(key) + ":" +
                        canonicalK2R(value.get(key))
                }
        is JSONArray ->
            (0 until value.length()).joinToString(
                prefix = "[",
                postfix = "]",
                separator = ",",
            ) { index ->
                canonicalK2R(value.get(index))
            }
        is String -> JSONObject.quote(value)
        is Boolean -> if (value) "true" else "false"
        is Int, is Long, is Short, is Byte -> value.toString()
        else -> error(
            "unsupported K2R JSON type: " +
                value::class.java.name
        )
    }
