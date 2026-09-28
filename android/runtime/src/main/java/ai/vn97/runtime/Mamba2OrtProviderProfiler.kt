package ai.vn97.runtime

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.nio.LongBuffer
import java.security.MessageDigest
import org.json.JSONArray
import org.json.JSONObject

data class Mamba2OrtProfileMeasurement(
    val requestedProvider: OrtProviderKind,
    val actualProvider: OrtProviderKind,
    val fallbackUsed: Boolean,
    val warmupIterations: Int,
    val steadyIterations: Int,
    val sessionCreateNanos: Long,
    val latencyP50Nanos: Long,
    val latencyP95Nanos: Long,
    val memoryBeforeBytes: Long,
    val memoryAfterBytes: Long,
    val thermalBefore: Int,
    val thermalAfter: Int,
) {
    init {
        require(warmupIterations >= 1)
        require(steadyIterations >= 3)
        require(sessionCreateNanos > 0L)
        require(latencyP50Nanos > 0L)
        require(latencyP95Nanos >= latencyP50Nanos)
        require(memoryBeforeBytes > 0L)
        require(memoryAfterBytes > 0L)
        require(thermalBefore in 0..6)
        require(thermalAfter in 0..6)
        require(requestedProvider == actualProvider || fallbackUsed)
    }

    fun toJson(): JSONObject =
        JSONObject()
            .put("requested_provider", requestedProvider.name)
            .put("actual_provider", actualProvider.name)
            .put("fallback_used", fallbackUsed)
            .put("warmup_iterations", warmupIterations)
            .put("steady_iterations", steadyIterations)
            .put("session_create_ns", sessionCreateNanos)
            .put("latency_p50_ns", latencyP50Nanos)
            .put("latency_p95_ns", latencyP95Nanos)
            .put("memory_before_bytes", memoryBeforeBytes)
            .put("memory_after_bytes", memoryAfterBytes)
            .put("thermal_before", thermalBefore)
            .put("thermal_after", thermalAfter)
}

data class Mamba2OrtProfileWorkload(
    val validLength: Int,
    val measurements: List<Mamba2OrtProfileMeasurement>,
) {
    init {
        require(validLength > 0)
        require(measurements.isNotEmpty())
        require(
            measurements.map { it.requestedProvider }.distinct().size ==
                measurements.size
        )
        require(
            measurements.any {
                it.requestedProvider == OrtProviderKind.CPU &&
                    it.actualProvider == OrtProviderKind.CPU &&
                    !it.fallbackUsed
            }
        ) {
            "G0.7 workload requires direct CPU measurement"
        }
    }

    fun toJson(): JSONObject {
        val values = JSONArray()
        measurements.forEach { values.put(it.toJson()) }
        return JSONObject()
            .put("valid_length", validLength)
            .put("measurements", values)
    }
}

data class Mamba2OrtProfileReport(
    val runtimeId: String,
    val graphFilename: String,
    val maxChunkSize: Int,
    val device: OrtDeviceCapabilities,
    val workloads: List<Mamba2OrtProfileWorkload>,
) {
    init {
        requireG07ProfileSha(runtimeId, "G0.7 runtime ID")
        require(maxChunkSize in setOf(8, 16, 32))
        require(graphFilename == "recurrent-" + maxChunkSize + ".onnx")
        require(
            workloads.map { it.validLength }.sorted() ==
                requiredG07ProfileLengths(maxChunkSize)
        )
    }

    fun toJson(): JSONObject {
        val workloadArray = JSONArray()
        workloads.sortedBy { it.validLength }.forEach {
            workloadArray.put(it.toJson())
        }
        val body = JSONObject()
            .put("schema", PROFILE_SCHEMA)
            .put("runtime_id", runtimeId)
            .put("graph_filename", graphFilename)
            .put("max_chunk_size", maxChunkSize)
            .put(
                "device",
                JSONObject()
                    .put("sdk_int", device.sdkInt)
                    .put("logical_cores", device.logicalCores)
                    .put("hardware", device.hardware)
                    .put(
                        "soc_manufacturer",
                        device.socManufacturer,
                    )
                    .put("soc_model", device.socModel)
                    .put(
                        "profiled_available_memory_bytes",
                        device.availableMemoryBytes,
                    ),
            )
            .put("workloads", workloadArray)
            .put("device_measured", true)
            .put("synthetic", false)
            .put("same_weights_semantics", true)
            .put("production_activation_authorized", false)

        val prefix = "VN97M2G07PROFILE1" +
            String(charArrayOf(0.toChar()))
        val receipt = sha256G07Profile(
            prefix.toByteArray(Charsets.US_ASCII) +
                canonicalJsonG07Profile(body)
                    .toByteArray(Charsets.US_ASCII)
        )
        return JSONObject(canonicalJsonG07Profile(body))
            .put("receipt_id", receipt)
    }

    fun writeAtomic(target: File) {
        require(!target.isDirectory)
        target.parentFile?.mkdirs()
        val temp = File(target.parentFile, target.name + ".tmp")
        if (temp.exists()) {
            check(temp.delete())
        }
        temp.writeText(
            canonicalJsonG07Profile(toJson()) + "\n",
            Charsets.US_ASCII,
        )
        if (target.exists()) {
            check(target.delete())
        }
        check(temp.renameTo(target))
    }

    companion object {
        const val PROFILE_SCHEMA = "VN97M2G07PROFILE1"
    }
}

class VN97Mamba2OrtProviderProfiler(
    private val environment: OrtEnvironment =
        OrtEnvironment.getEnvironment(),
    private val sessionFactory: VN97OrtSessionFactory =
        VN97OrtSessionFactory(environment),
) {
    fun profile(
        context: Context,
        runtimeRoot: File,
        qnnBackendPath: String? = null,
        providers: List<OrtProviderKind> = listOf(
            OrtProviderKind.QNN,
            OrtProviderKind.NNAPI,
            OrtProviderKind.XNNPACK,
            OrtProviderKind.CPU,
        ),
        warmupIterations: Int = 2,
        steadyIterations: Int = 8,
    ): Mamba2OrtProfileReport {
        require(providers.isNotEmpty())
        require(providers.distinct().size == providers.size)
        require(OrtProviderKind.CPU in providers) {
            "G0.7 profiling requires CPU baseline"
        }
        require(warmupIterations >= 1)
        require(steadyIterations >= 3)

        val app = context.applicationContext
        val runtime = Mamba2OrtRuntimePackage.load(runtimeRoot)
        val initialDevice = OrtDeviceProbe.inspect(
            app,
            qnnBackendPath,
        )
        val requiredLengths = requiredG07ProfileLengths(
            runtime.maxChunkSize
        )
        val results = linkedMapOf<
            Int,
            MutableList<Mamba2OrtProfileMeasurement>
            >()
        requiredLengths.forEach {
            results[it] = mutableListOf()
        }

        Mamba2ProfileBuffers(
            environment,
            runtime,
        ).use { buffers ->
            for (provider in providers) {
                if (!providerAvailableForG07Profile(
                        provider,
                        initialDevice,
                    )
                ) {
                    continue
                }
                val threads = minOf(
                    4,
                    initialDevice.logicalCores,
                ).coerceAtLeast(1)
                val createStart = System.nanoTime()
                val handle = try {
                    sessionFactory.createForProvider(
                        modelPath = runtime.graphFile.absolutePath,
                        provider = provider,
                        xnnpackThreads = threads,
                        device = initialDevice,
                    )
                } catch (_: Throwable) {
                    continue
                }
                val createNanos = elapsedG07Profile(createStart)
                handle.use { opened ->
                    validateG07ProfileSession(opened.session)
                    for (validLength in requiredLengths) {
                        buffers.prepare(validLength)
                        repeat(warmupIterations) {
                            buffers.run(opened.session)
                        }

                        val before = OrtDeviceProbe.inspect(
                            app,
                            qnnBackendPath,
                        )
                        val samples = ArrayList<Long>(
                            steadyIterations
                        )
                        repeat(steadyIterations) {
                            val start = System.nanoTime()
                            buffers.run(opened.session)
                            samples += elapsedG07Profile(start)
                        }
                        val after = OrtDeviceProbe.inspect(
                            app,
                            qnnBackendPath,
                        )
                        val measurement = Mamba2OrtProfileMeasurement(
                            requestedProvider = provider,
                            actualProvider = opened.primaryProvider,
                            fallbackUsed =
                                opened.primaryProvider != provider,
                            warmupIterations = warmupIterations,
                            steadyIterations = steadyIterations,
                            sessionCreateNanos = createNanos,
                            latencyP50Nanos =
                                percentileG07Profile(samples, 0.50),
                            latencyP95Nanos =
                                percentileG07Profile(samples, 0.95),
                            memoryBeforeBytes =
                                before.availableMemoryBytes,
                            memoryAfterBytes =
                                after.availableMemoryBytes,
                            thermalBefore = before.thermalStatus,
                            thermalAfter = after.thermalStatus,
                        )
                        results.getValue(validLength).add(measurement)
                    }
                }
            }
        }

        val workloads = requiredLengths.map { validLength ->
            Mamba2OrtProfileWorkload(
                validLength = validLength,
                measurements = results.getValue(validLength)
                    .sortedBy { it.requestedProvider.name },
            )
        }
        return Mamba2OrtProfileReport(
            runtimeId = runtime.runtimeId,
            graphFilename = runtime.graphFilename,
            maxChunkSize = runtime.maxChunkSize,
            device = initialDevice,
            workloads = workloads,
        )
    }
}

private class Mamba2ProfileBuffers(
    environment: OrtEnvironment,
    private val runtime: Mamba2OrtRuntimePackage,
) : AutoCloseable {
    private val inputBuffer =
        directLongBufferG07Profile(runtime.maxChunkSize)
    private val validLengthBuffer = directLongBufferG07Profile(1)
    private val convInputBuffer =
        directFloatBufferG07Profile(runtime.convStateElements)
    private val ssmInputBuffer =
        directFloatBufferG07Profile(runtime.ssmStateElements)
    private val convOutputBuffer =
        directFloatBufferG07Profile(runtime.convStateElements)
    private val ssmOutputBuffer =
        directFloatBufferG07Profile(runtime.ssmStateElements)
    private val logitsBuffer = directFloatBufferG07Profile(
        checkedG07ProfileCount(
            "G0.7 logits",
            runtime.maxChunkSize.toLong(),
            runtime.vocabSize.toLong(),
        )
    )

    private val inputTensor = OnnxTensor.createTensor(
        environment,
        inputBuffer,
        longArrayOf(1L, runtime.maxChunkSize.toLong()),
    )
    private val validLengthTensor = OnnxTensor.createTensor(
        environment,
        validLengthBuffer,
        longArrayOf(1L),
    )
    private val convInputTensor = OnnxTensor.createTensor(
        environment,
        convInputBuffer,
        longArrayOf(
            runtime.nLayers.toLong(),
            1L,
            runtime.convDim.toLong(),
            runtime.dConv.toLong(),
        ),
    )
    private val ssmInputTensor = OnnxTensor.createTensor(
        environment,
        ssmInputBuffer,
        longArrayOf(
            runtime.nLayers.toLong(),
            1L,
            runtime.nHeads.toLong(),
            runtime.headDim.toLong(),
            runtime.dState.toLong(),
        ),
    )
    private val convOutputTensor = OnnxTensor.createTensor(
        environment,
        convOutputBuffer,
        longArrayOf(
            runtime.nLayers.toLong(),
            1L,
            runtime.convDim.toLong(),
            runtime.dConv.toLong(),
        ),
    )
    private val ssmOutputTensor = OnnxTensor.createTensor(
        environment,
        ssmOutputBuffer,
        longArrayOf(
            runtime.nLayers.toLong(),
            1L,
            runtime.nHeads.toLong(),
            runtime.headDim.toLong(),
            runtime.dState.toLong(),
        ),
    )
    private val logitsTensor = OnnxTensor.createTensor(
        environment,
        logitsBuffer,
        longArrayOf(
            1L,
            runtime.maxChunkSize.toLong(),
            runtime.vocabSize.toLong(),
        ),
    )

    private val inputs = linkedMapOf(
        "input_ids" to inputTensor,
        "valid_length" to validLengthTensor,
        "conv_state" to convInputTensor,
        "ssm_state" to ssmInputTensor,
    )
    private val outputs = linkedMapOf(
        "logits" to logitsTensor,
        "next_conv_state" to convOutputTensor,
        "next_ssm_state" to ssmOutputTensor,
    )

    init {
        zero(convInputBuffer)
        zero(ssmInputBuffer)
    }

    fun prepare(validLength: Int) {
        require(validLength in 1..runtime.maxChunkSize)
        for (index in 0 until runtime.maxChunkSize) {
            inputBuffer.put(index, 0L)
        }
        validLengthBuffer.put(0, validLength.toLong())
    }

    fun run(session: OrtSession) {
        session.run(inputs, outputs).use {
            // Outputs are pinned and deliberately not fed into the next run.
        }
    }

    private fun zero(buffer: FloatBuffer) {
        for (index in 0 until buffer.capacity()) {
            buffer.put(index, 0.0f)
        }
    }

    override fun close() {
        inputTensor.close()
        validLengthTensor.close()
        convInputTensor.close()
        ssmInputTensor.close()
        convOutputTensor.close()
        ssmOutputTensor.close()
        logitsTensor.close()
    }
}

private fun validateG07ProfileSession(session: OrtSession) {
    require(
        session.inputNames == setOf(
            "input_ids",
            "valid_length",
            "conv_state",
            "ssm_state",
        )
    ) {
        "G0.7 ORT session input contract mismatch"
    }
    require(
        session.outputNames == setOf(
            "logits",
            "next_conv_state",
            "next_ssm_state",
        )
    ) {
        "G0.7 ORT session output contract mismatch"
    }
}

private fun requiredG07ProfileLengths(maxChunk: Int): List<Int> =
    listOf(1, 8, 16, 32).filter {
        it == 1 || it <= maxChunk
    }

private fun providerAvailableForG07Profile(
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

private fun percentileG07Profile(
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

private fun elapsedG07Profile(start: Long): Long =
    (System.nanoTime() - start).coerceAtLeast(1L)

private fun directFloatBufferG07Profile(count: Int): FloatBuffer {
    require(count > 0)
    require(count <= Int.MAX_VALUE / Float.SIZE_BYTES)
    return ByteBuffer.allocateDirect(
        count * Float.SIZE_BYTES
    ).order(ByteOrder.nativeOrder()).asFloatBuffer()
}

private fun directLongBufferG07Profile(count: Int): LongBuffer {
    require(count > 0)
    require(count <= Int.MAX_VALUE / Long.SIZE_BYTES)
    return ByteBuffer.allocateDirect(
        count * Long.SIZE_BYTES
    ).order(ByteOrder.nativeOrder()).asLongBuffer()
}

private fun checkedG07ProfileCount(
    label: String,
    vararg dimensions: Long,
): Int {
    var product = 1L
    for (dimension in dimensions) {
        require(dimension > 0L)
        require(product <= Int.MAX_VALUE.toLong() / dimension) {
            label + " exceeds direct-buffer bounds"
        }
        product *= dimension
    }
    return product.toInt()
}

private fun requireG07ProfileSha(
    value: String,
    label: String,
) {
    require(
        value.length == 64 &&
            value.all { it in '0'..'9' || it in 'a'..'f' }
    ) {
        label + " must be lowercase SHA-256"
    }
}

private fun sha256G07Profile(bytes: ByteArray): String =
    MessageDigest.getInstance("SHA-256")
        .digest(bytes)
        .joinToString("") {
            "%02x".format(it.toInt() and 0xff)
        }

private fun canonicalJsonG07Profile(value: Any?): String {
    return when (value) {
        JSONObject.NULL, null -> "null"
        is JSONObject -> {
            value.keys().asSequence().toList().sorted().joinToString(
                prefix = "{",
                postfix = "}",
                separator = ",",
            ) { key ->
                JSONObject.quote(key) + ":" +
                    canonicalJsonG07Profile(value.get(key))
            }
        }
        is JSONArray -> {
            (0 until value.length()).joinToString(
                prefix = "[",
                postfix = "]",
                separator = ",",
            ) { index ->
                canonicalJsonG07Profile(value.get(index))
            }
        }
        is String -> JSONObject.quote(value)
        is Boolean -> if (value) "true" else "false"
        is Int, is Long, is Short, is Byte -> value.toString()
        else -> error(
            "unsupported G0.7 profile JSON type: " +
                value::class.java.name
        )
    }
}
