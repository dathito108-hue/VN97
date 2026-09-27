package ai.vn97.runtime

import ai.onnxruntime.OnnxJavaType
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.nio.LongBuffer

data class Mamba2OrtInvocationResult(
    val sequenceLength: Int,
    val provider: OrtProviderKind,
    val elapsedNanos: Long,
) {
    init {
        require(sequenceLength > 0)
        require(elapsedNanos > 0L)
    }
}

data class Mamba2OrtRunResult(
    val logits: FloatArray,
    val invocations: List<Mamba2OrtInvocationResult>,
    val sequenceStart: Long,
    val sequenceEnd: Long,
) {
    init {
        require(logits.isNotEmpty())
        require(logits.all { it.isFinite() })
        require(invocations.isNotEmpty())
        require(sequenceStart >= 0L)
        require(sequenceEnd >= sequenceStart)
        require(
            sequenceEnd - sequenceStart ==
                invocations.sumOf { it.sequenceLength }.toLong()
        ) {
            "G0.6 sequence accounting mismatch"
        }
    }
}

class Mamba2OrtExecutionException(
    message: String,
    val lastProvider: OrtProviderKind,
    val elapsedNanos: Long,
    cause: Throwable?,
) : IllegalStateException(message, cause)

private data class Mamba2OrtSessionKey(
    val provider: OrtProviderKind,
    val xnnpackThreads: Int,
)

private class Mamba2OrtSessionCache(
    private val maxEntries: Int,
) : AutoCloseable {
    private val sessions =
        LinkedHashMap<Mamba2OrtSessionKey, OrtSessionHandle>(
            4,
            0.75f,
            true,
        )

    init {
        require(maxEntries > 0)
    }

    fun getOrCreate(
        key: Mamba2OrtSessionKey,
        create: () -> OrtSessionHandle,
    ): OrtSessionHandle {
        sessions[key]?.let { return it }
        val opened = create()
        sessions[key] = opened
        while (sessions.size > maxEntries) {
            val iterator = sessions.entries.iterator()
            val eldest = iterator.next()
            iterator.remove()
            eldest.value.close()
        }
        return opened
    }

    fun invalidate(key: Mamba2OrtSessionKey) {
        sessions.remove(key)?.close()
    }

    override fun close() {
        val values = sessions.values.toList()
        sessions.clear()
        values.forEach { it.close() }
    }
}

private data class Mamba2OrtStateSlot(
    val convBytes: ByteBuffer,
    val ssmBytes: ByteBuffer,
    val convTensor: OnnxTensor,
    val ssmTensor: OnnxTensor,
) : AutoCloseable {
    fun zero() {
        for (index in 0 until convBytes.capacity()) {
            convBytes.put(index, 0)
        }
        for (index in 0 until ssmBytes.capacity()) {
            ssmBytes.put(index, 0)
        }
    }

    override fun close() {
        convTensor.close()
        ssmTensor.close()
    }
}

private data class Mamba2OrtGraphBuffers(
    val sequenceLength: Int,
    val vocabSize: Int,
    val inputBuffer: LongBuffer,
    val inputTensor: OnnxTensor,
    val logitsBuffer: FloatBuffer,
    val logitsTensor: OnnxTensor,
) : AutoCloseable {
    fun putTokens(tokens: IntArray) {
        require(tokens.size == sequenceLength)
        for (index in tokens.indices) {
            inputBuffer.put(index, tokens[index].toLong())
        }
    }

    fun finalLogits(): FloatArray {
        val output = FloatArray(vocabSize)
        val offset = (sequenceLength - 1) * vocabSize
        require(
            offset.toLong() + vocabSize.toLong() <=
                logitsBuffer.capacity().toLong()
        )
        for (index in 0 until vocabSize) {
            output[index] = logitsBuffer.get(offset + index)
        }
        return output
    }

    override fun close() {
        inputTensor.close()
        logitsTensor.close()
    }
}

class VN97Mamba2OrtExecutor private constructor(
    private val context: Context,
    val runtimePackage: Mamba2OrtRuntimePackage,
    private val qnnBackendPath: String?,
    private val environment: OrtEnvironment,
    private val sessionFactory: VN97OrtSessionFactory,
    maxCachedSessions: Int,
) : AutoCloseable {
    private val initialDevice = OrtDeviceProbe.inspect(
        context,
        qnnBackendPath,
    )
    private val sessionCache = Mamba2OrtSessionCache(maxCachedSessions)
    private val stateSlots = arrayOf(
        createStateSlot(),
        createStateSlot(),
    )
    private var graphBuffers: Mamba2OrtGraphBuffers? = null
    private var currentStateSlot = 0
    private var sequencePosition = 0L
    private var movingLatencyMs: Double? = null
    private var closed = false

    init {
        stateSlots.forEach { it.zero() }
    }

    companion object {
        fun open(
            context: Context,
            runtimeRoot: File,
            qnnBackendPath: String? = null,
            maxCachedSessions: Int = 2,
        ): VN97Mamba2OrtExecutor {
            val applicationContext = context.applicationContext
            val runtimePackage = Mamba2OrtRuntimePackage.load(runtimeRoot)
            val environment = OrtEnvironment.getEnvironment()
            return VN97Mamba2OrtExecutor(
                context = applicationContext,
                runtimePackage = runtimePackage,
                qnnBackendPath = qnnBackendPath,
                environment = environment,
                sessionFactory = VN97OrtSessionFactory(environment),
                maxCachedSessions = maxCachedSessions,
            )
        }
    }

    @Synchronized
    fun currentSequencePosition(): Long {
        requireOpen()
        return sequencePosition
    }

    @Synchronized
    fun resetRecurrentState(
        clearSessions: Boolean = false,
    ) {
        requireOpen()
        stateSlots.forEach { it.zero() }
        currentStateSlot = 0
        sequencePosition = 0L
        movingLatencyMs = null
        if (clearSessions) {
            sessionCache.close()
        }
    }

    @Synchronized
    fun prefill(
        inputIds: IntArray,
        deadlineMs: Double = 250.0,
    ): Mamba2OrtRunResult {
        requireOpen()
        require(inputIds.isNotEmpty()) {
            "G0.6 prefill input must not be empty"
        }
        requireTokens(inputIds)

        val start = sequencePosition
        val invocations = ArrayList<Mamba2OrtInvocationResult>()
        var offset = 0
        var finalLogits: FloatArray? = null
        while (offset < inputIds.size) {
            val take = minOf(
                runtimePackage.maxSequenceLength,
                inputIds.size - offset,
            )
            val tokens = inputIds.copyOfRange(offset, offset + take)
            val outcome = executeSequence(
                tokens = tokens,
                deadlineMs = deadlineMs,
                realtime = false,
                stateDependency = 0.2,
            )
            invocations += outcome.first
            finalLogits = outcome.second
            offset += take
        }
        return Mamba2OrtRunResult(
            logits = requireNotNull(finalLogits),
            invocations = invocations,
            sequenceStart = start,
            sequenceEnd = sequencePosition,
        )
    }

    @Synchronized
    fun step(
        tokenId: Int,
        deadlineMs: Double = 100.0,
    ): Mamba2OrtRunResult {
        requireOpen()
        requireTokens(intArrayOf(tokenId))
        val start = sequencePosition
        val outcome = executeSequence(
            tokens = intArrayOf(tokenId),
            deadlineMs = deadlineMs,
            realtime = true,
            stateDependency = 1.0,
        )
        return Mamba2OrtRunResult(
            logits = outcome.second,
            invocations = listOf(outcome.first),
            sequenceStart = start,
            sequenceEnd = sequencePosition,
        )
    }

    private fun executeSequence(
        tokens: IntArray,
        deadlineMs: Double,
        realtime: Boolean,
        stateDependency: Double,
    ): Pair<Mamba2OrtInvocationResult, FloatArray> {
        require(tokens.isNotEmpty())
        require(tokens.size <= runtimePackage.maxSequenceLength)
        val device = OrtDeviceProbe.inspect(
            context,
            qnnBackendPath,
        )
        val decision = OrtAdaptiveScheduler.decide(
            device = device,
            workload = OrtAdaptiveWorkload(
                sequenceLength = tokens.size,
                deadlineMs = deadlineMs,
                realtime = realtime,
                stateDependency = stateDependency,
                preferAccelerator = true,
            ),
            telemetry = OrtAdaptiveTelemetry(
                movingLatencyMs = movingLatencyMs,
                thermalStatus = device.thermalStatus,
                availableMemoryBytes = device.availableMemoryBytes,
            ),
        )
        val buffers = buffersFor(tokens.size)
        buffers.putTokens(tokens)

        val current = stateSlots[currentStateSlot]
        val nextIndex = 1 - currentStateSlot
        val next = stateSlots[nextIndex]
        val inputs = linkedMapOf(
            "input_ids" to buffers.inputTensor,
            "conv_state" to current.convTensor,
            "ssm_state" to current.ssmTensor,
        )
        val outputs = linkedMapOf(
            "logits" to buffers.logitsTensor,
            "next_conv_state" to next.convTensor,
            "next_ssm_state" to next.ssmTensor,
        )
        val nextPosition = Math.addExact(
            sequencePosition,
            tokens.size.toLong(),
        )

        var lastFailure: Throwable? = null
        var lastProvider = decision.providers.last()
        var totalNanos = 0L

        for (provider in decision.providers) {
            lastProvider = provider
            val key = Mamba2OrtSessionKey(
                provider = provider,
                xnnpackThreads = decision.xnnpackThreads,
            )
            val started = System.nanoTime()
            try {
                val handle = sessionCache.getOrCreate(key) {
                    val opened = sessionFactory.createForProvider(
                        modelPath =
                            runtimePackage.graphFile.absolutePath,
                        provider = provider,
                        xnnpackThreads = decision.xnnpackThreads,
                        device = initialDevice,
                    )
                    try {
                        validateSessionContract(opened.session)
                        opened
                    } catch (error: Throwable) {
                        opened.close()
                        throw error
                    }
                }
                handle.session.run(inputs, outputs).use {
                    // Outputs are pinned to reusable direct buffers.
                }
                val elapsed = elapsedNanosG06(started)
                totalNanos = safeAddNanosG06(totalNanos, elapsed)
                val logits = buffers.finalLogits()
                require(logits.all { it.isFinite() }) {
                    "G0.6 provider produced non-finite logits"
                }
                currentStateSlot = nextIndex
                sequencePosition = nextPosition
                movingLatencyMs = elapsed.toDouble() / 1_000_000.0
                return Pair(
                    Mamba2OrtInvocationResult(
                        sequenceLength = tokens.size,
                        provider = provider,
                        elapsedNanos = totalNanos,
                    ),
                    logits,
                )
            } catch (error: Exception) {
                totalNanos = safeAddNanosG06(
                    totalNanos,
                    elapsedNanosG06(started),
                )
                sessionCache.invalidate(key)
                lastFailure = error
            } catch (error: UnsatisfiedLinkError) {
                totalNanos = safeAddNanosG06(
                    totalNanos,
                    elapsedNanosG06(started),
                )
                sessionCache.invalidate(key)
                lastFailure = error
            }
        }

        throw Mamba2OrtExecutionException(
            message = "all G0.6 provider attempts failed",
            lastProvider = lastProvider,
            elapsedNanos = totalNanos.coerceAtLeast(1L),
            cause = lastFailure,
        )
    }

    private fun buffersFor(
        sequenceLength: Int,
    ): Mamba2OrtGraphBuffers {
        require(sequenceLength in 1..runtimePackage.maxSequenceLength)
        graphBuffers?.let {
            if (it.sequenceLength == sequenceLength) {
                return it
            }
            it.close()
        }
        return createGraphBuffers(sequenceLength).also {
            graphBuffers = it
        }
    }

    private fun createGraphBuffers(
        sequenceLength: Int,
    ): Mamba2OrtGraphBuffers {
        val inputBuffer = directLongBufferG06(sequenceLength)
        val inputTensor = OnnxTensor.createTensor(
            environment,
            inputBuffer,
            longArrayOf(1L, sequenceLength.toLong()),
        )
        val logitsCount = checkedIntProductG06Executor(
            "G0.6 logits",
            sequenceLength.toLong(),
            runtimePackage.vocabSize.toLong(),
        )
        val logitsBuffer = directFloatBufferG06(logitsCount)
        val logitsTensor = OnnxTensor.createTensor(
            environment,
            logitsBuffer,
            longArrayOf(
                1L,
                sequenceLength.toLong(),
                runtimePackage.vocabSize.toLong(),
            ),
        )
        return Mamba2OrtGraphBuffers(
            sequenceLength = sequenceLength,
            vocabSize = runtimePackage.vocabSize,
            inputBuffer = inputBuffer,
            inputTensor = inputTensor,
            logitsBuffer = logitsBuffer,
            logitsTensor = logitsTensor,
        )
    }

    private fun createStateSlot(): Mamba2OrtStateSlot {
        val type = when (runtimePackage.stateDType) {
            Mamba2OrtStateDType.FLOAT32 -> OnnxJavaType.FLOAT
            Mamba2OrtStateDType.FLOAT16 -> OnnxJavaType.FLOAT16
            Mamba2OrtStateDType.BFLOAT16 -> OnnxJavaType.BFLOAT16
        }
        val convBytes = directByteBufferG06(
            runtimePackage.convStateElements,
            runtimePackage.stateDType.elementBytes,
            "G0.6 conv state",
        )
        val ssmBytes = directByteBufferG06(
            runtimePackage.ssmStateElements,
            runtimePackage.stateDType.elementBytes,
            "G0.6 SSM state",
        )
        val convTensor = OnnxTensor.createTensor(
            environment,
            convBytes,
            runtimePackage.convStateShape,
            type,
        )
        val ssmTensor = OnnxTensor.createTensor(
            environment,
            ssmBytes,
            runtimePackage.ssmStateShape,
            type,
        )
        return Mamba2OrtStateSlot(
            convBytes = convBytes,
            ssmBytes = ssmBytes,
            convTensor = convTensor,
            ssmTensor = ssmTensor,
        )
    }

    private fun validateSessionContract(session: OrtSession) {
        require(
            session.inputNames == setOf(
                "input_ids",
                "conv_state",
                "ssm_state",
            )
        ) {
            "G0.6 ORT session input contract mismatch"
        }
        require(
            session.outputNames == setOf(
                "logits",
                "next_conv_state",
                "next_ssm_state",
            )
        ) {
            "G0.6 ORT session output contract mismatch"
        }
    }

    private fun requireTokens(tokens: IntArray) {
        require(tokens.isNotEmpty())
        require(
            tokens.all {
                it >= 0 && it < runtimePackage.vocabSize
            }
        ) {
            "G0.6 token ID outside vocabulary"
        }
    }

    @Synchronized
    override fun close() {
        if (closed) return
        closed = true
        graphBuffers?.close()
        graphBuffers = null
        sessionCache.close()
        stateSlots.forEach { it.close() }
    }

    private fun requireOpen() {
        check(!closed) {
            "G0.6 Mamba-2 ONNX executor is closed"
        }
    }
}

private fun directByteBufferG06(
    elements: Int,
    elementBytes: Int,
    label: String,
): ByteBuffer {
    require(elements > 0)
    require(elementBytes in setOf(2, 4))
    val bytes = Math.multiplyExact(elements, elementBytes)
    require(bytes > 0) {
        "$label direct buffer size must be positive"
    }
    return ByteBuffer.allocateDirect(bytes)
        .order(ByteOrder.nativeOrder())
}

private fun directFloatBufferG06(count: Int): FloatBuffer =
    directByteBufferG06(
        count,
        Float.SIZE_BYTES,
        "G0.6 float buffer",
    ).asFloatBuffer()

private fun directLongBufferG06(count: Int): LongBuffer {
    require(count > 0)
    val bytes = Math.multiplyExact(count, Long.SIZE_BYTES)
    return ByteBuffer.allocateDirect(bytes)
        .order(ByteOrder.nativeOrder())
        .asLongBuffer()
}

private fun checkedIntProductG06Executor(
    label: String,
    vararg dimensions: Long,
): Int {
    var product = 1L
    for (dimension in dimensions) {
        require(dimension > 0L)
        require(product <= Int.MAX_VALUE.toLong() / dimension) {
            "$label exceeds JVM direct-buffer element bound"
        }
        product *= dimension
    }
    return product.toInt()
}

private fun elapsedNanosG06(start: Long): Long =
    (System.nanoTime() - start).coerceAtLeast(1L)

private fun safeAddNanosG06(left: Long, right: Long): Long =
    if (Long.MAX_VALUE - left < right) {
        Long.MAX_VALUE
    } else {
        left + right
    }
