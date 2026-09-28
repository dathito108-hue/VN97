package ai.vn97.runtime

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.LongBuffer
import java.util.LinkedHashMap

data class Mamba2OrtInvocationResult(
    val validLength: Int,
    val provider: OrtProviderKind,
    val elapsedNanos: Long,
) {
    init {
        require(validLength > 0)
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
                invocations.sumOf { it.validLength }.toLong()
        ) {
            "G0.6 sequence accounting mismatch"
        }
    }
}

private data class Mamba2SessionKey(
    val provider: OrtProviderKind,
    val xnnpackThreads: Int,
)

private class Mamba2SessionCache(
    private val maxEntries: Int,
) : AutoCloseable {
    private val sessions =
        LinkedHashMap<Mamba2SessionKey, OrtSessionHandle>(
            4,
            0.75f,
            true,
        )

    init {
        require(maxEntries > 0)
    }

    fun getOrCreate(
        key: Mamba2SessionKey,
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

    fun invalidate(key: Mamba2SessionKey) {
        sessions.remove(key)?.close()
    }

    override fun close() {
        val values = sessions.values.toList()
        sessions.clear()
        values.forEach { it.close() }
    }
}

class VN97Mamba2OrtExecutor private constructor(
    private val context: Context,
    val runtimePackage: Mamba2OrtRuntimePackage,
    private val qnnBackendPath: String?,
    private val environment: OrtEnvironment,
    private val sessionFactory: VN97OrtSessionFactory,
    private val tuningProfile: Mamba2OrtTuningProfile?,
    maxCachedSessions: Int,
) : AutoCloseable {
    private val initialDevice = OrtDeviceProbe.inspect(
        context,
        qnnBackendPath,
    )
    private val sessions = Mamba2SessionCache(maxCachedSessions)
    private val convShape = longArrayOf(
        runtimePackage.nLayers.toLong(),
        1L,
        runtimePackage.convDim.toLong(),
        runtimePackage.dConv.toLong(),
    )
    private val ssmShape = longArrayOf(
        runtimePackage.nLayers.toLong(),
        1L,
        runtimePackage.nHeads.toLong(),
        runtimePackage.headDim.toLong(),
        runtimePackage.dState.toLong(),
    )
    private val valueDtype =
        Mamba2OrtValueDtype.fromRuntime(runtimePackage.stateDtype)
    private val stateSlots = arrayOf(
        createStateSlot(),
        createStateSlot(),
    )
    private val graphBuffers = createGraphBuffers()
    private var currentStateSlot = 0
    private var sequencePosition = 0L
    private var closed = false

    init {
        stateSlots.forEach { it.zero() }
    }

    companion object {
        fun open(
            context: Context,
            runtimeRoot: File,
            qnnBackendPath: String? = null,
            tuningFile: File? = null,
            maxCachedSessions: Int = 2,
        ): VN97Mamba2OrtExecutor {
            val app = context.applicationContext
            val runtime = Mamba2OrtRuntimePackage.load(runtimeRoot)
            val device = OrtDeviceProbe.inspect(
                app,
                qnnBackendPath,
            )
            val tuning = tuningFile?.let {
                Mamba2OrtTuningProfile.load(
                    file = it,
                    expectedRuntimeId = runtime.runtimeId,
                    currentDevice = device,
                )
            }
            val environment = OrtEnvironment.getEnvironment()
            return VN97Mamba2OrtExecutor(
                context = app,
                runtimePackage = runtime,
                qnnBackendPath = qnnBackendPath,
                environment = environment,
                sessionFactory = VN97OrtSessionFactory(environment),
                tuningProfile = tuning,
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
    fun hasDeviceMeasuredTuning(): Boolean {
        requireOpen()
        return tuningProfile != null
    }

    @Synchronized
    fun activeTuningId(): String? {
        requireOpen()
        return tuningProfile?.tuningId
    }

    @Synchronized
    fun resetRecurrentState() {
        requireOpen()
        stateSlots.forEach { it.zero() }
        currentStateSlot = 0
        sequencePosition = 0L
    }

    @Synchronized
    fun prefill(
        inputIds: IntArray,
        deadlineMs: Double = 250.0,
    ): Mamba2OrtRunResult {
        requireOpen()
        requireTokens(inputIds)
        val start = sequencePosition
        val invocations = mutableListOf<Mamba2OrtInvocationResult>()
        var offset = 0
        var finalLogits: FloatArray? = null
        while (offset < inputIds.size) {
            val end = minOf(
                inputIds.size,
                offset + runtimePackage.maxChunkSize,
            )
            val tokens = inputIds.copyOfRange(offset, end)
            val outcome = executeChunk(
                tokens = tokens,
                realtime = false,
                deadlineMs = deadlineMs,
            )
            invocations += outcome.first
            finalLogits = outcome.second
            offset = end
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
        requireToken(tokenId)
        val start = sequencePosition
        val outcome = executeChunk(
            tokens = intArrayOf(tokenId),
            realtime = true,
            deadlineMs = deadlineMs,
        )
        return Mamba2OrtRunResult(
            logits = outcome.second,
            invocations = listOf(outcome.first),
            sequenceStart = start,
            sequenceEnd = sequencePosition,
        )
    }

    private fun executeChunk(
        tokens: IntArray,
        realtime: Boolean,
        deadlineMs: Double,
    ): Pair<Mamba2OrtInvocationResult, FloatArray> {
        require(tokens.isNotEmpty())
        require(tokens.size <= runtimePackage.maxChunkSize)
        requireTokens(tokens)
        graphBuffers.put(tokens)

        val device = OrtDeviceProbe.inspect(
            context,
            qnnBackendPath,
        )
        val tunedDecision = tuningProfile?.decide(
            validLength = tokens.size,
            currentDevice = device,
        )
        val conservativeDecision = if (tunedDecision == null) {
            OrtAdaptiveScheduler.decide(
                device = device,
                workload = OrtAdaptiveWorkload(
                    sequenceLength = tokens.size,
                    deadlineMs = deadlineMs,
                    realtime = realtime,
                    stateDependency = if (realtime) 1.0 else 0.2,
                    preferAccelerator = true,
                ),
            )
        } else {
            null
        }
        val providers = tunedDecision?.providers
            ?: requireNotNull(conservativeDecision).providers
        val xnnpackThreads = tunedDecision?.xnnpackThreads
            ?: requireNotNull(conservativeDecision).xnnpackThreads
        require(providers.isNotEmpty())
        require(providers.last() == OrtProviderKind.CPU)

        val current = stateSlots[currentStateSlot]
        val nextIndex = 1 - currentStateSlot
        val next = stateSlots[nextIndex]

        val inputs = linkedMapOf(
            "input_ids" to graphBuffers.inputTensor,
            "valid_length" to graphBuffers.validLengthTensor,
            "conv_state" to current.conv.tensor,
            "ssm_state" to current.ssm.tensor,
        )
        val outputs = linkedMapOf(
            "logits" to graphBuffers.logits.tensor,
            "next_conv_state" to next.conv.tensor,
            "next_ssm_state" to next.ssm.tensor,
        )

        var lastFailure: Throwable? = null
        var totalNanos = 0L
        for (provider in providers) {
            val key = Mamba2SessionKey(
                provider = provider,
                xnnpackThreads = xnnpackThreads,
            )
            val attemptStart = System.nanoTime()
            try {
                val handle = sessions.getOrCreate(key) {
                    val opened = sessionFactory.createForProvider(
                        modelPath = runtimePackage.graphFile.absolutePath,
                        provider = provider,
                        xnnpackThreads = xnnpackThreads,
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
                    // Outputs are pinned directly into reusable buffers.
                }
                val elapsed = elapsedMamba2(attemptStart)
                totalNanos = safeAddMamba2(totalNanos, elapsed)
                val logits = graphBuffers.finalLogits(tokens.size)
                require(logits.all { it.isFinite() }) {
                    "G0.6 provider produced non-finite logits"
                }
                currentStateSlot = nextIndex
                sequencePosition = Math.addExact(
                    sequencePosition,
                    tokens.size.toLong(),
                )
                return Mamba2OrtInvocationResult(
                    validLength = tokens.size,
                    provider = provider,
                    elapsedNanos = totalNanos,
                ) to logits
            } catch (error: Exception) {
                totalNanos = safeAddMamba2(
                    totalNanos,
                    elapsedMamba2(attemptStart),
                )
                sessions.invalidate(key)
                lastFailure = error
            } catch (error: UnsatisfiedLinkError) {
                totalNanos = safeAddMamba2(
                    totalNanos,
                    elapsedMamba2(attemptStart),
                )
                sessions.invalidate(key)
                lastFailure = error
            }
        }
        throw IllegalStateException(
            "all G0.6 ONNX providers failed",
            lastFailure,
        )
    }

    private fun createStateSlot(): Mamba2StateSlot {
        val conv = Mamba2OrtTensorStorage.create(
            environment, valueDtype, convShape,
            runtimePackage.convStateElements,
        )
        try {
            val ssm = Mamba2OrtTensorStorage.create(
                environment, valueDtype, ssmShape,
                runtimePackage.ssmStateElements,
            )
            return Mamba2StateSlot(conv, ssm)
        } catch (error: Throwable) {
            conv.close()
            throw error
        }
    }

    private fun createGraphBuffers(): Mamba2GraphBuffers {
        val input = directLongBufferMamba2(runtimePackage.maxChunkSize)
        val valid = directLongBufferMamba2(1)
        val logitsCount = checkedMamba2BufferCount(
            "G0.6 logits",
            runtimePackage.maxChunkSize.toLong(),
            runtimePackage.vocabSize.toLong(),
        )
        val logits = Mamba2OrtTensorStorage.create(
            environment,
            valueDtype,
            longArrayOf(
                1L,
                runtimePackage.maxChunkSize.toLong(),
                runtimePackage.vocabSize.toLong(),
            ),
            logitsCount,
        )
        return Mamba2GraphBuffers(
            inputBuffer = input,
            validLengthBuffer = valid,
            inputTensor = OnnxTensor.createTensor(
                environment,
                input,
                longArrayOf(
                    1L,
                    runtimePackage.maxChunkSize.toLong(),
                ),
            ),
            validLengthTensor = OnnxTensor.createTensor(
                environment,
                valid,
                longArrayOf(1L),
            ),
            logits = logits,
            maxChunkSize = runtimePackage.maxChunkSize,
            vocabSize = runtimePackage.vocabSize,
        )
    }

    private fun validateSessionContract(session: OrtSession) {
        require(
            session.inputNames == setOf(
                "input_ids",
                "valid_length",
                "conv_state",
                "ssm_state",
            )
        ) {
            "G0.6 ORT session inputs differ from G0.5 contract"
        }
        require(
            session.outputNames == setOf(
                "logits",
                "next_conv_state",
                "next_ssm_state",
            )
        ) {
            "G0.6 ORT session outputs differ from G0.5 contract"
        }
    }

    private fun requireTokens(tokens: IntArray) {
        require(tokens.isNotEmpty())
        tokens.forEach { requireToken(it) }
    }

    private fun requireToken(token: Int) {
        require(token in 0 until runtimePackage.vocabSize) {
            "G0.6 token outside Mamba-2 vocabulary"
        }
    }

    private fun requireOpen() {
        check(!closed) {
            "G0.6 Mamba-2 ORT executor is closed"
        }
    }

    @Synchronized
    override fun close() {
        if (closed) return
        closed = true
        sessions.close()
        graphBuffers.close()
        stateSlots.forEach { it.close() }
    }
}


private fun directLongBufferMamba2(count: Int): LongBuffer {
    require(count > 0)
    require(count <= Int.MAX_VALUE / Long.SIZE_BYTES)
    return ByteBuffer.allocateDirect(
        count * Long.SIZE_BYTES
    ).order(ByteOrder.nativeOrder()).asLongBuffer()
}

private fun checkedMamba2BufferCount(
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

private fun elapsedMamba2(start: Long): Long =
    (System.nanoTime() - start).coerceAtLeast(1L)

private fun safeAddMamba2(left: Long, right: Long): Long =
    if (Long.MAX_VALUE - left < right) Long.MAX_VALUE else left + right
