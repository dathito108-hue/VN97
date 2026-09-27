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
import java.util.LinkedHashMap

data class OrtProductionInvocationResult(
    val graphFilename: String,
    val sequenceLength: Int,
    val provider: OrtProviderKind,
    val elapsedNanos: Long,
) {
    init {
        require(graphFilename.isNotBlank())
        require(sequenceLength > 0)
        require(elapsedNanos > 0L)
    }
}

private data class OrtProductionInvocationOutcome(
    val result: OrtProductionInvocationResult,
    val logits: FloatArray,
)

data class OrtProductionRunResult(
    val logits: FloatArray,
    val invocations: List<OrtProductionInvocationResult>,
    val mode: OrtAdaptiveMode,
    val reason: String,
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
            "F1 run sequence accounting mismatch"
        }
    }
}

class OrtProductionExecutionException(
    message: String,
    val graphFilename: String,
    val lastProvider: OrtProviderKind,
    val elapsedNanos: Long,
    cause: Throwable?,
) : IllegalStateException(message, cause)

private data class OrtProductionSessionKey(
    val graphFilename: String,
    val provider: OrtProviderKind,
    val xnnpackThreads: Int,
)

private class OrtProductionSessionCache(
    private val maxEntries: Int,
) : AutoCloseable {
    private val sessions =
        LinkedHashMap<OrtProductionSessionKey, OrtSessionHandle>(
            4,
            0.75f,
            true,
        )

    init {
        require(maxEntries > 0) {
            "F1 session cache bound must be positive"
        }
    }

    fun getOrCreate(
        key: OrtProductionSessionKey,
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

    fun invalidate(key: OrtProductionSessionKey) {
        sessions.remove(key)?.close()
    }

    fun clear() {
        val values = sessions.values.toList()
        sessions.clear()
        values.forEach { it.close() }
    }

    override fun close() {
        clear()
    }
}

private data class OrtProductionStateSlot(
    val convBuffer: FloatBuffer,
    val ssmBuffer: FloatBuffer,
    val convTensor: OnnxTensor,
    val ssmTensor: OnnxTensor,
) : AutoCloseable {
    fun zero() {
        for (index in 0 until convBuffer.capacity()) {
            convBuffer.put(index, 0.0f)
        }
        for (index in 0 until ssmBuffer.capacity()) {
            ssmBuffer.put(index, 0.0f)
        }
    }

    override fun close() {
        convTensor.close()
        ssmTensor.close()
    }
}

private data class OrtProductionGraphBuffers(
    val inputBuffer: LongBuffer,
    val inputTensor: OnnxTensor,
    val logitsBuffer: FloatBuffer,
    val logitsTensor: OnnxTensor,
    val sequenceLength: Int,
    val vocabSize: Int,
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
            offset >= 0 &&
                offset.toLong() + vocabSize.toLong() <=
                logitsBuffer.capacity().toLong()
        ) {
            "F1 logits buffer bounds mismatch"
        }
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

class VN97OrtProductionExecutor private constructor(
    private val context: Context,
    val runtimePackage: OrtProductionPackage,
    private val qnnBackendPath: String?,
    maxCachedSessions: Int,
    private val environment: OrtEnvironment,
    private val sessionFactory: VN97OrtSessionFactory,
) : OrtHardeningExecutor, AutoCloseable {
    private val initialDevice = OrtDeviceProbe.inspect(
        context,
        qnnBackendPath,
    )
    private var autotuner = VN97OrtAutotuner(
        runtimePackage.tuning,
        initialDevice,
        runtimePackage.bundleId,
    )
    private val sessionCache = OrtProductionSessionCache(
        maxCachedSessions
    )
    private val graphBuffers =
        mutableMapOf<String, OrtProductionGraphBuffers>()

    private val convShape = longArrayOf(
        runtimePackage.activeLayers.toLong(),
        1L,
        runtimePackage.dInner.toLong(),
        runtimePackage.dConvState.toLong(),
    )
    private val ssmShape = longArrayOf(
        runtimePackage.activeLayers.toLong(),
        1L,
        runtimePackage.dInner.toLong(),
        runtimePackage.dState.toLong(),
    )
    private val stateSlots = arrayOf(
        createStateSlot(),
        createStateSlot(),
    )
    private var currentStateSlot = 0
    private var movingLatencyMs: Double? = null
    private var sequencePosition = 0L
    private var closed = false

    init {
        runtimePackage.tuning.requireCompatible(
            runtimePackage.bundleId,
            initialDevice,
        )
        stateSlots.forEach { it.zero() }
    }

    companion object {
        fun open(
            context: Context,
            runtimeRoot: File,
            tuningFile: File,
            qnnBackendPath: String? = null,
            maxCachedSessions: Int = 1,
        ): VN97OrtProductionExecutor {
            val applicationContext = context.applicationContext
            val device = OrtDeviceProbe.inspect(
                applicationContext,
                qnnBackendPath,
            )
            val runtimePackage = OrtProductionPackage.load(
                rootDir = runtimeRoot,
                tuningFile = tuningFile,
                currentDevice = device,
            )
            val environment = OrtEnvironment.getEnvironment()
            return VN97OrtProductionExecutor(
                context = applicationContext,
                runtimePackage = runtimePackage,
                qnnBackendPath = qnnBackendPath,
                maxCachedSessions = maxCachedSessions,
                environment = environment,
                sessionFactory = VN97OrtSessionFactory(environment),
            )
        }
    }

    @Synchronized
    fun currentSequencePosition(): Long {
        requireOpen()
        return sequencePosition
    }

    @Synchronized
    fun prefill(
        inputIds: IntArray,
        deadlineMs: Double = 250.0,
        stateDependency: Double = 0.2,
    ): OrtProductionRunResult {
        requireOpen()
        require(inputIds.isNotEmpty()) {
            "F1 prefill input must not be empty"
        }
        return runTokens(
            inputIds = inputIds,
            workload = OrtAdaptiveWorkload(
                sequenceLength = inputIds.size,
                deadlineMs = deadlineMs,
                realtime = false,
                stateDependency = stateDependency,
                preferAccelerator = true,
            ),
        )
    }

    @Synchronized
    fun step(
        tokenId: Int,
        deadlineMs: Double = 100.0,
    ): OrtProductionRunResult {
        requireOpen()
        return runTokens(
            inputIds = intArrayOf(tokenId),
            workload = OrtAdaptiveWorkload(
                sequenceLength = 1,
                deadlineMs = deadlineMs,
                realtime = true,
                stateDependency = 1.0,
                preferAccelerator = true,
            ),
        )
    }

    @Synchronized
    fun resetRecurrentState(
        clearSessions: Boolean = false,
        resetControlState: Boolean = false,
    ) {
        requireOpen()
        stateSlots.forEach { it.zero() }
        currentStateSlot = 0
        sequencePosition = 0L
        movingLatencyMs = null
        if (clearSessions) {
            sessionCache.clear()
        }
        if (resetControlState) {
            val device = OrtDeviceProbe.inspect(
                context,
                qnnBackendPath,
            )
            runtimePackage.tuning.requireCompatible(
                runtimePackage.bundleId,
                device,
            )
            autotuner = VN97OrtAutotuner(
                runtimePackage.tuning,
                device,
                runtimePackage.bundleId,
            )
        }
    }

    @Synchronized
    override fun resetState() {
        resetRecurrentState(
            clearSessions = true,
            resetControlState = true,
        )
    }

    @Synchronized
    override fun execute(
        invocation: OrtTunedInvocation,
    ): OrtHardeningExecutionResult {
        requireOpen()
        val tokens = IntArray(invocation.sequenceLength)
        return try {
            val outcome = executeInvocation(
                invocation,
                tokens,
            )
            val result = outcome.result
            OrtHardeningExecutionResult(
                graphFilename = result.graphFilename,
                provider = result.provider,
                elapsedNanos = result.elapsedNanos,
                success = true,
            )
        } catch (error: OrtProductionExecutionException) {
            OrtHardeningExecutionResult(
                graphFilename = error.graphFilename,
                provider = error.lastProvider,
                elapsedNanos = error.elapsedNanos.coerceAtLeast(1L),
                success = false,
            )
        }
    }

    @Synchronized
    override fun close() {
        if (closed) return
        closed = true
        sessionCache.close()
        graphBuffers.values.forEach { it.close() }
        graphBuffers.clear()
        stateSlots.forEach { it.close() }
    }

    private fun runTokens(
        inputIds: IntArray,
        workload: OrtAdaptiveWorkload,
    ): OrtProductionRunResult {
        require(inputIds.size == workload.sequenceLength)
        requireTokens(inputIds)

        val device = OrtDeviceProbe.inspect(
            context,
            qnnBackendPath,
        )
        runtimePackage.tuning.requireCompatible(
            runtimePackage.bundleId,
            device,
        )
        val decision = autotuner.decide(
            workload,
            OrtAdaptiveTelemetry(
                movingLatencyMs = movingLatencyMs,
                thermalStatus = device.thermalStatus,
                availableMemoryBytes = device.availableMemoryBytes,
            ),
        )
        val totalPlanned = decision.invocations.sumOf {
            it.sequenceLength
        }
        require(totalPlanned == inputIds.size) {
            "F1 E4 decision lost token coverage"
        }

        val startPosition = sequencePosition
        val results = ArrayList<OrtProductionInvocationResult>(
            decision.invocations.size
        )
        var offset = 0
        var finalLogits: FloatArray? = null
        for (invocation in decision.invocations) {
            val end = offset + invocation.sequenceLength
            val tokens = inputIds.copyOfRange(offset, end)
            val outcome = executeInvocation(
                invocation,
                tokens,
            )
            results += outcome.result
            finalLogits = outcome.logits
            offset = end
        }
        require(offset == inputIds.size)
        return OrtProductionRunResult(
            logits = requireNotNull(finalLogits),
            invocations = results,
            mode = decision.mode,
            reason = decision.reason,
            sequenceStart = startPosition,
            sequenceEnd = sequencePosition,
        )
    }

    private fun executeInvocation(
        invocation: OrtTunedInvocation,
        tokens: IntArray,
    ): OrtProductionInvocationOutcome {
        val graph = runtimePackage.graphs[invocation.graphFilename]
            ?: throw IllegalArgumentException(
                "F1 invocation references unknown graph"
            )
        require(graph.sequenceLength == invocation.sequenceLength) {
            "F1 invocation sequence length differs from graph"
        }
        require(tokens.size == graph.sequenceLength)
        requireTokens(tokens)
        require(invocation.providers.isNotEmpty())
        require(invocation.providers.last() == OrtProviderKind.CPU)

        val buffers = graphBuffers.getOrPut(graph.filename) {
            createGraphBuffers(graph)
        }
        buffers.putTokens(tokens)

        val current = stateSlots[currentStateSlot]
        val nextIndex = 1 - currentStateSlot
        val next = stateSlots[nextIndex]
        val inputs = linkedMapOf(
            "input_ids" to buffers.inputTensor,
            "conv_state" to current.convTensor,
            "ssm_state" to current.ssmTensor,
        )
        val pinnedOutputs = linkedMapOf(
            "logits" to buffers.logitsTensor,
            "next_conv_state" to next.convTensor,
            "next_ssm_state" to next.ssmTensor,
        )

        val nextSequencePosition = Math.addExact(
            sequencePosition,
            graph.sequenceLength.toLong(),
        )
        var lastFailure: Throwable? = null
        var lastProvider = invocation.providers.last()
        var totalAttemptNanos = 0L

        for (provider in invocation.providers) {
            lastProvider = provider
            val key = OrtProductionSessionKey(
                graphFilename = graph.filename,
                provider = provider,
                xnnpackThreads = invocation.xnnpackThreads,
            )
            val attemptStart = System.nanoTime()
            try {
                val handle = sessionCache.getOrCreate(key) {
                    val opened = sessionFactory.createForProvider(
                        modelPath = graph.file.absolutePath,
                        provider = provider,
                        xnnpackThreads = invocation.xnnpackThreads,
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
                handle.session.run(
                    inputs,
                    pinnedOutputs,
                ).use {
                    // Outputs are pinned into reusable direct buffers.
                }
                val elapsed = elapsedNanosF1(attemptStart)
                totalAttemptNanos = safeAddNanosF1(
                    totalAttemptNanos,
                    elapsed,
                )
                val logits = buffers.finalLogits()
                require(logits.all { it.isFinite() }) {
                    "F1 provider produced non-finite final logits"
                }

                currentStateSlot = nextIndex
                sequencePosition = nextSequencePosition
                autotuner.recordProviderSuccess(provider)
                updateLatencyFeedback(elapsed)
                return OrtProductionInvocationOutcome(
                    result = OrtProductionInvocationResult(
                        graphFilename = graph.filename,
                        sequenceLength = graph.sequenceLength,
                        provider = provider,
                        elapsedNanos = totalAttemptNanos,
                    ),
                    logits = logits,
                )
            } catch (error: Exception) {
                val elapsed = elapsedNanosF1(attemptStart)
                totalAttemptNanos = safeAddNanosF1(
                    totalAttemptNanos,
                    elapsed,
                )
                sessionCache.invalidate(key)
                autotuner.recordProviderFailure(provider)
                lastFailure = error
            } catch (error: UnsatisfiedLinkError) {
                val elapsed = elapsedNanosF1(attemptStart)
                totalAttemptNanos = safeAddNanosF1(
                    totalAttemptNanos,
                    elapsed,
                )
                sessionCache.invalidate(key)
                autotuner.recordProviderFailure(provider)
                lastFailure = error
            }
        }

        throw OrtProductionExecutionException(
            message = "all F1 provider attempts failed for " +
                graph.filename,
            graphFilename = graph.filename,
            lastProvider = lastProvider,
            elapsedNanos = totalAttemptNanos.coerceAtLeast(1L),
            cause = lastFailure,
        )
    }

    private fun createStateSlot(): OrtProductionStateSlot {
        val convBuffer = directFloatBufferF1(
            runtimePackage.convStateElements
        )
        val ssmBuffer = directFloatBufferF1(
            runtimePackage.ssmStateElements
        )
        val convTensor = OnnxTensor.createTensor(
            environment,
            convBuffer,
            convShape,
        )
        val ssmTensor = OnnxTensor.createTensor(
            environment,
            ssmBuffer,
            ssmShape,
        )
        return OrtProductionStateSlot(
            convBuffer,
            ssmBuffer,
            convTensor,
            ssmTensor,
        )
    }

    private fun createGraphBuffers(
        graph: OrtProductionGraph,
    ): OrtProductionGraphBuffers {
        val inputBuffer = directLongBufferF1(graph.sequenceLength)
        val inputShape = if (graph.kind == "step") {
            longArrayOf(1L)
        } else {
            longArrayOf(1L, graph.sequenceLength.toLong())
        }
        val inputTensor = OnnxTensor.createTensor(
            environment,
            inputBuffer,
            inputShape,
        )

        val logitsCount = checkedBufferCountF1(
            "logits",
            graph.sequenceLength.toLong(),
            runtimePackage.vocabSize.toLong(),
        )
        val logitsBuffer = directFloatBufferF1(logitsCount)
        val logitsShape = if (graph.kind == "step") {
            longArrayOf(
                1L,
                runtimePackage.vocabSize.toLong(),
            )
        } else {
            longArrayOf(
                1L,
                graph.sequenceLength.toLong(),
                runtimePackage.vocabSize.toLong(),
            )
        }
        val logitsTensor = OnnxTensor.createTensor(
            environment,
            logitsBuffer,
            logitsShape,
        )
        return OrtProductionGraphBuffers(
            inputBuffer = inputBuffer,
            inputTensor = inputTensor,
            logitsBuffer = logitsBuffer,
            logitsTensor = logitsTensor,
            sequenceLength = graph.sequenceLength,
            vocabSize = runtimePackage.vocabSize,
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
            "F1 ORT session input names differ from E2 contract"
        }
        require(
            session.outputNames == setOf(
                "logits",
                "next_conv_state",
                "next_ssm_state",
            )
        ) {
            "F1 ORT session output names differ from E2 contract"
        }
    }

    private fun requireTokens(inputIds: IntArray) {
        require(inputIds.isNotEmpty())
        require(
            inputIds.all {
                it >= 0 && it < runtimePackage.vocabSize
            }
        ) {
            "F1 token ID is outside runtime vocabulary"
        }
    }

    private fun updateLatencyFeedback(elapsedNanos: Long) {
        val millis = elapsedNanos.toDouble() / 1_000_000.0
        movingLatencyMs = movingLatencyMs?.let {
            it * 0.75 + millis * 0.25
        } ?: millis
    }

    private fun requireOpen() {
        check(!closed) {
            "F1 ONNX production executor is closed"
        }
    }
}

private fun directFloatBufferF1(count: Int): FloatBuffer {
    require(count > 0)
    require(count <= Int.MAX_VALUE / Float.SIZE_BYTES) {
        "F1 float buffer exceeds direct allocation bound"
    }
    return ByteBuffer.allocateDirect(
        count * Float.SIZE_BYTES
    ).order(ByteOrder.nativeOrder()).asFloatBuffer()
}

private fun directLongBufferF1(count: Int): LongBuffer {
    require(count > 0)
    require(count <= Int.MAX_VALUE / Long.SIZE_BYTES) {
        "F1 long buffer exceeds direct allocation bound"
    }
    return ByteBuffer.allocateDirect(
        count * Long.SIZE_BYTES
    ).order(ByteOrder.nativeOrder()).asLongBuffer()
}

private fun checkedBufferCountF1(
    label: String,
    vararg dimensions: Long,
): Int {
    var product = 1L
    for (dimension in dimensions) {
        require(dimension > 0L) {
            label + " dimension must be positive"
        }
        require(product <= Int.MAX_VALUE.toLong() / dimension) {
            label + " exceeds JVM direct-buffer element bound"
        }
        product *= dimension
    }
    return product.toInt()
}

private fun elapsedNanosF1(startNanos: Long): Long =
    (System.nanoTime() - startNanos).coerceAtLeast(1L)

private fun safeAddNanosF1(left: Long, right: Long): Long =
    if (Long.MAX_VALUE - left < right) {
        Long.MAX_VALUE
    } else {
        left + right
    }
