package ai.vn97.runtime

import java.nio.ByteBuffer
import java.nio.CharBuffer
import java.nio.charset.CodingErrorAction
import java.nio.charset.StandardCharsets

enum class NativeSamplerStatus(val code: Int) {
    OK(0),
    NULL_ARGUMENT(1),
    INVALID_CONFIG(2),
    EMPTY_LOGITS(3),
    NON_FINITE(4),
    OUT_OF_MEMORY(5),
    WORKSPACE_TOO_SMALL(6);

    companion object {
        internal fun fromCode(code: Int): NativeSamplerStatus =
            entries.firstOrNull { it.code == code }
                ?: throw IllegalStateException("unknown native sampler status: $code")
    }
}

class NativeSamplerException(
    val status: NativeSamplerStatus,
) : IllegalStateException("native sampler failed: ${status.name} (${status.code})")

data class NativeSamplerConfig(
    val temperature: Float = 0.8f,
    val topK: Int = 40,
    val topP: Float = 0.95f,
    val seed: Long = 97L,
) {
    init {
        require(temperature.isFinite() && temperature >= 0.0f) {
            "temperature must be finite and non-negative"
        }
        require(topK >= 0) { "topK must be non-negative; zero disables the limit" }
        require(topP.isFinite() && topP > 0.0f && topP <= 1.0f) {
            "topP must be finite and in (0, 1]"
        }
    }
}

internal object NativeSamplerBindings {
    init {
        System.loadLibrary("vn97_jni")
    }

    external fun nativeSample(
        logits: FloatArray,
        temperature: Float,
        topK: Int,
        topP: Float,
        seed: Long,
        step: Long,
        tokenOut: IntArray,
    ): Int
}

object NativeSampler {
    fun sample(logits: FloatArray, config: NativeSamplerConfig, step: Long): Int {
        require(logits.isNotEmpty()) { "logits must not be empty" }
        require(step >= 0L) { "sampler step must be non-negative" }
        val tokenOut = IntArray(1)
        val status = NativeSamplerStatus.fromCode(
            NativeSamplerBindings.nativeSample(
                logits,
                config.temperature,
                config.topK,
                config.topP,
                config.seed,
                step,
                tokenOut,
            )
        )
        if (status != NativeSamplerStatus.OK) throw NativeSamplerException(status)
        check(tokenOut[0] in logits.indices) { "native sampler returned an out-of-range token" }
        return tokenOut[0]
    }
}

enum class NativeGenerationStatus(val code: Int) {
    OK(0),
    NULL_ARGUMENT(1),
    INVALID_CONFIG(2),
    INVALID_HANDLE(3),
    MODEL_ERROR(4),
    RUNTIME_ERROR(5),
    SAMPLER_ERROR(6),
    SEQUENCE_MISMATCH(7),
    COUNTER_OVERFLOW(8),
    OUT_OF_MEMORY(9),
    FINISHED(10);

    companion object {
        internal fun fromCode(code: Int): NativeGenerationStatus =
            entries.firstOrNull { it.code == code }
                ?: throw IllegalStateException("unknown native generation status: $code")
    }
}

class NativeGenerationException(
    val status: NativeGenerationStatus,
    operation: String,
) : IllegalStateException("$operation failed: ${status.name} (${status.code})")

data class NativeGenerationConfig(
    val maxNewTokens: Int = 256,
    val eosToken: Int = 2,
    val addBosOnFreshSession: Boolean = true,
    val addTextControl: Boolean = true,
    val sampler: NativeSamplerConfig = NativeSamplerConfig(),
) {
    init {
        require(maxNewTokens > 0) { "maxNewTokens must be positive" }
        require(eosToken >= -1) { "eosToken must be -1 or non-negative" }
    }
}

enum class NativeGenerationStopReason {
    EOS,
    TOKEN_LIMIT,
    CANCELLED,
}

data class NativeGenerationChunk(
    val tokenIndex: Int,
    val tokenId: Int?,
    val text: String,
    val final: Boolean,
)

data class NativeGenerationResult(
    val tokenIds: IntArray,
    val text: String,
    val stopReason: NativeGenerationStopReason,
    val sequencePosition: Long,
)

internal object NativeGenerationBindings {
    init {
        System.loadLibrary("vn97_jni")
    }

    external fun nativeOpen(
        modelHandle: Long,
        runtimeHandle: Long,
        promptIds: IntArray,
        temperature: Float,
        topK: Int,
        topP: Float,
        seed: Long,
        eosToken: Int,
        handleOut: LongArray,
    ): Int
    external fun nativeNext(handle: Long, tokenOut: IntArray, eosOut: IntArray): Int
    external fun nativeDestroy(handle: Long): Int
}

private data class NativeCursorToken(val tokenId: Int, val eos: Boolean)

private class NativeGenerationCursor private constructor(private var handle: Long) : AutoCloseable {
    companion object {
        fun open(
            modelHandle: Long,
            runtimeHandle: Long,
            promptIds: IntArray,
            config: NativeGenerationConfig,
        ): NativeGenerationCursor {
            val out = LongArray(1)
            checkGenerationStatus(
                NativeGenerationBindings.nativeOpen(
                    modelHandle,
                    runtimeHandle,
                    promptIds,
                    config.sampler.temperature,
                    config.sampler.topK,
                    config.sampler.topP,
                    config.sampler.seed,
                    config.eosToken,
                    out,
                ),
                "generation cursor open",
            )
            check(out[0] != 0L) { "native generation returned a zero handle" }
            return NativeGenerationCursor(out[0])
        }
    }

    fun next(): NativeCursorToken {
        check(handle != 0L) { "native generation cursor is closed" }
        val token = IntArray(1)
        val eos = IntArray(1)
        checkGenerationStatus(
            NativeGenerationBindings.nativeNext(handle, token, eos),
            "generation next token",
        )
        require(token[0] >= 0) { "native generation returned a negative token" }
        return NativeCursorToken(token[0], eos[0] != 0)
    }

    override fun close() {
        val old = handle
        handle = 0L
        if (old != 0L) {
            val status = NativeGenerationStatus.fromCode(NativeGenerationBindings.nativeDestroy(old))
            if (status != NativeGenerationStatus.OK && status != NativeGenerationStatus.INVALID_HANDLE) {
                throw NativeGenerationException(status, "generation cursor destroy")
            }
        }
    }
}

internal class Utf8StreamAccumulator {
    private val decoder = StandardCharsets.UTF_8.newDecoder()
        .onMalformedInput(CodingErrorAction.REPLACE)
        .onUnmappableCharacter(CodingErrorAction.REPLACE)
    private var pending = ByteArray(0)
    private var finished = false

    fun append(bytes: ByteArray): String {
        check(!finished) { "UTF-8 stream is already finished" }
        return decode(bytes, endOfInput = false)
    }

    fun finish(): String {
        check(!finished) { "UTF-8 stream is already finished" }
        finished = true
        return decode(ByteArray(0), endOfInput = true)
    }

    private fun decode(bytes: ByteArray, endOfInput: Boolean): String {
        val combined = ByteArray(pending.size + bytes.size)
        pending.copyInto(combined, 0)
        bytes.copyInto(combined, pending.size)
        val input = ByteBuffer.wrap(combined)
        val output = CharBuffer.allocate(maxOf(8, combined.size + 8))
        val result = decoder.decode(input, output, endOfInput)
        check(!result.isOverflow) { "UTF-8 decoder output bound was exceeded" }
        if (endOfInput) {
            check(!result.isError) { "UTF-8 decoder failed at end of stream" }
            val flush = decoder.flush(output)
            check(!flush.isOverflow && !flush.isError) { "UTF-8 decoder flush failed" }
            pending = ByteArray(0)
        } else {
            pending = ByteArray(input.remaining())
            input.get(pending)
        }
        output.flip()
        return output.toString()
    }
}

private fun NativeActivatedModel.decodeTokenBytes(tokenId: Int): ByteArray {
    require(tokenId >= 0) { "token ID must be non-negative" }
    return decodeBytes(intArrayOf(tokenId), skipControl = true)
}

fun NativeRuntimeSession.generateStreaming(
    model: NativeActivatedModel,
    prompt: String,
    config: NativeGenerationConfig = NativeGenerationConfig(),
    onChunk: (NativeGenerationChunk) -> Boolean,
): NativeGenerationResult {
    require(model.info.hasTokenizer) { "streaming generation requires VN97TK1 in the activated model" }
    val before = info()
    require(before.lifecycle == RuntimeLifecycle.ACTIVE) { "runtime must be ACTIVE for generation" }
    require(before.config.batch == 1) { "streaming generation currently requires batch=1" }

    val promptIds = model.encodeUtf8(
        prompt,
        addBos = config.addBosOnFreshSession && before.sequencePosition == 0L,
        addText = config.addTextControl,
        addEos = false,
    )
    require(promptIds.isNotEmpty()) { "encoded prompt must not be empty" }

    return model.withHandle { modelHandle ->
        withHandle { runtimeHandle ->
            NativeGenerationCursor.open(modelHandle, runtimeHandle, promptIds, config).use { cursor ->
                val generated = IntArray(config.maxNewTokens)
                var generatedCount = 0
                val text = StringBuilder()
                val utf8 = Utf8StreamAccumulator()
                var stopReason = NativeGenerationStopReason.TOKEN_LIMIT

                for (index in 0 until config.maxNewTokens) {
                    val next = cursor.next()
                    generated[generatedCount++] = next.tokenId
                    val piece = utf8.append(model.decodeTokenBytes(next.tokenId))
                    text.append(piece)
                    val keepGoing = onChunk(
                        NativeGenerationChunk(
                            tokenIndex = index,
                            tokenId = next.tokenId,
                            text = piece,
                            final = false,
                        )
                    )
                    if (next.eos) {
                        stopReason = NativeGenerationStopReason.EOS
                        break
                    }
                    if (!keepGoing) {
                        stopReason = NativeGenerationStopReason.CANCELLED
                        break
                    }
                }

                val tail = utf8.finish()
                if (tail.isNotEmpty()) {
                    text.append(tail)
                    onChunk(
                        NativeGenerationChunk(
                            tokenIndex = generatedCount,
                            tokenId = null,
                            text = tail,
                            final = true,
                        )
                    )
                }

                NativeGenerationResult(
                    tokenIds = generated.copyOf(generatedCount),
                    text = text.toString(),
                    stopReason = stopReason,
                    sequencePosition = info().sequencePosition,
                )
            }
        }
    }
}

private fun checkGenerationStatus(code: Int, operation: String) {
    val status = NativeGenerationStatus.fromCode(code)
    if (status != NativeGenerationStatus.OK) {
        throw NativeGenerationException(status, operation)
    }
}
