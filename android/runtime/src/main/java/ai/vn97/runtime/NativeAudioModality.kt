package ai.vn97.runtime

enum class NativeModalityStatus(val code: Int) {
    OK(0),
    NULL_ARGUMENT(1),
    INVALID_SHAPE(2),
    INVALID_PARAMETER(3),
    SIZE_OVERFLOW(4),
    OUTPUT_TOO_SMALL(5);

    companion object {
        fun fromCode(code: Int): NativeModalityStatus =
            entries.firstOrNull { it.code == code }
                ?: throw IllegalStateException("unknown native modality status: " + code)
    }
}

class NativeModalityException(
    val status: NativeModalityStatus,
    operation: String,
) : IllegalStateException(
    operation + " failed: " + status.name + " (" + status.code + ")"
)

data class NativePreparedAudio(
    val frameSize: Int,
    val hopSize: Int,
    val frameCount: Int,
    val normalizedFrames: FloatArray,
) {
    init {
        require(frameSize > 0)
        require(hopSize > 0)
        require(frameCount > 0)
        require(normalizedFrames.size == frameSize * frameCount)
    }
}

internal object NativeAudioModalityBindings {
    init {
        System.loadLibrary("vn97_jni")
    }

    external fun nativeFrameCount(
        sampleCount: Int,
        frameSize: Int,
        hopSize: Int,
        out: IntArray,
    ): Int

    external fun nativePrepare(
        pcm16: ShortArray,
        frameSize: Int,
        hopSize: Int,
        eps: Float,
        out: FloatArray,
    ): Int
}

object NativeAudioModality {
    const val SAMPLE_RATE_HZ = 16_000
    const val FRAME_SIZE = 320
    const val HOP_SIZE = 320
    const val MAX_UTTERANCE_SAMPLES = SAMPLE_RATE_HZ * 30

    fun preparePcm16(
        pcm16: ShortArray,
        frameSize: Int = FRAME_SIZE,
        hopSize: Int = HOP_SIZE,
        eps: Float = 1e-5f,
    ): NativePreparedAudio {
        require(pcm16.isNotEmpty()) { "pcm16 must not be empty" }
        require(pcm16.size <= MAX_UTTERANCE_SAMPLES) {
            "voice utterance exceeds 30 second bound"
        }
        require(frameSize > 0 && hopSize > 0)
        require(eps.isFinite() && eps > 0f)

        val countOut = IntArray(1)
        checkStatus(
            NativeAudioModalityBindings.nativeFrameCount(
                pcm16.size,
                frameSize,
                hopSize,
                countOut,
            ),
            "audio frame count",
        )
        val frameCount = countOut[0]
        require(frameCount > 0) { "voice utterance is shorter than one audio frame" }

        val totalValues = Math.multiplyExact(frameCount, frameSize)
        val output = FloatArray(totalValues)
        checkStatus(
            NativeAudioModalityBindings.nativePrepare(
                pcm16,
                frameSize,
                hopSize,
                eps,
                output,
            ),
            "audio frame preprocessing",
        )
        return NativePreparedAudio(
            frameSize = frameSize,
            hopSize = hopSize,
            frameCount = frameCount,
            normalizedFrames = output,
        )
    }

    private fun checkStatus(code: Int, operation: String) {
        val status = NativeModalityStatus.fromCode(code)
        if (status != NativeModalityStatus.OK) {
            throw NativeModalityException(status, operation)
        }
    }
}
