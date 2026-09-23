package ai.vn97.runtime

data class NativePreparedVision(
    val width: Int,
    val height: Int,
    val channels: Int,
    val patchSize: Int,
    val patchCount: Int,
    val inputFeatures: Int,
    val normalizedPatches: FloatArray,
) {
    init {
        require(width > 0 && height > 0)
        require(channels == NativeVisionModality.CHANNELS)
        require(patchSize == NativeVisionModality.PATCH_SIZE)
        require(patchCount > 0)
        require(inputFeatures == channels * patchSize * patchSize + 2)
        require(normalizedPatches.size == patchCount * inputFeatures)
    }
}

internal object NativeVisionModalityBindings {
    init {
        System.loadLibrary("vn97_jni")
    }

    external fun nativePatchCount(
        height: Int,
        width: Int,
        patchSize: Int,
        out: IntArray,
    ): Int

    external fun nativePrepareRgb888(
        rgb888: ByteArray,
        width: Int,
        height: Int,
        patchSize: Int,
        eps: Float,
        out: FloatArray,
    ): Int
}

object NativeVisionModality {
    const val CHANNELS = 3
    const val PATCH_SIZE = 16
    const val EPS = 1e-5f
    const val MAX_WIDTH = 224
    const val MAX_HEIGHT = 224

    fun prepareRgb888(
        rgb888: ByteArray,
        width: Int,
        height: Int,
    ): NativePreparedVision {
        require(width > 0 && height > 0) {
            "vision dimensions must be positive"
        }
        require(width <= MAX_WIDTH && height <= MAX_HEIGHT) {
            "vision frame exceeds 224x224 production bound"
        }
        require(width % PATCH_SIZE == 0 && height % PATCH_SIZE == 0) {
            "vision dimensions must be divisible by 16"
        }
        val pixelCount = Math.multiplyExact(width, height)
        val expectedBytes = Math.multiplyExact(pixelCount, CHANNELS)
        require(rgb888.size == expectedBytes) {
            "RGB888 byte count does not match width/height"
        }

        val countOut = IntArray(1)
        checkStatus(
            NativeVisionModalityBindings.nativePatchCount(
                height,
                width,
                PATCH_SIZE,
                countOut,
            ),
            "vision patch count",
        )
        val patchCount = countOut[0]
        val features = CHANNELS * PATCH_SIZE * PATCH_SIZE + 2
        val output = FloatArray(Math.multiplyExact(patchCount, features))
        checkStatus(
            NativeVisionModalityBindings.nativePrepareRgb888(
                rgb888,
                width,
                height,
                PATCH_SIZE,
                EPS,
                output,
            ),
            "vision patch preprocessing",
        )
        return NativePreparedVision(
            width = width,
            height = height,
            channels = CHANNELS,
            patchSize = PATCH_SIZE,
            patchCount = patchCount,
            inputFeatures = features,
            normalizedPatches = output,
        )
    }

    private fun checkStatus(code: Int, operation: String) {
        val status = NativeModalityStatus.fromCode(code)
        if (status != NativeModalityStatus.OK) {
            throw NativeModalityException(status, operation)
        }
    }
}
