package ai.vn97.app

import ai.vn97.runtime.NativePreparedVision
import ai.vn97.runtime.NativeVisionModality
import android.graphics.Bitmap

internal object VN97BitmapVision {
    fun prepare(bitmap: Bitmap): NativePreparedVision {
        require(bitmap.width > 0 && bitmap.height > 0) {
            "bitmap dimensions must be positive"
        }
        val side = minOf(bitmap.width, bitmap.height)
        val left = (bitmap.width - side) / 2
        val top = (bitmap.height - side) / 2
        val cropped = Bitmap.createBitmap(
            bitmap,
            left,
            top,
            side,
            side,
        )
        val scaled = if (
            cropped.width == NativeVisionModality.MAX_WIDTH &&
            cropped.height == NativeVisionModality.MAX_HEIGHT
        ) {
            cropped
        } else {
            Bitmap.createScaledBitmap(
                cropped,
                NativeVisionModality.MAX_WIDTH,
                NativeVisionModality.MAX_HEIGHT,
                true,
            )
        }
        try {
            val width = scaled.width
            val height = scaled.height
            val pixels = IntArray(width * height)
            scaled.getPixels(
                pixels,
                0,
                width,
                0,
                0,
                width,
                height,
            )
            val rgb = ByteArray(width * height * 3)
            var out = 0
            for (pixel in pixels) {
                rgb[out++] = ((pixel ushr 16) and 0xff).toByte()
                rgb[out++] = ((pixel ushr 8) and 0xff).toByte()
                rgb[out++] = (pixel and 0xff).toByte()
            }
            return NativeVisionModality.prepareRgb888(
                rgb,
                width,
                height,
            )
        } finally {
            if (scaled !== cropped) {
                scaled.recycle()
            }
            if (cropped !== bitmap) {
                cropped.recycle()
            }
        }
    }
}
