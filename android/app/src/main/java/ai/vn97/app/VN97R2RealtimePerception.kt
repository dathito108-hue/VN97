package ai.vn97.app

import ai.vn97.runtime.NativePreparedVision
import kotlin.math.abs
import kotlin.math.sqrt

/**
 * Deterministic R2 visual evidence adapter.
 *
 * This is not a second AI backend. It converts the already prepared frame into
 * bounded numeric evidence for the text-only R2 cognition graph until a signed
 * R2 vision ONNX graph is packaged.
 */
object VN97R2RealtimePerception {
    fun summarizeVision(
        prepared: NativePreparedVision,
    ): String {
        val values = prepared.normalizedPatches
        require(values.isNotEmpty())
        var absSum = 0.0
        var squareSum = 0.0
        var positive = 0
        var negative = 0
        var zeroish = 0
        var fingerprint = 0xcbf29ce484222325UL
        val stride = maxOf(1, values.size / MAX_FINGERPRINT_SAMPLES)
        var index = 0
        var sampled = 0
        while (index < values.size && sampled < MAX_FINGERPRINT_SAMPLES) {
            val value = values[index]
            require(value.isFinite()) {
                "prepared vision contains non-finite evidence"
            }
            val d = value.toDouble()
            absSum += abs(d)
            squareSum += d * d
            when {
                value > ZEROISH -> positive += 1
                value < -ZEROISH -> negative += 1
                else -> zeroish += 1
            }
            val quantized =
                (value.coerceIn(-4f, 4f) * 1024f).toInt()
            fingerprint =
                (fingerprint xor quantized.toUInt().toULong()) *
                    0x100000001b3UL
            index += stride
            sampled += 1
        }
        require(sampled > 0)
        val meanAbs = absSum / sampled.toDouble()
        val rms = sqrt(squareSum / sampled.toDouble())
        return buildString {
            append("VN97R2VISIONFEATURE1")
            append(" width=")
            append(prepared.width)
            append(" height=")
            append(prepared.height)
            append(" patches=")
            append(prepared.patchCount)
            append(" samples=")
            append(sampled)
            append(" mean_abs_milli=")
            append((meanAbs * 1000.0).toInt())
            append(" rms_milli=")
            append((rms * 1000.0).toInt())
            append(" positive=")
            append(positive)
            append(" negative=")
            append(negative)
            append(" zeroish=")
            append(zeroish)
            append(" fingerprint=")
            append(fingerprint.toString(16).padStart(16, '0'))
            append(" semantic_vision=false")
        }
    }

    private const val ZEROISH = 1e-3f
    private const val MAX_FINGERPRINT_SAMPLES = 1024
}
