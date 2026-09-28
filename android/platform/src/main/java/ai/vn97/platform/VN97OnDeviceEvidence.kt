package ai.vn97.platform

import ai.vn97.runtime.VN97G06Model
import android.content.Context
import android.os.BatteryManager
import android.os.Build
import android.os.Debug
import android.os.PowerManager
import android.os.SystemClock
import java.util.Locale
import kotlin.math.ceil

data class VN97MobileEvidenceConfig(
    val warmupRuns: Int = 1,
    val measuredRuns: Int = 5,
    val decodeTokens: Int = 16,
    val prompt: String = "VN97 mobile production evidence probe.",
) {
    init {
        require(warmupRuns >= 0) { "warmupRuns must be non-negative" }
        require(measuredRuns >= 3) { "measuredRuns must be at least 3" }
        require(decodeTokens > 0) { "decodeTokens must be positive" }
        require(prompt.isNotBlank()) { "prompt must be non-blank" }
    }
}

data class VN97LatencyEvidence(
    val p50Ms: Double,
    val p95Ms: Double,
) {
    init {
        require(p50Ms.isFinite() && p50Ms >= 0.0)
        require(p95Ms.isFinite() && p95Ms >= p50Ms)
    }
}

data class VN97MobileEvidenceRecord(
    val modelImageSha256: String,
    val manufacturer: String,
    val model: String,
    val sdkInt: Int,
    val abi: String,
    val runs: Int,
    val textPrefill: VN97LatencyEvidence,
    val textDecodePerToken: VN97LatencyEvidence,
    val speechPrefill: VN97LatencyEvidence?,
    val peakPssKib: Long,
    val thermalStatusMax: Int,
    val batteryEnergyCounterDeltaNwh: Long?,
) {
    init {
        require(modelImageSha256.matches(Regex("[0-9a-f]{64}")))
        require(manufacturer.isNotBlank() && model.isNotBlank() && abi.isNotBlank())
        require(sdkInt > 0 && runs > 0)
        require(peakPssKib > 0)
        require(thermalStatusMax in 0..6)
    }

    fun toCanonicalJson(): String {
        val energy = batteryEnergyCounterDeltaNwh?.toString() ?: "null"
        val speech = speechPrefill?.let(::latencyJson) ?: "null"
        return buildString {
            append("{")
            append("\"battery_energy_counter_delta_nwh\":")
            append(energy)
            append(",\"device\":{")
            append("\"abi\":")
            append(jsonString(abi))
            append(",\"manufacturer\":")
            append(jsonString(manufacturer))
            append(",\"model\":")
            append(jsonString(model))
            append(",\"sdk_int\":")
            append(sdkInt)
            append("}")
            append(",\"deployment_identity_sha256\":")
            append(jsonString(modelImageSha256))
            append(",\"peak_pss_kib\":")
            append(peakPssKib)
            append(",\"runs\":")
            append(runs)
            append(",\"schema\":\"VN97G06MOBEVID1\"")
            append(",\"speech_prefill\":")
            append(speech)
            append(",\"text_decode_per_token\":")
            append(latencyJson(textDecodePerToken))
            append(",\"text_prefill\":")
            append(latencyJson(textPrefill))
            append(",\"thermal_status_max\":")
            append(thermalStatusMax)
            append("}")
        }
    }

    private fun latencyJson(value: VN97LatencyEvidence): String =
        "{\"p50_ms\":${number(value.p50Ms)},\"p95_ms\":${number(value.p95Ms)}}"

    private fun number(value: Double): String =
        String.format(Locale.US, "%.6f", value)
            .trimEnd('0')
            .trimEnd('.')
            .ifEmpty { "0" }

    private fun jsonString(value: String): String = buildString {
        append('"')
        for (ch in value) {
            when (ch) {
                '"' -> append("\\\"")
                '\\' -> append("\\\\")
                '\b' -> append("\\b")
                '\u000c' -> append("\\f")
                '\n' -> append("\\n")
                '\r' -> append("\\r")
                '\t' -> append("\\t")
                else -> {
                    if (ch.code < 0x20) {
                        append("\\u")
                        append(ch.code.toString(16).padStart(4, '0'))
                    } else {
                        append(ch)
                    }
                }
            }
        }
        append('"')
    }
}

class VN97OnDeviceEvidenceCollector(
    context: Context,
) {
    private val appContext = context.applicationContext
    private val powerManager =
        appContext.getSystemService(Context.POWER_SERVICE) as PowerManager
    private val batteryManager =
        appContext.getSystemService(Context.BATTERY_SERVICE) as BatteryManager

    fun collect(
        model: VN97G06Model,
        config: VN97MobileEvidenceConfig = VN97MobileEvidenceConfig(),
    ): VN97MobileEvidenceRecord {
        require(model.info.hasTokenizer) {
            "mobile evidence requires G06 tokenizer"
        }

        val promptIds = model.tokenizer.encode(config.prompt)
        require(promptIds.isNotEmpty())
        repeat(config.warmupRuns) {
            measureOne(
                model = model,
                promptIds = promptIds,
                decodeTokens = config.decodeTokens,
            )
        }

        val startEnergy = readEnergyCounterOrNull()
        var maxThermal = thermalStatus()
        var peakPss = Debug.getPss().coerceAtLeast(1L)
        val prefillMs = ArrayList<Double>(config.measuredRuns)
        val decodeMsPerToken = ArrayList<Double>(config.measuredRuns)
        repeat(config.measuredRuns) {
            val observation = measureOne(
                model = model,
                promptIds = promptIds,
                decodeTokens = config.decodeTokens,
            )
            prefillMs += observation.textPrefillMs
            decodeMsPerToken += observation.textDecodeMsPerToken
            peakPss = maxOf(peakPss, Debug.getPss().coerceAtLeast(1L))
            maxThermal = maxOf(maxThermal, thermalStatus())
        }

        val endEnergy = readEnergyCounterOrNull()
        val energyDelta = if (startEnergy != null && endEnergy != null) {
            endEnergy - startEnergy
        } else {
            null
        }

        return VN97MobileEvidenceRecord(
            modelImageSha256 = model.info.modelId.toLowerHex(),
            manufacturer = Build.MANUFACTURER.ifBlank { "unknown" },
            model = Build.MODEL.ifBlank { "unknown" },
            sdkInt = Build.VERSION.SDK_INT,
            abi = Build.SUPPORTED_ABIS.firstOrNull().orEmpty().ifBlank { "unknown" },
            runs = config.measuredRuns,
            textPrefill = percentiles(prefillMs),
            textDecodePerToken = percentiles(decodeMsPerToken),
            speechPrefill = null,
            peakPssKib = peakPss,
            thermalStatusMax = maxThermal,
            batteryEnergyCounterDeltaNwh = energyDelta,
        )
    }

    private fun measureOne(
        model: VN97G06Model,
        promptIds: IntArray,
        decodeTokens: Int,
    ): Observation {
        model.requireOpen()
        return ai.vn97.runtime.VN97G06CognitionInference.open(appContext, model).use {
            val timing = it.measurePrefillDecode(promptIds, decodeTokens)
            Observation(nanosToMs(timing.first), nanosToMs(timing.second) / decodeTokens.toDouble())
        }
    }

    private fun thermalStatus(): Int =
        if (Build.VERSION.SDK_INT >= 29) {
            powerManager.currentThermalStatus.coerceIn(0, 6)
        } else {
            0
        }

    private fun readEnergyCounterOrNull(): Long? {
        val value = batteryManager.getLongProperty(
            BatteryManager.BATTERY_PROPERTY_ENERGY_COUNTER
        )
        return if (value == Long.MIN_VALUE || value < 0L) null else value
    }

    private fun percentiles(values: List<Double>): VN97LatencyEvidence {
        require(values.isNotEmpty())
        val sorted = values.sorted()
        return VN97LatencyEvidence(
            p50Ms = percentile(sorted, 0.50),
            p95Ms = percentile(sorted, 0.95),
        )
    }

    private fun percentile(sorted: List<Double>, q: Double): Double {
        val index = ceil(q * sorted.size).toInt().coerceIn(1, sorted.size) - 1
        return sorted[index]
    }

    private fun argmax(values: FloatArray): Int {
        require(values.isNotEmpty())
        var bestIndex = 0
        var bestValue = values[0]
        for (index in 1 until values.size) {
            if (values[index] > bestValue) {
                bestValue = values[index]
                bestIndex = index
            }
        }
        return bestIndex
    }

    private fun nanosToMs(value: Long): Double =
        value.toDouble() / 1_000_000.0

    private data class Observation(
        val textPrefillMs: Double,
        val textDecodeMsPerToken: Double,
    )
}

private fun ByteArray.toLowerHex(): String = joinToString(separator = "") { byte ->
    "%02x".format(Locale.US, byte.toInt() and 0xff)
}
