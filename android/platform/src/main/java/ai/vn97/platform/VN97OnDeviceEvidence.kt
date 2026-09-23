package ai.vn97.platform

import ai.vn97.runtime.NativeActivatedModel
import ai.vn97.runtime.NativeAudioModality
import ai.vn97.runtime.NativeBackend
import ai.vn97.runtime.NativePreparedAudio
import ai.vn97.runtime.NativeRuntimeConfig
import ai.vn97.runtime.NativeRuntimeSession
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
    val speechFrames: Int = 8,
    val prompt: String = "VN97 mobile production evidence probe.",
) {
    init {
        require(warmupRuns >= 0) { "warmupRuns must be non-negative" }
        require(measuredRuns >= 3) { "measuredRuns must be at least 3" }
        require(decodeTokens > 0) { "decodeTokens must be positive" }
        require(speechFrames > 0) { "speechFrames must be positive" }
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
    val peakPssKib: Int,
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
            append(",\"model_image_sha256\":")
            append(jsonString(modelImageSha256))
            append(",\"peak_pss_kib\":")
            append(peakPssKib)
            append(",\"runs\":")
            append(runs)
            append(",\"schema\":\"VN97MOBEVID1\"")
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
                '"' -> append("\\"")
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
        model: NativeActivatedModel,
        config: VN97MobileEvidenceConfig = VN97MobileEvidenceConfig(),
    ): VN97MobileEvidenceRecord {
        require(model.info.hasTokenizer) {
            "mobile evidence requires tokenizer-bearing VN97MI1"
        }

        val promptIds = model.encodeUtf8(
            config.prompt,
            addBos = true,
            addText = true,
            addEos = false,
        )
        require(promptIds.isNotEmpty()) {
            "mobile evidence prompt encoded to no tokens"
        }

        val preparedSpeech = if (model.info.hasAudioProjection) {
            val sampleCount = Math.multiplyExact(
                model.info.audioFrameSize,
                config.speechFrames,
            )
            NativeAudioModality.preparePcm16(
                ShortArray(sampleCount) { index ->
                    (((index % 97) - 48) * 128).toShort()
                }
            )
        } else {
            null
        }

        repeat(config.warmupRuns) {
            measureOne(
                model = model,
                promptIds = promptIds,
                preparedSpeech = preparedSpeech,
                decodeTokens = config.decodeTokens,
            )
        }

        val startEnergy = readEnergyCounterOrNull()
        var maxThermal = thermalStatus()
        var peakPss = Debug.getPss().coerceAtLeast(1)
        val prefillMs = ArrayList<Double>(config.measuredRuns)
        val decodeMsPerToken = ArrayList<Double>(config.measuredRuns)
        val speechMs = if (preparedSpeech != null) {
            ArrayList<Double>(config.measuredRuns)
        } else {
            null
        }

        repeat(config.measuredRuns) {
            val observation = measureOne(
                model = model,
                promptIds = promptIds,
                preparedSpeech = preparedSpeech,
                decodeTokens = config.decodeTokens,
            )
            prefillMs += observation.textPrefillMs
            decodeMsPerToken += observation.textDecodeMsPerToken
            observation.speechPrefillMs?.let { value ->
                speechMs?.add(value)
            }
            peakPss = maxOf(peakPss, Debug.getPss().coerceAtLeast(1))
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
            speechPrefill = speechMs?.let(::percentiles),
            peakPssKib = peakPss,
            thermalStatusMax = maxThermal,
            batteryEnergyCounterDeltaNwh = energyDelta,
        )
    }

    private fun measureOne(
        model: NativeActivatedModel,
        promptIds: IntArray,
        preparedSpeech: NativePreparedAudio?,
        decodeTokens: Int,
    ): Observation {
        val runtimeConfig = NativeRuntimeConfig(
            layers = model.info.layers,
            batch = 1,
            dModel = model.info.dModel,
            dState = model.info.dState,
            recurrentBackend = NativeBackend.AUTO,
            packedBackend = NativeBackend.AUTO,
        )

        val textSession = NativeRuntimeSession.create(runtimeConfig)
        try {
            textSession.activate()
            val prefillStart = SystemClock.elapsedRealtimeNanos()
            var logits = textSession.prefill(model, promptIds)
            val prefillEnd = SystemClock.elapsedRealtimeNanos()

            val decodeStart = SystemClock.elapsedRealtimeNanos()
            repeat(decodeTokens) {
                val token = argmax(logits)
                logits = textSession.inferStep(model, intArrayOf(token))
            }
            val decodeEnd = SystemClock.elapsedRealtimeNanos()

            val speechMs = preparedSpeech?.let { prepared ->
                val speechSession = NativeRuntimeSession.create(runtimeConfig)
                try {
                    speechSession.activate()
                    val start = SystemClock.elapsedRealtimeNanos()
                    speechSession.prefillAudio(model, prepared)
                    val end = SystemClock.elapsedRealtimeNanos()
                    nanosToMs(end - start)
                } finally {
                    speechSession.close()
                }
            }

            return Observation(
                textPrefillMs = nanosToMs(prefillEnd - prefillStart),
                textDecodeMsPerToken =
                    nanosToMs(decodeEnd - decodeStart) / decodeTokens.toDouble(),
                speechPrefillMs = speechMs,
            )
        } finally {
            textSession.close()
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
        val speechPrefillMs: Double?,
    )
}

private fun ByteArray.toLowerHex(): String = joinToString(separator = "") { byte ->
    "%02x".format(Locale.US, byte.toInt() and 0xff)
}
