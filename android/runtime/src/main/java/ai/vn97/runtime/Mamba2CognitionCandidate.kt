package ai.vn97.runtime

import android.content.Context
import java.io.File
import java.nio.file.Files
import java.security.MessageDigest
import java.util.Random
import kotlin.math.exp
import kotlin.math.sqrt
import org.json.JSONArray
import org.json.JSONObject

data class Mamba2CandidateSamplerConfig(
    val temperature: Float = 0.0f,
    val topK: Int = 1,
    val topP: Float = 1.0f,
    val seed: Long = 97L,
) {
    init {
        require(temperature >= 0.0f && temperature.isFinite())
        require(topK > 0)
        require(topP > 0.0f && topP <= 1.0f)
    }
}

data class Mamba2CandidateGenerationResult(
    val text: String,
    val tokenIds: IntArray,
    val stopReason: NativeGenerationStopReason,
    val sequencePosition: Long,
)

data class Mamba2CognitionBridgeBinding(
    val bridgeId: String,
    val runtimeId: String,
    val g05ManifestId: String,
    val capsuleId: String,
    val sourceWeightSha256: String,
    val tokenizerId: String,
    val tuningId: String,
    val deviceProfileReceiptId: String,
    val tokenIdSpace: Int,
    val runtimeLogitsSize: Int,
    val eosTokenId: Int,
) {
    init {
        requireG09Sha(bridgeId, "G0.9 bridge ID")
        requireG09Sha(runtimeId, "G0.9 runtime ID")
        requireG09Sha(g05ManifestId, "G0.9 G0.5 manifest ID")
        requireG09Sha(capsuleId, "G0.9 capsule ID")
        requireG09Sha(sourceWeightSha256, "G0.9 source weight SHA-256")
        requireG09Sha(tokenizerId, "G0.9 tokenizer ID")
        requireG09Sha(tuningId, "G0.9 tuning ID")
        requireG09Sha(
            deviceProfileReceiptId,
            "G0.9 device profile receipt ID",
        )
        require(tokenIdSpace == 50_277)
        require(runtimeLogitsSize == 50_288)
        require(eosTokenId in 0 until tokenIdSpace)
    }

    fun requireCompatible(
        runtime: Mamba2OrtRuntimePackage,
        tokenizer: Mamba2TokenizerPackage,
        activeTuningId: String?,
    ) {
        require(runtime.runtimeId == runtimeId) {
            "G0.9 binding runtime identity mismatch"
        }
        require(runtime.g05ManifestId == g05ManifestId) {
            "G0.9 binding G0.5 manifest identity mismatch"
        }
        require(runtime.capsuleId == capsuleId) {
            "G0.9 binding runtime capsule identity mismatch"
        }
        require(runtime.sourceWeightSha256 == sourceWeightSha256) {
            "G0.9 binding source-weight identity mismatch"
        }
        require(tokenizer.tokenizerId == tokenizerId) {
            "G0.9 binding tokenizer identity mismatch"
        }
        require(tokenizer.capsuleId == capsuleId) {
            "G0.9 binding tokenizer capsule identity mismatch"
        }
        require(tokenizer.tokenIdSpace == tokenIdSpace) {
            "G0.9 binding tokenizer ID-space mismatch"
        }
        require(tokenizer.runtimeLogitsSize == runtimeLogitsSize) {
            "G0.9 binding tokenizer logits-width mismatch"
        }
        require(runtime.vocabSize == runtimeLogitsSize) {
            "G0.9 binding runtime logits-width mismatch"
        }
        require(tokenizer.eosTokenId == eosTokenId) {
            "G0.9 binding EOS identity mismatch"
        }
        require(activeTuningId == tuningId) {
            "G0.9 binding requires exact device-measured tuning"
        }
    }

    companion object {
        const val SCHEMA = "VN97M2G09BRIDGE1"
        const val FILENAME = "binding.vn97m2g09.json"

        fun load(file: File): Mamba2CognitionBridgeBinding {
            require(
                file.isFile &&
                    !Files.isSymbolicLink(file.toPath())
            ) {
                "G0.9 bridge binding must be regular non-symlink file"
            }
            val root = JSONObject(file.readText(Charsets.US_ASCII))
            val expectedFields = setOf(
                "schema",
                "runtime_id",
                "g05_manifest_id",
                "capsule_id",
                "source_weight_sha256",
                "tokenizer_id",
                "tuning_id",
                "device_profile_receipt_id",
                "token_id_space",
                "runtime_logits_size",
                "eos_token_id",
                "same_weights_semantics",
                "same_token_ids_required",
                "padded_logits_must_be_masked",
                "device_measured_tuning_required",
                "candidate_validation_only",
                "production_activation_authorized",
                "bridge_id",
            )
            require(root.keys().asSequence().toSet() == expectedFields) {
                "G0.9 bridge binding fields mismatch"
            }
            require(root.getString("schema") == SCHEMA)
            require(root.getBoolean("same_weights_semantics"))
            require(root.getBoolean("same_token_ids_required"))
            require(root.getBoolean("padded_logits_must_be_masked"))
            require(root.getBoolean("device_measured_tuning_required"))
            require(root.getBoolean("candidate_validation_only"))
            require(!root.getBoolean("production_activation_authorized"))

            val bridgeId = root.getString("bridge_id")
            requireG09Sha(bridgeId, "G0.9 bridge ID")
            val body = JSONObject(root.toString())
            body.remove("bridge_id")
            val prefix = "VN97M2G09BRIDGE1" +
                String(charArrayOf(0.toChar()))
            val expected = sha256G09(
                prefix.toByteArray(Charsets.US_ASCII) +
                    canonicalJsonG09(body)
                        .toByteArray(Charsets.US_ASCII)
            )
            require(bridgeId == expected) {
                "G0.9 bridge binding identity mismatch"
            }

            return Mamba2CognitionBridgeBinding(
                bridgeId = bridgeId,
                runtimeId = root.getString("runtime_id"),
                g05ManifestId = root.getString("g05_manifest_id"),
                capsuleId = root.getString("capsule_id"),
                sourceWeightSha256 =
                    root.getString("source_weight_sha256"),
                tokenizerId = root.getString("tokenizer_id"),
                tuningId = root.getString("tuning_id"),
                deviceProfileReceiptId =
                    root.getString("device_profile_receipt_id"),
                tokenIdSpace = root.getInt("token_id_space"),
                runtimeLogitsSize =
                    root.getInt("runtime_logits_size"),
                eosTokenId = root.getInt("eos_token_id"),
            )
        }
    }
}

/**
 * G0.9 candidate-only text cognition bridge.
 *
 * The app opens this engine through VN97G06CognitionInference after G0.10
 * validation. Direct opening remains available for candidate diagnostics. It exists
 * so the exact G0.6/G0.7/G0.8 stack can be validated end-to-end before G0.10
 * evidence promotion. There is no fallback to the old F1/VN97TK1 inference
 * path inside this class.
 */
class VN97Mamba2CognitionCandidate private constructor(
    private val executor: VN97Mamba2OrtExecutor,
    private val tokenizer: VN97GptNeoXTokenizer,
    val binding: Mamba2CognitionBridgeBinding,
    private val config: NativeCognitionRuntimeConfig,
) : NativeCognitionInference, AutoCloseable {
    private val lock = Any()
    private var closed = false

    companion object {
        fun open(
            context: Context,
            runtimeRoot: File,
            tokenizerRoot: File,
            tuningFile: File,
            bindingFile: File,
            qnnBackendPath: String? = null,
            config: NativeCognitionRuntimeConfig =
                NativeCognitionRuntimeConfig(),
        ): VN97Mamba2CognitionCandidate {
            val packageInfo = Mamba2TokenizerPackage.load(tokenizerRoot)
            val tokenizer = VN97GptNeoXTokenizer(packageInfo)
            val executor = VN97Mamba2OrtExecutor.open(
                context = context,
                runtimeRoot = runtimeRoot,
                qnnBackendPath = qnnBackendPath,
                tuningFile = tuningFile,
                maxCachedSessions = 1,
            )
            try {
                require(executor.hasDeviceMeasuredTuning()) {
                    "G0.9 candidate requires device-measured G0.7 tuning"
                }
                val binding = Mamba2CognitionBridgeBinding.load(bindingFile)
                binding.requireCompatible(
                    runtime = executor.runtimePackage,
                    tokenizer = packageInfo,
                    activeTuningId = executor.activeTuningId(),
                )
                return VN97Mamba2CognitionCandidate(
                    executor = executor,
                    tokenizer = tokenizer,
                    binding = binding,
                    config = config,
                )
            } catch (error: Throwable) {
                executor.close()
                throw error
            }
        }
    }

    override fun generateOperation(
        operation: NativeCognitionOperation,
        requestJson: String,
    ): String {
        val prompt = NativeCognitionPrompt.build(
            operation = operation,
            requestJson = requestJson,
            maxPromptUtf8Bytes =
                config.adapterConfig.maxPromptUtf8Bytes,
        )
        return generateText(
            prompt = prompt,
            maxNewTokens =
                config.adapterConfig.maxNewTokens(operation),
        ).text
    }

    fun generateText(
        prompt: String,
        maxNewTokens: Int,
        sampler: Mamba2CandidateSamplerConfig =
            Mamba2CandidateSamplerConfig(),
    ): Mamba2CandidateGenerationResult = synchronized(lock) {
        check(!closed) { "G0.9 cognition candidate is closed" }
        require(maxNewTokens > 0)
        val promptIds = tokenizer.encode(prompt)
        require(promptIds.isNotEmpty()) {
            "G0.9 tokenizer produced an empty prompt"
        }
        require(promptIds.size <= config.inferenceLimits.maxPromptTokens) {
            "G0.9 prompt exceeds cognition token budget"
        }
        require(
            promptIds.all { it in 0 until binding.tokenIdSpace }
        ) {
            "G0.9 prompt contains token outside source vocabulary"
        }

        executor.resetRecurrentState()
        var run = executor.prefill(promptIds)
        val random = Random(sampler.seed)
        val generated = decodeG06Tokens(
            maxNewTokens = maxNewTokens,
            eosTokenId = binding.eosTokenId,
            sampleNext = {
                val logits = run.logits
                tokenizer.maskInvalidPaddedLogits(logits)
                sampleG09(
                    logits = logits,
                    tokenIdSpace = binding.tokenIdSpace,
                    config = sampler,
                    random = random,
                )
            },
            consumeToken = { token -> run = executor.step(token) },
        )
        val stop = if (generated.stoppedAtEos) NativeGenerationStopReason.EOS
            else NativeGenerationStopReason.TOKEN_LIMIT
        val ids = generated.tokenIds
        val bytes = tokenizer.decodeBytes(
            ids,
            skipEos = true,
        )
        require(
            bytes.size <= config.inferenceLimits.maxOutputUtf8Bytes
        ) {
            "G0.9 cognition output exceeds UTF-8 byte budget"
        }
        Mamba2CandidateGenerationResult(
            text = decodeStrictUtf8(bytes),
            tokenIds = ids,
            stopReason = stop,
            sequencePosition = executor.currentSequencePosition(),
        )
    }

    /** Timings from this same engine; does not open a second model for diagnostics. */
    fun measurePrefillDecode(promptIds: IntArray, decodeTokens: Int): Pair<Long, Long> = synchronized(lock) {
        check(!closed)
        require(promptIds.isNotEmpty() && promptIds.size <= config.inferenceLimits.maxPromptTokens)
        require(promptIds.all { it in 0 until binding.tokenIdSpace })
        require(decodeTokens in 1..1024)
        executor.resetRecurrentState()
        val start = System.nanoTime()
        var run = executor.prefill(promptIds)
        val prefillEnd = System.nanoTime()
        repeat(decodeTokens) {
            tokenizer.maskInvalidPaddedLogits(run.logits)
            val token = sampleG09(run.logits, binding.tokenIdSpace, Mamba2CandidateSamplerConfig(), Random(97))
            run = executor.step(token)
        }
        Pair(prefillEnd - start, System.nanoTime() - prefillEnd)
    }

    override fun embedText(
        text: String,
        vectorDim: Int,
    ): FloatArray = synchronized(lock) {
        check(!closed) { "G0.9 cognition candidate is closed" }
        require(vectorDim == executor.runtimePackage.dModel) {
            "G0.9 retrieval vector dimension must match Mamba-2 dModel"
        }
        g06TokenFeatures(tokenizer.encode(text), vectorDim)
    }

    override fun close() {
        synchronized(lock) {
            if (closed) return
            closed = true
            executor.close()
        }
    }
}

private fun sampleG09(
    logits: FloatArray,
    tokenIdSpace: Int,
    config: Mamba2CandidateSamplerConfig,
    random: Random,
): Int {
    require(logits.size >= tokenIdSpace)
    for (index in 0 until tokenIdSpace) {
        val value = logits[index]
        require(!value.isNaN() && value != Float.POSITIVE_INFINITY) {
            "G0.9 logits contain invalid value"
        }
    }
    for (index in tokenIdSpace until logits.size) {
        require(logits[index] == Float.NEGATIVE_INFINITY) {
            "G0.9 padded logits were not masked"
        }
    }

    if (config.temperature == 0.0f || config.topK == 1) {
        var best = 0
        var bestValue = Float.NEGATIVE_INFINITY
        for (index in 0 until tokenIdSpace) {
            if (logits[index] > bestValue) {
                best = index
                bestValue = logits[index]
            }
        }
        require(bestValue.isFinite()) {
            "G0.9 no finite source-vocabulary logit"
        }
        return best
    }

    val order = (0 until tokenIdSpace)
        .filter { logits[it].isFinite() }
        .sortedByDescending { logits[it] }
        .take(minOf(config.topK, tokenIdSpace))
    require(order.isNotEmpty()) {
        "G0.9 sampler has no finite token candidates"
    }
    val maximum = logits[order.first()]
    val weights = DoubleArray(order.size)
    var total = 0.0
    for (index in order.indices) {
        val weight = exp(
            ((logits[order[index]] - maximum) / config.temperature)
                .toDouble()
        )
        weights[index] = weight
        total += weight
    }
    require(total.isFinite() && total > 0.0)
    var cumulative = 0.0
    var keep = order.size
    for (index in order.indices) {
        cumulative += weights[index] / total
        if (cumulative >= config.topP) {
            keep = index + 1
            break
        }
    }
    var keptTotal = 0.0
    for (index in 0 until keep) {
        keptTotal += weights[index]
    }
    var draw = random.nextDouble() * keptTotal
    for (index in 0 until keep) {
        draw -= weights[index]
        if (draw <= 0.0) {
            return order[index]
        }
    }
    return order[keep - 1]
}

private fun requireG09Sha(value: String, label: String) {
    require(
        value.length == 64 &&
            value.all { it in '0'..'9' || it in 'a'..'f' }
    ) {
        label + " must be lowercase SHA-256"
    }
}

private fun sha256G09(bytes: ByteArray): String =
    MessageDigest.getInstance("SHA-256")
        .digest(bytes)
        .joinToString("") {
            "%02x".format(it.toInt() and 0xff)
        }

private fun canonicalJsonG09(value: Any?): String {
    return when (value) {
        JSONObject.NULL, null -> "null"
        is JSONObject -> {
            value.keys().asSequence().toList().sorted().joinToString(
                prefix = "{",
                postfix = "}",
                separator = ",",
            ) { key ->
                JSONObject.quote(key) + ":" +
                    canonicalJsonG09(value.get(key))
            }
        }
        is JSONArray -> {
            (0 until value.length()).joinToString(
                prefix = "[",
                postfix = "]",
                separator = ",",
            ) { index ->
                canonicalJsonG09(value.get(index))
            }
        }
        is String -> JSONObject.quote(value)
        is Boolean -> if (value) "true" else "false"
        is Int, is Long, is Short, is Byte -> value.toString()
        else -> error(
            "unsupported G0.9 canonical JSON type: " +
                value::class.java.name
        )
    }
}
