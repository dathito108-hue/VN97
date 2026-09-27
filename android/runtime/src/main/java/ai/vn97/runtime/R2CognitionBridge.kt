package ai.vn97.runtime

import android.content.Context
import java.io.File
import java.nio.charset.StandardCharsets
import java.security.MessageDigest
import kotlin.math.sqrt
import org.json.JSONArray
import org.json.JSONObject

enum class VN97R2ReasoningMode {
    FAST,
    DEEP,
}

data class VN97R2AssistantBinding(
    val bindingId: String,
    val runtimeId: String,
    val bundleId: String,
    val checkpointSha256: String,
    val checkpointStage: String,
    val tokenizerModelSha256: String,
    val architectureFingerprint: String,
    val profile: String,
    val activeLayers: Int,
    val vocabSize: Int,
    val dModel: Int,
    val nLayers: Int,
    val dState: Int,
) {
    init {
        requireSha256F2(bindingId, "bindingId")
        requireSha256F2(runtimeId, "runtimeId")
        requireSha256F2(bundleId, "bundleId")
        requireSha256F2(checkpointSha256, "checkpointSha256")
        requireSha256F2(tokenizerModelSha256, "tokenizerModelSha256")
        requireSha256F2(architectureFingerprint, "architectureFingerprint")
        require(checkpointStage.isNotBlank())
        require(profile.isNotBlank())
        require(activeLayers in 1..nLayers)
        require(vocabSize > 1)
        require(dModel > 0)
        require(dState > 0)
    }

    fun requireCompatible(
        model: NativeActivatedModel,
        runtime: OrtProductionPackage,
    ) {
        val modelId = model.info.modelId.lowerHexF2()
        require(model.info.hasTokenizer) {
            "F2 requires VN97TK1 in the activated artifact"
        }
        require(modelId == tokenizerModelSha256) {
            "F2 tokenizer artifact identity mismatch"
        }
        require(runtime.runtimeId == runtimeId) {
            "F2 runtime identity mismatch"
        }
        require(runtime.bundleId == bundleId) {
            "F2 bundle identity mismatch"
        }
        require(runtime.architectureFingerprint == architectureFingerprint) {
            "F2 architecture identity mismatch"
        }
        require(runtime.modelProfile == profile) {
            "F2 profile identity mismatch"
        }
        require(runtime.activeLayers == activeLayers) {
            "F2 active-layer identity mismatch"
        }
        require(runtime.vocabSize == vocabSize && model.info.vocabSize == vocabSize) {
            "F2 tokenizer/runtime vocabulary mismatch"
        }
        require(model.info.dModel == dModel) {
            "F2 activated tokenizer model dModel mismatch"
        }
        require(model.info.layers == nLayers) {
            "F2 activated tokenizer model layer count mismatch"
        }
        require(model.info.dState == dState && runtime.dState == dState) {
            "F2 state geometry mismatch"
        }
    }

    companion object {
        const val SCHEMA = "VN97R2F2BIND1"
        const val FILENAME = "assistant.vn97r2f2.json"

        fun load(file: File): VN97R2AssistantBinding {
            require(file.isFile && !java.nio.file.Files.isSymbolicLink(file.toPath())) {
                "F2 binding must be a regular non-symlink file"
            }
            val root = JSONObject(file.readText(StandardCharsets.UTF_8))
            val fields = setOf(
                "schema",
                "runtime_id",
                "bundle_id",
                "checkpoint_sha256",
                "checkpoint_stage",
                "tokenizer_model_sha256",
                "architecture_fingerprint",
                "profile",
                "active_layers",
                "vocab_size",
                "d_model",
                "n_layers",
                "d_state",
                "same_weights_semantics",
                "legacy_inference_fallback",
                "binding_id",
            )
            require(root.keys().asSequence().toSet() == fields) {
                "F2 binding fields mismatch"
            }
            require(root.getString("schema") == SCHEMA) {
                "F2 binding schema mismatch"
            }
            require(root.getBoolean("same_weights_semantics"))
            require(!root.getBoolean("legacy_inference_fallback")) {
                "F2 legacy inference fallback is forbidden"
            }
            val bindingId = root.getString("binding_id")
            requireSha256F2(bindingId, "bindingId")
            val body = JSONObject(root.toString())
            body.remove("binding_id")
            val prefix = "VN97R2F2BIND1" + String(charArrayOf(0.toChar()))
            val actual = sha256F2(
                prefix.toByteArray(StandardCharsets.UTF_8) +
                    canonicalJsonF2(body).toByteArray(StandardCharsets.UTF_8)
            )
            require(bindingId == actual) {
                "F2 binding identity mismatch"
            }
            return VN97R2AssistantBinding(
                bindingId = bindingId,
                runtimeId = root.getString("runtime_id"),
                bundleId = root.getString("bundle_id"),
                checkpointSha256 = root.getString("checkpoint_sha256"),
                checkpointStage = root.getString("checkpoint_stage"),
                tokenizerModelSha256 = root.getString("tokenizer_model_sha256"),
                architectureFingerprint = root.getString("architecture_fingerprint"),
                profile = root.getString("profile"),
                activeLayers = root.getInt("active_layers"),
                vocabSize = root.getInt("vocab_size"),
                dModel = root.getInt("d_model"),
                nLayers = root.getInt("n_layers"),
                dState = root.getInt("d_state"),
            )
        }
    }
}

data class VN97R2GenerationConfig(
    val mode: VN97R2ReasoningMode = VN97R2ReasoningMode.DEEP,
    val sampler: NativeSamplerConfig = NativeSamplerConfig(),
    val eosToken: Int = 2,
    val prefillDeadlineMs: Double = 250.0,
    val stepDeadlineMs: Double = 100.0,
) {
    init {
        require(eosToken >= -1)
        require(prefillDeadlineMs.isFinite() && prefillDeadlineMs > 0.0)
        require(stepDeadlineMs.isFinite() && stepDeadlineMs > 0.0)
    }
}

class VN97R2CognitionInference private constructor(
    private val model: NativeActivatedModel,
    private val executor: VN97OrtProductionExecutor,
    val binding: VN97R2AssistantBinding,
    private val cognitionConfig: NativeCognitionRuntimeConfig,
    private val generationConfig: VN97R2GenerationConfig,
) : NativeCognitionInference, AutoCloseable {
    private val lock = Any()
    private var closed = false

    companion object {
        fun open(
            context: Context,
            model: NativeActivatedModel,
            runtimeRoot: File,
            tuningFile: File,
            cognitionConfig: NativeCognitionRuntimeConfig =
                NativeCognitionRuntimeConfig(),
            generationConfig: VN97R2GenerationConfig =
                VN97R2GenerationConfig(),
            qnnBackendPath: String? = null,
            maxCachedSessions: Int = 1,
        ): VN97R2CognitionInference {
            val binding = VN97R2AssistantBinding.load(
                File(runtimeRoot, VN97R2AssistantBinding.FILENAME)
            )
            val executor = VN97OrtProductionExecutor.open(
                context = context,
                runtimeRoot = runtimeRoot,
                tuningFile = tuningFile,
                qnnBackendPath = qnnBackendPath,
                maxCachedSessions = maxCachedSessions,
            )
            return try {
                binding.requireCompatible(model, executor.runtimePackage)
                VN97R2CognitionInference(
                    model = model,
                    executor = executor,
                    binding = binding,
                    cognitionConfig = cognitionConfig,
                    generationConfig = generationConfig,
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
                cognitionConfig.adapterConfig.maxPromptUtf8Bytes,
        )
        return generateText(
            prompt,
            cognitionConfig.adapterConfig.maxNewTokens(operation),
        )
    }

    fun generateText(
        prompt: String,
        maxNewTokens: Int,
        mode: VN97R2ReasoningMode = generationConfig.mode,
    ): String = synchronized(lock) {
        requireOpen()
        require(maxNewTokens > 0)
        val inputIds = encodePrompt(prompt)
        executor.resetRecurrentState()
        var logits = executor.prefill(
            inputIds = inputIds,
            deadlineMs = when (mode) {
                VN97R2ReasoningMode.FAST ->
                    minOf(generationConfig.prefillDeadlineMs, 140.0)
                VN97R2ReasoningMode.DEEP ->
                    generationConfig.prefillDeadlineMs
            },
            stateDependency = when (mode) {
                VN97R2ReasoningMode.FAST -> 0.35
                VN97R2ReasoningMode.DEEP -> 0.20
            },
        ).logits
        val generated = IntArray(maxNewTokens)
        var count = 0
        for (index in 0 until maxNewTokens) {
            val token = NativeSampler.sample(
                logits = logits,
                config = generationConfig.sampler,
                step = index.toLong(),
            )
            if (token == generationConfig.eosToken) {
                break
            }
            generated[count++] = token
            logits = executor.step(
                tokenId = token,
                deadlineMs = when (mode) {
                    VN97R2ReasoningMode.FAST ->
                        minOf(generationConfig.stepDeadlineMs, 65.0)
                    VN97R2ReasoningMode.DEEP ->
                        generationConfig.stepDeadlineMs
                },
            ).logits
        }
        val bytes = model.decodeBytes(
            generated.copyOf(count),
            skipControl = true,
        )
        if (bytes.size > cognitionConfig.inferenceLimits.maxOutputUtf8Bytes) {
            throw NativeCognitionContractException(
                "R2 cognition output exceeds UTF-8 byte budget"
            )
        }
        decodeStrictUtf8(bytes)
    }

    override fun embedText(text: String, vectorDim: Int): FloatArray =
        synchronized(lock) {
            requireOpen()
            require(vectorDim == binding.dModel) {
                "R2 VN97MEM1 embedding dimension must equal bound dModel"
            }
            val ids = encodePrompt(text)
            executor.resetRecurrentState()
            val logits = executor.prefill(
                inputIds = ids,
                deadlineMs = generationConfig.prefillDeadlineMs,
                stateDependency = 0.20,
            ).logits
            logitsSketchEmbeddingF2(logits, vectorDim)
        }

    fun reset() = synchronized(lock) {
        requireOpen()
        executor.resetRecurrentState(
            clearSessions = false,
            resetControlState = false,
        )
    }

    override fun close() = synchronized(lock) {
        if (closed) return
        closed = true
        executor.close()
    }

    private fun encodePrompt(text: String): IntArray {
        val ids = model.encodeUtf8(
            text,
            addBos = true,
            addText = true,
            addEos = false,
        )
        require(ids.isNotEmpty()) {
            "R2 encoded prompt must not be empty"
        }
        if (ids.size > cognitionConfig.inferenceLimits.maxPromptTokens) {
            throw NativeCognitionContractException(
                "R2 prompt exceeds cognition token budget"
            )
        }
        return ids
    }

    private fun requireOpen() {
        check(!closed) {
            "R2 cognition bridge is closed"
        }
    }
}

/**
 * VN97R2LOGITEMB1: deterministic same-model retrieval embedding when the F1
 * graph contract exposes logits but not a hidden-state output. It never invokes
 * the legacy model backend. The sketch is versioned and normalized so VN97MEM1
 * can bind/migrate by checkpoint identity rather than silently mixing spaces.
 */
internal fun logitsSketchEmbeddingF2(
    logits: FloatArray,
    vectorDim: Int,
): FloatArray {
    require(logits.isNotEmpty() && logits.all { it.isFinite() })
    require(vectorDim > 0)
    val output = FloatArray(vectorDim)
    for (index in logits.indices) {
        val bucket = index % vectorDim
        val sign = if (((index / vectorDim) and 1) == 0) 1.0f else -1.0f
        output[bucket] += logits[index] * sign
    }
    var normSquared = 0.0
    for (value in output) {
        val v = value.toDouble()
        normSquared += v * v
    }
    val norm = sqrt(normSquared)
    require(norm.isFinite() && norm > 0.0) {
        "R2 logit embedding norm is zero or invalid"
    }
    val inverse = (1.0 / norm).toFloat()
    for (index in output.indices) {
        output[index] *= inverse
    }
    return output
}

private fun requireSha256F2(value: String, label: String) {
    require(
        value.length == 64 &&
            value.all { it in '0'..'9' || it in 'a'..'f' }
    ) {
        "$label must be lowercase SHA-256"
    }
}

private fun sha256F2(bytes: ByteArray): String =
    MessageDigest.getInstance("SHA-256")
        .digest(bytes)
        .joinToString("") { "%02x".format(it.toInt() and 0xff) }

private fun ByteArray.lowerHexF2(): String =
    joinToString("") { "%02x".format(it.toInt() and 0xff) }

private fun canonicalJsonF2(value: Any?): String = when (value) {
    JSONObject.NULL, null -> "null"
    is JSONObject -> value.keys().asSequence().toList().sorted().joinToString(
        prefix = "{",
        postfix = "}",
        separator = ",",
    ) { key ->
        JSONObject.quote(key) + ":" + canonicalJsonF2(value.get(key))
    }
    is JSONArray -> (0 until value.length()).joinToString(
        prefix = "[",
        postfix = "]",
        separator = ",",
    ) { index -> canonicalJsonF2(value.get(index)) }
    is String -> JSONObject.quote(value)
    is Boolean -> if (value) "true" else "false"
    is Int, is Long, is Short, is Byte -> value.toString()
    else -> error("unsupported F2 canonical JSON type: " + value::class.java.name)
}
