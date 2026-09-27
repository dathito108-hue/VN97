package ai.vn97.runtime

import android.content.Context
import java.io.File
import java.nio.ByteBuffer
import java.nio.charset.CodingErrorAction
import java.nio.charset.StandardCharsets
import java.util.Random
import kotlin.math.exp
import kotlin.math.sqrt
import org.json.JSONObject

private const val R2_F2_BINDING_SCHEMA = "VN97R2F2BRIDGE1"

data class VN97R2SamplerConfig(
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

data class VN97R2GenerationResult(
    val text: String,
    val tokenIds: IntArray,
    val stopReason: NativeGenerationStopReason,
    val sequencePosition: Long,
)

data class VN97R2BridgeBinding(
    val runtimeId: String,
    val bundleId: String,
    val tokenizerModelId: String,
    val vocabSize: Int,
    val sameWeightsSemantics: Boolean,
) {
    init {
        require(runtimeId.matches(Regex("^[0-9a-f]{64}$")))
        require(bundleId.matches(Regex("^[0-9a-f]{64}$")))
        require(tokenizerModelId.matches(Regex("^[0-9a-f]{64}$")))
        require(vocabSize > 1)
        require(sameWeightsSemantics)
    }

    fun requireCompatible(
        model: NativeActivatedModel,
        runtime: OrtProductionPackage,
    ) {
        require(runtime.runtimeId == runtimeId) {
            "F2 binding runtime identity mismatch"
        }
        require(runtime.bundleId == bundleId) {
            "F2 binding ONNX bundle identity mismatch"
        }
        require(runtime.vocabSize == vocabSize) {
            "F2 binding runtime vocab mismatch"
        }
        require(model.info.vocabSize == vocabSize) {
            "F2 binding tokenizer vocab mismatch"
        }
        require(model.info.hasTokenizer) {
            "F2 requires VN97TK1 tokenizer"
        }
        require(model.info.modelId.toHexF2() == tokenizerModelId) {
            "F2 tokenizer/checkpoint identity mismatch"
        }
        require(runtime.activeLayers <= model.info.layers) {
            "F2 runtime layer profile exceeds activated checkpoint geometry"
        }
    }

    companion object {
        fun load(file: File): VN97R2BridgeBinding {
            require(
                file.isFile &&
                    !java.nio.file.Files.isSymbolicLink(file.toPath())
            ) {
                "F2 bridge binding must be a regular file"
            }
            val root = JSONObject(file.readText(Charsets.UTF_8))
            val expected = setOf(
                "schema",
                "runtime_id",
                "bundle_id",
                "tokenizer_model_id",
                "vocab_size",
                "same_weights_semantics",
            )
            require(root.keys().asSequence().toSet() == expected) {
                "F2 bridge binding fields mismatch"
            }
            require(root.getString("schema") == R2_F2_BINDING_SCHEMA)
            return VN97R2BridgeBinding(
                runtimeId = root.getString("runtime_id"),
                bundleId = root.getString("bundle_id"),
                tokenizerModelId = root.getString("tokenizer_model_id"),
                vocabSize = root.getInt("vocab_size"),
                sameWeightsSemantics =
                    root.getBoolean("same_weights_semantics"),
            )
        }
    }
}

/**
 * Canonical R2 text intelligence bridge.
 *
 * The activated VN97 image contributes the signed VN97TK1 tokenizer identity.
 * Every text logit and recurrent state transition comes from F1 ONNX Runtime.
 * There is deliberately no NativeRuntimeSession/legacy inference fallback.
 */
class VN97R2CognitionInference private constructor(
    private val model: NativeActivatedModel,
    private val executor: VN97OrtProductionExecutor,
    private val config: NativeCognitionRuntimeConfig,
) : NativeCognitionInference, AutoCloseable {
    private val lock = Any()
    private var closed = false

    init {
        require(model.info.hasTokenizer)
        require(model.info.vocabSize == executor.runtimePackage.vocabSize)
    }

    companion object {
        const val ROOT_DIR = "vn97-r2"
        const val RUNTIME_DIR = "runtime"
        const val TUNING_FILE = "tuning.vn97r2e4.json"
        const val BINDING_FILE = "binding.vn97r2f2.json"

        fun open(
            context: Context,
            model: NativeActivatedModel,
            config: NativeCognitionRuntimeConfig =
                NativeCognitionRuntimeConfig(),
        ): VN97R2CognitionInference {
            val root = File(context.noBackupFilesDir, ROOT_DIR)
            val runtimeRoot = File(root, RUNTIME_DIR)
            val tuningFile = File(root, TUNING_FILE)
            val bindingFile = File(root, BINDING_FILE)
            require(root.isDirectory) {
                "VN97 R2 runtime is not installed"
            }
            val executor = VN97OrtProductionExecutor.open(
                context = context,
                runtimeRoot = runtimeRoot,
                tuningFile = tuningFile,
                maxCachedSessions = 2,
            )
            try {
                VN97R2BridgeBinding.load(bindingFile)
                    .requireCompatible(
                        model = model,
                        runtime = executor.runtimePackage,
                    )
                return VN97R2CognitionInference(
                    model = model,
                    executor = executor,
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
            sampler = VN97R2SamplerConfig(),
        ).text
    }

    fun generateText(
        prompt: String,
        maxNewTokens: Int,
        sampler: VN97R2SamplerConfig = VN97R2SamplerConfig(),
    ): VN97R2GenerationResult = synchronized(lock) {
        check(!closed) { "F2 cognition bridge is closed" }
        require(maxNewTokens > 0)
        val promptIds = model.encodeUtf8(
            text = prompt,
            addBos = true,
            addText = true,
            addEos = false,
        )
        require(promptIds.isNotEmpty()) {
            "F2 tokenizer produced an empty prompt"
        }
        executor.resetRecurrentState()
        var run = executor.prefill(promptIds)
        val generated = ArrayList<Int>(maxNewTokens)
        val random = Random(sampler.seed)
        var stop = NativeGenerationStopReason.TOKEN_LIMIT

        for (index in 0 until maxNewTokens) {
            val token = sampleF2(
                logits = run.logits,
                config = sampler,
                random = random,
            )
            if (token == 2) {
                stop = NativeGenerationStopReason.EOS
                break
            }
            generated.add(token)
            run = executor.step(token)
        }

        val ids = generated.toIntArray()
        val bytes = model.decodeBytes(ids, skipControl = true)
        require(
            bytes.size <=
                config.inferenceLimits.maxOutputUtf8Bytes
        ) {
            "F2 generation exceeds cognition output byte budget"
        }
        VN97R2GenerationResult(
            text = decodeStrictUtf8F2(bytes),
            tokenIds = ids,
            stopReason = stop,
            sequencePosition =
                executor.currentSequencePosition(),
        )
    }

    fun chat(
        systemPrompt: String,
        userMessage: String,
        deep: Boolean,
        sampler: VN97R2SamplerConfig =
            if (deep) {
                VN97R2SamplerConfig(
                    temperature = 0.2f,
                    topK = 32,
                    topP = 0.95f,
                    seed = 97L,
                )
            } else {
                VN97R2SamplerConfig()
            },
    ): VN97R2GenerationResult {
        val prompt = NativeChatPrompt.build(
            systemPrompt = systemPrompt,
            userMessage = userMessage,
            includeSystem = true,
            limits = NativeChatLimits(),
        )
        return generateText(
            prompt = prompt,
            maxNewTokens = if (deep) 1024 else 256,
            sampler = sampler,
        )
    }

    /**
     * VN97MEM1 retrieval support with no hidden inference backend.
     *
     * This is a deterministic VN97TK1 token-feature projection. It supplies only
     * retrieval geometry; reasoning/generation stays exclusively on R2 ONNX.
     */
    override fun embedText(
        text: String,
        vectorDim: Int,
    ): FloatArray = synchronized(lock) {
        check(!closed) { "F2 cognition bridge is closed" }
        require(vectorDim == model.info.dModel) {
            "F2 memory vector dimension must match VN97 dModel"
        }
        val ids = model.encodeUtf8(
            text = text,
            addBos = false,
            addText = false,
            addEos = false,
        )
        val out = FloatArray(vectorDim)
        ids.forEachIndexed { position, token ->
            val mixed = mixF2(token, position)
            val index = (mixed and Int.MAX_VALUE) % vectorDim
            val sign =
                if ((mixed ushr 31) == 0) 1.0f else -1.0f
            out[index] +=
                sign / sqrt((position + 1).toFloat())
        }
        var norm2 = 0.0
        out.forEach {
            norm2 += it.toDouble() * it.toDouble()
        }
        if (norm2 != 0.0) {
            val inv = (1.0 / sqrt(norm2)).toFloat()
            for (i in out.indices) {
                out[i] *= inv
            }
        }
        out
    }

    override fun close() {
        synchronized(lock) {
            if (closed) return
            closed = true
            executor.close()
        }
    }
}

private fun sampleF2(
    logits: FloatArray,
    config: VN97R2SamplerConfig,
    random: Random,
): Int {
    require(logits.isNotEmpty())
    logits.forEach {
        require(it.isFinite()) {
            "F2 logits contain non-finite value"
        }
    }
    if (config.temperature == 0.0f || config.topK == 1) {
        var best = 0
        var bestValue = Float.NEGATIVE_INFINITY
        for (i in logits.indices) {
            if (logits[i] > bestValue) {
                bestValue = logits[i]
                best = i
            }
        }
        return best
    }

    val order = logits.indices
        .sortedByDescending { logits[it] }
        .take(minOf(config.topK, logits.size))
    val max = order.maxOf { logits[it] }
    val weights = DoubleArray(order.size)
    var total = 0.0
    for (i in order.indices) {
        val weight = exp(
            ((logits[order[i]] - max) / config.temperature)
                .toDouble()
        )
        weights[i] = weight
        total += weight
    }
    var cumulative = 0.0
    var keep = order.size
    for (i in order.indices) {
        cumulative += weights[i] / total
        if (cumulative >= config.topP) {
            keep = i + 1
            break
        }
    }
    var keptTotal = 0.0
    for (i in 0 until keep) {
        keptTotal += weights[i]
    }
    var draw = random.nextDouble() * keptTotal
    for (i in 0 until keep) {
        draw -= weights[i]
        if (draw <= 0.0) {
            return order[i]
        }
    }
    return order[keep - 1]
}

private fun mixF2(token: Int, position: Int): Int {
    var value =
        token * -0x61c88647 +
            position * 0x045d9f3b
    value = value xor (value ushr 16)
    value *= 0x045d9f3b
    value = value xor (value ushr 16)
    return value
}

private fun decodeStrictUtf8F2(bytes: ByteArray): String =
    StandardCharsets.UTF_8
        .newDecoder()
        .onMalformedInput(CodingErrorAction.REPORT)
        .onUnmappableCharacter(CodingErrorAction.REPORT)
        .decode(ByteBuffer.wrap(bytes))
        .toString()

private fun ByteArray.toHexF2(): String =
    joinToString("") {
        "%02x".format(it.toInt() and 0xff)
    }
