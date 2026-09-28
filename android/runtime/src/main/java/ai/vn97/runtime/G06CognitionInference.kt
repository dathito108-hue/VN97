package ai.vn97.runtime

import android.content.Context
import java.io.File
import java.nio.file.Files
import kotlin.math.sqrt

/** A verified G06 deployment identity, with no MI1/SSM1 model allocation. */
class VN97G06Model private constructor(
    val root: File,
    val runtime: Mamba2OrtRuntimePackage,
    val tokenizer: VN97GptNeoXTokenizer,
    val binding: Mamba2CognitionBridgeBinding,
    val promotion: Mamba2ProductionPromotion,
) : AutoCloseable {
    data class Info(
        val modelId: ByteArray,
        val dModel: Int,
        val layers: Int,
        val dState: Int,
        val vocabSize: Int,
        val hasTokenizer: Boolean = true,
    )

    val info = Info(
        promotion.promotionId.chunked(2).map { it.toInt(16).toByte() }.toByteArray(),
        runtime.dModel, runtime.nLayers, runtime.dState, tokenizer.packageInfo.tokenIdSpace,
    )
    private var closed = false

    @Synchronized fun requireOpen() = check(!closed) { "G06 deployment is closed" }
    @Synchronized override fun close() { closed = true }

    fun embedText(text: String, vectorDim: Int): FloatArray {
        requireOpen()
        require(vectorDim == info.dModel) { "G06 memory dimension mismatch" }
        return g06TokenFeatures(tokenizer.encode(text), vectorDim)
    }

    companion object {
        const val ROOT_DIR = "vn97-g06"
        const val TUNING_FILE = "tuning.vn97m2g07.json"

        fun openOrNull(root: File, context: Context): VN97G06Model? {
            if (!root.exists()) return null
            require(root.isDirectory && !Files.isSymbolicLink(root.toPath())) {
                "G06 deployment root must be a regular directory"
            }
            val runtime = Mamba2OrtRuntimePackage.load(File(root, "runtime"))
            val tokenizer = Mamba2TokenizerPackage.load(File(root, "tokenizer"))
            val binding = Mamba2CognitionBridgeBinding.load(File(root, Mamba2CognitionBridgeBinding.FILENAME))
            val tuning = Mamba2OrtTuningProfile.load(File(root, TUNING_FILE), runtime.runtimeId, OrtDeviceProbe.inspect(context.applicationContext, null))
            binding.requireCompatible(runtime, tokenizer, tuning.tuningId)
            val promotion = Mamba2ProductionPromotion.load(File(root, Mamba2ProductionPromotion.FILENAME))
            promotion.requireCompatible(binding, runtime, tokenizer, tuning)
            return VN97G06Model(root, runtime, VN97GptNeoXTokenizer(tokenizer), binding, promotion)
        }
    }
}

/** Sole app inference entry: the existing G06 executor, G08 tokenizer and G09 bridge. */
class VN97G06CognitionInference private constructor(
    private val model: VN97G06Model,
    private val engine: VN97Mamba2CognitionCandidate,
) : NativeCognitionInference, AutoCloseable {
    override fun generateOperation(operation: NativeCognitionOperation, requestJson: String): String =
        engine.generateOperation(operation, requestJson)

    fun generateText(prompt: String, maxNewTokens: Int): Mamba2CandidateGenerationResult =
        engine.generateText(prompt, maxNewTokens)

    fun chat(systemPrompt: String, userMessage: String, deep: Boolean): Mamba2CandidateGenerationResult =
        generateText(
            NativeChatPrompt.build(systemPrompt, userMessage, true, NativeChatLimits()),
            if (deep) 1024 else 256,
        )

    override fun embedText(text: String, vectorDim: Int): FloatArray = model.embedText(text, vectorDim)
    override fun close() = engine.close()

    companion object {
        fun embedTokenizerFeatures(model: VN97G06Model, text: String, vectorDim: Int): FloatArray =
            model.embedText(text, vectorDim)

        fun open(
            context: Context,
            model: VN97G06Model,
            config: NativeCognitionRuntimeConfig = NativeCognitionRuntimeConfig(),
        ): VN97G06CognitionInference {
            model.requireOpen()
            val engine = VN97Mamba2CognitionCandidate.open(
                context = context,
                runtimeRoot = File(model.root, "runtime"),
                tokenizerRoot = File(model.root, "tokenizer"),
                tuningFile = File(model.root, VN97G06Model.TUNING_FILE),
                bindingFile = File(model.root, Mamba2CognitionBridgeBinding.FILENAME),
                config = config,
            )
            return try {
                check(engine.binding == model.binding) { "G06 deployment changed while opening" }
                VN97G06CognitionInference(model, engine)
            } catch (error: Throwable) {
                engine.close()
                throw error
            }
        }
    }
}

/** Deterministic retrieval features, never a second inference model. */
internal fun g06TokenFeatures(ids: IntArray, vectorDim: Int): FloatArray {
    require(vectorDim > 0)
    val output = FloatArray(vectorDim)
    ids.forEachIndexed { position, token ->
        var mixed = token * -0x61c88647 + position * 0x045d9f3b
        mixed = mixed xor (mixed ushr 16)
        mixed *= 0x045d9f3b
        mixed = mixed xor (mixed ushr 16)
        output[(mixed and Int.MAX_VALUE) % vectorDim] +=
            (if ((mixed ushr 31) == 0) 1.0f else -1.0f) / sqrt((position + 1).toFloat())
    }
    val norm = sqrt(output.sumOf { it.toDouble() * it.toDouble() })
    if (norm > 0.0) for (i in output.indices) output[i] = (output[i] / norm).toFloat()
    return output
}
