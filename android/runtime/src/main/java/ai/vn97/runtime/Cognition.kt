package ai.vn97.runtime

import java.nio.ByteBuffer
import java.nio.charset.CodingErrorAction
import java.nio.charset.StandardCharsets
import kotlin.math.sqrt

private const val COGNITION_PROTOCOL = "VN97COG1"

enum class NativeCognitionOperation(
    val wireName: String,
    val schema: String,
) {
    PLAN(
        "plan",
        "{\"steps\":[{\"kind\":\"REASON|RETRIEVE|VERIFY|RESPOND|EXTERNAL\",\"objective\":\"string\",\"dependencies\":[1],\"requires_verification\":false,\"min_confidence\":0.0}]}",
    ),
    MEMORY_QUERY(
        "memory_query",
        "{\"query\":\"string\",\"top_k\":5,\"kinds\":[\"EPISODIC\",\"SEMANTIC\"]|null,\"semantic_weight\":1.0,\"recency_weight\":0.0,\"importance_weight\":0.0,\"recency_half_life_ns\":86400000000000}",
    ),
    STEP(
        "step",
        "{\"result\":\"string\",\"confidence\":0.0}",
    ),
    VERIFY(
        "verify",
        "{\"passed\":true,\"note\":\"string\"}",
    ),
    EXTERNAL_INTENT(
        "external_intent",
        "{\"capability_id\":\"string\",\"scope\":{\"key\":\"string\"},\"payload\":{\"key\":\"json-value\"}}",
    ),
}

data class NativeCognitionInferenceLimits(
    val maxPromptTokens: Int = 4096,
    val maxOutputUtf8Bytes: Int = 64 * 1024,
) {
    init {
        require(maxPromptTokens > 0) { "maxPromptTokens must be positive" }
        require(maxOutputUtf8Bytes > 0) { "maxOutputUtf8Bytes must be positive" }
    }
}

data class NativeCognitionAdapterConfig(
    val maxPromptUtf8Bytes: Int = 128 * 1024,
    val planNewTokens: Int = 768,
    val memoryQueryNewTokens: Int = 384,
    val stepNewTokens: Int = 1024,
    val verificationNewTokens: Int = 384,
    val externalIntentNewTokens: Int = 512,
) {
    init {
        require(maxPromptUtf8Bytes > 0) { "maxPromptUtf8Bytes must be positive" }
        require(planNewTokens > 0) { "planNewTokens must be positive" }
        require(memoryQueryNewTokens > 0) { "memoryQueryNewTokens must be positive" }
        require(stepNewTokens > 0) { "stepNewTokens must be positive" }
        require(verificationNewTokens > 0) { "verificationNewTokens must be positive" }
        require(externalIntentNewTokens > 0) { "externalIntentNewTokens must be positive" }
    }

    fun maxNewTokens(operation: NativeCognitionOperation): Int = when (operation) {
        NativeCognitionOperation.PLAN -> planNewTokens
        NativeCognitionOperation.MEMORY_QUERY -> memoryQueryNewTokens
        NativeCognitionOperation.STEP -> stepNewTokens
        NativeCognitionOperation.VERIFY -> verificationNewTokens
        NativeCognitionOperation.EXTERNAL_INTENT -> externalIntentNewTokens
    }
}

data class NativeCognitionRuntimeConfig(
    val recurrentBackend: NativeBackend = NativeBackend.AUTO,
    val packedBackend: NativeBackend = NativeBackend.AUTO,
    val inferenceLimits: NativeCognitionInferenceLimits = NativeCognitionInferenceLimits(),
    val adapterConfig: NativeCognitionAdapterConfig = NativeCognitionAdapterConfig(),
)

class NativeCognitionInferenceException(
    message: String,
    cause: Throwable? = null,
) : IllegalStateException(message, cause)

class NativeCognitionContractException(
    message: String,
    cause: Throwable? = null,
) : IllegalArgumentException(message, cause)

internal object NativeCognitionPrompt {
    fun build(
        operation: NativeCognitionOperation,
        requestJson: String,
        maxPromptUtf8Bytes: Int,
    ): String {
        require(maxPromptUtf8Bytes > 0) { "maxPromptUtf8Bytes must be positive" }
        requireCanonicalRequestJson(requestJson)
        val prompt = buildString {
            append(COGNITION_PROTOCOL)
            append('\n')
            append("operation=")
            append(operation.wireName)
            append('\n')
            append("Return exactly one UTF-8 JSON object. No markdown fences. No prose outside JSON.\n")
            append("schema=")
            append(operation.schema)
            append('\n')
            append("request=")
            append(requestJson)
        }
        if (prompt.toByteArray(StandardCharsets.UTF_8).size > maxPromptUtf8Bytes) {
            throw NativeCognitionContractException(
                "VN97 cognition prompt exceeds UTF-8 byte budget"
            )
        }
        return prompt
    }

    private fun requireCanonicalRequestJson(value: String) {
        if (value.isEmpty() || value != value.trim() ||
            !value.startsWith('{') || !value.endsWith('}') ||
            '\n' in value || '\r' in value || '\u0000' in value) {
            throw NativeCognitionContractException(
                "requestJson must be one bounded canonical JSON object line"
            )
        }
    }
}

interface NativeCognitionInference {
    fun generateOperation(
        operation: NativeCognitionOperation,
        requestJson: String,
    ): String

    fun embedText(text: String, vectorDim: Int): FloatArray
}

class NativeCognitionInferenceEngine(
    private val model: NativeActivatedModel,
    private val config: NativeCognitionRuntimeConfig = NativeCognitionRuntimeConfig(),
) : NativeCognitionInference {
    init {
        require(model.info.hasTokenizer) {
            "native cognition requires VN97TK1 in the activated model"
        }
    }

    fun buildPrompt(
        operation: NativeCognitionOperation,
        requestJson: String,
    ): String = NativeCognitionPrompt.build(
        operation = operation,
        requestJson = requestJson,
        maxPromptUtf8Bytes = config.adapterConfig.maxPromptUtf8Bytes,
    )

    override fun generateOperation(
        operation: NativeCognitionOperation,
        requestJson: String,
    ): String = generateText(
        buildPrompt(operation, requestJson),
        maxNewTokens = config.adapterConfig.maxNewTokens(operation),
    )

    fun generateText(prompt: String, maxNewTokens: Int): String {
        require(maxNewTokens > 0) { "maxNewTokens must be positive" }
        val promptIds = encodePrompt(prompt)
        val generated = withFreshSession { session ->
            session.generateGreedy(
                model = model,
                promptIds = promptIds,
                maxNewTokens = maxNewTokens,
                eosToken = 2,
            )
        }
        val bytes = model.decodeBytes(generated, skipControl = true)
        if (bytes.size > config.inferenceLimits.maxOutputUtf8Bytes) {
            throw NativeCognitionContractException(
                "VN97 cognition output exceeds UTF-8 byte budget"
            )
        }
        return decodeStrictUtf8(bytes)
    }

    override fun embedText(text: String, vectorDim: Int): FloatArray {
        require(vectorDim > 0) { "vectorDim must be positive" }
        if (vectorDim != model.info.dModel) {
            throw NativeCognitionContractException(
                "native VN97 retrieval embedding requires vectorDim == model dModel"
            )
        }
        val promptIds = encodePrompt(text)
        val hidden = withFreshSession { session ->
            session.prefillHidden(model, promptIds)
        }
        if (hidden.size != vectorDim) {
            throw NativeCognitionInferenceException(
                "native hidden embedding has unexpected dimension"
            )
        }
        var normSquared = 0.0
        for (value in hidden) {
            if (!value.isFinite()) {
                throw NativeCognitionInferenceException(
                    "VN97 retrieval embedding contains non-finite values"
                )
            }
            val v = value.toDouble()
            normSquared += v * v
        }
        val norm = sqrt(normSquared)
        if (!norm.isFinite() || norm == 0.0) {
            throw NativeCognitionInferenceException(
                "VN97 retrieval embedding has zero/invalid norm"
            )
        }
        return hidden
    }

    private fun encodePrompt(text: String): IntArray {
        val ids = model.encodeUtf8(
            text,
            addBos = true,
            addText = true,
            addEos = false,
        )
        if (ids.isEmpty()) {
            throw NativeCognitionContractException("encoded cognition prompt is empty")
        }
        if (ids.size > config.inferenceLimits.maxPromptTokens) {
            throw NativeCognitionContractException(
                "prompt exceeds VN97 cognition token budget"
            )
        }
        return ids
    }

    private fun <T> withFreshSession(block: (NativeRuntimeSession) -> T): T {
        val runtimeConfig = NativeRuntimeConfig(
            layers = model.info.layers,
            batch = 1,
            dModel = model.info.dModel,
            dState = model.info.dState,
            recurrentBackend = config.recurrentBackend,
            packedBackend = config.packedBackend,
        )
        NativeRuntimeSession.create(runtimeConfig).use { session ->
            session.activate()
            return block(session)
        }
    }
}

internal fun decodeStrictUtf8(bytes: ByteArray): String {
    val decoder = StandardCharsets.UTF_8.newDecoder()
        .onMalformedInput(CodingErrorAction.REPORT)
        .onUnmappableCharacter(CodingErrorAction.REPORT)
    return try {
        decoder.decode(ByteBuffer.wrap(bytes)).toString()
    } catch (exc: Exception) {
        throw NativeCognitionContractException(
            "model output is not valid UTF-8 transport",
            exc,
        )
    }
}
