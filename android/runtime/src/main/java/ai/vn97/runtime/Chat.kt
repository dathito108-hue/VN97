package ai.vn97.runtime

import java.nio.charset.StandardCharsets
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicReference

private const val CHAT_PROTOCOL = "VN97CHAT1"

/** Bounded chat-surface limits. Runtime/model bounds remain authoritative underneath. */
data class NativeChatLimits(
    val maxSystemPromptUtf8Bytes: Int = 8 * 1024,
    val maxUserMessageUtf8Bytes: Int = 16 * 1024,
    val maxPromptUtf8Bytes: Int = 32 * 1024,
    val maxResponseUtf8Bytes: Int = 64 * 1024,
) {
    init {
        require(maxSystemPromptUtf8Bytes > 0) { "maxSystemPromptUtf8Bytes must be positive" }
        require(maxUserMessageUtf8Bytes > 0) { "maxUserMessageUtf8Bytes must be positive" }
        require(maxPromptUtf8Bytes > 0) { "maxPromptUtf8Bytes must be positive" }
        require(maxResponseUtf8Bytes > 0) { "maxResponseUtf8Bytes must be positive" }
        require(maxPromptUtf8Bytes >= maxUserMessageUtf8Bytes) {
            "maxPromptUtf8Bytes must accommodate one maximum user message"
        }
    }
}

data class NativeChatConfig(
    val systemPrompt: String = "",
    val generation: NativeGenerationConfig = NativeGenerationConfig(),
    val limits: NativeChatLimits = NativeChatLimits(),
) {
    init {
        val systemBytes = utf8Size(systemPrompt)
        require(systemBytes <= limits.maxSystemPromptUtf8Bytes) {
            "systemPrompt exceeds maxSystemPromptUtf8Bytes"
        }
    }
}

class NativeChatCancellation {
    private val cancelled = AtomicBoolean(false)

    fun cancel() {
        cancelled.set(true)
    }

    fun isCancelled(): Boolean = cancelled.get()
}

enum class NativeChatStopReason {
    EOS,
    TOKEN_LIMIT,
    CANCELLED,
    RESPONSE_LIMIT,
}

data class NativeChatDelta(
    val tokenIndex: Int,
    val tokenId: Int?,
    val text: String,
    val final: Boolean,
)

data class NativeChatTurnResult(
    val userMessage: String,
    val assistantText: String,
    val tokenIds: IntArray,
    val stopReason: NativeChatStopReason,
    val sequenceStart: Long,
    val sequenceEnd: Long,
) {
    init {
        require(sequenceEnd >= sequenceStart) { "chat sequence position moved backwards" }
        require(sequenceEnd - sequenceStart >= tokenIds.size.toLong()) {
            "chat result cannot contain more generated tokens than committed runtime steps"
        }
    }

    val promptTokensConsumed: Long
        get() = sequenceEnd - sequenceStart - tokenIds.size.toLong()
}

internal object NativeChatPrompt {
    fun build(
        systemPrompt: String,
        userMessage: String,
        includeSystem: Boolean,
        limits: NativeChatLimits,
    ): String {
        require(userMessage.isNotBlank()) { "user message must not be blank" }
        val userBytes = utf8Size(userMessage)
        require(userBytes <= limits.maxUserMessageUtf8Bytes) {
            "user message exceeds maxUserMessageUtf8Bytes"
        }
        if (includeSystem) {
            require(utf8Size(systemPrompt) <= limits.maxSystemPromptUtf8Bytes) {
                "system prompt exceeds maxSystemPromptUtf8Bytes"
            }
        }

        val prompt = buildString {
            append(CHAT_PROTOCOL)
            append('\n')
            append("{\"messages\":[")
            var needsComma = false
            if (includeSystem && systemPrompt.isNotEmpty()) {
                appendMessage("system", systemPrompt)
                needsComma = true
            }
            if (needsComma) append(',')
            appendMessage("user", userMessage)
            append("],\"next_role\":\"assistant\"}")
        }
        require(utf8Size(prompt) <= limits.maxPromptUtf8Bytes) {
            "canonical chat prompt exceeds maxPromptUtf8Bytes"
        }
        return prompt
    }

    private fun StringBuilder.appendMessage(role: String, content: String) {
        append("{\"role\":\"")
        append(role)
        append("\",\"content\":\"")
        appendJsonEscaped(content)
        append("\"}")
    }

    private fun StringBuilder.appendJsonEscaped(value: String) {
        val hex = "0123456789abcdef"
        for (ch in value) {
            when (ch) {
                '"' -> append("\\\"")
                '\\' -> append("\\\\")
                '\b' -> append("\\b")
                '\u000C' -> append("\\f")
                '\n' -> append("\\n")
                '\r' -> append("\\r")
                '\t' -> append("\\t")
                else -> {
                    if (ch.code < 0x20) {
                        append("\\u")
                        append(hex[(ch.code ushr 12) and 0xf])
                        append(hex[(ch.code ushr 8) and 0xf])
                        append(hex[(ch.code ushr 4) and 0xf])
                        append(hex[ch.code and 0xf])
                    } else {
                        append(ch)
                    }
                }
            }
        }
    }
}


class NativeChatController(
    private val owner: NativeRuntimeOwner,
    private val model: NativeActivatedModel,
    private val config: NativeChatConfig = NativeChatConfig(),
) {
    private val activeCancellation = AtomicReference<NativeChatCancellation?>(null)

    fun isGenerating(): Boolean = activeCancellation.get() != null

    fun cancelActive(): Boolean {
        val cancellation = activeCancellation.get() ?: return false
        cancellation.cancel()
        return true
    }

    fun send(
        userMessage: String,
        onDelta: (NativeChatDelta) -> Boolean,
    ): NativeChatTurnResult {
        val cancellation = NativeChatCancellation()
        check(activeCancellation.compareAndSet(null, cancellation)) {
            "a chat generation is already active"
        }
        return try {
            owner.chatStreaming(
                model = model,
                userMessage = userMessage,
                config = config,
                cancellation = cancellation,
                onDelta = onDelta,
            )
        } finally {
            activeCancellation.compareAndSet(cancellation, null)
        }
    }
}

fun NativeRuntimeSession.chatStreaming(
    model: NativeActivatedModel,
    userMessage: String,
    config: NativeChatConfig = NativeChatConfig(),
    cancellation: NativeChatCancellation = NativeChatCancellation(),
    onDelta: (NativeChatDelta) -> Boolean,
): NativeChatTurnResult {
    val before = info()
    require(before.lifecycle == RuntimeLifecycle.ACTIVE) { "runtime must be ACTIVE for chat" }
    require(before.config.batch == 1) { "chat streaming currently requires batch=1" }

    val binding = modelBinding()
    if (binding.bound) {
        require(binding.modelId.contentEquals(model.info.modelId)) {
            "runtime is bound to a different activated model"
        }
    } else {
        require(before.sequencePosition == 0L) {
            "unbound runtime with nonzero sequence position cannot start chat"
        }
    }

    val prompt = NativeChatPrompt.build(
        systemPrompt = config.systemPrompt,
        userMessage = userMessage,
        includeSystem = before.sequencePosition == 0L,
        limits = config.limits,
    )

    if (cancellation.isCancelled()) {
        return NativeChatTurnResult(
            userMessage = userMessage,
            assistantText = "",
            tokenIds = IntArray(0),
            stopReason = NativeChatStopReason.CANCELLED,
            sequenceStart = before.sequencePosition,
            sequenceEnd = before.sequencePosition,
        )
    }

    var responseBytes = 0
    var responseLimitReached = false
    val generated = generateStreaming(
        model = model,
        prompt = prompt,
        config = config.generation.copy(
            addBosOnFreshSession = before.sequencePosition == 0L,
            addTextControl = true,
        ),
    ) { chunk ->
        val deltaBytes = utf8Size(chunk.text)
        val nextBytes = responseBytes.toLong() + deltaBytes.toLong()
        responseBytes = if (nextBytes > Int.MAX_VALUE.toLong()) Int.MAX_VALUE else nextBytes.toInt()
        responseLimitReached = responseBytes > config.limits.maxResponseUtf8Bytes
        val delivered = onDelta(
            NativeChatDelta(
                tokenIndex = chunk.tokenIndex,
                tokenId = chunk.tokenId,
                text = chunk.text,
                final = chunk.final,
            )
        )
        delivered && !cancellation.isCancelled() && !responseLimitReached
    }

    val mappedStop = when {
        responseLimitReached -> NativeChatStopReason.RESPONSE_LIMIT
        generated.stopReason == NativeGenerationStopReason.EOS -> NativeChatStopReason.EOS
        generated.stopReason == NativeGenerationStopReason.TOKEN_LIMIT -> NativeChatStopReason.TOKEN_LIMIT
        else -> NativeChatStopReason.CANCELLED
    }
    check(generated.sequencePosition >= before.sequencePosition) {
        "native generation sequence position moved backwards"
    }
    return NativeChatTurnResult(
        userMessage = userMessage,
        assistantText = generated.text,
        tokenIds = generated.tokenIds,
        stopReason = mappedStop,
        sequenceStart = before.sequencePosition,
        sequenceEnd = generated.sequencePosition,
    )
}

private fun utf8Size(value: String): Int = value.toByteArray(StandardCharsets.UTF_8).size
