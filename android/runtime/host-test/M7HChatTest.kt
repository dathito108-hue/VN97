import ai.vn97.runtime.*

private inline fun expectFailure(block: () -> Unit) {
    var failed = false
    try {
        block()
    } catch (_: IllegalArgumentException) {
        failed = true
    }
    check(failed)
}

fun main() {
    val limits = NativeChatLimits(
        maxSystemPromptUtf8Bytes = 32,
        maxUserMessageUtf8Bytes = 64,
        maxPromptUtf8Bytes = 256,
        maxResponseUtf8Bytes = 128,
    )

    val fresh = NativeChatPrompt.build(
        systemPrompt = "VN97",
        userMessage = "a\"b\nc €",
        includeSystem = true,
        limits = limits,
    )
    check(fresh.startsWith("VN97CHAT1\n"))
    check("\"role\":\"system\"" in fresh)
    check("\"role\":\"user\"" in fresh)
    check("\\\"" in fresh)
    check("\\n" in fresh)
    check("€" in fresh)

    val continued = NativeChatPrompt.build(
        systemPrompt = "SHOULD_NOT_REPEAT",
        userMessage = "second turn",
        includeSystem = false,
        limits = limits,
    )
    check("\"role\":\"system\"" !in continued)
    check("SHOULD_NOT_REPEAT" !in continued)
    check("\"next_role\":\"assistant\"" in continued)

    expectFailure {
        NativeChatPrompt.build(
            systemPrompt = "",
            userMessage = "x".repeat(65),
            includeSystem = false,
            limits = limits,
        )
    }
    expectFailure {
        NativeChatConfig(
            systemPrompt = "x".repeat(33),
            limits = limits,
        )
    }

    val cancellation = NativeChatCancellation()
    check(!cancellation.isCancelled())
    cancellation.cancel()
    check(cancellation.isCancelled())

    val result = NativeChatTurnResult(
        userMessage = "hello",
        assistantText = "world",
        tokenIds = intArrayOf(7, 8),
        stopReason = NativeChatStopReason.EOS,
        sequenceStart = 10,
        sequenceEnd = 15,
    )
    check(result.promptTokensConsumed == 3L)

    expectFailure {
        NativeChatTurnResult(
            userMessage = "bad",
            assistantText = "",
            tokenIds = intArrayOf(1, 2),
            stopReason = NativeChatStopReason.CANCELLED,
            sequenceStart = 5,
            sequenceEnd = 6,
        )
    }

    println("M7H_CHAT_PROTOCOL_PASS")
}
