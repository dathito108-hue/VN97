import ai.vn97.runtime.*

private inline fun expectContract(block: () -> Unit) {
    var failed = false
    try {
        block()
    } catch (_: NativeCognitionContractException) {
        failed = true
    }
    check(failed)
}

fun main() {
    val prompt = NativeCognitionPrompt.build(
        NativeCognitionOperation.STEP,
        "{\"a\":1}",
        4096,
    )
    check(prompt.startsWith("VN97COG1\noperation=step\n"))
    check("schema={\"result\":\"string\",\"confidence\":0.0}" in prompt)
    check(prompt.endsWith("request={\"a\":1}"))

    expectContract {
        NativeCognitionPrompt.build(
            NativeCognitionOperation.PLAN,
            " {\"a\":1}",
            4096,
        )
    }
    expectContract {
        NativeCognitionPrompt.build(
            NativeCognitionOperation.PLAN,
            "{\"a\":1}\n{\"b\":2}",
            4096,
        )
    }

    check(decodeStrictUtf8("€".toByteArray()) == "€")
    expectContract {
        decodeStrictUtf8(byteArrayOf(0xC3.toByte(), 0x28))
    }

    val adapter = NativeCognitionAdapterConfig()
    check(adapter.maxNewTokens(NativeCognitionOperation.PLAN) == 768)
    check(adapter.maxNewTokens(NativeCognitionOperation.MEMORY_QUERY) == 384)
    check(adapter.maxNewTokens(NativeCognitionOperation.STEP) == 1024)
    check(adapter.maxNewTokens(NativeCognitionOperation.VERIFY) == 384)
    check(adapter.maxNewTokens(NativeCognitionOperation.EXTERNAL_INTENT) == 512)
    check(NativeCognitionInferenceLimits().maxPromptTokens == 4096)
    check(NativeCognitionInferenceLimits().maxOutputUtf8Bytes == 64 * 1024)

    println("M7I_COGNITION_CONTRACT_PASS")
}
