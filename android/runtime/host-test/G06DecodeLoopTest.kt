package ai.vn97.runtime

private fun checkDecode(limit: Int, predictions: List<Int>, expected: List<Int>, consumed: List<Int>, eos: Boolean) {
    var samples = 0
    val steps = mutableListOf<Int>()
    val events = mutableListOf<String>()
    val result = decodeG06Tokens(
        maxNewTokens = limit,
        eosTokenId = 99,
        sampleNext = {
            predictions[samples++].also { events += "sample:$it" }
        },
        consumeToken = { steps += it; events += "step:$it" },
    )
    check(result.tokenIds.toList() == expected)
    check(result.stoppedAtEos == eos)
    check(steps == consumed)
    check(samples == expected.size + if (eos) 1 else 0)
    // Each additional sample requires exactly the preceding emitted token's
    // step. Neither EOS nor an unused final prediction may invoke ONNX.
    check(events == buildList {
        expected.forEachIndexed { index, token ->
            add("sample:$token")
            if (index < consumed.size) add("step:$token")
        }
        if (eos) add("sample:99")
    })
}

fun main() {
    checkDecode(1, listOf(7), listOf(7), emptyList(), false)
    checkDecode(4, listOf(7, 8, 9, 10), listOf(7, 8, 9, 10), listOf(7, 8, 9), false)
    checkDecode(4, listOf(99), emptyList(), emptyList(), true)
    checkDecode(4, listOf(7, 8, 99), listOf(7, 8), listOf(7, 8), true)
    checkDecode(3, listOf(7, 8, 99), listOf(7, 8), listOf(7, 8), true)
    check(runCatching { decodeG06Tokens(0, 99, { error("must not sample") }, { error("must not step") }) }.isFailure)
    val failure = IllegalStateException("inference failed")
    val actual = runCatching { decodeG06Tokens(3, 99, { 7 }, { throw failure }) }.exceptionOrNull()
    check(actual === failure)
    println("G06 decode loop: token parity, terminal step elimination, EOS and error propagation PASS")
}
