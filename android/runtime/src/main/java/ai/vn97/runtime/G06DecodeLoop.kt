package ai.vn97.runtime

internal data class G06DecodedTokens(
    val tokenIds: IntArray,
    val stoppedAtEos: Boolean,
)

/**
 * Prefill has already produced logits for the first token. Consume a sampled
 * token only when another prediction is needed. The last emitted token at the
 * token limit is therefore not part of the recurrent state. Callers report the
 * executor's actual sequence position; independent G06 operations reset state.
 */
internal fun decodeG06Tokens(
    maxNewTokens: Int,
    eosTokenId: Int,
    sampleNext: () -> Int,
    consumeToken: (Int) -> Unit,
): G06DecodedTokens {
    require(maxNewTokens > 0)
    val generated = ArrayList<Int>(minOf(maxNewTokens, 256))
    repeat(maxNewTokens) { index ->
        val token = sampleNext()
        if (token == eosTokenId) {
            return G06DecodedTokens(generated.toIntArray(), stoppedAtEos = true)
        }
        generated += token
        if (index + 1 < maxNewTokens) consumeToken(token)
    }
    return G06DecodedTokens(generated.toIntArray(), stoppedAtEos = false)
}
