package ai.vn97.runtime

private fun binding(
    runtime: Char = 'a',
    tokenizer: Char = 'c',
    dtype: String = "float16",
    ssmLast: Long = 8,
) = VN97Mamba2PrefixBinding(
    runtimeId = runtime.toString().repeat(64),
    graphSha256 = "b".repeat(64),
    tokenizerModelId = tokenizer.toString().repeat(64),
    vocabSize = 32,
    stateDtype = dtype,
    convStateShape = listOf(2, 1, 4, 3),
    ssmStateShape = listOf(2, 1, 2, 4, ssmLast),
)

private fun snapshot(
    binding: VN97Mamba2PrefixBinding,
    tokens: IntArray,
    bytes: Int = 16,
) = VN97Mamba2PrefixSnapshot.capture(
    binding = binding,
    prefixTokenIds = tokens,
    convState = ByteArray(bytes / 2) { 1 },
    ssmState = ByteArray(bytes - bytes / 2) { 2 },
)

fun main() {
    val original = intArrayOf(1, 2, 3, 4)
    val active = binding()
    val first = snapshot(active, original)
    val cache = VN97Mamba2ExactPrefixCache(maxEntries = 2, maxStateBytes = 40)
    check(cache.put(first))
    check(cache.find(active, original) === first)
    check(cache.find(active, intArrayOf(1, 2, 9, 4)) == null)
    check(cache.find(active, intArrayOf(1, 2, 3)) == null)
    check(cache.find(binding(tokenizer = 'd'), original) == null)
    check(cache.find(binding(dtype = "float32"), original) == null)
    check(cache.find(binding(ssmLast = 9), original) == null)
    check(cache.find(binding(runtime = 'd'), original) == null)

    val secondTokens = intArrayOf(5, 6)
    val second = snapshot(active, secondTokens)
    val thirdTokens = intArrayOf(7, 8)
    val third = snapshot(active, thirdTokens)
    check(cache.put(second))
    check(cache.find(active, original) === first) // refresh first in access order
    check(cache.put(third))
    check(cache.find(active, secondTokens) == null)
    check(cache.find(active, original) === first)
    check(cache.find(active, thirdTokens) === third)
    check(cache.entryCount() == 2)
    check(cache.retainedStateBytes() == 32L)

    val oversized = snapshot(active, intArrayOf(10), bytes = 41)
    check(!cache.put(oversized))
    check(cache.find(active, intArrayOf(10)) == null)
    check(cache.invalidateIncompatible(binding(tokenizer = 'e')) == 2)
    check(cache.entryCount() == 0)
    check(cache.retainedStateBytes() == 0L)

    check(runCatching {
        VN97Mamba2ExactPrefixCache.hashPrefix(intArrayOf(32), active.vocabSize)
    }.isFailure)
    check(active.bindingId.length == 64)
    println("G06 exact-prefix cache: identity/edit/bounds/LRU contracts PASS")
}
