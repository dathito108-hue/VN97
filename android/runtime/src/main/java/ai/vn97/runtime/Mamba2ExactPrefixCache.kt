package ai.vn97.runtime

import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.security.MessageDigest
import java.util.LinkedHashMap

class VN97Mamba2PrefixBinding(
    val runtimeId: String,
    val graphSha256: String,
    val tokenizerModelId: String,
    val vocabSize: Int,
    val stateDtype: String,
    convStateShape: List<Long>,
    ssmStateShape: List<Long>,
) {
    val convStateShape: List<Long> = convStateShape.toList()
    val ssmStateShape: List<Long> = ssmStateShape.toList()

    init {
        listOf(runtimeId, graphSha256, tokenizerModelId).forEach {
            require(SHA256_HEX.matches(it)) { "prefix binding identity is invalid" }
        }
        require(vocabSize > 1)
        require(stateDtype in setOf("float16", "float32"))
        require(convStateShape.isNotEmpty() && convStateShape.all { it > 0L })
        require(ssmStateShape.isNotEmpty() && ssmStateShape.all { it > 0L })
    }

    val bindingId: String by lazy(LazyThreadSafetyMode.PUBLICATION) {
        sha256Prefix(
            buildString {
                append("VN97M2PREFIXBIND1\u0000")
                append(runtimeId).append('\u0000')
                append(graphSha256).append('\u0000')
                append(tokenizerModelId).append('\u0000')
                append(vocabSize).append('\u0000')
                append(stateDtype).append('\u0000')
                append(convStateShape.joinToString(",")).append('\u0000')
                append(ssmStateShape.joinToString(","))
            }.toByteArray(Charsets.US_ASCII)
        )
    }

    override fun equals(other: Any?): Boolean = other is VN97Mamba2PrefixBinding &&
        runtimeId == other.runtimeId &&
        graphSha256 == other.graphSha256 &&
        tokenizerModelId == other.tokenizerModelId &&
        vocabSize == other.vocabSize &&
        stateDtype == other.stateDtype &&
        convStateShape == other.convStateShape &&
        ssmStateShape == other.ssmStateShape

    override fun hashCode(): Int {
        var result = runtimeId.hashCode()
        result = 31 * result + graphSha256.hashCode()
        result = 31 * result + tokenizerModelId.hashCode()
        result = 31 * result + vocabSize
        result = 31 * result + stateDtype.hashCode()
        result = 31 * result + convStateShape.hashCode()
        return 31 * result + ssmStateShape.hashCode()
    }
}

class VN97Mamba2PrefixSnapshot private constructor(
    val binding: VN97Mamba2PrefixBinding,
    val sequencePosition: Long,
    val prefixTokenCount: Int,
    val prefixTokenSha256: String,
    private val convState: ByteArray,
    private val ssmState: ByteArray,
) {
    val stateBytes: Long = convState.size.toLong() + ssmState.size.toLong()
    val cacheKey: String = binding.bindingId + ":" + prefixTokenSha256

    init {
        require(sequencePosition == prefixTokenCount.toLong())
        require(prefixTokenCount > 0)
        require(SHA256_HEX.matches(prefixTokenSha256))
        require(convState.isNotEmpty() && ssmState.isNotEmpty())
        require(stateBytes > 0L)
    }

    fun matches(
        expectedBinding: VN97Mamba2PrefixBinding,
        prefixTokenIds: IntArray,
    ): Boolean = binding == expectedBinding &&
        prefixTokenIds.size == prefixTokenCount &&
        prefixTokenSha256 == VN97Mamba2ExactPrefixCache.hashPrefix(
            prefixTokenIds,
            expectedBinding.vocabSize,
        )

    internal fun restoreInto(
        conv: Mamba2OrtTensorStorage,
        ssm: Mamba2OrtTensorStorage,
    ) {
        require(conv.rawByteSize == convState.size)
        require(ssm.rawByteSize == ssmState.size)
        conv.writeRawBytes(convState)
        ssm.writeRawBytes(ssmState)
    }

    companion object {
        internal fun capture(
            binding: VN97Mamba2PrefixBinding,
            prefixTokenIds: IntArray,
            convState: ByteArray,
            ssmState: ByteArray,
        ): VN97Mamba2PrefixSnapshot {
            require(prefixTokenIds.isNotEmpty())
            return VN97Mamba2PrefixSnapshot(
                binding = binding,
                sequencePosition = prefixTokenIds.size.toLong(),
                prefixTokenCount = prefixTokenIds.size,
                prefixTokenSha256 = VN97Mamba2ExactPrefixCache.hashPrefix(
                    prefixTokenIds,
                    binding.vocabSize,
                ),
                // Both arrays are newly allocated by the internal executor capture path.
                // Avoid doubling the very large recurrent-state peak during capture.
                convState = convState,
                ssmState = ssmState,
            )
        }
    }
}

/** Bounded in-memory LRU. It never treats a shorter, edited or incompatible prefix as a hit. */
class VN97Mamba2ExactPrefixCache(
    private val maxEntries: Int,
    private val maxStateBytes: Long,
) {
    private val entries = LinkedHashMap<String, VN97Mamba2PrefixSnapshot>(
        maxEntries.coerceAtMost(16),
        0.75f,
        true,
    )
    private var stateBytes = 0L

    init {
        require(maxEntries > 0)
        require(maxStateBytes > 0L)
    }

    @Synchronized
    fun put(snapshot: VN97Mamba2PrefixSnapshot): Boolean {
        if (snapshot.stateBytes > maxStateBytes) return false
        entries.remove(snapshot.cacheKey)?.let { stateBytes -= it.stateBytes }
        entries[snapshot.cacheKey] = snapshot
        stateBytes = Math.addExact(stateBytes, snapshot.stateBytes)
        trim()
        return entries[snapshot.cacheKey] === snapshot
    }

    @Synchronized
    fun find(
        binding: VN97Mamba2PrefixBinding,
        prefixTokenIds: IntArray,
    ): VN97Mamba2PrefixSnapshot? {
        if (prefixTokenIds.isEmpty()) return null
        val hash = hashPrefix(prefixTokenIds, binding.vocabSize)
        val snapshot = entries[binding.bindingId + ":" + hash] ?: return null
        return snapshot.takeIf { it.matches(binding, prefixTokenIds) }
    }

    @Synchronized
    fun invalidateIncompatible(binding: VN97Mamba2PrefixBinding): Int {
        val incompatible = entries.filterValues { it.binding != binding }.keys.toList()
        incompatible.forEach { key ->
            entries.remove(key)?.let { stateBytes -= it.stateBytes }
        }
        return incompatible.size
    }

    @Synchronized
    fun clear() {
        entries.clear()
        stateBytes = 0L
    }

    @Synchronized
    fun entryCount(): Int = entries.size

    @Synchronized
    fun retainedStateBytes(): Long = stateBytes

    private fun trim() {
        while (entries.size > maxEntries || stateBytes > maxStateBytes) {
            val iterator = entries.entries.iterator()
            val eldest = iterator.next()
            iterator.remove()
            stateBytes -= eldest.value.stateBytes
        }
    }

    companion object {
        internal fun hashPrefix(tokenIds: IntArray, vocabSize: Int): String {
            require(tokenIds.isNotEmpty())
            require(vocabSize > 1)
            val digest = MessageDigest.getInstance("SHA-256")
            digest.update("VN97M2PREFIXTOKENS1\u0000".toByteArray(Charsets.US_ASCII))
            val encoded = ByteBuffer.allocate(Int.SIZE_BYTES).order(ByteOrder.BIG_ENDIAN)
            tokenIds.forEach { token ->
                require(token in 0 until vocabSize) { "prefix token is outside vocabulary" }
                encoded.clear()
                encoded.putInt(token)
                digest.update(encoded.array())
            }
            return digest.digest().joinToString("") { "%02x".format(it) }
        }
    }
}

private val SHA256_HEX = Regex("^[0-9a-f]{64}$")

private fun sha256Prefix(bytes: ByteArray): String = MessageDigest.getInstance("SHA-256")
    .digest(bytes).joinToString("") { "%02x".format(it) }
