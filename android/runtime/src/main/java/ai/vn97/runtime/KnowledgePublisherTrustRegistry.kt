package ai.vn97.runtime

import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.charset.CodingErrorAction
import java.nio.charset.StandardCharsets
import java.nio.channels.FileChannel
import java.nio.file.AtomicMoveNotSupportedException
import java.nio.file.Files
import java.nio.file.LinkOption
import java.nio.file.StandardCopyOption
import java.nio.file.StandardOpenOption

const val VN97_KNOWLEDGE_PUBLISHER_TRUST_FILE =
    "knowledge-publisher-trust.vn97kptr1.json"

private const val KNOWLEDGE_TRUST_LOCK =
    ".knowledge-publisher-trust.vn97kptr1.lock"
private const val KNOWLEDGE_TRUST_MAX_BYTES = 64 * 1024
private const val KNOWLEDGE_TRUST_MAX_KEYS = 64

class VN97KnowledgePublisherTrustRegistry(
    root: File,
) {
    private data class StoredKey(
        val keyId: String,
        val publicKey: ByteArray,
    )

    private val rootPath =
        root.toPath().toAbsolutePath().normalize()
    private val trustPath =
        rootPath.resolve(VN97_KNOWLEDGE_PUBLISHER_TRUST_FILE)
    private val lockPath =
        rootPath.resolve(KNOWLEDGE_TRUST_LOCK)

    init {
        Files.createDirectories(rootPath)
        if (
            !Files.isDirectory(
                rootPath,
                LinkOption.NOFOLLOW_LINKS,
            ) ||
            Files.isSymbolicLink(rootPath)
        ) {
            throw VN97CapabilityTrustException(
                "knowledge publisher trust root must be a non-symlink directory"
            )
        }
        rejectSymlink(
            trustPath.toFile(),
            "knowledge publisher trust file",
        )
        rejectSymlink(
            lockPath.toFile(),
            "knowledge publisher trust lock",
        )
    }

    /**
     * Returns true only when the exact key_id/public-key binding is already
     * durable. A key ID can never be rebound to different bytes.
     */
    fun requireCompatibleOrUnenrolled(
        keyId: String,
        publicKey: ByteArray,
    ): Boolean {
        val candidate = knowledgePublisherKey(keyId, publicKey)
        return locked {
            val existing =
                readLocked().firstOrNull { it.keyId == keyId }
                    ?: return@locked false
            if (
                !existing.publicKey.contentEquals(
                    candidate.publicKey()
                )
            ) {
                throw VN97CapabilityTrustException(
                    "knowledge publisher key_id is already bound to different public-key bytes"
                )
            }
            true
        }
    }

    /**
     * Explicitly persists a publisher under a fixed data-only policy:
     * namespace knowledge / knowledge.* and kind knowledge only.
     */
    fun enrollKnowledgePublisher(
        keyId: String,
        publicKey: ByteArray,
    ): VN97CapabilityTrustStore {
        val candidate = knowledgePublisherKey(keyId, publicKey)
        return locked {
            val keys = readLocked().toMutableList()
            val existingIndex =
                keys.indexOfFirst { it.keyId == keyId }
            if (existingIndex >= 0) {
                val existing = keys[existingIndex]
                if (
                    !existing.publicKey.contentEquals(
                        candidate.publicKey()
                    )
                ) {
                    throw VN97CapabilityTrustException(
                        "knowledge publisher key_id cannot be rebound"
                    )
                }
            } else {
                if (keys.size >= KNOWLEDGE_TRUST_MAX_KEYS) {
                    throw VN97CapabilityTrustException(
                        "knowledge publisher trust registry key limit reached"
                    )
                }
                keys += StoredKey(
                    keyId,
                    candidate.publicKey(),
                )
                writeLocked(keys)
            }
            val persisted =
                readLocked().firstOrNull { it.keyId == keyId }
                    ?: throw VN97CapabilityTrustException(
                        "knowledge publisher enrollment did not persist"
                    )
            VN97CapabilityTrustStore(
                listOf(
                    knowledgePublisherKey(
                        persisted.keyId,
                        persisted.publicKey,
                    )
                )
            )
        }
    }

    fun requireTrustStore(
        keyId: String,
    ): VN97CapabilityTrustStore = locked {
        val stored =
            readLocked().firstOrNull { it.keyId == keyId }
                ?: throw VN97CapabilityTrustException(
                    "knowledge publisher key_id is not enrolled"
                )
        VN97CapabilityTrustStore(
            listOf(
                knowledgePublisherKey(
                    stored.keyId,
                    stored.publicKey,
                )
            )
        )
    }

    private fun knowledgePublisherKey(
        keyId: String,
        publicKey: ByteArray,
    ): VN97TrustedPublisherKey =
        VN97TrustedPublisherKey(
            keyId = keyId,
            publicKey = publicKey.copyOf(),
            capabilityPrefixes = setOf("knowledge"),
            allowedKinds = setOf("knowledge"),
            minVersion = 1L,
            maxVersion = 0xffff_ffffL,
            revoked = false,
        )

    private fun readLocked(): List<StoredKey> {
        rejectSymlink(
            trustPath.toFile(),
            "knowledge publisher trust file",
        )
        if (
            !Files.exists(
                trustPath,
                LinkOption.NOFOLLOW_LINKS,
            )
        ) {
            return emptyList()
        }
        if (
            !Files.isRegularFile(
                trustPath,
                LinkOption.NOFOLLOW_LINKS,
            )
        ) {
            throw VN97CapabilityTrustException(
                "knowledge publisher trust file must be regular"
            )
        }
        val size = Files.size(trustPath)
        if (
            size <= 0L ||
            size > KNOWLEDGE_TRUST_MAX_BYTES.toLong()
        ) {
            throw VN97CapabilityTrustException(
                "knowledge publisher trust file size is outside bounds"
            )
        }
        val bytes = Files.readAllBytes(trustPath)
        if (bytes.size.toLong() != size) {
            throw VN97CapabilityTrustException(
                "knowledge publisher trust file changed while reading"
            )
        }
        val text = strictUtf8(bytes)
        val root = try {
            VnStrictJson.parseObject(
                text,
                VnJsonLimits(
                    maxInputUtf8Bytes =
                        KNOWLEDGE_TRUST_MAX_BYTES,
                    maxDepth = 8,
                    maxNodes = 512,
                    maxStringUtf8Bytes = 4096,
                ),
            )
        } catch (exc: RuntimeException) {
            throw VN97CapabilityTrustException(
                "knowledge publisher trust registry is not strict JSON",
                exc,
            )
        }
        if (
            VnStrictJson.canonical(root) != text ||
            root.values.keys != setOf("keys", "schema") ||
            root.string("schema") != "VN97KPTR1"
        ) {
            throw VN97CapabilityTrustException(
                "knowledge publisher trust schema/canonical form is invalid"
            )
        }
        val rawKeys = root.array("keys").values
        if (
            rawKeys.isEmpty() ||
            rawKeys.size > KNOWLEDGE_TRUST_MAX_KEYS
        ) {
            throw VN97CapabilityTrustException(
                "knowledge publisher trust key count is invalid"
            )
        }
        val out = ArrayList<StoredKey>(rawKeys.size)
        var previous: String? = null
        rawKeys.forEach { raw ->
            val obj = raw as? VnJsonObject
                ?: throw VN97CapabilityTrustException(
                    "knowledge publisher key entry must be object"
                )
            if (
                obj.values.keys !=
                setOf("key_id", "public_key")
            ) {
                throw VN97CapabilityTrustException(
                    "knowledge publisher key fields are invalid"
                )
            }
            val keyId = obj.string("key_id")
            val publicHex = obj.string("public_key")
            if (
                publicHex.length != 64 ||
                publicHex.any {
                    it !in "0123456789abcdef"
                }
            ) {
                throw VN97CapabilityTrustException(
                    "knowledge publisher public key encoding is invalid"
                )
            }
            val publicKey = publicHex.hexToBytes()
            knowledgePublisherKey(keyId, publicKey)
            if (previous != null && keyId <= previous!!) {
                throw VN97CapabilityTrustException(
                    "knowledge publisher keys must be sorted and unique"
                )
            }
            previous = keyId
            out += StoredKey(keyId, publicKey)
        }
        return out
    }

    private fun writeLocked(keys: List<StoredKey>) {
        if (
            keys.isEmpty() ||
            keys.size > KNOWLEDGE_TRUST_MAX_KEYS
        ) {
            throw VN97CapabilityTrustException(
                "knowledge publisher trust key count is invalid"
            )
        }
        val sorted = keys.sortedBy { it.keyId }
        if (sorted.map { it.keyId }.toSet().size != sorted.size) {
            throw VN97CapabilityTrustException(
                "knowledge publisher trust contains duplicate key_id"
            )
        }
        val root = VnStrictJson.objectOf(
            "keys" to VnStrictJson.array(
                sorted.map { key ->
                    VnStrictJson.objectOf(
                        "key_id" to
                            VnStrictJson.string(key.keyId),
                        "public_key" to
                            VnStrictJson.string(
                                key.publicKey.toHex()
                            ),
                    )
                }
            ),
            "schema" to
                VnStrictJson.string("VN97KPTR1"),
        )
        val bytes =
            VnStrictJson.canonical(root)
                .toByteArray(StandardCharsets.UTF_8)
        if (bytes.size > KNOWLEDGE_TRUST_MAX_BYTES) {
            throw VN97CapabilityTrustException(
                "knowledge publisher trust exceeds byte bound"
            )
        }
        rejectSymlink(
            trustPath.toFile(),
            "knowledge publisher trust file",
        )
        val temp =
            Files.createTempFile(
                rootPath,
                ".vn97kptr-",
                ".tmp",
            )
        try {
            FileOutputStream(temp.toFile()).use { out ->
                out.write(bytes)
                out.flush()
                out.fd.sync()
            }
            try {
                Files.move(
                    temp,
                    trustPath,
                    StandardCopyOption.ATOMIC_MOVE,
                    StandardCopyOption.REPLACE_EXISTING,
                )
            } catch (
                exc: AtomicMoveNotSupportedException
            ) {
                throw VN97CapabilityTrustException(
                    "knowledge publisher trust requires atomic replace",
                    exc,
                )
            }
            FileChannel.open(
                rootPath,
                StandardOpenOption.READ,
            ).use { it.force(true) }
        } finally {
            Files.deleteIfExists(temp)
        }
    }

    private fun <T> locked(block: () -> T): T {
        rejectSymlink(
            lockPath.toFile(),
            "knowledge publisher trust lock",
        )
        FileChannel.open(
            lockPath,
            StandardOpenOption.CREATE,
            StandardOpenOption.WRITE,
            LinkOption.NOFOLLOW_LINKS,
        ).use { channel ->
            channel.lock().use {
                return block()
            }
        }
    }

    private fun rejectSymlink(
        file: File,
        label: String,
    ) {
        if (
            Files.exists(
                file.toPath(),
                LinkOption.NOFOLLOW_LINKS,
            ) &&
            Files.isSymbolicLink(file.toPath())
        ) {
            throw VN97CapabilityTrustException(
                "$label must not be a symlink"
            )
        }
    }

    private fun VnJsonObject.string(key: String): String =
        (values[key] as? VnJsonString)?.value
            ?: throw VN97CapabilityTrustException(
                "knowledge trust $key must be string"
            )

    private fun VnJsonObject.array(
        key: String,
    ): VnJsonArray =
        values[key] as? VnJsonArray
            ?: throw VN97CapabilityTrustException(
                "knowledge trust $key must be array"
            )

    private fun String.hexToBytes(): ByteArray =
        ByteArray(length / 2) { index ->
            substring(
                index * 2,
                index * 2 + 2,
            ).toInt(16).toByte()
        }

    private fun ByteArray.toHex(): String =
        joinToString("") {
            "%02x".format(it.toInt() and 0xff)
        }

    private fun strictUtf8(bytes: ByteArray): String =
        try {
            StandardCharsets.UTF_8.newDecoder()
                .onMalformedInput(
                    CodingErrorAction.REPORT
                )
                .onUnmappableCharacter(
                    CodingErrorAction.REPORT
                )
                .decode(ByteBuffer.wrap(bytes))
                .toString()
        } catch (exc: Exception) {
            throw VN97CapabilityTrustException(
                "knowledge publisher trust is not strict UTF-8",
                exc,
            )
        }
}
