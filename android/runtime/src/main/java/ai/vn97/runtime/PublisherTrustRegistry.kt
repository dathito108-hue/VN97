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

const val VN97_PUBLISHER_TRUST_FILE = "publisher-trust.vn97ptr1.json"
private const val VN97_PUBLISHER_TRUST_LOCK = ".publisher-trust.vn97ptr1.lock"
private const val VN97_PUBLISHER_TRUST_MAX_BYTES = 64 * 1024
private const val VN97_PUBLISHER_TRUST_MAX_KEYS = 64

class VN97PublisherTrustRegistry(
    root: File,
) {
    private data class StoredKey(
        val keyId: String,
        val publicKey: ByteArray,
    )

    private val rootPath = root.toPath().toAbsolutePath().normalize()
    private val trustPath = rootPath.resolve(VN97_PUBLISHER_TRUST_FILE)
    private val lockPath = rootPath.resolve(VN97_PUBLISHER_TRUST_LOCK)

    init {
        Files.createDirectories(rootPath)
        if (!Files.isDirectory(rootPath, LinkOption.NOFOLLOW_LINKS) ||
            Files.isSymbolicLink(rootPath)
        ) {
            throw VN97CapabilityTrustException(
                "publisher trust root must be a non-symlink directory"
            )
        }
        rejectSymlink(trustPath.toFile(), "publisher trust file")
        rejectSymlink(lockPath.toFile(), "publisher trust lock")
    }

    /**
     * Returns true when the exact key is already enrolled, false when the key_id is new.
     * Reusing an existing key_id with different public-key bytes fails closed.
     */
    fun requireCompatibleOrUnenrolled(
        keyId: String,
        publicKey: ByteArray,
    ): Boolean {
        val candidate = modelPublisherKey(keyId, publicKey)
        return locked {
            val existing = readLocked().firstOrNull { it.keyId == keyId }
                ?: return@locked false
            if (!existing.publicKey.contentEquals(candidate.publicKey())) {
                throw VN97CapabilityTrustException(
                    "publisher key_id is already bound to different public-key bytes"
                )
            }
            true
        }
    }

    /**
     * Explicitly enroll one model publisher under fixed, narrow policy and return a trust store
     * reconstructed from the durable registry bytes.
     */
    fun enrollModelPublisher(
        keyId: String,
        publicKey: ByteArray,
    ): VN97CapabilityTrustStore {
        val candidate = modelPublisherKey(keyId, publicKey)
        return locked {
            val keys = readLocked().toMutableList()
            val existingIndex = keys.indexOfFirst { it.keyId == keyId }
            if (existingIndex >= 0) {
                val existing = keys[existingIndex]
                if (!existing.publicKey.contentEquals(candidate.publicKey())) {
                    throw VN97CapabilityTrustException(
                        "publisher key_id cannot be rebound to a different public key"
                    )
                }
            } else {
                if (keys.size >= VN97_PUBLISHER_TRUST_MAX_KEYS) {
                    throw VN97CapabilityTrustException(
                        "publisher trust registry key limit reached"
                    )
                }
                keys += StoredKey(keyId, candidate.publicKey())
                writeLocked(keys)
            }

            val persisted = readLocked().firstOrNull { it.keyId == keyId }
                ?: throw VN97CapabilityTrustException(
                    "publisher trust enrollment did not persist"
                )
            VN97CapabilityTrustStore(
                listOf(modelPublisherKey(persisted.keyId, persisted.publicKey))
            )
        }
    }

    fun requireTrustStore(keyId: String): VN97CapabilityTrustStore = locked {
        val stored = readLocked().firstOrNull { it.keyId == keyId }
            ?: throw VN97CapabilityTrustException(
                "publisher key_id is not enrolled"
            )
        VN97CapabilityTrustStore(
            listOf(modelPublisherKey(stored.keyId, stored.publicKey))
        )
    }

    private fun modelPublisherKey(
        keyId: String,
        publicKey: ByteArray,
    ): VN97TrustedPublisherKey =
        VN97TrustedPublisherKey(
            keyId = keyId,
            publicKey = publicKey.copyOf(),
            capabilityPrefixes = setOf("model"),
            allowedKinds = setOf("weights"),
            minVersion = 1L,
            maxVersion = 0xffff_ffffL,
            revoked = false,
        )

    private fun readLocked(): List<StoredKey> {
        rejectSymlink(trustPath.toFile(), "publisher trust file")
        if (!Files.exists(trustPath, LinkOption.NOFOLLOW_LINKS)) {
            return emptyList()
        }
        if (!Files.isRegularFile(trustPath, LinkOption.NOFOLLOW_LINKS)) {
            throw VN97CapabilityTrustException(
                "publisher trust file must be a regular file"
            )
        }
        val size = Files.size(trustPath)
        if (size <= 0L || size > VN97_PUBLISHER_TRUST_MAX_BYTES.toLong()) {
            throw VN97CapabilityTrustException(
                "publisher trust file size is outside bounds"
            )
        }
        val bytes = Files.readAllBytes(trustPath)
        if (bytes.size.toLong() != size) {
            throw VN97CapabilityTrustException(
                "publisher trust file changed while being read"
            )
        }
        val text = strictUtf8(bytes)
        val root = try {
            VnStrictJson.parseObject(
                text,
                VnJsonLimits(
                    maxInputUtf8Bytes = VN97_PUBLISHER_TRUST_MAX_BYTES,
                    maxDepth = 8,
                    maxNodes = 512,
                    maxStringUtf8Bytes = 4096,
                ),
            )
        } catch (exc: RuntimeException) {
            throw VN97CapabilityTrustException(
                "publisher trust registry is not strict JSON",
                exc,
            )
        }
        if (VnStrictJson.canonical(root) != text ||
            root.values.keys != setOf("keys", "schema") ||
            root.string("schema") != "VN97PTR1"
        ) {
            throw VN97CapabilityTrustException(
                "publisher trust registry schema/canonical form is invalid"
            )
        }
        val rawKeys = root.array("keys").values
        if (rawKeys.isEmpty() || rawKeys.size > VN97_PUBLISHER_TRUST_MAX_KEYS) {
            throw VN97CapabilityTrustException(
                "publisher trust registry key count is invalid"
            )
        }
        val out = ArrayList<StoredKey>(rawKeys.size)
        var previous: String? = null
        for (raw in rawKeys) {
            val obj = raw as? VnJsonObject
                ?: throw VN97CapabilityTrustException(
                    "publisher trust key entry must be object"
                )
            if (obj.values.keys != setOf("key_id", "public_key")) {
                throw VN97CapabilityTrustException(
                    "publisher trust key entry fields are invalid"
                )
            }
            val keyId = obj.string("key_id")
            val publicKeyHex = obj.string("public_key")
            if (publicKeyHex.length != 64 ||
                publicKeyHex.any { it !in "0123456789abcdef" }
            ) {
                throw VN97CapabilityTrustException(
                    "publisher trust public key encoding is invalid"
                )
            }
            val publicKey = publicKeyHex.hexToBytes()
            modelPublisherKey(keyId, publicKey)
            if (previous != null && keyId <= previous) {
                throw VN97CapabilityTrustException(
                    "publisher trust keys must be sorted and unique"
                )
            }
            previous = keyId
            out += StoredKey(keyId, publicKey)
        }
        return out
    }

    private fun writeLocked(keys: List<StoredKey>) {
        if (keys.isEmpty() || keys.size > VN97_PUBLISHER_TRUST_MAX_KEYS) {
            throw VN97CapabilityTrustException(
                "publisher trust registry key count is invalid"
            )
        }
        val sorted = keys.sortedBy { it.keyId }
        if (sorted.map { it.keyId }.toSet().size != sorted.size) {
            throw VN97CapabilityTrustException(
                "publisher trust registry contains duplicate key_id"
            )
        }
        val root = VnStrictJson.objectOf(
            "keys" to VnStrictJson.array(
                sorted.map { key ->
                    VnStrictJson.objectOf(
                        "key_id" to VnStrictJson.string(key.keyId),
                        "public_key" to VnStrictJson.string(key.publicKey.toHex()),
                    )
                }
            ),
            "schema" to VnStrictJson.string("VN97PTR1"),
        )
        val bytes = VnStrictJson.canonical(root)
            .toByteArray(StandardCharsets.UTF_8)
        if (bytes.size > VN97_PUBLISHER_TRUST_MAX_BYTES) {
            throw VN97CapabilityTrustException(
                "publisher trust registry exceeds byte limit"
            )
        }
        rejectSymlink(trustPath.toFile(), "publisher trust file")
        val temp = Files.createTempFile(rootPath, ".vn97ptr-", ".tmp")
        try {
            FileOutputStream(temp.toFile()).use { output ->
                output.write(bytes)
                output.flush()
                output.fd.sync()
            }
            try {
                Files.move(
                    temp,
                    trustPath,
                    StandardCopyOption.ATOMIC_MOVE,
                    StandardCopyOption.REPLACE_EXISTING,
                )
            } catch (exc: AtomicMoveNotSupportedException) {
                throw VN97CapabilityTrustException(
                    "publisher trust filesystem lacks atomic replace",
                    exc,
                )
            }
            FileChannel.open(rootPath, StandardOpenOption.READ).use {
                it.force(true)
            }
        } finally {
            Files.deleteIfExists(temp)
        }
    }

    private fun <T> locked(block: () -> T): T {
        rejectSymlink(lockPath.toFile(), "publisher trust lock")
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

    private fun rejectSymlink(file: File, label: String) {
        if (Files.exists(file.toPath(), LinkOption.NOFOLLOW_LINKS) &&
            Files.isSymbolicLink(file.toPath())
        ) {
            throw VN97CapabilityTrustException("$label must not be a symlink")
        }
    }

    private fun VnJsonObject.string(key: String): String =
        (values[key] as? VnJsonString)?.value
            ?: throw VN97CapabilityTrustException(
                "publisher trust $key must be string"
            )

    private fun VnJsonObject.array(key: String): VnJsonArray =
        values[key] as? VnJsonArray
            ?: throw VN97CapabilityTrustException(
                "publisher trust $key must be array"
            )

    private fun String.hexToBytes(): ByteArray =
        ByteArray(length / 2) { index ->
            substring(index * 2, index * 2 + 2).toInt(16).toByte()
        }

    private fun ByteArray.toHex(): String =
        joinToString("") { "%02x".format(it.toInt() and 0xff) }

    private fun strictUtf8(bytes: ByteArray): String = try {
        StandardCharsets.UTF_8.newDecoder()
            .onMalformedInput(CodingErrorAction.REPORT)
            .onUnmappableCharacter(CodingErrorAction.REPORT)
            .decode(ByteBuffer.wrap(bytes))
            .toString()
    } catch (exc: Exception) {
        throw VN97CapabilityTrustException(
            "publisher trust registry is not strict UTF-8",
            exc,
        )
    }
}
