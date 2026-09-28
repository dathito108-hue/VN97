package ai.vn97.platform

import android.content.Context
import android.security.keystore.KeyGenParameterSpec
import android.security.keystore.KeyProperties
import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import java.io.DataInputStream
import java.io.DataOutputStream
import java.io.File
import java.nio.file.AtomicMoveNotSupportedException
import java.nio.file.Files
import java.nio.file.LinkOption
import java.nio.file.StandardCopyOption
import java.security.KeyStore
import javax.crypto.Cipher
import javax.crypto.KeyGenerator
import javax.crypto.SecretKey
import javax.crypto.spec.GCMParameterSpec

class AndroidExnessCredentialVault(
    context: Context,
    private val keyAlias: String =
        "vn97.exness.credentials.v1",
) {
    private val root =
        File(context.noBackupFilesDir, ROOT_DIR)
    private val target =
        File(root, VAULT_FILE)

    init {
        ensureRoot()
    }

    @Synchronized
    fun store(
        apiKey: String,
        accountId: String,
        baseUrl: String,
        privateKeySecret: ByteArray,
    ) {
        val clear =
            VN97ExnessCredentialPayloadCodec.encode(
                apiKey = apiKey,
                accountId = accountId,
                baseUrl = baseUrl,
                privateKeySecret = privateKeySecret,
            )
        try {
            val cipher =
                Cipher.getInstance(TRANSFORMATION)
            cipher.init(
                Cipher.ENCRYPT_MODE,
                loadOrCreateKey(),
            )
            val iv = cipher.iv
            require(iv.size in 12..32) {
                "unexpected Exness vault GCM IV length"
            }
            cipher.updateAAD(AAD)
            val ciphertext =
                cipher.doFinal(clear)
            try {
                writeEnvelope(
                    iv = iv,
                    ciphertext = ciphertext,
                )
            } finally {
                ciphertext.fill(0)
            }
        } finally {
            clear.fill(0)
        }
    }

    @Synchronized
    fun loadOrNull():
        VN97ExnessCredentialPayload? {
        if (!target.exists()) {
            return null
        }
        require(
            target.isFile &&
                !Files.isSymbolicLink(target.toPath())
        ) {
            "Exness credential vault must be a regular non-symlink file"
        }
        val bytes = Files.readAllBytes(target.toPath())
        require(bytes.size in 1..MAX_ENVELOPE_BYTES) {
            "Exness credential vault exceeds byte bound"
        }
        val envelope =
            try {
                decodeEnvelope(bytes)
            } finally {
                bytes.fill(0)
            }
        try {
            val cipher =
                Cipher.getInstance(TRANSFORMATION)
            cipher.init(
                Cipher.DECRYPT_MODE,
                loadExistingKey(),
                GCMParameterSpec(
                    GCM_TAG_BITS,
                    envelope.iv,
                ),
            )
            cipher.updateAAD(AAD)
            val clear =
                cipher.doFinal(envelope.ciphertext)
            try {
                return VN97ExnessCredentialPayloadCodec
                    .decode(clear)
            } finally {
                clear.fill(0)
            }
        } finally {
            envelope.close()
        }
    }

    @Synchronized
    fun hasCredentials(): Boolean =
        target.exists() &&
            target.isFile &&
            !Files.isSymbolicLink(target.toPath())

    @Synchronized
    fun clear() {
        if (target.exists()) {
            require(
                target.isFile &&
                    !Files.isSymbolicLink(target.toPath())
            ) {
                "refusing to delete unexpected Exness vault target"
            }
            check(target.delete()) {
                "failed to delete Exness credential vault"
            }
        }
        val keyStore =
            KeyStore.getInstance(ANDROID_KEYSTORE)
                .apply { load(null) }
        if (keyStore.containsAlias(keyAlias)) {
            keyStore.deleteEntry(keyAlias)
        }
    }

    private fun writeEnvelope(
        iv: ByteArray,
        ciphertext: ByteArray,
    ) {
        ensureRoot()
        val tmp =
            File(
                root,
                VAULT_FILE + ".tmp-" +
                    System.nanoTime().toString(16),
            )
        require(
            !tmp.exists() &&
                !Files.isSymbolicLink(tmp.toPath())
        )
        val envelope =
            ByteArrayOutputStream().use { out ->
                DataOutputStream(out).use { data ->
                    data.writeUTF(OUTER_SCHEMA)
                    data.writeInt(iv.size)
                    data.write(iv)
                    data.writeInt(ciphertext.size)
                    data.write(ciphertext)
                }
                out.toByteArray()
            }
        try {
            require(envelope.size <= MAX_ENVELOPE_BYTES) {
                "Exness credential vault envelope exceeds byte bound"
            }
            Files.write(tmp.toPath(), envelope)
            try {
                Files.move(
                    tmp.toPath(),
                    target.toPath(),
                    StandardCopyOption.ATOMIC_MOVE,
                    StandardCopyOption.REPLACE_EXISTING,
                )
            } catch (_: AtomicMoveNotSupportedException) {
                Files.move(
                    tmp.toPath(),
                    target.toPath(),
                    StandardCopyOption.REPLACE_EXISTING,
                )
            }
        } finally {
            envelope.fill(0)
            runCatching { Files.deleteIfExists(tmp.toPath()) }
        }
    }

    private fun decodeEnvelope(
        bytes: ByteArray,
    ): VaultEnvelope {
        DataInputStream(
            ByteArrayInputStream(bytes)
        ).use { data ->
            require(data.readUTF() == OUTER_SCHEMA) {
                "unsupported Exness vault schema"
            }
            val ivLength = data.readInt()
            require(ivLength in 12..32) {
                "invalid Exness vault IV length"
            }
            val iv = ByteArray(ivLength)
            data.readFully(iv)
            val cipherLength = data.readInt()
            require(
                cipherLength in 16..MAX_ENVELOPE_BYTES
            ) {
                "invalid Exness vault ciphertext length"
            }
            val ciphertext =
                ByteArray(cipherLength)
            data.readFully(ciphertext)
            require(data.read() == -1) {
                "Exness vault contains trailing data"
            }
            return VaultEnvelope(
                iv = iv,
                ciphertext = ciphertext,
            )
        }
    }

    private fun ensureRoot() {
        if (!root.exists()) {
            check(root.mkdirs()) {
                "failed to create Exness credential vault directory"
            }
        }
        require(
            root.isDirectory &&
                !Files.isSymbolicLink(root.toPath())
        ) {
            "Exness credential vault root must be a real directory"
        }
    }

    private fun loadExistingKey():
        SecretKey {
        val keyStore =
            KeyStore.getInstance(ANDROID_KEYSTORE)
                .apply { load(null) }
        return keyStore.getKey(keyAlias, null)
            as? SecretKey
            ?: error(
                "Exness credential vault key is missing"
            )
    }

    private fun loadOrCreateKey():
        SecretKey {
        val keyStore =
            KeyStore.getInstance(ANDROID_KEYSTORE)
                .apply { load(null) }
        val existing =
            keyStore.getKey(keyAlias, null)
        if (existing != null) {
            return existing as? SecretKey
                ?: error(
                    "Exness credential alias is not a symmetric key"
                )
        }
        return KeyGenerator.getInstance(
            KeyProperties.KEY_ALGORITHM_AES,
            ANDROID_KEYSTORE,
        ).run {
            init(
                KeyGenParameterSpec.Builder(
                    keyAlias,
                    KeyProperties.PURPOSE_ENCRYPT or
                        KeyProperties.PURPOSE_DECRYPT,
                )
                    .setKeySize(256)
                    .setBlockModes(
                        KeyProperties.BLOCK_MODE_GCM
                    )
                    .setEncryptionPaddings(
                        KeyProperties.ENCRYPTION_PADDING_NONE
                    )
                    .setRandomizedEncryptionRequired(true)
                    .build()
            )
            generateKey()
        }
    }

    private data class VaultEnvelope(
        val iv: ByteArray,
        val ciphertext: ByteArray,
    ) : AutoCloseable {
        override fun close() {
            iv.fill(0)
            ciphertext.fill(0)
        }
    }

    companion object {
        private const val ROOT_DIR = "vn97-revenue"
        private const val VAULT_FILE =
            "exness-credentials.vn97vault1"
        private const val OUTER_SCHEMA =
            "VN97EXNVAULT1"
        private const val TRANSFORMATION =
            "AES/GCM/NoPadding"
        private const val ANDROID_KEYSTORE =
            "AndroidKeyStore"
        private const val GCM_TAG_BITS = 128
        private const val MAX_ENVELOPE_BYTES =
            64 * 1024

        private val AAD =
            "VN97EXNVAULT1|AES-256-GCM"
                .toByteArray(
                    java.nio.charset.StandardCharsets.UTF_8
                )
    }
}
