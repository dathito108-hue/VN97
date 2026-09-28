package ai.vn97.platform

import java.nio.charset.StandardCharsets
import java.security.KeyFactory
import java.security.PrivateKey
import java.security.spec.PKCS8EncodedKeySpec
import java.util.Base64

object VN97ExnessPrivateKeyCodec {
    private const val MAX_SECRET_BYTES = 16 * 1024
    private val PEM_BEGIN =
        "-----BEGIN PRIVATE KEY-----"
    private val PEM_END =
        "-----END PRIVATE KEY-----"
    private val HEX_32 =
        Regex("^[0-9a-fA-F]{64}$")

    fun decode(
        secret: ByteArray,
    ): PrivateKey {
        require(secret.size in 1..MAX_SECRET_BYTES) {
            "Exness private key secret is outside byte bound"
        }
        val candidate =
            decodeCandidate(secret)
        try {
            val pkcs8 =
                if (candidate.size == 32) {
                    wrapRawEd25519Seed(candidate)
                } else {
                    candidate.copyOf()
                }
            try {
                val key =
                    KeyFactory.getInstance("Ed25519")
                        .generatePrivate(
                            PKCS8EncodedKeySpec(pkcs8)
                        )
                require(
                    key.algorithm.equals(
                        "Ed25519",
                        ignoreCase = true,
                    ) ||
                        key.algorithm.equals(
                            "EdDSA",
                            ignoreCase = true,
                        )
                ) {
                    "decoded Exness secret is not Ed25519"
                }
                return key
            } finally {
                pkcs8.fill(0)
            }
        } catch (exc: Exception) {
            throw IllegalArgumentException(
                "unsupported or invalid Exness Ed25519 private key format",
                exc,
            )
        } finally {
            candidate.fill(0)
        }
    }

    private fun decodeCandidate(
        secret: ByteArray,
    ): ByteArray {
        if (secret.size == 32) {
            return secret.copyOf()
        }
        val ascii =
            runCatching {
                String(
                    secret,
                    StandardCharsets.US_ASCII,
                ).trim()
            }.getOrNull()
                ?: return secret.copyOf()

        if (ascii.startsWith(PEM_BEGIN)) {
            require(ascii.endsWith(PEM_END)) {
                "incomplete Exness PKCS#8 PEM private key"
            }
            val body =
                ascii.removePrefix(PEM_BEGIN)
                    .removeSuffix(PEM_END)
                    .filterNot(Char::isWhitespace)
            require(body.isNotEmpty()) {
                "empty Exness PKCS#8 PEM private key"
            }
            return decodeBase64(body)
        }

        if (HEX_32.matches(ascii)) {
            return ByteArray(32) { index ->
                ascii.substring(
                    index * 2,
                    index * 2 + 2,
                ).toInt(16).toByte()
            }
        }

        if (
            ascii.isNotEmpty() &&
            ascii.all {
                it.code in 0x21..0x7e
            }
        ) {
            val decoded =
                runCatching {
                    decodeBase64(ascii)
                }.getOrNull()
            if (decoded != null) {
                return decoded
            }
        }

        return secret.copyOf()
    }

    private fun decodeBase64(
        value: String,
    ): ByteArray {
        val standard =
            runCatching {
                Base64.getDecoder().decode(value)
            }.getOrNull()
        if (standard != null) {
            return standard
        }
        return Base64.getUrlDecoder()
            .decode(value)
    }

    private fun wrapRawEd25519Seed(
        seed: ByteArray,
    ): ByteArray {
        require(seed.size == 32)
        val prefix =
            byteArrayOf(
                0x30, 0x2e,
                0x02, 0x01, 0x00,
                0x30, 0x05,
                0x06, 0x03, 0x2b, 0x65, 0x70,
                0x04, 0x22,
                0x04, 0x20,
            )
        return ByteArray(
            prefix.size + seed.size
        ).also { out ->
            prefix.copyInto(out)
            seed.copyInto(
                destination = out,
                destinationOffset = prefix.size,
            )
        }
    }
}
