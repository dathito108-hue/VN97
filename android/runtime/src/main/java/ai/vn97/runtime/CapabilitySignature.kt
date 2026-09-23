package ai.vn97.runtime

import java.nio.ByteBuffer
import java.nio.charset.CodingErrorAction
import java.nio.charset.StandardCharsets
import java.security.MessageDigest

const val VN97_SIGNATURE_MAX_BYTES = 16 * 1024

class VN97CapabilitySignatureException(
    message: String,
    cause: Throwable? = null,
) : IllegalStateException(message, cause)

data class VN97CapabilitySignatureEnvelope(
    val keyId: String,
    val packageSha256: String,
    val capabilityId: String,
    val capabilityVersion: Long,
    val signature: ByteArray,
    val envelopeSha256: String,
) {
    init {
        require(signature.size == 64) {
            "signature must be exactly 64 bytes"
        }
    }

    fun signingMessage(): ByteArray {
        val claims = VnStrictJson.objectOf(
            "algorithm" to VnStrictJson.string("ed25519"),
            "capability_id" to VnStrictJson.string(capabilityId),
            "capability_version" to VnJsonNumber(capabilityVersion.toString()),
            "key_id" to VnStrictJson.string(keyId),
            "package_sha256" to VnStrictJson.string(packageSha256),
            "schema" to VnStrictJson.string("VN97SIG1"),
        )
        return "VN97CAP1-SIGNATURE-V1\u0000"
            .toByteArray(StandardCharsets.UTF_8) +
            VnStrictJson.canonical(claims).toByteArray(StandardCharsets.UTF_8)
    }
}

object VN97CapabilitySignatureParser {
    fun parse(bytes: ByteArray): VN97CapabilitySignatureEnvelope {
        if (bytes.isEmpty() || bytes.size > VN97_SIGNATURE_MAX_BYTES) {
            fail("VN97SIG1 envelope size is invalid")
        }
        val text = strictUtf8(bytes)
        val root = try {
            VnStrictJson.parseObject(
                text,
                VnJsonLimits(
                    maxInputUtf8Bytes = VN97_SIGNATURE_MAX_BYTES,
                    maxDepth = 8,
                    maxNodes = 64,
                    maxStringUtf8Bytes = 4096,
                ),
            )
        } catch (exc: RuntimeException) {
            throw VN97CapabilitySignatureException(
                "VN97SIG1 is not strict JSON",
                exc,
            )
        }
        if (VnStrictJson.canonical(root) != text) {
            fail("VN97SIG1 must be canonical JSON")
        }
        if (root.values.keys != setOf(
                "schema",
                "algorithm",
                "key_id",
                "package_sha256",
                "capability_id",
                "capability_version",
                "signature",
            )
        ) {
            fail("VN97SIG1 schema/keys mismatch")
        }
        if (root.string("schema") != "VN97SIG1" ||
            root.string("algorithm") != "ed25519"
        ) {
            fail("VN97SIG1 schema/algorithm mismatch")
        }
        val keyId = requireId(root.string("key_id"), "key_id")
        val packageSha = requireSha(root.string("package_sha256"), "package_sha256")
        val capabilityId = requireId(
            root.string("capability_id"),
            "capability_id",
        )
        val version = root.unsigned32("capability_version")
        val signatureHex = root.string("signature")
        if (signatureHex.length != 128 ||
            signatureHex.any { it !in "0123456789abcdef" }
        ) {
            fail("VN97SIG1 signature encoding is invalid")
        }

        return VN97CapabilitySignatureEnvelope(
            keyId = keyId,
            packageSha256 = packageSha,
            capabilityId = capabilityId,
            capabilityVersion = version,
            signature = signatureHex.hexToBytes(),
            envelopeSha256 = MessageDigest.getInstance("SHA-256")
                .digest(bytes)
                .toHex(),
        )
    }

    private fun requireId(value: String, label: String): String {
        if (value.isEmpty() ||
            value.length > 128 ||
            value[0] !in 'a'..'z' ||
            value.any {
                it !in 'a'..'z' &&
                    it !in '0'..'9' &&
                    it != '.' && it != '_' && it != '-'
            }
        ) {
            fail("VN97SIG1 $label is invalid")
        }
        return value
    }

    private fun requireSha(value: String, label: String): String {
        if (value.length != 64 ||
            value.any { it !in "0123456789abcdef" }
        ) {
            fail("VN97SIG1 $label is invalid")
        }
        return value
    }

    private fun VnJsonObject.string(key: String): String =
        (values[key] as? VnJsonString)?.value
            ?: fail("VN97SIG1 $key must be string")

    private fun VnJsonObject.unsigned32(key: String): Long {
        val raw = (values[key] as? VnJsonNumber)?.canonical
            ?: fail("VN97SIG1 $key must be integer")
        if (raw.any { it == '.' || it == 'e' || it == 'E' }) {
            fail("VN97SIG1 $key must be integer")
        }
        val value = raw.toLongOrNull()
            ?: fail("VN97SIG1 $key is outside Long range")
        if (value !in 1L..0xffff_ffffL) {
            fail("VN97SIG1 $key exceeds unsigned 32-bit range")
        }
        return value
    }

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
        throw VN97CapabilitySignatureException(
            "VN97SIG1 is not strict UTF-8",
            exc,
        )
    }

    private fun fail(message: String): Nothing =
        throw VN97CapabilitySignatureException(message)
}
