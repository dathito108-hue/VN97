package ai.vn97.platform

import java.nio.charset.StandardCharsets
import java.security.MessageDigest
import java.security.PrivateKey
import java.security.Signature
import java.util.Base64

data class VN97ExnessSignedRequest(
    val method: String,
    val pathWithQuery: String,
    val body: ByteArray,
    val headers: Map<String, String>,
) {
    init {
        require(method in ALLOWED_METHODS)
        require(pathWithQuery.startsWith("/"))
        require(headers.keys.containsAll(REQUIRED_HEADERS))
    }

    companion object {
        private val ALLOWED_METHODS =
            setOf("GET", "POST", "PUT", "PATCH", "DELETE")
        private val REQUIRED_HEADERS =
            setOf(
                "EXN-API-KEY",
                "EXN-IDEMPOTENCY-KEY",
                "EXN-TIMESTAMP",
                "EXN-SIGN-VERSION",
                "EXN-DATA",
                "EXN-SIGN",
            )
    }
}

object VN97ExnessApiSigner {
    private const val SIGN_VERSION = 1
    private const val MAX_API_KEY_BYTES = 256
    private const val MAX_IDEMPOTENCY_BYTES = 256
    private const val MAX_PATH_BYTES = 8 * 1024
    private const val MAX_BODY_BYTES = 256 * 1024

    fun sign(
        apiKey: String,
        privateKey: PrivateKey,
        method: String,
        pathWithQuery: String,
        body: ByteArray = ByteArray(0),
        idempotencyKey: String = "",
        timestampMillis: Long,
    ): VN97ExnessSignedRequest {
        val normalizedMethod = method.uppercase()
        require(
            normalizedMethod in
                setOf("GET", "POST", "PUT", "PATCH", "DELETE")
        ) {
            "unsupported Exness HTTP method"
        }
        require(method == normalizedMethod) {
            "Exness HTTP method must already be uppercase"
        }
        require(timestampMillis >= 0L) {
            "Exness timestamp must be non-negative"
        }
        requireBoundedAscii(
            apiKey,
            MAX_API_KEY_BYTES,
            "Exness API key",
        )
        require(apiKey.isNotBlank()) {
            "Exness API key must not be blank"
        }
        requireBoundedAscii(
            idempotencyKey,
            MAX_IDEMPOTENCY_BYTES,
            "Exness idempotency key",
        )
        validatePath(pathWithQuery)
        require(body.size <= MAX_BODY_BYTES) {
            "Exness request body exceeds byte bound"
        }

        if (normalizedMethod == "GET") {
            require(idempotencyKey.isEmpty()) {
                "signed Exness GET must use an empty idempotency key"
            }
        } else {
            require(idempotencyKey.isNotBlank()) {
                "mutating Exness request requires an idempotency key"
            }
        }

        val bodyHash =
            base64UrlNoPadding(
                MessageDigest.getInstance("SHA-256")
                    .digest(body)
            )
        val payload =
            buildString {
                append("{\"api_key\":")
                appendJsonString(apiKey)
                append(",\"idempotency_key\":")
                appendJsonString(idempotencyKey)
                append(",\"timestamp\":")
                append(timestampMillis)
                append(",\"sign_version\":")
                append(SIGN_VERSION)
                append(",\"method\":")
                appendJsonString(normalizedMethod)
                append(",\"path\":")
                appendJsonString(pathWithQuery)
                append(",\"body_hash\":")
                appendJsonString(bodyHash)
                append('}')
            }
        val payloadBytes =
            payload.toByteArray(StandardCharsets.UTF_8)
        val signatureBytes =
            Signature.getInstance("Ed25519").run {
                initSign(privateKey)
                update(payloadBytes)
                sign()
            }

        val headers =
            linkedMapOf(
                "EXN-API-KEY" to apiKey,
                "EXN-IDEMPOTENCY-KEY" to idempotencyKey,
                "EXN-TIMESTAMP" to timestampMillis.toString(),
                "EXN-SIGN-VERSION" to SIGN_VERSION.toString(),
                "EXN-DATA" to base64UrlNoPadding(payloadBytes),
                "EXN-SIGN" to base64UrlNoPadding(signatureBytes),
            )

        return VN97ExnessSignedRequest(
            method = normalizedMethod,
            pathWithQuery = pathWithQuery,
            body = body.copyOf(),
            headers = headers,
        )
    }

    private fun validatePath(
        value: String,
    ) {
        require(value.startsWith("/") && value.length > 1) {
            "Exness path must be absolute and non-empty"
        }
        requireBoundedAscii(
            value,
            MAX_PATH_BYTES,
            "Exness path",
        )
        require('#' !in value && ' ' !in value) {
            "Exness path contains forbidden characters"
        }
        val query = value.substringAfter('?', "")
        if (query.isNotEmpty()) {
            val keys =
                query.split('&')
                    .filter { it.isNotEmpty() }
                    .map { component ->
                        component.substringBefore('=')
                    }
            require(keys.all { it.isNotEmpty() }) {
                "Exness query contains empty parameter name"
            }
            require(keys.distinct().size == keys.size) {
                "Exness query contains duplicate parameters"
            }
        }
    }

    private fun requireBoundedAscii(
        value: String,
        maxBytes: Int,
        label: String,
    ) {
        val bytes =
            value.toByteArray(StandardCharsets.UTF_8)
        require(bytes.size <= maxBytes) {
            "$label exceeds byte bound"
        }
        require(
            value.all { it.code in 0x21..0x7e }
        ) {
            "$label must contain printable ASCII only"
        }
    }

    private fun base64UrlNoPadding(
        value: ByteArray,
    ): String =
        Base64.getUrlEncoder()
            .withoutPadding()
            .encodeToString(value)

    private fun appendJsonString(
        value: String,
    ): String =
        buildString {
            append('"')
            value.forEach { ch ->
                when (ch) {
                    '"' -> append("\\\"")
                    '\\' -> append("\\\\")
                    '\b' -> append("\\b")
                    12.toChar() -> append("\\f")
                    '\n' -> append("\\n")
                    '\r' -> append("\\r")
                    '\t' -> append("\\t")
                    else -> append(ch)
                }
            }
            append('"')
        }
}
