package ai.vn97.platform

import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import java.io.DataInputStream
import java.io.DataOutputStream
import java.net.URI
import java.nio.charset.StandardCharsets

class VN97ExnessCredentialPayload internal constructor(
    val apiKey: String,
    val accountId: String,
    val baseUrl: String,
    private val privateKeySecret: ByteArray,
) : AutoCloseable {
    private var closed = false

    init {
        VN97ExnessCredentialPayloadCodec.validateMetadata(
            apiKey = apiKey,
            accountId = accountId,
            baseUrl = baseUrl,
        )
        require(
            privateKeySecret.size in 1..
                VN97ExnessCredentialPayloadCodec.MAX_SECRET_BYTES
        ) {
            "Exness private key secret is outside byte bound"
        }
    }

    @Synchronized
    fun <T> usePrivateKeySecret(
        block: (ByteArray) -> T,
    ): T {
        check(!closed) {
            "Exness credential payload is closed"
        }
        val copy = privateKeySecret.copyOf()
        try {
            return block(copy)
        } finally {
            copy.fill(0)
        }
    }

    @Synchronized
    override fun close() {
        if (!closed) {
            privateKeySecret.fill(0)
            closed = true
        }
    }

    override fun toString(): String =
        "VN97ExnessCredentialPayload(" +
            "apiKey=<redacted>, accountId=$accountId, " +
            "baseUrl=$baseUrl, privateKeySecret=<redacted>)"
}

object VN97ExnessCredentialPayloadCodec {
    internal const val MAX_SECRET_BYTES = 16 * 1024
    private const val MAX_API_KEY_BYTES = 512
    private const val MAX_ACCOUNT_ID_BYTES = 128
    private const val MAX_BASE_URL_BYTES = 2 * 1024
    private const val MAX_PAYLOAD_BYTES = 32 * 1024
    private const val SCHEMA = "VN97EXNCRED1"

    fun encode(
        apiKey: String,
        accountId: String,
        baseUrl: String,
        privateKeySecret: ByteArray,
    ): ByteArray {
        validateMetadata(apiKey, accountId, baseUrl)
        require(privateKeySecret.size in 1..MAX_SECRET_BYTES) {
            "Exness private key secret is outside byte bound"
        }
        val secretCopy = privateKeySecret.copyOf()
        return try {
            val out = ByteArrayOutputStream()
            DataOutputStream(out).use { data ->
                writeUtf8(data, SCHEMA)
                writeUtf8(data, apiKey)
                writeUtf8(data, accountId)
                writeUtf8(data, baseUrl)
                data.writeInt(secretCopy.size)
                data.write(secretCopy)
            }
            out.toByteArray().also {
                require(it.size <= MAX_PAYLOAD_BYTES) {
                    "Exness credential payload exceeds byte bound"
                }
            }
        } finally {
            secretCopy.fill(0)
        }
    }

    fun decode(
        encoded: ByteArray,
    ): VN97ExnessCredentialPayload {
        require(encoded.size in 1..MAX_PAYLOAD_BYTES) {
            "Exness credential payload is outside byte bound"
        }
        DataInputStream(ByteArrayInputStream(encoded)).use { data ->
            require(readUtf8(data, 64) == SCHEMA) {
                "unsupported Exness credential payload schema"
            }
            val apiKey = readUtf8(data, MAX_API_KEY_BYTES)
            val accountId =
                readUtf8(data, MAX_ACCOUNT_ID_BYTES)
            val baseUrl =
                readUtf8(data, MAX_BASE_URL_BYTES)
            val secretLength = data.readInt()
            require(secretLength in 1..MAX_SECRET_BYTES) {
                "Exness private key secret length is invalid"
            }
            val secret = ByteArray(secretLength)
            try {
                data.readFully(secret)
                require(data.read() == -1) {
                    "Exness credential payload has trailing data"
                }
                validateMetadata(
                    apiKey = apiKey,
                    accountId = accountId,
                    baseUrl = baseUrl,
                )
                return VN97ExnessCredentialPayload(
                    apiKey = apiKey,
                    accountId = accountId,
                    baseUrl = baseUrl,
                    privateKeySecret = secret.copyOf(),
                )
            } finally {
                secret.fill(0)
            }
        }
    }

    internal fun validateMetadata(
        apiKey: String,
        accountId: String,
        baseUrl: String,
    ) {
        requireBoundedPrintableAscii(
            value = apiKey,
            maxBytes = MAX_API_KEY_BYTES,
            label = "Exness API key",
        )
        require(apiKey.isNotBlank()) {
            "Exness API key must not be blank"
        }
        require(
            accountId.isNotBlank() &&
                accountId.length <= MAX_ACCOUNT_ID_BYTES &&
                accountId.all { it.isDigit() }
        ) {
            "Exness account ID must contain decimal digits only"
        }
        requireBoundedPrintableAscii(
            value = baseUrl,
            maxBytes = MAX_BASE_URL_BYTES,
            label = "Exness base URL",
        )
        val uri = URI(baseUrl)
        require(uri.scheme.equals("https", ignoreCase = true)) {
            "Exness base URL must use HTTPS"
        }
        require(!uri.host.isNullOrBlank()) {
            "Exness base URL host is missing"
        }
        require(
            uri.rawUserInfo == null &&
                uri.rawQuery == null &&
                uri.rawFragment == null &&
                (uri.rawPath.isNullOrEmpty() || uri.rawPath == "/")
        ) {
            "Exness base URL must not contain credentials, path, query or fragment"
        }
    }

    private fun writeUtf8(
        data: DataOutputStream,
        value: String,
    ) {
        val bytes =
            value.toByteArray(StandardCharsets.UTF_8)
        data.writeInt(bytes.size)
        data.write(bytes)
    }

    private fun readUtf8(
        data: DataInputStream,
        maxBytes: Int,
    ): String {
        val length = data.readInt()
        require(length in 0..maxBytes) {
            "Exness credential string length is invalid"
        }
        val bytes = ByteArray(length)
        data.readFully(bytes)
        return String(bytes, StandardCharsets.UTF_8)
    }

    private fun requireBoundedPrintableAscii(
        value: String,
        maxBytes: Int,
        label: String,
    ) {
        val bytes =
            value.toByteArray(StandardCharsets.UTF_8)
        require(bytes.size in 1..maxBytes) {
            "$label is missing or exceeds byte bound"
        }
        require(value.all { it.code in 0x21..0x7e }) {
            "$label must contain printable ASCII only"
        }
    }
}
