package ai.vn97.platform

import java.io.ByteArrayOutputStream
import java.net.URL
import java.security.PrivateKey
import javax.net.ssl.HttpsURLConnection
import org.json.JSONObject

data class VN97ExnessReadOnlyValidation(
    val accountId: String,
    val accessPointBaseUrl: String,
    val instrumentCount: Int,
) {
    init {
        require(accountId.isNotEmpty())
        require(instrumentCount >= 0)
    }
}

class VN97ExnessHttpException(
    val httpStatus: Int,
    val apiErrorCode: String?,
) : IllegalStateException(
    "Exness API request failed with HTTP $httpStatus" +
        if (apiErrorCode == null) {
            ""
        } else {
            " / code $apiErrorCode"
        }
)

class VN97ExnessReadOnlyClient(
    private val apiKey: String,
    private val privateKey: PrivateKey,
    discoveryBaseUrl: String,
    private val accountId: String,
    private val connectTimeoutMs: Int = 5_000,
    private val readTimeoutMs: Int = 5_000,
    private val maxResponseBytes: Int = 256 * 1024,
) {
    private val discoveryBaseUrl =
        VN97ExnessEndpointPolicy.normalizeBaseUrl(
            discoveryBaseUrl
        )

    init {
        require(apiKey.isNotBlank())
        VN97ExnessEndpointPolicy.accessPointPath(
            accountId
        )
        require(connectTimeoutMs in 500..30_000)
        require(readTimeoutMs in 500..30_000)
        require(
            maxResponseBytes in 1_024..1_048_576
        )
    }

    fun validate():
        VN97ExnessReadOnlyValidation {
        val accessPoint =
            discoverAccessPoint()
        val account =
            getJson(
                baseUrl = accessPoint,
                path =
                    VN97ExnessEndpointPolicy
                        .accountDetailsPath(accountId),
            )
        account.opt("account_id")?.let {
            require(it.toString() == accountId) {
                "Exness account response identity mismatch"
            }
        }

        val instruments =
            getJson(
                baseUrl = accessPoint,
                path =
                    VN97ExnessEndpointPolicy
                        .instrumentListPath(accountId),
            )
        val array =
            instruments.optJSONArray("instruments")
                ?: throw IllegalStateException(
                    "Exness instrument response is missing instruments"
                )
        return VN97ExnessReadOnlyValidation(
            accountId = accountId,
            accessPointBaseUrl = accessPoint,
            instrumentCount = array.length(),
        )
    }

    fun discoverAccessPoint(): String {
        val response =
            getJson(
                baseUrl = discoveryBaseUrl,
                path =
                    VN97ExnessEndpointPolicy
                        .accessPointPath(accountId),
            )
        val returnedAccount =
            response.opt("account_id")
        if (returnedAccount != null) {
            require(
                returnedAccount.toString() ==
                    accountId
            ) {
                "Exness access-point response identity mismatch"
            }
        }
        val accessPoint =
            response.optString(
                "access_point",
                "",
            )
        require(accessPoint.isNotBlank()) {
            "Exness access-point response is missing access_point"
        }
        return VN97ExnessEndpointPolicy
            .normalizeBaseUrl(accessPoint)
    }

    private fun getJson(
        baseUrl: String,
        path: String,
    ): JSONObject {
        val normalized =
            VN97ExnessEndpointPolicy
                .normalizeBaseUrl(baseUrl)
        val signed =
            VN97ExnessApiSigner.sign(
                apiKey = apiKey,
                privateKey = privateKey,
                method = "GET",
                pathWithQuery = path,
                timestampMillis =
                    System.currentTimeMillis(),
            )
        val url = URL(normalized + path)
        val connection =
            url.openConnection()
                as? HttpsURLConnection
                ?: error(
                    "Exness endpoint did not create HTTPS connection"
                )
        try {
            connection.requestMethod = "GET"
            connection.instanceFollowRedirects = false
            connection.connectTimeout =
                connectTimeoutMs
            connection.readTimeout =
                readTimeoutMs
            connection.useCaches = false
            connection.setRequestProperty(
                "Accept",
                "application/json",
            )
            signed.headers.forEach {
                    (name, value) ->
                connection.setRequestProperty(
                    name,
                    value,
                )
            }
            val status = connection.responseCode
            val body =
                readBounded(
                    if (status in 200..299) {
                        connection.inputStream
                    } else {
                        connection.errorStream
                            ?: connection.inputStream
                    }
                )
            try {
                if (status !in 200..299) {
                    val code =
                        runCatching {
                            JSONObject(
                                body.toString(
                                    Charsets.UTF_8
                                )
                            ).opt("error_code")
                                ?.toString()
                        }.getOrNull()
                    throw VN97ExnessHttpException(
                        httpStatus = status,
                        apiErrorCode = code,
                    )
                }
                return JSONObject(
                    body.toString(Charsets.UTF_8)
                )
            } finally {
                body.fill(0)
            }
        } finally {
            connection.disconnect()
        }
    }

    private fun readBounded(
        input: java.io.InputStream,
    ): ByteArray =
        input.use { stream ->
            val out = ByteArrayOutputStream()
            val buffer = ByteArray(8 * 1024)
            try {
                while (true) {
                    val read =
                        stream.read(buffer)
                    if (read < 0) break
                    require(
                        out.size() + read <=
                            maxResponseBytes
                    ) {
                        "Exness API response exceeds byte bound"
                    }
                    out.write(buffer, 0, read)
                }
                out.toByteArray()
            } finally {
                buffer.fill(0)
            }
        }
}
