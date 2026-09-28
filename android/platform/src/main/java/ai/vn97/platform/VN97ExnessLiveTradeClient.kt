package ai.vn97.platform

import java.io.ByteArrayOutputStream
import java.net.URL
import java.nio.charset.StandardCharsets
import java.security.PrivateKey
import javax.net.ssl.HttpsURLConnection
import org.json.JSONObject

class VN97ExnessSubmissionAmbiguousException(
    val idempotencyKey: String,
    cause: Throwable,
) : IllegalStateException(
    "Exness submission outcome is ambiguous; reconcile before any retry",
    cause,
)

class VN97ExnessLiveTradeClient(
    private val apiKey: String,
    private val privateKey: PrivateKey,
    discoveryBaseUrl: String,
    private val accountId: String,
    private val connectTimeoutMs: Int = 5_000,
    private val readTimeoutMs: Int = 5_000,
    private val maxResponseBytes: Int = 256 * 1024,
    private val operationPollMillis: Long = 1_000L,
    private val operationTimeoutMillis: Long = 30_000L,
) {
    private val readOnly =
        VN97ExnessReadOnlyClient(
            apiKey = apiKey,
            privateKey = privateKey,
            discoveryBaseUrl =
                discoveryBaseUrl,
            accountId = accountId,
            connectTimeoutMs =
                connectTimeoutMs,
            readTimeoutMs =
                readTimeoutMs,
            maxResponseBytes =
                maxResponseBytes,
        )

    init {
        require(
            operationPollMillis in
                250L..5_000L
        )
        require(
            operationTimeoutMillis in
                1_000L..120_000L
        )
    }

    fun openPositionAndConfirm(
        request: VN97RevenueOpenPositionRequest,
    ): VN97RevenueTradeResult {
        require(request.accountId == accountId) {
            "Exness live request account mismatch"
        }
        val accessPoint =
            readOnly.discoverAccessPoint()
        val path =
            "/v1/trading/accounts/" +
                accountId +
                "/positions"
        val body =
            JSONObject()
                .put(
                    "instrument",
                    request.instrument,
                )
                .put(
                    "side",
                    request.side.wireValue,
                )
                .put(
                    "volume",
                    request.volume,
                )
                .toString()
                .toByteArray(
                    StandardCharsets.UTF_8
                )
        val ack =
            try {
                requestJson(
                    baseUrl = accessPoint,
                    method = "POST",
                    path = path,
                    body = body,
                    idempotencyKey =
                        request.idempotencyKey,
                )
            } catch (
                exc: VN97ExnessHttpException
            ) {
                throw exc
            } catch (exc: Throwable) {
                throw VN97ExnessSubmissionAmbiguousException(
                    idempotencyKey =
                        request.idempotencyKey,
                    cause = exc,
                )
            } finally {
                body.fill(0)
            }

        val operationId =
            ack.optString(
                "operation_id",
                "",
            )
        require(
            operationId.matches(
                IDENTIFIER
            )
        ) {
            "Exness ACK is missing a valid operation_id"
        }
        val clientRequestId =
            ack.optString(
                "client_request_id",
                "",
            ).ifBlank { null }
        if (clientRequestId != null) {
            require(
                clientRequestId ==
                    request.idempotencyKey
            ) {
                "Exness ACK client_request_id mismatch"
            }
        }

        val deadline =
            System.nanoTime() +
                operationTimeoutMillis *
                    1_000_000L
        while (true) {
            val operation =
                requestJson(
                    baseUrl = accessPoint,
                    method = "GET",
                    path =
                        "/v1/trading/accounts/" +
                            accountId +
                            "/operations/" +
                            operationId,
                )
            val status =
                operation.optString(
                    "status",
                    "",
                ).lowercase()
            when (status) {
                "confirmed" ->
                    return terminalResult(
                        operationId,
                        clientRequestId,
                        VN97RevenueTradeTerminalStatus
                            .CONFIRMED,
                        operation,
                    )
                "rejected" ->
                    return terminalResult(
                        operationId,
                        clientRequestId,
                        VN97RevenueTradeTerminalStatus
                            .REJECTED,
                        operation,
                    )
                "failed" ->
                    return terminalResult(
                        operationId,
                        clientRequestId,
                        VN97RevenueTradeTerminalStatus
                            .FAILED,
                        operation,
                    )
            }
            check(
                System.nanoTime() < deadline
            ) {
                "Exness operation confirmation timed out; do not resubmit automatically"
            }
            Thread.sleep(operationPollMillis)
        }
    }

    private fun terminalResult(
        operationId: String,
        clientRequestId: String?,
        status: VN97RevenueTradeTerminalStatus,
        json: JSONObject,
    ) = VN97RevenueTradeResult(
        operationId = operationId,
        clientRequestId = clientRequestId,
        status = status,
        errorCode =
            json.opt("error_code")
                ?.toString()
                ?.take(128),
    )

    private fun requestJson(
        baseUrl: String,
        method: String,
        path: String,
        body: ByteArray = ByteArray(0),
        idempotencyKey: String = "",
    ): JSONObject {
        val normalized =
            VN97ExnessEndpointPolicy
                .normalizeBaseUrl(baseUrl)
        val signed =
            VN97ExnessApiSigner.sign(
                apiKey = apiKey,
                privateKey = privateKey,
                method = method,
                pathWithQuery = path,
                body = body,
                idempotencyKey =
                    idempotencyKey,
                timestampMillis =
                    System.currentTimeMillis(),
            )
        val connection =
            URL(normalized + path)
                .openConnection()
                as? HttpsURLConnection
                ?: error(
                    "Exness endpoint did not create HTTPS connection"
                )
        try {
            connection.requestMethod = method
            connection.instanceFollowRedirects =
                false
            connection.connectTimeout =
                connectTimeoutMs
            connection.readTimeout =
                readTimeoutMs
            connection.useCaches = false
            connection.setRequestProperty(
                "Accept",
                "application/json",
            )
            if (method != "GET") {
                connection.doOutput = true
                connection.setRequestProperty(
                    "Content-Type",
                    "application/json",
                )
            }
            signed.headers.forEach {
                    (name, value) ->
                connection.setRequestProperty(
                    name,
                    value,
                )
            }
            if (method != "GET") {
                connection.outputStream.use {
                    it.write(body)
                    it.flush()
                }
            }

            val status =
                connection.responseCode
            val response =
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
                                response.toString(
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
                    response.toString(
                        Charsets.UTF_8
                    )
                )
            } finally {
                response.fill(0)
            }
        } finally {
            connection.disconnect()
        }
    }

    private fun readBounded(
        input: java.io.InputStream,
    ): ByteArray =
        input.use { stream ->
            val out =
                ByteArrayOutputStream()
            val buffer =
                ByteArray(8 * 1024)
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
                    out.write(
                        buffer,
                        0,
                        read,
                    )
                }
                out.toByteArray()
            } finally {
                buffer.fill(0)
            }
        }

    companion object {
        private val IDENTIFIER =
            Regex(
                "^[A-Za-z0-9._:-]{1,256}$"
            )
    }
}
