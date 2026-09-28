package ai.vn97.platform

import java.nio.charset.StandardCharsets
import java.security.KeyPairGenerator
import java.security.Signature
import java.util.Base64

fun main() {
    val pair =
        KeyPairGenerator.getInstance("Ed25519")
            .generateKeyPair()

    val get =
        VN97ExnessApiSigner.sign(
            apiKey = "exnsk_fixture",
            privateKey = pair.private,
            method = "GET",
            pathWithQuery =
                "/v1/configuration/accounts/123456/account",
            timestampMillis = 1773656070123L,
        )
    check(get.headers["EXN-IDEMPOTENCY-KEY"] == "")
    check(get.headers["EXN-SIGN-VERSION"] == "1")

    val payloadBytes =
        Base64.getUrlDecoder()
            .decode(get.headers.getValue("EXN-DATA"))
    val payload =
        String(payloadBytes, StandardCharsets.UTF_8)
    check(
        "\"body_hash\":\"47DEQpj8HBSa-_TImW-5JCeuQeRkm5NMpJWZG3hSuFU\"" in
            payload
    )
    val signature =
        Base64.getUrlDecoder()
            .decode(get.headers.getValue("EXN-SIGN"))
    val verified =
        Signature.getInstance("Ed25519").run {
            initVerify(pair.public)
            update(payloadBytes)
            verify(signature)
        }
    check(verified)

    val body =
        "{\"instrument\":\"XAUUSD\",\"side\":\"buy\",\"volume\":\"0.01\"}"
            .toByteArray(StandardCharsets.UTF_8)
    val post =
        VN97ExnessApiSigner.sign(
            apiKey = "exnsk_fixture",
            privateKey = pair.private,
            method = "POST",
            pathWithQuery =
                "/v1/trading/accounts/123456/positions",
            body = body,
            idempotencyKey =
                "vn97-00000000000000000001",
            timestampMillis = 1773656071123L,
        )
    check(
        post.headers.getValue("EXN-IDEMPOTENCY-KEY") ==
            "vn97-00000000000000000001"
    )
    check(post.body.contentEquals(body))

    check(
        runCatching {
            VN97ExnessApiSigner.sign(
                apiKey = "exnsk_fixture",
                privateKey = pair.private,
                method = "GET",
                pathWithQuery = "/v1/test",
                idempotencyKey = "not-empty",
                timestampMillis = 1L,
            )
        }.isFailure
    )
    check(
        runCatching {
            VN97ExnessApiSigner.sign(
                apiKey = "exnsk_fixture",
                privateKey = pair.private,
                method = "POST",
                pathWithQuery = "/v1/test",
                timestampMillis = 1L,
            )
        }.isFailure
    )
    check(
        runCatching {
            VN97ExnessApiSigner.sign(
                apiKey = "exnsk_fixture",
                privateKey = pair.private,
                method = "GET",
                pathWithQuery = "/v1/test?a=1&a=2",
                timestampMillis = 1L,
            )
        }.isFailure
    )

    println("M20D Exness signed request core: PASS")
}
