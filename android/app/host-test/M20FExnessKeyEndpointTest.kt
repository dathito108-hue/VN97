package ai.vn97.platform

import java.security.KeyPairGenerator
import java.util.Base64

fun main() {
    val pair =
        KeyPairGenerator.getInstance("Ed25519")
            .generateKeyPair()
    val encoded = pair.private.encoded

    val direct =
        VN97ExnessPrivateKeyCodec.decode(encoded)
    check(
        direct.algorithm.equals(
            "EdDSA",
            ignoreCase = true,
        ) ||
            direct.algorithm.equals(
                "Ed25519",
                ignoreCase = true,
            )
    )

    val pem =
        "-----BEGIN PRIVATE KEY-----\n" +
            Base64.getEncoder()
                .encodeToString(encoded) +
            "\n-----END PRIVATE KEY-----"
    val fromPem =
        VN97ExnessPrivateKeyCodec.decode(
            pem.toByteArray(Charsets.US_ASCII)
        )
    check(fromPem.format == "PKCS#8")

    val fromBase64 =
        VN97ExnessPrivateKeyCodec.decode(
            Base64.getUrlEncoder()
                .withoutPadding()
                .encode(encoded)
        )
    check(fromBase64.format == "PKCS#8")

    val seed =
        ByteArray(32) { index ->
            (index + 7).toByte()
        }
    val fromSeed =
        VN97ExnessPrivateKeyCodec.decode(seed)
    check(fromSeed.format == "PKCS#8")
    seed.fill(0)

    check(
        VN97ExnessEndpointPolicy
            .normalizeBaseUrl(
                "api.exness.com"
            ) ==
            "https://api.exness.com"
    )
    check(
        VN97ExnessEndpointPolicy
            .normalizeBaseUrl(
                "https://ap-qwerty123.trading.exness.com"
            ) ==
            "https://ap-qwerty123.trading.exness.com"
    )
    check(
        VN97ExnessEndpointPolicy
            .accessPointPath("123456") ==
            "/v1/trading/access-point?account_id=123456"
    )
    check(
        VN97ExnessEndpointPolicy
            .accountDetailsPath("123456") ==
            "/v1/configuration/accounts/123456/account"
    )
    check(
        runCatching {
            VN97ExnessEndpointPolicy
                .normalizeBaseUrl(
                    "https://api.exness.com.evil.test"
                )
        }.isFailure
    )
    check(
        runCatching {
            VN97ExnessEndpointPolicy
                .normalizeBaseUrl(
                    "http://api.exness.com"
                )
        }.isFailure
    )

    encoded.fill(0)
    println("M20F Exness key/endpoint contracts: PASS")
}
