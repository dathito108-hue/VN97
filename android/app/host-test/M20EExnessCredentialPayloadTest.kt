package ai.vn97.platform

fun main() {
    val secret =
        ByteArray(32) { index ->
            (index + 1).toByte()
        }
    val encoded =
        VN97ExnessCredentialPayloadCodec.encode(
            apiKey = "exnsk_fixture",
            accountId = "123456",
            baseUrl = "https://api.exness.com",
            privateKeySecret = secret,
        )
    check(
        !encoded.contentEquals(secret)
    )

    VN97ExnessCredentialPayloadCodec
        .decode(encoded)
        .use { decoded ->
            check(decoded.apiKey == "exnsk_fixture")
            check(decoded.accountId == "123456")
            check(
                decoded.baseUrl ==
                    "https://api.exness.com"
            )
            decoded.usePrivateKeySecret { loaded ->
                check(loaded.contentEquals(secret))
            }
            check(
                "<redacted>" in decoded.toString()
            )
            check(
                "exnsk_fixture" !in decoded.toString()
            )
        }

    check(
        runCatching {
            VN97ExnessCredentialPayloadCodec.encode(
                apiKey = "exnsk_fixture",
                accountId = "not-numeric",
                baseUrl = "https://api.exness.com",
                privateKeySecret = secret,
            )
        }.isFailure
    )
    check(
        runCatching {
            VN97ExnessCredentialPayloadCodec.encode(
                apiKey = "exnsk_fixture",
                accountId = "123456",
                baseUrl =
                    "http://api.exness.com",
                privateKeySecret = secret,
            )
        }.isFailure
    )
    check(
        runCatching {
            VN97ExnessCredentialPayloadCodec.encode(
                apiKey = "exnsk_fixture",
                accountId = "123456",
                baseUrl =
                    "https://user:pass@api.exness.com/",
                privateKeySecret = secret,
            )
        }.isFailure
    )
    check(
        runCatching {
            VN97ExnessCredentialPayloadCodec.encode(
                apiKey = "exnsk_fixture",
                accountId = "123456",
                baseUrl =
                    "https://api.exness.com/path",
                privateKeySecret = secret,
            )
        }.isFailure
    )
    check(
        runCatching {
            VN97ExnessCredentialPayloadCodec.decode(
                encoded + byteArrayOf(1)
            )
        }.isFailure
    )

    secret.fill(0)
    encoded.fill(0)
    println("M20E Exness credential payload: PASS")
}
