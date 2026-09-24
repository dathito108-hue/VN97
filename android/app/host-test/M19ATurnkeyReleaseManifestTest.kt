package ai.vn97.app

import java.nio.charset.StandardCharsets
import java.security.MessageDigest
import java.util.Base64

private inline fun expectM19AFailure(
    label: String,
    block: () -> Unit,
) {
    check(runCatching(block).isFailure) {
        "expected M19A failure: $label"
    }
}

private fun sha256(
    bytes: ByteArray,
): String =
    MessageDigest
        .getInstance("SHA-256")
        .digest(bytes)
        .joinToString("") {
            "%02x".format(
                it.toInt() and 0xff
            )
        }

private fun manifest(
    modelSha: String,
    signature: ByteArray,
    publisher: ByteArray,
    versionCode: Int =
        BuildConfig.VERSION_CODE,
    versionName: String =
        BuildConfig.VERSION_NAME,
): ByteArray {
    val versionB64 =
        Base64.getEncoder()
            .encodeToString(
                versionName.toByteArray(
                    StandardCharsets.UTF_8
                )
            )
    return buildString {
        append("VN97REL1\n")
        append(
            "application_id=" +
                BuildConfig.APPLICATION_ID +
                "\n"
        )
        append(
            "version_code=" +
                versionCode +
                "\n"
        )
        append(
            "version_name_b64=" +
                versionB64 +
                "\n"
        )
        append("model_bytes=1024\n")
        append(
            "model_sha256=" +
                modelSha +
                "\n"
        )
        append(
            "signature_bytes=" +
                signature.size +
                "\n"
        )
        append(
            "signature_sha256=" +
                sha256(signature) +
                "\n"
        )
        append(
            "publisher_bytes=" +
                publisher.size +
                "\n"
        )
        append(
            "publisher_sha256=" +
                sha256(publisher) +
                "\n"
        )
    }.toByteArray(StandardCharsets.UTF_8)
}

fun main() {
    val modelSha = "11".repeat(32)
    val signature =
        """{"schema":"VN97SIG1"}"""
            .toByteArray(
                StandardCharsets.UTF_8
            )
    val publisher =
        ByteArray(32) {
            index ->
            (index + 1).toByte()
        }

    val parsed =
        VN97TurnkeyReleaseManifest
            .parse(
                manifest(
                    modelSha,
                    signature,
                    publisher,
                )
            )
    parsed.requireBuildIdentity()
    parsed.requireSignatureBytes(
        signature
    )
    parsed.requireTrustedReview(
        packageSha256 = modelSha,
        publisherKeySha256 =
            sha256(publisher),
    )

    expectM19AFailure(
        "wrong package identity"
    ) {
        parsed.requireTrustedReview(
            packageSha256 =
                "22".repeat(32),
            publisherKeySha256 =
                sha256(publisher),
        )
    }

    expectM19AFailure(
        "wrong publisher identity"
    ) {
        parsed.requireTrustedReview(
            packageSha256 = modelSha,
            publisherKeySha256 =
                "33".repeat(32),
        )
    }

    expectM19AFailure(
        "changed signature"
    ) {
        parsed.requireSignatureBytes(
            signature + 0x00
        )
    }

    expectM19AFailure(
        "version code mismatch"
    ) {
        VN97TurnkeyReleaseManifest
            .parse(
                manifest(
                    modelSha,
                    signature,
                    publisher,
                    versionCode =
                        BuildConfig
                            .VERSION_CODE +
                            1,
                )
            )
            .requireBuildIdentity()
    }

    expectM19AFailure(
        "version name mismatch"
    ) {
        VN97TurnkeyReleaseManifest
            .parse(
                manifest(
                    modelSha,
                    signature,
                    publisher,
                    versionName =
                        "1.0.0-rc999",
                )
            )
            .requireBuildIdentity()
    }

    expectM19AFailure(
        "publisher size"
    ) {
        VN97TurnkeyReleaseManifest
            .parse(
                manifest(
                    modelSha,
                    signature,
                    ByteArray(31),
                )
            )
    }

    expectM19AFailure(
        "trailing data"
    ) {
        VN97TurnkeyReleaseManifest
            .parse(
                manifest(
                    modelSha,
                    signature,
                    publisher,
                ) +
                    "extra".toByteArray()
            )
    }

    expectM19AFailure(
        "invalid sha"
    ) {
        VN97TurnkeyReleaseManifest
            .parse(
                manifest(
                    "zz".repeat(32),
                    signature,
                    publisher,
                )
            )
    }

    expectM19AFailure(
        "malformed utf8"
    ) {
        val invalid =
            manifest(
                modelSha,
                signature,
                publisher,
            ).copyOf()
        invalid[0] = 0xc3.toByte()
        invalid[1] = 0x28.toByte()
        VN97TurnkeyReleaseManifest
            .parse(invalid)
    }

    expectM19AFailure(
        "oversized manifest"
    ) {
        VN97TurnkeyReleaseManifest
            .parse(
                ByteArray(
                    VN97TurnkeyReleaseManifest
                        .MAX_MANIFEST_BYTES +
                        1
                ) {
                    'a'.code.toByte()
                }
            )
    }

    println(
        "M19A turnkey release manifest contracts: PASS"
    )
}
