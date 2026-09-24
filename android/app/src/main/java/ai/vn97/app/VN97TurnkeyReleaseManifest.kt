package ai.vn97.app

import java.nio.charset.CodingErrorAction
import java.nio.charset.StandardCharsets
import java.security.MessageDigest
import java.util.Base64

internal data class VN97TurnkeyReleaseManifest(
    val applicationId: String,
    val versionCode: Int,
    val versionName: String,
    val modelBytes: Long,
    val modelSha256: String,
    val signatureBytes: Long,
    val signatureSha256: String,
    val publisherBytes: Long,
    val publisherSha256: String,
) {
    init {
        require(applicationId == "ai.vn97.app") {
            "turnkey release application id mismatch"
        }
        require(versionCode > 0)
        require(versionName.isNotBlank())
        require(modelBytes in MIN_MODEL_BYTES..MAX_MODEL_BYTES)
        require(signatureBytes in 1L..MAX_SIGNATURE_BYTES)
        require(publisherBytes == PUBLISHER_BYTES)
        requireSha(modelSha256, "model")
        requireSha(signatureSha256, "signature")
        requireSha(publisherSha256, "publisher")
    }

    fun requireBuildIdentity() {
        check(applicationId == BuildConfig.APPLICATION_ID) {
            "turnkey release manifest application id does not match APK"
        }
        check(versionCode == BuildConfig.VERSION_CODE) {
            "turnkey release manifest version code does not match APK"
        }
        check(versionName == BuildConfig.VERSION_NAME) {
            "turnkey release manifest version name does not match APK"
        }
    }

    fun requireAssetSizes(
        modelBytes: Long,
        signatureBytes: Long,
        publisherBytes: Long,
    ) {
        check(this.modelBytes == modelBytes) {
            "turnkey release model size does not match manifest"
        }
        check(this.signatureBytes == signatureBytes) {
            "turnkey release signature size does not match manifest"
        }
        check(this.publisherBytes == publisherBytes) {
            "turnkey release publisher size does not match manifest"
        }
    }

    fun requireTrustedReview(
        packageSha256: String,
        publisherKeySha256: String,
    ) {
        check(modelSha256 == packageSha256) {
            "turnkey release package identity does not match manifest"
        }
        check(publisherSha256 == publisherKeySha256) {
            "turnkey release publisher identity does not match manifest"
        }
    }

    fun requireSignatureBytes(
        bytes: ByteArray,
    ) {
        check(bytes.size.toLong() == signatureBytes) {
            "turnkey release signature size changed"
        }
        check(sha256(bytes) == signatureSha256) {
            "turnkey release signature identity does not match manifest"
        }
    }

    companion object {
        private const val SCHEMA = "VN97REL1"
        private const val MIN_MODEL_BYTES = 96L
        private const val MAX_MODEL_BYTES =
            512L * 1024L * 1024L
        private const val MAX_SIGNATURE_BYTES =
            16L * 1024L
        private const val PUBLISHER_BYTES = 32L
        const val MAX_MANIFEST_BYTES = 4096

        fun parse(
            bytes: ByteArray,
        ): VN97TurnkeyReleaseManifest {
            require(bytes.isNotEmpty()) {
                "turnkey release manifest is empty"
            }
            require(bytes.size <= MAX_MANIFEST_BYTES) {
                "turnkey release manifest exceeds byte bound"
            }
            val text =
                StandardCharsets.UTF_8
                    .newDecoder()
                    .onMalformedInput(
                        CodingErrorAction.REPORT
                    )
                    .onUnmappableCharacter(
                        CodingErrorAction.REPORT
                    )
                    .decode(
                        java.nio.ByteBuffer.wrap(
                            bytes
                        )
                    )
                    .toString()
            require(text.endsWith("\n")) {
                "turnkey release manifest must end with newline"
            }
            val lines = text.split('\n')
            require(lines.size == 11 && lines.last().isEmpty()) {
                "turnkey release manifest line count is invalid"
            }
            require(lines[0] == SCHEMA) {
                "turnkey release manifest schema mismatch"
            }
            val values =
                lines.subList(1, 10)
                    .associate { line ->
                        val index =
                            line.indexOf('=')
                        require(index > 0) {
                            "turnkey release manifest field is malformed"
                        }
                        line.substring(0, index) to
                            line.substring(index + 1)
                    }
            require(
                values.keys ==
                    setOf(
                        "application_id",
                        "version_code",
                        "version_name_b64",
                        "model_bytes",
                        "model_sha256",
                        "signature_bytes",
                        "signature_sha256",
                        "publisher_bytes",
                        "publisher_sha256",
                    )
            ) {
                "turnkey release manifest fields mismatch"
            }
            val versionNameBytes =
                Base64.getDecoder()
                    .decode(
                        checkNotNull(
                            values[
                                "version_name_b64"
                            ]
                        )
                    )
            val versionName =
                StandardCharsets.UTF_8
                    .newDecoder()
                    .onMalformedInput(
                        CodingErrorAction.REPORT
                    )
                    .onUnmappableCharacter(
                        CodingErrorAction.REPORT
                    )
                    .decode(
                        java.nio.ByteBuffer.wrap(
                            versionNameBytes
                        )
                    )
                    .toString()
            return VN97TurnkeyReleaseManifest(
                applicationId =
                    checkNotNull(
                        values["application_id"]
                    ),
                versionCode =
                    checkNotNull(
                        values["version_code"]
                    ).toInt(),
                versionName = versionName,
                modelBytes =
                    checkNotNull(
                        values["model_bytes"]
                    ).toLong(),
                modelSha256 =
                    checkNotNull(
                        values["model_sha256"]
                    ),
                signatureBytes =
                    checkNotNull(
                        values["signature_bytes"]
                    ).toLong(),
                signatureSha256 =
                    checkNotNull(
                        values["signature_sha256"]
                    ),
                publisherBytes =
                    checkNotNull(
                        values["publisher_bytes"]
                    ).toLong(),
                publisherSha256 =
                    checkNotNull(
                        values["publisher_sha256"]
                    ),
            )
        }

        private fun requireSha(
            value: String,
            label: String,
        ) {
            require(
                value.length == 64 &&
                    value.all {
                        it in "0123456789abcdef"
                    }
            ) {
                "turnkey release $label SHA-256 is invalid"
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
    }
}
