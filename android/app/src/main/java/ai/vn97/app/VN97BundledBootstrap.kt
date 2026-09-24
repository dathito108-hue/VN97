package ai.vn97.app

import java.io.ByteArrayOutputStream
import java.io.IOException

enum class VN97BundledBootstrapOutcome {
    ABSENT,
    ALREADY_ACTIVE,
    ACTIVATED,
}

/**
 * APK-bundled trust bootstrap.
 *
 * Shipping all three assets makes the APK signer/distributor the source of the
 * bootstrap bytes. The bytes still pass through the exact canonical
 * stage/trust/compatibility/activation pipeline.
 *
 * A turnkey release also ships VN97REL1. That manifest binds the APK
 * application/version identity to the exact public bootstrap identities and is
 * checked before any new bundled model is activated.
 */
class VN97BundledBootstrap(
    private val application: VN97Application,
    private val provisioner: VN97AppProvisioner,
) {
    fun activateIfPresent(
        required: Boolean = false,
    ): VN97BundledBootstrapOutcome {
        val releaseManifest =
            if (required) {
                readRequiredReleaseManifest()
                    .also {
                        it.requireBuildIdentity()
                    }
            } else {
                null
            }

        if (
            releaseManifest != null &&
            provisioner.currentModelActivation()
                ?.packageSha256 ==
                releaseManifest.modelSha256
        ) {
            return VN97BundledBootstrapOutcome
                .ALREADY_ACTIVE
        }

        if (provisioner.pendingReview() != null) {
            if (required) {
                throw IllegalStateException(
                    "turnkey VN97 bootstrap cannot activate while another provisioning review is pending"
                )
            }
            return VN97BundledBootstrapOutcome.ABSENT
        }

        val entries = try {
            application.assets
                .list(ASSET_ROOT)
                ?.toSet()
                .orEmpty()
        } catch (exc: IOException) {
            throw IllegalStateException(
                "bundled VN97 bootstrap assets could not be listed",
                exc,
            )
        }

        return when (
            VN97BootstrapAssetContract
                .classify(entries)
        ) {
            VN97BootstrapAssetState.ABSENT -> {
                if (required) {
                    throw IllegalStateException(
                        "turnkey VN97 build is missing its bundled signed model"
                    )
                }
                VN97BundledBootstrapOutcome.ABSENT
            }

            VN97BootstrapAssetState.INCOMPLETE ->
                throw IllegalStateException(
                    "bundled VN97 bootstrap is incomplete; package, signature, and publisher key must ship together"
                )

            VN97BootstrapAssetState.COMPLETE -> {
                val signature =
                    readBoundedAsset(
                        "model.vn97sig1",
                        maxBytes = 16 * 1024,
                    )
                val publisherKey =
                    readBoundedAsset(
                        "publisher.ed25519",
                        maxBytes = 256,
                    )
                releaseManifest
                    ?.requireSignatureBytes(
                        signature
                    )
                if (
                    releaseManifest != null &&
                    publisherKey.size.toLong() !=
                        releaseManifest
                            .publisherBytes
                ) {
                    throw IllegalStateException(
                        "turnkey release publisher size does not match manifest"
                    )
                }

                val review =
                    application.assets.open(
                        "$ASSET_ROOT/model.vn97cap1"
                    ).use { packageInput ->
                        provisioner.review(
                            packageInput =
                                packageInput,
                            signatureBytes =
                                signature,
                            publisherKeyBytes =
                                publisherKey,
                        )
                    }

                releaseManifest
                    ?.requireTrustedReview(
                        packageSha256 =
                            review.packageSha256,
                        publisherKeySha256 =
                            review.publisherKeySha256,
                    )

                provisioner.activateReviewed()
                VN97BundledBootstrapOutcome.ACTIVATED
            }
        }
    }

    private fun readRequiredReleaseManifest():
        VN97TurnkeyReleaseManifest {
        val bytes =
            readBoundedAssetAtPath(
                path =
                    "$RELEASE_ROOT/$RELEASE_MANIFEST",
                maxBytes =
                    VN97TurnkeyReleaseManifest
                        .MAX_MANIFEST_BYTES,
            )
        return VN97TurnkeyReleaseManifest
            .parse(bytes)
    }

    private fun readBoundedAsset(
        name: String,
        maxBytes: Int,
    ): ByteArray =
        readBoundedAssetAtPath(
            path = "$ASSET_ROOT/$name",
            maxBytes = maxBytes,
        )

    private fun readBoundedAssetAtPath(
        path: String,
        maxBytes: Int,
    ): ByteArray {
        return application.assets
            .open(path)
            .use { input ->
                val out =
                    ByteArrayOutputStream()
                val buffer = ByteArray(4096)
                var total = 0
                while (true) {
                    val count =
                        input.read(buffer)
                    if (count < 0) break
                    if (count == 0) {
                        throw IllegalStateException(
                            "bundled asset read made no progress: $path"
                        )
                    }
                    total =
                        Math.addExact(
                            total,
                            count,
                        )
                    if (total > maxBytes) {
                        throw IllegalArgumentException(
                            "bundled asset exceeds byte limit: $path"
                        )
                    }
                    out.write(
                        buffer,
                        0,
                        count,
                    )
                }
                if (total == 0) {
                    throw IllegalArgumentException(
                        "bundled asset is empty: $path"
                    )
                }
                out.toByteArray()
            }
    }

    companion object {
        private const val ASSET_ROOT =
            "vn97-bootstrap"
        private const val RELEASE_ROOT =
            "vn97-release"
        private const val RELEASE_MANIFEST =
            "release.vn97rel1"
    }
}
