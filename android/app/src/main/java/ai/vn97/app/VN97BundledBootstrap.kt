package ai.vn97.app

import java.io.IOException

enum class VN97BundledBootstrapOutcome {
    ABSENT,
    ACTIVATED,
}

/**
 * Optional APK-bundled trust bootstrap.
 *
 * Shipping all three assets makes the APK signer/distributor the source of the bootstrap bytes.
 * The bytes still pass through the exact M10D -> M10E -> M10F -> M10G trust/activation pipeline.
 * Missing assets do not manufacture a model or a READY state.
 */
class VN97BundledBootstrap(
    private val application: VN97Application,
    private val provisioner: VN97AppProvisioner,
) {
    fun activateIfPresent(): VN97BundledBootstrapOutcome {
        if (provisioner.pendingReview() != null) {
            return VN97BundledBootstrapOutcome.ABSENT
        }

        val entries = try {
            application.assets.list(ASSET_ROOT)?.toSet().orEmpty()
        } catch (exc: IOException) {
            throw IllegalStateException(
                "bundled VN97 bootstrap assets could not be listed",
                exc,
            )
        }

        return when (VN97BootstrapAssetContract.classify(entries)) {
            VN97BootstrapAssetState.ABSENT ->
                VN97BundledBootstrapOutcome.ABSENT

            VN97BootstrapAssetState.INCOMPLETE ->
                throw IllegalStateException(
                    "bundled VN97 bootstrap is incomplete; package, signature, and publisher key must ship together"
                )

            VN97BootstrapAssetState.COMPLETE -> {
                val signature = readBoundedAsset(
                    "model.vn97sig1",
                    maxBytes = 16 * 1024,
                )
                val publisherKey = readBoundedAsset(
                    "publisher.ed25519",
                    maxBytes = 256,
                )
                application.assets.open(
                    "$ASSET_ROOT/model.vn97cap1"
                ).use { packageInput ->
                    provisioner.review(
                        packageInput = packageInput,
                        signatureBytes = signature,
                        publisherKeyBytes = publisherKey,
                    )
                }
                provisioner.activateReviewed()
                VN97BundledBootstrapOutcome.ACTIVATED
            }
        }
    }

    private fun readBoundedAsset(
        name: String,
        maxBytes: Int,
    ): ByteArray {
        val path = "$ASSET_ROOT/$name"
        return application.assets.open(path).use { input ->
            val out = java.io.ByteArrayOutputStream()
            val buffer = ByteArray(4096)
            var total = 0
            while (true) {
                val count = input.read(buffer)
                if (count < 0) break
                if (count == 0) {
                    throw IllegalStateException(
                        "bundled bootstrap asset read made no progress: $name"
                    )
                }
                total += count
                if (total > maxBytes) {
                    throw IllegalArgumentException(
                        "bundled bootstrap asset exceeds byte limit: $name"
                    )
                }
                out.write(buffer, 0, count)
            }
            if (total == 0) {
                throw IllegalArgumentException(
                    "bundled bootstrap asset is empty: $name"
                )
            }
            out.toByteArray()
        }
    }

    companion object {
        private const val ASSET_ROOT = "vn97-bootstrap"
    }
}
