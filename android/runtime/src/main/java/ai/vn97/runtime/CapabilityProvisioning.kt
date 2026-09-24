package ai.vn97.runtime

import java.io.InputStream
import java.security.MessageDigest

data class VN97ModelProvisioningReview(
    val capabilityId: String,
    val capabilityVersion: Long,
    val packageSha256: String,
    val publisherKeyId: String,
    val publisherKeySha256: String,
    val publisherPreviouslyTrusted: Boolean,
    val sourceOrigin: String,
    val sourceLicense: String,
    val planSha256: String,
)

private fun requireProvisioningSha(
    value: String,
    label: String,
) {
    require(
        value.length == 64 &&
            value.all {
                it in "0123456789abcdef"
            }
    ) {
        "$label must be lowercase SHA-256 hex"
    }
}

class VN97ModelImageProvisioningSession(
    private val stager: VN97CapabilityStager,
    private val stageRoot: java.io.File,
    private val profile: VN97CompatibilityProfile,
    private val backend: VN97CapabilityActivationBackend,
    private val coordinator: VN97CapabilityActivationCoordinator,
    private val trustRegistry: VN97PublisherTrustRegistry,
) {
    private data class Pending(
        val verified: VN97VerifiedCapability,
        val plan: VN97CompatibilityPlan,
        val publisherPublicKey: ByteArray,
        val review: VN97ModelProvisioningReview,
    )

    private var pending: Pending? = null

    @Synchronized
    fun recover(): VN97InventorySnapshot =
        coordinator.recover(mapOf(backend.backendId to backend))

    @Synchronized
    fun review(
        packageInput: InputStream,
        signatureBytes: ByteArray,
        publisherPublicKey: ByteArray,
    ): VN97ModelProvisioningReview {
        check(pending == null) {
            "a provisioning review is already pending"
        }
        require(publisherPublicKey.size == 32) {
            "publisher Ed25519 public key must be exactly 32 bytes"
        }

        val staged = stager.stage(packageInput, signatureBytes)
        val trustKey = VN97TrustedPublisherKey(
            keyId = staged.signature.keyId,
            publicKey = publisherPublicKey.copyOf(),
            capabilityPrefixes = setOf("model"),
            allowedKinds = setOf("weights"),
            minVersion = 1L,
            maxVersion = 0xffff_ffffL,
            revoked = false,
        )
        val previouslyTrusted = trustRegistry.requireCompatibleOrUnenrolled(
            staged.signature.keyId,
            publisherPublicKey,
        )
        val trustStore = VN97CapabilityTrustStore(listOf(trustKey))
        val verified = VN97CapabilityTrustVerifier.verify(
            staged = staged,
            stageRoot = stageRoot,
            trustStore = trustStore,
        )
        val plan = VN97CapabilityCompatibility.plan(
            verified = verified,
            profile = profile,
        )
        requireProductionModelIdentity(verified, plan)

        val publicKeyDigest = MessageDigest.getInstance("SHA-256")
            .digest(publisherPublicKey)
            .toHex()
        val source = verified.parsed.manifest.source
        val review = VN97ModelProvisioningReview(
            capabilityId = verified.parsed.manifest.capabilityId,
            capabilityVersion = verified.parsed.manifest.capabilityVersion,
            packageSha256 = verified.parsed.packageSha256,
            publisherKeyId = verified.publisherKeyId,
            publisherKeySha256 = publicKeyDigest,
            publisherPreviouslyTrusted = previouslyTrusted,
            sourceOrigin = source.origin,
            sourceLicense = source.license,
            planSha256 = plan.sha256(),
        )
        pending = Pending(
            verified = verified,
            plan = plan,
            publisherPublicKey = publisherPublicKey.copyOf(),
            review = review,
        )
        return review
    }

    @Synchronized
    fun pendingReview(): VN97ModelProvisioningReview? = pending?.review

    @Synchronized
    fun clearReview() {
        pending = null
    }

    @Synchronized
    fun activateReviewed(): VN97CapabilityInventoryItem =
        activatePendingControlled(
            expectedPackageSha256 = null,
            expectedPlanSha256 = null,
        )

    @Synchronized
    fun activateReviewedControlled(
        expectedPackageSha256: String,
        expectedPlanSha256: String,
    ): VN97CapabilityInventoryItem {
        requireProvisioningSha(
            expectedPackageSha256,
            "expectedPackageSha256",
        )
        requireProvisioningSha(
            expectedPlanSha256,
            "expectedPlanSha256",
        )
        return activatePendingControlled(
            expectedPackageSha256 =
                expectedPackageSha256,
            expectedPlanSha256 =
                expectedPlanSha256,
        )
    }

    private fun activatePendingControlled(
        expectedPackageSha256: String?,
        expectedPlanSha256: String?,
    ): VN97CapabilityInventoryItem {
        val current = checkNotNull(pending) {
            "no provisioning review is pending"
        }
        if (expectedPackageSha256 != null) {
            check(
                current.review.packageSha256 ==
                    expectedPackageSha256
            ) {
                "controlled activation package identity changed"
            }
        }
        if (expectedPlanSha256 != null) {
            check(
                current.review.planSha256 ==
                    expectedPlanSha256
            ) {
                "controlled activation plan identity changed"
            }
        }

        // Reconcile any crash-left activation before a new activation attempt.
        recover()
        val durableTrustStore =
            trustRegistry.enrollModelPublisher(
                keyId =
                    current.verified.publisherKeyId,
                publicKey =
                    current.publisherPublicKey,
            )
        return try {
            coordinator.activate(
                verified = current.verified,
                plan = current.plan,
                profile = profile,
                backend = backend,
                stageRoot = stageRoot,
                trustStore = durableTrustStore,
                adapters = emptyList(),
                allowLossy = false,
                allowSameVersionReplace = false,
                allowDowngrade = false,
            )
        } finally {
            pending = null
        }
    }

    private fun requireProductionModelIdentity(
        verified: VN97VerifiedCapability,
        plan: VN97CompatibilityPlan,
    ) {
        val manifest = verified.parsed.manifest
        if (manifest.capabilityId != VN97ModelImageActivationBackend.CAPABILITY_ID ||
            manifest.kind != "weights" ||
            manifest.sections.size != 1
        ) {
            throw VN97CapabilityActivationException(
                "provisioning accepts only model.language weights"
            )
        }
        val section = manifest.sections.single()
        val sectionPlan = plan.sections.singleOrNull()
            ?: throw VN97CapabilityActivationException(
                "provisioning requires exactly one model-image section"
            )
        if (section.role != VN97ModelImageActivationBackend.MODEL_IMAGE_ROLE ||
            section.format != VN97ModelImageActivationBackend.MODEL_IMAGE_FORMAT ||
            plan.disposition != VN97CompatibilityDisposition.DIRECT ||
            sectionPlan.role != section.role ||
            sectionPlan.sourceFormat != section.format ||
            sectionPlan.targetFormat != section.format ||
            sectionPlan.adapterId != null ||
            sectionPlan.lossy
        ) {
            throw VN97CapabilityActivationException(
                "provisioning requires direct VN97MI1 model_image"
            )
        }
    }

    private fun ByteArray.toHex(): String =
        joinToString("") { "%02x".format(it.toInt() and 0xff) }
}
