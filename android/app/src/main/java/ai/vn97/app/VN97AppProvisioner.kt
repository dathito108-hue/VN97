package ai.vn97.app

import ai.vn97.platform.AndroidVN97CapabilityProvisioner
import ai.vn97.runtime.VN97ImprovementCandidateLedger
import ai.vn97.runtime.VN97ImprovementCandidateRecord
import ai.vn97.runtime.VN97ImprovementCandidateSpec
import ai.vn97.runtime.VN97ModelImageActivationBackend
import ai.vn97.runtime.VN97ModelImageProvisioningSession
import ai.vn97.runtime.VN97ModelProvisioningReview
import android.net.Uri
import java.io.ByteArrayOutputStream
import java.io.InputStream

class VN97AppProvisioner(
    private val application: VN97Application,
) {
    private val platform = AndroidVN97CapabilityProvisioner(application)
    private val session = VN97ModelImageProvisioningSession(
        stager = platform.createStager(),
        stageRoot = platform.stageRoot,
        profile = platform.compatibilityProfile,
        backend = platform.activationBackend,
        coordinator = platform.activationCoordinator,
        trustRegistry = ai.vn97.runtime.VN97PublisherTrustRegistry(
            platform.capabilityRoot
        ),
    )

    private val improvementLedger =
        VN97ImprovementCandidateLedger(
            java.io.File(
                platform.capabilityRoot,
                "self-improvement",
            )
        )

    fun recoverPending() {
        session.recover()
    }

    fun pendingReview(): VN97ModelProvisioningReview? =
        session.pendingReview()

    fun clearReview() {
        session.clearReview()
    }

    fun pendingImprovementCandidate():
        VN97ImprovementCandidateRecord? {
        val review =
            session.pendingReview()
                ?: return null
        return improvementLedger
            .reviewedForPackageOrNull(
                review.packageSha256
            )
    }

    fun reviewSelfImprovement(
        packageUri: Uri,
        signatureUri: Uri,
        publisherKeyUri: Uri,
        objective: String,
    ): Pair<
        VN97ModelProvisioningReview,
        VN97ImprovementCandidateRecord
    > {
        require(objective.isNotBlank()) {
            "self-improvement objective must not be blank"
        }
        val review = review(
            packageUri = packageUri,
            signatureUri = signatureUri,
            publisherKeyUri = publisherKeyUri,
        )
        return try {
            val baseline =
                checkNotNull(
                    platform.currentModelActivation()
                ) {
                    "self-improvement requires an active canonical model baseline"
                }
            check(
                baseline.capabilityId ==
                    VN97ModelImageActivationBackend
                        .CAPABILITY_ID
            ) {
                "active baseline is not canonical model.language"
            }
            val spec =
                VN97ImprovementCandidateSpec(
                    baselineActivationId =
                        baseline.activationId,
                    baselineArtifactSha256 =
                        baseline.artifactSha256,
                    baselinePackageSha256 =
                        baseline.packageSha256,
                    baselineCapabilityVersion =
                        baseline.capabilityVersion,
                    candidatePackageSha256 =
                        review.packageSha256,
                    candidateCapabilityVersion =
                        review.capabilityVersion,
                    candidatePublisherKeyId =
                        review.publisherKeyId,
                    candidatePublisherKeySha256 =
                        review.publisherKeySha256,
                    candidatePlanSha256 =
                        review.planSha256,
                    sourceOrigin =
                        review.sourceOrigin,
                    sourceLicense =
                        review.sourceLicense,
                    objective = objective,
                )
            review to
                improvementLedger
                    .registerReviewed(spec)
        } catch (exc: Throwable) {
            session.clearReview()
            throw exc
        }
    }

    fun rejectImprovementCandidate(
        reason: String =
            "rejected by user before evaluation",
    ): VN97ImprovementCandidateRecord {
        val current =
            checkNotNull(
                pendingImprovementCandidate()
            ) {
                "no controlled self-improvement candidate is pending"
            }
        return improvementLedger.reject(
            candidateId =
                current.candidateId,
            reason = reason,
        ).also {
            session.clearReview()
        }
    }

    fun review(
        packageUri: Uri,
        signatureUri: Uri,
        publisherKeyUri: Uri,
    ): VN97ModelProvisioningReview {
        val resolver = application.contentResolver
        val signatureBytes = readBounded(
            signatureUri,
            maxBytes = 16 * 1024,
            label = "VN97SIG1",
        )
        val publisherKeyBytes = readBounded(
            publisherKeyUri,
            maxBytes = 256,
            label = "publisher public key",
        )
        val packageInput = checkNotNull(resolver.openInputStream(packageUri)) {
            "VN97CAP1 document could not be opened"
        }
        return packageInput.use { input ->
            review(
                packageInput = input,
                signatureBytes = signatureBytes,
                publisherKeyBytes = publisherKeyBytes,
            )
        }
    }

    fun review(
        packageInput: InputStream,
        signatureBytes: ByteArray,
        publisherKeyBytes: ByteArray,
    ): VN97ModelProvisioningReview =
        session.review(
            packageInput = packageInput,
            signatureBytes = signatureBytes,
            publisherPublicKey = parsePublisherKey(publisherKeyBytes),
        )

    fun activateReviewed() =
        session.pendingReview()
            ?.let { review ->
                check(
                    improvementLedger
                        .reviewedForPackageOrNull(
                            review.packageSha256
                        ) == null
                ) {
                    "controlled self-improvement candidate cannot use normal activation before evaluation and promotion gates"
                }
                session.activateReviewed()
            }
            ?: session.activateReviewed()

    private fun readBounded(
        uri: Uri,
        maxBytes: Int,
        label: String,
    ): ByteArray {
        val input = checkNotNull(application.contentResolver.openInputStream(uri)) {
            "$label document could not be opened"
        }
        return input.use { stream ->
            val out = ByteArrayOutputStream()
            val buffer = ByteArray(4096)
            var total = 0
            while (true) {
                val count = stream.read(buffer)
                if (count < 0) break
                if (count == 0) {
                    throw IllegalStateException("$label read made no progress")
                }
                total += count
                if (total > maxBytes) {
                    throw IllegalArgumentException("$label exceeds byte limit")
                }
                out.write(buffer, 0, count)
            }
            if (total == 0) {
                throw IllegalArgumentException("$label is empty")
            }
            out.toByteArray()
        }
    }

    private fun parsePublisherKey(bytes: ByteArray): ByteArray {
        if (bytes.size == 32) return bytes.copyOf()
        val text = bytes.toString(Charsets.US_ASCII)
        if (text.length == 64 &&
            text.all { it in "0123456789abcdef" }
        ) {
            return ByteArray(32) { index ->
                text.substring(index * 2, index * 2 + 2)
                    .toInt(16)
                    .toByte()
            }
        }
        throw IllegalArgumentException(
            "publisher key must be raw 32-byte Ed25519 or 64 lowercase hex characters"
        )
    }
}
