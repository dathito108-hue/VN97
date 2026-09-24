package ai.vn97.app

import ai.vn97.platform.AndroidVN97CapabilityProvisioner
import ai.vn97.platform.VN97ControlledImprovementEvaluator
import ai.vn97.runtime.VN97ImprovementCandidateLedger
import ai.vn97.runtime.VN97ImprovementCandidateRecord
import ai.vn97.runtime.VN97ImprovementCandidateSpec
import ai.vn97.runtime.NativeActivatedInventoryModelLoader
import ai.vn97.runtime.VN97CapabilityInventoryItem
import ai.vn97.runtime.VN97ImprovementEvaluationRecord
import ai.vn97.runtime.VN97ImprovementPromotionLedger
import ai.vn97.runtime.VN97ImprovementPromotionRecord
import ai.vn97.runtime.VN97ImprovementPromotionSpec
import ai.vn97.runtime.VN97ImprovementPromotionState
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

    private val improvementEvaluator =
        VN97ControlledImprovementEvaluator(
            application
        )

    private val promotionLedger =
        VN97ImprovementPromotionLedger(
            java.io.File(
                platform.capabilityRoot,
                "self-improvement-promotions",
            )
        )

    fun recoverPending() {
        session.recover()
        recoverPromotionAttempts()
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

    fun evaluatePendingImprovementCandidate():
        VN97ImprovementEvaluationRecord =
        application.withSovereignExecution {
            val candidate =
                checkNotNull(
                    pendingImprovementCandidate()
                ) {
                    "no controlled self-improvement candidate is pending"
                }

            val wasOpen =
                application.assistant
                    .releaseForModelEvaluation()
            try {
                improvementEvaluator
                    .evaluate(candidate)
            } finally {
                if (wasOpen) {
                    application.assistant
                        .reloadActivatedModel()
                }
            }
        }

    fun pendingImprovementEvaluation():
        VN97ImprovementEvaluationRecord? {
        val candidate =
            pendingImprovementCandidate()
                ?: return null
        return improvementEvaluator
            .loadOrNull(
                candidate.candidateId
            )
    }

    fun promotePendingImprovementCandidate(
        userApproved: Boolean,
    ): VN97ImprovementPromotionRecord =
        application.withSovereignExecution {
            check(userApproved) {
                "controlled self-improvement promotion requires explicit user approval"
            }
            val candidate =
                checkNotNull(
                    pendingImprovementCandidate()
                ) {
                    "no controlled self-improvement candidate is pending"
                }
            val evaluation =
                checkNotNull(
                    improvementEvaluator
                        .loadOrNull(
                            candidate.candidateId
                        )
                ) {
                    "controlled promotion requires held-out evaluation"
                }
            requirePromotionEligibility(
                candidate,
                evaluation,
            )
            val review =
                checkNotNull(
                    session.pendingReview()
                ) {
                    "controlled promotion requires a current signed provisioning review"
                }
            check(
                review.packageSha256 ==
                    candidate.spec
                        .candidatePackageSha256 &&
                    review.capabilityVersion ==
                        candidate.spec
                            .candidateCapabilityVersion &&
                    review.publisherKeyId ==
                        candidate.spec
                            .candidatePublisherKeyId &&
                    review.publisherKeySha256 ==
                        candidate.spec
                            .candidatePublisherKeySha256 &&
                    review.planSha256 ==
                        candidate.spec
                            .candidatePlanSha256
            ) {
                "current signed review does not match controlled candidate"
            }
            requireExactBaseline(candidate)

            val intent =
                promotionLedger.begin(
                    VN97ImprovementPromotionSpec(
                        candidateId =
                            candidate.candidateId,
                        evaluationId =
                            evaluation.evaluationId,
                        baselineActivationId =
                            candidate.spec
                                .baselineActivationId,
                        baselineArtifactSha256 =
                            candidate.spec
                                .baselineArtifactSha256,
                        baselinePackageSha256 =
                            candidate.spec
                                .baselinePackageSha256,
                        baselineCapabilityVersion =
                            candidate.spec
                                .baselineCapabilityVersion,
                        candidatePackageSha256 =
                            candidate.spec
                                .candidatePackageSha256,
                        candidateArtifactSha256 =
                            evaluation
                                .candidateArtifactSha256,
                        candidateCapabilityVersion =
                            candidate.spec
                                .candidateCapabilityVersion,
                        candidatePlanSha256 =
                            candidate.spec
                                .candidatePlanSha256,
                        candidatePublisherKeyId =
                            candidate.spec
                                .candidatePublisherKeyId,
                        requestedWallTimeMillis =
                            System.currentTimeMillis(),
                    )
                )

            val wasOpen =
                application.assistant
                    .releaseForModelEvaluation()
            try {
                val activated =
                    session
                        .activateReviewedControlled(
                            expectedPackageSha256 =
                                candidate.spec
                                    .candidatePackageSha256,
                            expectedPlanSha256 =
                                candidate.spec
                                    .candidatePlanSha256,
                        )
                requireActivatedCandidate(
                    activated,
                    intent,
                )
                verifyActivatedModel(
                    activated
                )
                promotionLedger
                    .completePromoted(
                        promotionId =
                            intent.promotionId,
                        activationId =
                            activated.activationId,
                        artifactSha256 =
                            activated.artifactSha256,
                    )
            } catch (exc: Throwable) {
                reconcileFailedPromotion(
                    intent = intent,
                    cause = exc,
                )
            } finally {
                if (wasOpen) {
                    application.assistant
                        .reloadActivatedModel()
                }
            }
        }

    fun pendingPromotionAttempt():
        VN97ImprovementPromotionRecord? {
        val candidate =
            pendingImprovementCandidate()
                ?: return null
        return promotionLedger
            .pendingForCandidate(
                candidate.candidateId
            )
    }

    private fun requirePromotionEligibility(
        candidate:
            VN97ImprovementCandidateRecord,
        evaluation:
            VN97ImprovementEvaluationRecord,
    ) {
        check(
            evaluation.decision.passed
        ) {
            "held-out evaluation did not pass"
        }
        check(
            evaluation.candidateId ==
                candidate.candidateId &&
                evaluation.baselineActivationId ==
                    candidate.spec
                        .baselineActivationId &&
                evaluation.baselineArtifactSha256 ==
                    candidate.spec
                        .baselineArtifactSha256 &&
                evaluation.candidatePackageSha256 ==
                    candidate.spec
                        .candidatePackageSha256
        ) {
            "held-out evaluation identity does not match candidate"
        }
    }

    private fun requireExactBaseline(
        candidate:
            VN97ImprovementCandidateRecord,
    ): VN97CapabilityInventoryItem =
        checkNotNull(
            platform.currentModelActivation()
        ) {
            "controlled promotion requires active baseline"
        }.also { active ->
            check(
                active.activationId ==
                    candidate.spec
                        .baselineActivationId &&
                    active.artifactSha256 ==
                        candidate.spec
                            .baselineArtifactSha256 &&
                    active.packageSha256 ==
                        candidate.spec
                            .baselinePackageSha256 &&
                    active.capabilityVersion ==
                        candidate.spec
                            .baselineCapabilityVersion &&
                    active.capabilityId ==
                        VN97ModelImageActivationBackend
                            .CAPABILITY_ID
            ) {
                "active baseline no longer matches candidate binding"
            }
        }

    private fun requireActivatedCandidate(
        active: VN97CapabilityInventoryItem,
        intent:
            VN97ImprovementPromotionRecord,
    ) {
        val spec = intent.spec
        check(
            active.capabilityId ==
                VN97ModelImageActivationBackend
                    .CAPABILITY_ID &&
                active.packageSha256 ==
                    spec.candidatePackageSha256 &&
                active.artifactSha256 ==
                    spec.candidateArtifactSha256 &&
                active.capabilityVersion ==
                    spec.candidateCapabilityVersion &&
                active.planSha256 ==
                    spec.candidatePlanSha256 &&
                active.publisherKeyId ==
                    spec.candidatePublisherKeyId
        ) {
            "controlled promotion activated unexpected model identity"
        }
    }

    private fun verifyActivatedModel(
        active: VN97CapabilityInventoryItem,
    ) {
        val opened =
            checkNotNull(
                NativeActivatedInventoryModelLoader
                    .openOrNull(
                        platform.capabilityRoot
                    )
            ) {
                "promoted VN97 model could not be reopened"
            }
        opened.use { model ->
            check(
                model.info.modelId
                    .joinToString("") {
                        "%02x".format(
                            it.toInt() and 0xff
                        )
                    } ==
                    active.artifactSha256
            ) {
                "promoted native model identity does not match inventory"
            }
        }
    }

    private fun reconcileFailedPromotion(
        intent:
            VN97ImprovementPromotionRecord,
        cause: Throwable,
    ): VN97ImprovementPromotionRecord {
        val recoveryFailure =
            runCatching {
                session.recover()
            }.exceptionOrNull()
        if (recoveryFailure != null) {
            cause.addSuppressed(
                recoveryFailure
            )
            throw cause
        }

        val current =
            platform.currentModelActivation()
        val spec = intent.spec
        if (
            current != null &&
            current.packageSha256 ==
                spec.candidatePackageSha256 &&
            current.artifactSha256 ==
                spec.candidateArtifactSha256
        ) {
            val candidateActivation =
                current
            val restored =
                try {
                    platform.activationCoordinator
                        .rollback(
                            capabilityId =
                                VN97ModelImageActivationBackend
                                    .CAPABILITY_ID,
                            backend =
                                platform.activationBackend,
                        )
                } catch (rollbackExc: Throwable) {
                    cause.addSuppressed(
                        rollbackExc
                    )
                    throw cause
                }
            checkNotNull(restored) {
                "controlled promotion rollback removed baseline"
            }
            check(
                restored.activationId ==
                    spec.baselineActivationId &&
                    restored.artifactSha256 ==
                        spec.baselineArtifactSha256 &&
                    restored.packageSha256 ==
                        spec.baselinePackageSha256 &&
                    restored.capabilityVersion ==
                        spec.baselineCapabilityVersion
            ) {
                "controlled promotion rollback did not restore exact baseline"
            }
            verifyActivatedModel(restored)
            return promotionLedger
                .completeRolledBack(
                    promotionId =
                        intent.promotionId,
                    activationId =
                        candidateActivation
                            .activationId,
                    artifactSha256 =
                        candidateActivation
                            .artifactSha256,
                    detail =
                        "post-promotion verification failed; exact baseline restored: " +
                            (
                                cause.message ?:
                                    cause::class.java
                                        .simpleName
                            ),
                )
        }

        if (
            current != null &&
            current.activationId ==
                spec.baselineActivationId &&
            current.artifactSha256 ==
                spec.baselineArtifactSha256 &&
            current.packageSha256 ==
                spec.baselinePackageSha256 &&
            current.capabilityVersion ==
                spec.baselineCapabilityVersion
        ) {
            verifyActivatedModel(current)
            return promotionLedger
                .completeAborted(
                    promotionId =
                        intent.promotionId,
                    detail =
                        "promotion aborted before candidate became active: " +
                            (
                                cause.message ?:
                                    cause::class.java
                                        .simpleName
                            ),
                    baselineStillActive =
                        true,
                )
        }

        throw IllegalStateException(
            "promotion recovery found neither exact baseline nor exact candidate; manual recovery required",
            cause,
        )
    }

    private fun recoverPromotionAttempts() {
        for (
            intent in
                promotionLedger
                    .pendingAttempts()
        ) {
            val current =
                platform.currentModelActivation()
            val spec = intent.spec
            when {
                current != null &&
                    current.packageSha256 ==
                        spec.candidatePackageSha256 &&
                    current.artifactSha256 ==
                        spec.candidateArtifactSha256 &&
                    current.capabilityVersion ==
                        spec.candidateCapabilityVersion -> {
                    requireActivatedCandidate(
                        current,
                        intent,
                    )
                    verifyActivatedModel(current)
                    promotionLedger
                        .completePromoted(
                            promotionId =
                                intent.promotionId,
                            activationId =
                                current.activationId,
                            artifactSha256 =
                                current.artifactSha256,
                        )
                }
                current != null &&
                    current.activationId ==
                        spec.baselineActivationId &&
                    current.artifactSha256 ==
                        spec.baselineArtifactSha256 &&
                    current.packageSha256 ==
                        spec.baselinePackageSha256 &&
                    current.capabilityVersion ==
                        spec.baselineCapabilityVersion -> {
                    verifyActivatedModel(current)
                    promotionLedger
                        .completeAborted(
                            promotionId =
                                intent.promotionId,
                            detail =
                                "recovered pending promotion with exact baseline still active",
                            baselineStillActive =
                                true,
                        )
                }
                else ->
                    throw IllegalStateException(
                        "pending promotion recovery found divergent active model"
                    )
            }
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
        check(
            promotionLedger
                .pendingForCandidate(
                    current.candidateId
                ) == null
        ) {
            "cannot reject candidate while promotion attempt is pending"
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
