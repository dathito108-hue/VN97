package ai.vn97.app

import ai.vn97.platform.VN97RemoteCapabilityFetchResult
import ai.vn97.runtime.VN97AcquisitionProvenanceRecord
import ai.vn97.runtime.VN97KnowledgeAcquisitionResult
import ai.vn97.runtime.VN97KnowledgeAcquisitionReview
import ai.vn97.runtime.VN97KnowledgeGapProposal
import android.net.Uri
import java.io.ByteArrayOutputStream

class VN97AppKnowledgeAcquisition(
    private val application: VN97Application,
) {
    private val provenanceLock = Any()
    private var lastProposal: VN97KnowledgeGapProposal? = null
    private var lastFetch: VN97RemoteCapabilityFetchResult? = null
    private var lastReview: VN97KnowledgeAcquisitionReview? = null
    fun pendingReview(): VN97KnowledgeAcquisitionReview? =
        application.assistant.pendingKnowledgeReview()

    fun propose(
        goal: String,
    ): VN97KnowledgeGapProposal {
        check(application.assistant.openIfActivated()) {
            "trusted VN97 model is not active"
        }
        val proposal = application.assistant
            .proposeKnowledgeAcquisition(goal)
        synchronized(provenanceLock) {
            lastProposal = proposal
        }
        return proposal
    }

    fun clearReview() {
        application.assistant.clearKnowledgeReview()
        synchronized(provenanceLock) {
            lastReview = null
        }
    }

    fun noteRemoteFetch(
        result: VN97RemoteCapabilityFetchResult,
    ) {
        if (!result.approved) return
        synchronized(provenanceLock) {
            lastFetch = result
        }
    }

    fun review(
        packageUri: Uri,
        signatureUri: Uri,
        publisherKeyUri: Uri,
    ): VN97KnowledgeAcquisitionReview {
        check(application.assistant.openIfActivated()) {
            "trusted VN97 model is not active"
        }
        val resolver = application.contentResolver
        val signatureBytes = readBounded(
            signatureUri,
            MAX_SIGNATURE_BYTES,
            "VN97SIG1",
        )
        val publisherKey = parsePublisherKey(
            readBounded(
                publisherKeyUri,
                MAX_PUBLISHER_KEY_BYTES,
                "publisher public key",
            )
        )
        val packageInput =
            checkNotNull(resolver.openInputStream(packageUri)) {
                "VN97CAP1 document could not be opened"
            }
        val review = packageInput.use { input ->
            application.assistant.reviewKnowledgeCapability(
                packageInput = input,
                signatureBytes = signatureBytes,
                publisherPublicKey = publisherKey,
            )
        }
        synchronized(provenanceLock) {
            lastReview = review
            if (
                lastFetch?.artifact?.packageSha256 !=
                    review.packageSha256
            ) {
                lastFetch = null
            }
        }
        return review
    }

    fun reviewFetchedPackage(
        packageSha256: String,
        signatureUri: Uri,
        publisherKeyUri: Uri,
    ): VN97KnowledgeAcquisitionReview {
        check(application.assistant.openIfActivated()) {
            "trusted VN97 model is not active"
        }
        val signatureBytes = readBounded(
            signatureUri,
            MAX_SIGNATURE_BYTES,
            "VN97SIG1",
        )
        val publisherKey = parsePublisherKey(
            readBounded(
                publisherKeyUri,
                MAX_PUBLISHER_KEY_BYTES,
                "publisher public key",
            )
        )
        val artifact =
            application.platformRuntime
                .requireFetchedCapabilityArtifact(
                    packageSha256
                )
        val review = artifact.inputStream().use { input ->
            application.assistant.reviewKnowledgeCapability(
                packageInput = input,
                signatureBytes = signatureBytes,
                publisherPublicKey = publisherKey,
            )
        }
        synchronized(provenanceLock) {
            lastReview = review
        }
        return review
    }

    fun acquireReviewed(): VN97KnowledgeAcquisitionResult {
        val review = checkNotNull(
            application.assistant.pendingKnowledgeReview()
                ?: synchronized(provenanceLock) {
                    lastReview
                }
        ) {
            "no reviewed knowledge capability is pending"
        }
        val result =
            application.assistant.acquireReviewedKnowledge()
        check(
            result.packageSha256 == review.packageSha256 &&
                result.capabilityId == review.capabilityId &&
                result.capabilityVersion == review.capabilityVersion &&
                result.publisherKeyId == review.publisherKeyId
        ) {
            "knowledge acquisition result does not match reviewed identity"
        }

        val snapshot = synchronized(provenanceLock) {
            val proposal = lastProposal?.takeIf {
                it.needed &&
                    it.capabilityId == review.capabilityId
            }
            val fetch = lastFetch?.takeIf {
                it.approved &&
                    it.artifact?.packageSha256 ==
                    review.packageSha256
            }
            Triple(proposal, fetch, lastReview)
        }
        val proposal = snapshot.first
        val fetch = snapshot.second
        val artifact = fetch?.artifact

        application.platformRuntime
            .saveProductionKnowledgeAcquisitionProvenance(
                VN97AcquisitionProvenanceRecord(
                    packageSha256 = review.packageSha256,
                    capabilityId = review.capabilityId,
                    capabilityVersion = review.capabilityVersion,
                    publisherKeyId = review.publisherKeyId,
                    publisherKeySha256 = review.publisherKeySha256,
                    signatureSha256 = review.signatureSha256,
                    payloadSha256 = review.payloadSha256,
                    sourceOrigin = review.sourceOrigin,
                    sourceLicense = review.sourceLicense,
                    recordIds = result.recordIds,
                    alreadyAcquired = result.alreadyAcquired,
                    proposalId =
                        proposal?.proposalId ?: "",
                    proposalCapabilityId =
                        proposal?.capabilityId ?: "",
                    proposalEvidenceRecordIds =
                        proposal?.evidenceRecordIds ?: emptyList(),
                    fetchReceiptId =
                        fetch?.receiptId ?: "",
                    fetchCanonicalUrl =
                        artifact?.canonicalUrl ?: "",
                    createdWallTimeMillis =
                        System.currentTimeMillis(),
                )
            )
        synchronized(provenanceLock) {
            lastReview = null
            lastFetch = null
            lastProposal = null
        }
        return result
    }

    private fun readBounded(
        uri: Uri,
        maxBytes: Int,
        label: String,
    ): ByteArray {
        val input = checkNotNull(
            application.contentResolver.openInputStream(uri)
        ) {
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
                    throw IllegalStateException(
                        "$label read made no progress"
                    )
                }
                total += count
                require(total <= maxBytes) {
                    "$label exceeds byte limit"
                }
                out.write(buffer, 0, count)
            }
            require(total > 0) {
                "$label is empty"
            }
            out.toByteArray()
        }
    }

    private fun parsePublisherKey(
        bytes: ByteArray,
    ): ByteArray {
        if (bytes.size == 32) {
            return bytes.copyOf()
        }
        val text = bytes.toString(Charsets.US_ASCII)
        if (
            text.length == 64 &&
            text.all { it in "0123456789abcdef" }
        ) {
            return ByteArray(32) { index ->
                text.substring(
                    index * 2,
                    index * 2 + 2,
                ).toInt(16).toByte()
            }
        }
        throw IllegalArgumentException(
            "publisher key must be raw 32-byte Ed25519 or 64 lowercase hex characters"
        )
    }

    companion object {
        private const val MAX_SIGNATURE_BYTES = 16 * 1024
        private const val MAX_PUBLISHER_KEY_BYTES = 256
    }
}
