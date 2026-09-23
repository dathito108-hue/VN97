package ai.vn97.app

import ai.vn97.runtime.VN97KnowledgeAcquisitionResult
import ai.vn97.runtime.VN97KnowledgeAcquisitionReview
import ai.vn97.runtime.VN97KnowledgeGapProposal
import android.net.Uri
import java.io.ByteArrayOutputStream

class VN97AppKnowledgeAcquisition(
    private val application: VN97Application,
) {
    fun pendingReview(): VN97KnowledgeAcquisitionReview? =
        application.assistant.pendingKnowledgeReview()

    fun propose(
        goal: String,
    ): VN97KnowledgeGapProposal {
        check(application.assistant.openIfActivated()) {
            "trusted VN97 model is not active"
        }
        return application.assistant
            .proposeKnowledgeAcquisition(goal)
    }

    fun clearReview() {
        application.assistant.clearKnowledgeReview()
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
        return packageInput.use { input ->
            application.assistant.reviewKnowledgeCapability(
                packageInput = input,
                signatureBytes = signatureBytes,
                publisherPublicKey = publisherKey,
            )
        }
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
        return artifact.inputStream().use { input ->
            application.assistant.reviewKnowledgeCapability(
                packageInput = input,
                signatureBytes = signatureBytes,
                publisherPublicKey = publisherKey,
            )
        }
    }

    fun acquireReviewed(): VN97KnowledgeAcquisitionResult =
        application.assistant.acquireReviewedKnowledge()

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
