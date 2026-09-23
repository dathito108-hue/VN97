package ai.vn97.runtime

import java.io.File

enum class VN97BackendTransactionState { PREPARED, COMMITTED, ROLLED_BACK }
data class VN97BackendStatus(val state: VN97BackendTransactionState, val runtimeRevision: String? = null)
data class VN97PreparedActivation(val backendId: String, val token: String, val artifactSha256: String)
interface VN97CapabilityActivationBackend {
    val backendId: String
    fun prepare(verified: VN97VerifiedCapability, plan: VN97CompatibilityPlan): VN97PreparedActivation
    fun commit(token: String): String
    fun inspect(token: String): VN97BackendStatus
    fun rollback(token: String)
}
open class VN97CapabilityActivationException(message: String, cause: Throwable? = null) : IllegalStateException(message, cause)

data class VN97CapabilitySource(val origin: String, val sourceSha256: String, val license: String)
data class VN97CapabilitySection(val index: Int, val role: String, val format: String, val size: Long, val sha256: String, val packageOffset: Long)
data class VN97CapabilityManifest(val capabilityId: String, val capabilityVersion: Long, val kind: String, val source: VN97CapabilitySource, val sections: List<VN97CapabilitySection>)
data class VN97ParsedCapabilityPackage(val packageSha256: String, val manifest: VN97CapabilityManifest, val size: Long)
data class VN97StagedCapability(val parsed: VN97ParsedCapabilityPackage, val signature: Any? = null, val packageFile: File, val signatureFile: File = packageFile)
data class VN97VerifiedCapability(val staged: VN97StagedCapability, val parsed: VN97ParsedCapabilityPackage, val publisherKeyId: String, val signatureSha256: String)

enum class VN97CompatibilityDisposition { DIRECT, ADAPT_REQUIRED }
data class VN97FormatRule(val role: String, val acceptedFormats: List<String>)
data class VN97CompatibilityProfile(val profileId: String, val runtimeApiVersion: Long, val supportedKinds: Set<String>, val minCapabilityVersion: Long, val maxCapabilityVersion: Long, val formatRules: List<VN97FormatRule>)
data class VN97SectionPlan(val sectionIndex: Int, val role: String, val sourceFormat: String, val targetFormat: String, val adapterId: String?, val lossy: Boolean)
data class VN97CompatibilityPlan(val packageSha256: String, val capabilityId: String, val capabilityVersion: Long, val publisherKeyId: String, val signatureSha256: String, val profileId: String, val profileSha256: String, val runtimeApiVersion: Long, val disposition: VN97CompatibilityDisposition, val sections: List<VN97SectionPlan>)
