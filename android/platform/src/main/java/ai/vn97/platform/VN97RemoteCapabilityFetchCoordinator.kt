package ai.vn97.platform

import ai.vn97.runtime.NativePlanController
import ai.vn97.runtime.NativePlanStepSpec
import ai.vn97.runtime.NativeReasoningBudget
import ai.vn97.runtime.NativeStepKind
import java.io.File

data class VN97RemoteCapabilityFetchApproval(
    val canonicalUrl: String,
    val requestDigest: String,
    val presentationJson: String,
    val expiresNs: Long,
) {
    init {
        require(canonicalUrl.startsWith("https://"))
        require(
            requestDigest.length == 64 &&
                requestDigest.all {
                    it in "0123456789abcdef"
                }
        )
        require(presentationJson.isNotBlank())
        require(expiresNs > 0L)
    }
}

data class VN97RemoteCapabilityFetchResult(
    val approved: Boolean,
    val artifact: VN97FetchedCapabilityArtifact? = null,
    val receiptId: String = "",
    val replayed: Boolean = false,
) {
    init {
        if (approved) {
            require(artifact != null) {
                "approved remote capability fetch requires artifact"
            }
            require(
                receiptId.length == 64 &&
                    receiptId.all {
                        it in "0123456789abcdef"
                    }
            ) {
                "approved remote capability fetch requires M6 receipt"
            }
        } else {
            require(artifact == null)
            require(receiptId.isEmpty())
            require(!replayed)
        }
    }
}

class VN97RemoteCapabilityFetchCoordinator internal constructor(
    private val principal: String,
    private val approvals: M6ExternalApprovalHandoff,
    private val registryFactory: () -> M6TypedCapabilityRegistry,
    private val fabricFactory:
        (M6PolicyGrant) -> M6ExternalExecutionFabric,
) {
    private data class Pending(
        val canonicalUrl: String,
        val controller: NativePlanController,
        val request: M6ExternalActionRequest,
        val prompt: M6ApprovalPrompt,
        val fabric: M6ExternalExecutionFabric,
    )

    private var pending: Pending? = null

    init {
        require(principal.isNotBlank()) {
            "remote capability principal must not be blank"
        }
    }

    @Synchronized
    fun pendingApproval():
        VN97RemoteCapabilityFetchApproval? =
        pending?.let(::publicApproval)

    @Synchronized
    fun clearPending() {
        pending?.controller?.cancel(
            "remote capability fetch cleared by user"
        )
        pending = null
    }

    @Synchronized
    fun prepare(
        url: String,
        nowNs: Long,
    ): VN97RemoteCapabilityFetchApproval {
        check(pending == null) {
            "a remote capability fetch approval is already pending"
        }
        require(nowNs >= 0L) {
            "remote capability fetch time must be non-negative"
        }
        val canonical =
            canonicalCapabilityHttpsUrl(url)
        val controller =
            NativePlanController.create(
                goal =
                    "Fetch one exact remote VN97 knowledge capability artifact for explicit user review.",
                specs = listOf(
                    NativePlanStepSpec(
                        kind =
                            NativeStepKind.EXTERNAL,
                        objective =
                            FETCH_OBJECTIVE,
                        dependencies =
                            emptyList(),
                        requiresVerification = false,
                        minConfidence = 1.0,
                    )
                ),
                budget =
                    NativeReasoningBudget(
                        maxTransitions = 4,
                        maxMemoryQueries = 1,
                        maxMemoryHits = 1,
                        maxRetriesPerStep = 0,
                    ),
                createdNs = nowNs,
            )
        controller.beginStep(1)
        val request =
            M6ExternalActionRequest(
                planId = controller.plan.planId,
                stepId = 1,
                objective = FETCH_OBJECTIVE,
                capabilityId =
                    M6AndroidProductionCapabilities
                        .CAPABILITY_ARTIFACT_FETCH,
                scope =
                    M6CapabilityScope.fromMap(
                        mapOf(
                            M6AndroidProductionCapabilities
                                .CAPABILITY_URL_SCOPE to
                                canonical
                        )
                    ),
                payloadJson = "{}",
            )
        val registry = registryFactory()
        val descriptor =
            registry.validate(request)
        check(descriptor.approvalRequired) {
            "remote capability fetch must require M6 approval"
        }
        val grant =
            M6AndroidProductionCapabilities
                .userApprovedCapabilityArtifactGrant(
                    principal = principal,
                    url = canonical,
                )
        check(grant.scopeDigest == request.scope.digest) {
            "remote capability grant scope does not bind exact URL"
        }
        val fabric = fabricFactory(grant)
        val prompt = approvals.createPrompt(
            request = request,
            principal = principal,
            promptTtlNs = PROMPT_TTL_NS,
            nowNs = nowNs,
        )
        val created = Pending(
            canonicalUrl = canonical,
            controller = controller,
            request = request,
            prompt = prompt,
            fabric = fabric,
        )
        pending = created
        return publicApproval(created)
    }

    @Synchronized
    fun resolve(
        approved: Boolean,
        nowNs: Long,
    ): VN97RemoteCapabilityFetchResult {
        require(nowNs >= 0L) {
            "remote capability fetch time must be non-negative"
        }
        val current = checkNotNull(pending) {
            "no remote capability fetch approval is pending"
        }
        pending = null
        val token = approvals.resolve(
            prompt = current.prompt,
            approved = approved,
            approvalTtlNs =
                APPROVAL_TTL_NS,
            nowNs = nowNs,
        )
        if (!approved) {
            current.controller.cancel(
                "remote capability fetch rejected by user"
            )
            check(token == null)
            return VN97RemoteCapabilityFetchResult(
                approved = false,
            )
        }
        val approval = checkNotNull(token) {
            "approved remote capability fetch did not return token"
        }
        val execution =
            current.fabric.executeWaiting(
                controller =
                    current.controller,
                request = current.request,
                principal = principal,
                approval = approval,
                nowNs = nowNs,
            )
        val parsed =
            VN97FetchedCapabilityArtifact
                .parseWireResult(
                    execution.receipt.result
                )
        return VN97RemoteCapabilityFetchResult(
            approved = true,
            artifact =
                parsed.copy(
                    canonicalUrl =
                        current.canonicalUrl
                ),
            receiptId =
                execution.receipt.receiptId,
            replayed = execution.replayed,
        )
    }

    private fun publicApproval(
        value: Pending,
    ): VN97RemoteCapabilityFetchApproval =
        VN97RemoteCapabilityFetchApproval(
            canonicalUrl = value.canonicalUrl,
            requestDigest =
                value.request.requestDigest,
            presentationJson =
                value.prompt.presentationJson,
            expiresNs = value.prompt.expiresNs,
        )

    companion object {
        private const val FETCH_OBJECTIVE =
            "Fetch the exact approved remote VN97CAP1 knowledge artifact without trusting or activating it."
        private const val PROMPT_TTL_NS =
            300_000_000_000L
        private const val APPROVAL_TTL_NS =
            60_000_000_000L
    }
}
