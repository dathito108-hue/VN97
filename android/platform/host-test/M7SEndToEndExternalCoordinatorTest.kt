import ai.vn97.platform.*
import ai.vn97.runtime.*

private class ScriptedInference(
    outputs: List<String>,
) : NativeCognitionInference {
    private val queue = ArrayDeque(outputs)
    val calls = mutableListOf<NativeCognitionOperation>()

    override fun generateOperation(
        operation: NativeCognitionOperation,
        requestJson: String,
    ): String {
        check(requestJson.isNotEmpty())
        calls += operation
        check(queue.isNotEmpty()) { "unexpected cognition call: $operation" }
        return queue.removeFirst()
    }

    override fun embedText(text: String, vectorDim: Int): FloatArray {
        check(text.isNotEmpty())
        return FloatArray(vectorDim) { 1.0f }
    }
}

private class RecordingApprovalController : M6ApprovalControllerPort {
    var prompts = 0
    var resolves = 0

    override fun createPrompt(
        requestDigest: String,
        principal: String,
        presentationJson: String,
        promptTtlNs: Long,
        nowNs: Long,
    ): M6ApprovalPrompt {
        prompts += 1
        return M6ApprovalPrompt(
            promptId = "00".repeat(16),
            principal = principal,
            requestDigest = requestDigest,
            presentationJson = presentationJson,
            createdNs = nowNs,
            expiresNs = nowNs + promptTtlNs,
        )
    }

    override fun resolve(
        prompt: M6ApprovalPrompt,
        approved: Boolean,
        approvalTtlNs: Long,
        nowNs: Long,
    ): M6ApprovalToken? {
        resolves += 1
        return if (approved) {
            M6ApprovalToken(
                approvalId = "11".repeat(16),
                issuer = "local-user",
                principal = prompt.principal,
                requestDigest = prompt.requestDigest,
                issuedNs = nowNs,
                expiresNs = nowNs + approvalTtlNs,
                signature = "22".repeat(32),
            )
        } else {
            null
        }
    }

    override fun verify(
        token: M6ApprovalToken,
        requestDigest: String,
        principal: String,
        nowNs: Long,
    ) {
        check(token.requestDigest == requestDigest)
        check(token.principal == principal)
        check(nowNs in token.issuedNs until token.expiresNs)
    }
}

private fun descriptor() = M6CapabilityDescriptor(
    capabilityId = "test.echo",
    requiredScopeKeys = setOf("target"),
    approvalRequired = true,
    maxPayloadUtf8Bytes = 4096,
    maxLeaseNs = 30_000_000_000L,
    maxLeaseUses = 1,
)

private fun scope(target: String) = M6CapabilityScope.fromMap(
    mapOf("target" to target)
)

fun main() {
    val inference = ScriptedInference(
        listOf(
            // First request: approval required, then execute, then respond.
            """{"steps":[{"kind":"EXTERNAL","objective":"echo","dependencies":[],"requires_verification":false,"min_confidence":0.0},{"kind":"RESPOND","objective":"answer","dependencies":[1],"requires_verification":false,"min_confidence":0.0}]}""",
            """{"capability_id":"test.echo","scope":{"target":"alpha"},"payload":{"message":"one"}}""",
            """{"result":"first complete","confidence":1.0}""",
            // Restored plan: same request must replay without another approval/effect.
            """{"steps":[{"kind":"EXTERNAL","objective":"echo","dependencies":[],"requires_verification":false,"min_confidence":0.0},{"kind":"RESPOND","objective":"answer","dependencies":[1],"requires_verification":false,"min_confidence":0.0}]}""",
            """{"capability_id":"test.echo","scope":{"target":"alpha"},"payload":{"message":"one"}}""",
            """{"result":"replayed complete","confidence":1.0}""",
            // Third request: explicit user rejection, with a different request digest.
            """{"steps":[{"kind":"EXTERNAL","objective":"echo","dependencies":[],"requires_verification":false,"min_confidence":0.0},{"kind":"RESPOND","objective":"answer","dependencies":[1],"requires_verification":false,"min_confidence":0.0}]}""",
            """{"capability_id":"test.echo","scope":{"target":"beta"},"payload":{"message":"two"}}""",
        )
    )
    val cognition = NativeTypedCognitionAdapter(inference)
    val capability = descriptor()
    val binder = M6ExternalIntentBinder(listOf(capability))
    val approvals = RecordingApprovalController()
    var effects = 0
    val registry = M6TypedCapabilityRegistry().also { registry ->
        registry.register(
            capability,
            M6CapabilityHandler { action ->
                effects += 1
                M6ActionOutcome(
                    success = true,
                    result = "echo:${action.request.scope.asMap().getValue("target")}",
                    confidence = 0.95,
                )
            },
        )
        registry.seal()
    }
    val audit = M6InMemoryActionAudit()
    val authority = M6DenyByDefaultAuthorityGate(
        grants = listOf(
            M6PolicyGrant(
                principal = "runtime.user",
                capabilityId = "test.echo",
                scopeDigest = scope("alpha").digest,
                approvalRequired = true,
            ),
            M6PolicyGrant(
                principal = "runtime.user",
                capabilityId = "test.echo",
                scopeDigest = scope("beta").digest,
                approvalRequired = true,
            ),
        ),
        approvals = approvals,
    )
    val fabric = M6ExternalExecutionFabric(registry, authority, audit)
    val coordinator = M6EndToEndExternalCoordinator(
        cognition = cognition,
        binder = binder,
        approvals = M6ExternalApprovalHandoff(approvals),
        executionFabric = fabric,
    )

    val first = coordinator.buildPlan("first", createdNs = 1L)
    val awaiting = coordinator.advance(
        controller = first,
        principal = "runtime.user",
        nowNs = 10L,
    )
    check(awaiting.event == M6ExternalCoordinatorEvent.APPROVAL_REQUIRED)
    val pending = checkNotNull(awaiting.pendingApproval)
    check(effects == 0)
    check(approvals.prompts == 1)
    check(audit.receipts.size == 1)
    check(audit.receipts.single().status == M6ReceiptStatus.DENIED)

    val approved = coordinator.resolveApproval(
        controller = first,
        pending = pending,
        approved = true,
        nowNs = 20L,
    )
    check(approved.event == M6ExternalCoordinatorEvent.EXECUTED)
    check(checkNotNull(approved.execution).replayed.not())
    check(approved.cognition.boundary == NativeCognitionBoundary.COMPLETED)
    check(approved.cognition.finalResponse == "first complete")
    check(effects == 1)
    check(audit.receipts.size == 2)
    check(audit.receipts.last().status == M6ReceiptStatus.SUCCEEDED)

    val restored = coordinator.buildPlan("first", createdNs = 2L)
    val replay = coordinator.advance(
        controller = restored,
        principal = "runtime.user",
        nowNs = 30L,
    )
    check(replay.event == M6ExternalCoordinatorEvent.EXECUTED)
    check(checkNotNull(replay.execution).replayed)
    check(replay.cognition.boundary == NativeCognitionBoundary.COMPLETED)
    check(replay.cognition.finalResponse == "replayed complete")
    check(effects == 1)
    check(approvals.prompts == 1)
    check(audit.receipts.size == 2)

    val rejectedPlan = coordinator.buildPlan("third", createdNs = 3L)
    val rejectAwaiting = coordinator.advance(
        controller = rejectedPlan,
        principal = "runtime.user",
        nowNs = 40L,
    )
    val rejectPending = checkNotNull(rejectAwaiting.pendingApproval)
    val externalIntentCallsBeforeReject = inference.calls.count {
        it == NativeCognitionOperation.EXTERNAL_INTENT
    }
    val rejected = coordinator.resolveApproval(
        controller = rejectedPlan,
        pending = rejectPending,
        approved = false,
        nowNs = 41L,
    )
    check(rejected.event == M6ExternalCoordinatorEvent.APPROVAL_REJECTED)
    check(rejected.cognition.boundary == NativeCognitionBoundary.FAILED)
    check(effects == 1)
    check(
        inference.calls.count { it == NativeCognitionOperation.EXTERNAL_INTENT } ==
            externalIntentCallsBeforeReject
    )

    // M14B invariant: one coordinator advance can execute at most one external action.
    val frameInference = ScriptedInference(
        listOf(
            """{"steps":[{"kind":"EXTERNAL","objective":"first frame action","dependencies":[],"requires_verification":false,"min_confidence":0.0},{"kind":"EXTERNAL","objective":"second action requires next frame","dependencies":[1],"requires_verification":false,"min_confidence":0.0},{"kind":"RESPOND","objective":"finish","dependencies":[2],"requires_verification":false,"min_confidence":0.0}]}""",
            """{"capability_id":"test.frame","scope":{"target":"alpha"},"payload":{}}""",
        )
    )
    val frameCognition = NativeTypedCognitionAdapter(frameInference)
    val frameDescriptor = M6CapabilityDescriptor(
        capabilityId = "test.frame",
        requiredScopeKeys = setOf("target"),
        approvalRequired = false,
        maxPayloadUtf8Bytes = 16,
        maxLeaseNs = 10_000_000_000L,
        maxLeaseUses = 1,
        payloadSchemaJson = "{}",
    )
    var frameEffects = 0
    val frameRegistry = M6TypedCapabilityRegistry().also { registry ->
        registry.register(
            frameDescriptor,
            M6CapabilityHandler {
                frameEffects += 1
                M6ActionOutcome(true, "frame-action")
            },
        )
        registry.seal()
    }
    val frameScope =
        M6CapabilityScope.fromMap(mapOf("target" to "alpha"))
    val frameCoordinator = M6EndToEndExternalCoordinator(
        cognition = frameCognition,
        binder = M6ExternalIntentBinder(
            listOf(frameDescriptor)
        ),
        approvals = M6ExternalApprovalHandoff(
            RecordingApprovalController()
        ),
        executionFabric = M6ExternalExecutionFabric(
            frameRegistry,
            M6DenyByDefaultAuthorityGate(
                listOf(
                    M6PolicyGrant(
                        principal = "runtime.user",
                        capabilityId = "test.frame",
                        scopeDigest = frameScope.digest,
                        approvalRequired = false,
                        maxLeaseNs = 10_000_000_000L,
                        maxLeaseUses = 1,
                    )
                ),
                RecordingApprovalController(),
            ),
            M6InMemoryActionAudit(),
        ),
    )
    val frameController =
        frameCoordinator.buildPlan(
            "strict one-action frame",
            createdNs = 50L,
        )
    val firstFrame = frameCoordinator.advance(
        controller = frameController,
        principal = "runtime.user",
        maxCycles = 8,
        nowNs = 51L,
    )
    check(firstFrame.event == M6ExternalCoordinatorEvent.EXECUTED)
    check(firstFrame.execution != null)
    check(frameEffects == 1)
    check(
        firstFrame.cognition.boundary ==
            NativeCognitionBoundary.WAITING_EXTERNAL
    )
    check(
        frameController.plan.step(2).status ==
            NativeStepStatus.WAITING_EXTERNAL
    )
    check(
        frameInference.calls.count {
            it == NativeCognitionOperation.EXTERNAL_INTENT
        } == 1
    )

    println("M7S_END_TO_END_EXTERNAL_COORDINATOR_PASS")
}
