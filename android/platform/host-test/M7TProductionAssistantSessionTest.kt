import ai.vn97.platform.*
import ai.vn97.runtime.*

private class ScriptedInference(outputs: List<String>) : NativeCognitionInference {
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
            promptId = "%032x".format(prompts),
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
                approvalId = "%032x".format(resolves + 100),
                issuer = "local-user",
                principal = prompt.principal,
                requestDigest = prompt.requestDigest,
                issuedNs = nowNs,
                expiresNs = nowNs + approvalTtlNs,
                signature = "00".repeat(32),
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

private inline fun expectStateFailure(block: () -> Unit) {
    var failed = false
    try {
        block()
    } catch (_: IllegalStateException) {
        failed = true
    }
    check(failed)
}

fun main() {
    val planJson =
        """{"steps":[{"kind":"EXTERNAL","objective":"echo","dependencies":[],"requires_verification":false,"min_confidence":0.0},{"kind":"RESPOND","objective":"answer","dependencies":[1],"requires_verification":false,"min_confidence":0.0}]}"""
    val intentJson =
        """{"capability_id":"test.echo","scope":{"target":"alpha"},"payload":{"message":"one"}}"""
    val inference = ScriptedInference(
        listOf(
            planJson,
            intentJson,
            """{"result":"approved final","confidence":1.0}""",
            // Restored planner: successful M7P/M7Q-style audit replay, no new approval/effect.
            intentJson,
            """{"result":"replayed final","confidence":1.0}""",
            // Explicit rejection turn.
            planJson,
            """{"capability_id":"test.echo","scope":{"target":"beta"},"payload":{"message":"two"}}""",
        )
    )
    val cognition = NativeTypedCognitionAdapter(inference)
    val descriptor = M6CapabilityDescriptor(
        capabilityId = "test.echo",
        requiredScopeKeys = setOf("target"),
        approvalRequired = true,
        maxLeaseNs = 30_000_000_000L,
        maxLeaseUses = 1,
    )
    val binder = M6ExternalIntentBinder(listOf(descriptor))
    val approvals = RecordingApprovalController()
    var effects = 0
    val registry = M6TypedCapabilityRegistry().also { r ->
        r.register(
            descriptor,
            M6CapabilityHandler { action ->
                effects += 1
                M6ActionOutcome(
                    success = true,
                    result = "echo:${action.request.scope.asMap().getValue("target")}",
                    confidence = 0.95,
                )
            },
        )
        r.seal()
    }
    val alphaScope = M6CapabilityScope.fromMap(mapOf("target" to "alpha"))
    val betaScope = M6CapabilityScope.fromMap(mapOf("target" to "beta"))
    val authority = M6DenyByDefaultAuthorityGate(
        grants = listOf(
            M6PolicyGrant(
                principal = "runtime.user",
                capabilityId = "test.echo",
                scopeDigest = alphaScope.digest,
                approvalRequired = true,
            ),
            M6PolicyGrant(
                principal = "runtime.user",
                capabilityId = "test.echo",
                scopeDigest = betaScope.digest,
                approvalRequired = true,
            ),
        ),
        approvals = approvals,
    )
    val audit = M6InMemoryActionAudit()
    val fabric = M6ExternalExecutionFabric(registry, authority, audit)
    val coordinator = M6EndToEndExternalCoordinator(
        cognition = cognition,
        binder = binder,
        approvals = M6ExternalApprovalHandoff(approvals),
        executionFabric = fabric,
    )
    val session = VN97AssistantSession(coordinator)

    val waiting = session.startTurn(
        userMessage = "first",
        principal = "runtime.user",
        createdNs = 1L,
        nowNs = 10L,
    )
    check(waiting.state == VN97AssistantTurnState.APPROVAL_REQUIRED)
    check(session.hasActiveTurn)
    check(effects == 0)
    check(approvals.prompts == 1)
    check(audit.receipts.single().status == M6ReceiptStatus.DENIED)
    expectStateFailure {
        session.startTurn("parallel", "runtime.user", createdNs = 2L, nowNs = 11L)
    }

    val approved = session.resolveApproval(
        turn = waiting.turn,
        approval = checkNotNull(waiting.approval),
        approved = true,
        nowNs = 20L,
    )
    check(approved.state == VN97AssistantTurnState.COMPLETED)
    check(approved.finalResponse == "approved final")
    check(approved.executions.size == 1)
    check(!approved.executions.single().replayed)
    check(effects == 1)
    check(approvals.prompts == 1)
    check(!session.hasActiveTurn)
    check(audit.receipts.last().status == M6ReceiptStatus.SUCCEEDED)

    // Recreate the exact immutable planner definition and enter WAITING_EXTERNAL as a restored plan.
    val restored = NativePlanController.create(
        goal = "first",
        specs = listOf(
            NativePlanStepSpec(NativeStepKind.EXTERNAL, "echo"),
            NativePlanStepSpec(NativeStepKind.RESPOND, "answer", dependencies = listOf(1)),
        ),
        createdNs = 99L,
    )
    restored.beginStep(1)
    check(restored.plan.planId == waiting.turn.planId)
    val replay = session.resumeRestoredTurn(
        controller = restored,
        principal = "runtime.user",
        nowNs = 30L,
    )
    check(replay.state == VN97AssistantTurnState.COMPLETED)
    check(replay.finalResponse == "replayed final")
    check(replay.executions.single().replayed)
    check(effects == 1)
    check(approvals.prompts == 1)
    check(audit.receipts.size == 2)

    val rejectWait = session.startTurn(
        userMessage = "reject",
        principal = "runtime.user",
        createdNs = 3L,
        nowNs = 40L,
    )
    check(rejectWait.state == VN97AssistantTurnState.APPROVAL_REQUIRED)
    check(approvals.prompts == 2)
    val callsBeforeReject = inference.calls.count {
        it == NativeCognitionOperation.EXTERNAL_INTENT
    }
    val rejected = session.resolveApproval(
        turn = rejectWait.turn,
        approval = checkNotNull(rejectWait.approval),
        approved = false,
        nowNs = 41L,
    )
    check(rejected.state == VN97AssistantTurnState.APPROVAL_REJECTED)
    check(effects == 1)
    check(!session.hasActiveTurn)
    check(
        inference.calls.count { it == NativeCognitionOperation.EXTERNAL_INTENT } ==
            callsBeforeReject
    )

    check(
        inference.calls == listOf(
            NativeCognitionOperation.PLAN,
            NativeCognitionOperation.EXTERNAL_INTENT,
            NativeCognitionOperation.STEP,
            NativeCognitionOperation.EXTERNAL_INTENT,
            NativeCognitionOperation.STEP,
            NativeCognitionOperation.PLAN,
            NativeCognitionOperation.EXTERNAL_INTENT,
        )
    )

    println("M7T_PRODUCTION_ASSISTANT_TURN_SESSION_PASS")
}
