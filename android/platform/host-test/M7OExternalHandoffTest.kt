import ai.vn97.platform.*
import ai.vn97.runtime.*

private class RecordingApprovalController : M6ApprovalControllerPort {
    var lastPresentation = ""
    var lastDigest = ""

    override fun createPrompt(
        requestDigest: String,
        principal: String,
        presentationJson: String,
        promptTtlNs: Long,
        nowNs: Long,
    ): M6ApprovalPrompt {
        lastDigest = requestDigest
        lastPresentation = presentationJson
        return M6ApprovalPrompt(
            promptId = "00112233445566778899aabbccddeeff",
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
    ): M6ApprovalToken? = if (approved) {
        M6ApprovalToken(
            approvalId = "00112233445566778899aabbccddeeff",
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

private inline fun expectContract(block: () -> Unit) {
    var failed = false
    try {
        block()
    } catch (_: M6ExternalIntentContractException) {
        failed = true
    }
    check(failed)
}

fun main() {
    val controller = NativePlanController(
        NativePlan(
            planId = "ab".repeat(32),
            goal = "perform external action",
            steps = listOf(
                NativePlannerStep(
                    1,
                    NativePlanStepSpec(NativeStepKind.REASON, "prepare"),
                    NativeStepStatus.SUCCEEDED,
                ),
                NativePlannerStep(
                    2,
                    NativePlanStepSpec(NativeStepKind.REASON, "prepare again"),
                    NativeStepStatus.SUCCEEDED,
                ),
                NativePlannerStep(
                    3,
                    NativePlanStepSpec(
                        NativeStepKind.EXTERNAL,
                        "send exact external action",
                    ),
                    NativeStepStatus.WAITING_EXTERNAL,
                ),
            ),
            status = NativePlanStatus.WAITING_EXTERNAL,
        )
    )

    val descriptor = M6CapabilityDescriptor(
        capabilityId = "test.echo",
        requiredScopeKeys = setOf("target"),
        optionalScopeKeys = setOf("mode"),
        approvalRequired = true,
        maxPayloadUtf8Bytes = 4096,
    )
    val binder = M6ExternalIntentBinder(listOf(descriptor))
    val intent = NativeExternalIntent(
        capabilityId = "test.echo",
        scope = linkedMapOf(
            "target" to "alpha",
            "mode" to "fast",
        ),
        payloadJson = """{"message":"hello","n":1}""",
    )

    val request = binder.bind(controller, intent)
    check(
        request.scope.entries ==
            listOf("mode" to "fast", "target" to "alpha")
    )
    check(
        request.scope.digest ==
            "c1ccf2848b5cc4a0aa30dc1611c869d85205af7a61bc52a2520078e6572b02eb"
    )
    check(
        request.requestDigest ==
            "6c9db355eba8988e1ccc8df7515b03b10cb75c4e57e66fb8c62881b581e54184"
    )

    val expectedPresentation =
        """{"capability_id":"test.echo","objective":"send exact external action","payload":{"message":"hello","n":1},"plan_id":"abababababababababababababababababababababababababababababababab","request_digest":"6c9db355eba8988e1ccc8df7515b03b10cb75c4e57e66fb8c62881b581e54184","scope":{"mode":"fast","target":"alpha"},"step_id":3}"""
    check(request.approvalPresentationJson() == expectedPresentation)

    val backend = NativeTypedCognitionAdapter(intent)
    check(
        binder.proposeAndBind(controller, backend).requestDigest ==
            request.requestDigest
    )

    expectContract {
        binder.bind(
            controller,
            NativeExternalIntent(
                "test.other",
                mapOf("target" to "alpha"),
                "{}",
            ),
        )
    }
    expectContract {
        binder.bind(
            controller,
            NativeExternalIntent(
                "test.echo",
                mapOf("wrong" to "alpha"),
                "{}",
            ),
        )
    }

    val approvals = RecordingApprovalController()
    val handoff = M6ExternalApprovalHandoff(approvals)
    val prompt = handoff.createPrompt(
        request,
        principal = "runtime.user",
        promptTtlNs = 1000L,
        nowNs = 10L,
    )
    check(prompt.requestDigest == request.requestDigest)
    check(approvals.lastDigest == request.requestDigest)
    check(approvals.lastPresentation == expectedPresentation)
    val token = checkNotNull(
        handoff.resolve(
            prompt,
            approved = true,
            approvalTtlNs = 100L,
            nowNs = 20L,
        )
    )
    handoff.verify(token, request, "runtime.user", 21L)

    println("M7O_EXTERNAL_HANDOFF_PASS")
}
