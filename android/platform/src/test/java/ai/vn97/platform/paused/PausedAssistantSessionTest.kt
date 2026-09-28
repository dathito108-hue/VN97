package ai.vn97.platform.paused

import ai.vn97.platform.*
import ai.vn97.runtime.*

private class ScriptedInference(
    outputs: List<String>,
) : NativeCognitionInference {
    private val queue = ArrayDeque(outputs)
    var calls = 0

    override fun generateOperation(
        operation: NativeCognitionOperation,
        requestJson: String,
    ): String {
        check(requestJson.isNotEmpty())
        calls += 1
        check(queue.isNotEmpty()) { "unexpected cognition call: $operation" }
        return queue.removeFirst()
    }

    override fun embedText(text: String, vectorDim: Int): FloatArray {
        check(text.isNotBlank())
        return FloatArray(vectorDim) { 1.0f }
    }
}

private class RecordingMemory : NativeMemoryRetriever {
    override val vectorDim: Int = 4
    var calls = 0

    override fun retrieve(
        query: NativeMemoryQuery,
        topK: Int,
    ): List<NativeMemoryContextItem> {
        check(query.vector.size == vectorDim)
        check(topK == 1)
        calls += 1
        return listOf(
            NativeMemoryContextItem(
                recordId = 7L,
                content = "remembered",
                source = "m7v-test",
                score = 1.0,
                semanticScore = 1.0,
                recencyScore = 0.0,
                importanceScore = 0.5,
            )
        )
    }
}

private object NoopApprovalController : M6ApprovalControllerPort {
    override fun createPrompt(
        requestDigest: String,
        principal: String,
        presentationJson: String,
        promptTtlNs: Long,
        nowNs: Long,
    ): M6ApprovalPrompt = error("approval is not expected")

    override fun resolve(
        prompt: M6ApprovalPrompt,
        approved: Boolean,
        approvalTtlNs: Long,
        nowNs: Long,
    ): M6ApprovalToken? = error("approval is not expected")

    override fun verify(
        token: M6ApprovalToken,
        requestDigest: String,
        principal: String,
        nowNs: Long,
    ) = error("approval is not expected")
}

private inline fun expectState(block: () -> Unit) {
    var failed = false
    try {
        block()
    } catch (_: IllegalStateException) {
        failed = true
    }
    check(failed)
}

private fun coordinator(
    cognition: NativeTypedCognitionAdapter,
): M6EndToEndExternalCoordinator {
    val descriptor = M6CapabilityDescriptor(
        capabilityId = "test.echo",
        requiredScopeKeys = setOf("target"),
    )
    val registry = M6TypedCapabilityRegistry().also {
        it.register(
            descriptor,
            M6CapabilityHandler {
                M6ActionOutcome(success = true, result = "unused")
            },
        )
        it.seal()
    }
    val fabric = M6ExternalExecutionFabric(
        registry = registry,
        authority = M6DenyByDefaultAuthorityGate(
            grants = emptyList(),
            approvals = NoopApprovalController,
        ),
        audit = M6InMemoryActionAudit(),
    )
    return M6EndToEndExternalCoordinator(
        cognition = cognition,
        binder = M6ExternalIntentBinder(listOf(descriptor)),
        approvals = M6ExternalApprovalHandoff(NoopApprovalController),
        executionFabric = fabric,
    )
}

private fun exercisePausedSession() {
    val inference = ScriptedInference(
        listOf(
            """{"steps":[{"kind":"RETRIEVE","objective":"recall","dependencies":[],"requires_verification":false,"min_confidence":0.0},{"kind":"RESPOND","objective":"answer","dependencies":[1],"requires_verification":false,"min_confidence":0.0}]}""",
            """{"query":"needle","top_k":1,"kinds":["SEMANTIC"],"semantic_weight":1.0,"recency_weight":0.0,"importance_weight":0.0,"recency_half_life_ns":1000000000}""",
            """{"result":"memory-used","confidence":1.0}""",
            """{"result":"done","confidence":1.0}""",
        )
    )
    val memory = RecordingMemory()
    val foreignMemory = RecordingMemory()
    val session = VN97AssistantSession(
        coordinator = coordinator(NativeTypedCognitionAdapter(inference)),
        limits = VN97AssistantSessionLimits(maxCyclesPerAdvance = 1),
        defaultMemory = memory,
    )

    val yielded = session.startTurn(
        userMessage = "use sovereign memory",
        principal = "runtime.user",
        createdNs = 1L,
        nowNs = 1L,
    )
    check(yielded.state == VN97AssistantTurnState.YIELDED)
    check(memory.calls == 1)


    val callsAtYield = inference.calls
    expectState { session.resumePausedTurn(yielded.turn, nowNs = 2L) }
    check(session.hasActiveTurn && inference.calls == callsAtYield)
    yielded.turn.controller.pause("test pause after first completed step")
    val paused = session.continueTurn(yielded.turn, nowNs = 2L)
    check(paused.state == VN97AssistantTurnState.PAUSED)
    check(paused.turn === yielded.turn && session.hasActiveTurn)
    check(inference.calls == callsAtYield && memory.calls == 1)

    expectState { session.resumePausedTurn(paused.turn, memory = foreignMemory, nowNs = 3L) }
    check(runCatching { session.resumePausedTurn(paused.turn, nowNs = -1L) }
        .exceptionOrNull() is IllegalArgumentException)
    val impostor = VN97AssistantTurn(999L, paused.turn.controller, "runtime.user")
    expectState { session.resumePausedTurn(impostor, nowNs = 3L) }
    check(paused.turn.controller.plan.status == NativePlanStatus.PAUSED)
    check(inference.calls == callsAtYield && memory.calls == 1)

    val completed = session.resumePausedTurn(paused.turn, nowNs = 4L)
    check(completed.state == VN97AssistantTurnState.COMPLETED)
    check(completed.turn === paused.turn && completed.finalResponse == "done")
    check(memory.calls == 1) // Previously completed retrieval must not run again.
    check(!session.hasActiveTurn)
    val callsAtCompletion = inference.calls
    expectState { session.resumePausedTurn(completed.turn, nowNs = 5L) }
    check(inference.calls == callsAtCompletion)
}

class PausedAssistantSessionTest {
    @org.junit.Test fun resumePreservesIdentityMemoryAndCompletedSteps() { exercisePausedSession() }
}
