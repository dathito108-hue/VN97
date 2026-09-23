import ai.vn97.platform.*
import ai.vn97.runtime.*
import java.nio.file.Files

private class ScriptedInference(outputs: List<String>) : NativeCognitionInference {
    private val queue = ArrayDeque(outputs)

    override fun generateOperation(
        operation: NativeCognitionOperation,
        requestJson: String,
    ): String {
        check(requestJson.isNotEmpty())
        check(queue.isNotEmpty()) { "unexpected cognition call: $operation" }
        return queue.removeFirst()
    }

    override fun embedText(text: String, vectorDim: Int): FloatArray {
        check(text.isNotEmpty())
        return FloatArray(vectorDim) { 1.0f }
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

private class FakeTurnMemoryBackend : VN97TurnMemoryBackend {
    override val vectorDim = 4
    private val records = linkedMapOf<Long, NativeMemoryRecord>()
    var appendCalls = 0
    var failAppend = false

    override fun stats() = NativeMemoryStoreStats(
        vectorDim = vectorDim,
        recordCount = records.size.toLong(),
        lastRecordId = records.keys.maxOrNull() ?: 0L,
        validBytes = 0L,
    )

    override fun append(
        kind: NativeMemoryKind,
        timestampNs: Long,
        importance: Float,
        source: String,
        content: String,
        vector: FloatArray,
    ): Long {
        appendCalls += 1
        if (failAppend) error("injected persistence failure")
        check(vector.size == vectorDim)
        val id = (records.keys.maxOrNull() ?: 0L) + 1L
        records[id] = NativeMemoryRecord(
            id,
            timestampNs,
            0L,
            kind,
            importance,
            source,
            content,
        )
        return id
    }

    override fun record(recordId: Long): NativeMemoryRecord =
        records[recordId] ?: error("missing record")
}

private fun coordinator(response: String): M6EndToEndExternalCoordinator {
    val inference = ScriptedInference(
        listOf(
            """{"steps":[{"kind":"RESPOND","objective":"answer","dependencies":[],"requires_verification":false,"min_confidence":0.0}]}""",
            """{"result":"$response","confidence":1.0}""",
        )
    )
    val descriptor = M6CapabilityDescriptor(
        capabilityId = "test.unused",
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
    return M6EndToEndExternalCoordinator(
        cognition = NativeTypedCognitionAdapter(inference),
        binder = M6ExternalIntentBinder(listOf(descriptor)),
        approvals = M6ExternalApprovalHandoff(NoopApprovalController),
        executionFabric = M6ExternalExecutionFabric(
            registry = registry,
            authority = M6DenyByDefaultAuthorityGate(
                grants = emptyList(),
                approvals = NoopApprovalController,
            ),
            audit = M6InMemoryActionAudit(),
        ),
    )
}

fun main() {
    val embedder = VN97TurnMemoryEmbedder { _, dim -> FloatArray(dim) { 1.0f } }

    val okBackend = FakeTurnMemoryBackend()
    val okRoot = Files.createTempDirectory("m7x-ok-").toFile()
    VN97TurnMemoryWriter(okBackend, embedder, okRoot).use { writer ->
        val session = VN97AssistantSession(
            coordinator = coordinator("stored"),
            turnMemoryWriter = writer,
        )
        val completed = session.startTurn(
            userMessage = "remember this",
            principal = "runtime.user",
            createdNs = 10L,
            nowNs = 100L,
        )
        check(completed.state == VN97AssistantTurnState.COMPLETED)
        check(completed.finalResponse == "stored")
        check(completed.memoryCommit?.state == VN97AssistantMemoryCommitState.COMMITTED)
        check(completed.memoryCommit?.recordId == 1L)
        check(okBackend.appendCalls == 1)
        check(!session.hasActiveTurn)

        val alreadyCommitted = session.retryCompletedTurnMemoryCommit(
            update = completed,
            timestampNs = 101L,
        )
        check(alreadyCommitted == completed)
        check(okBackend.appendCalls == 1)
    }

    val flakyBackend = FakeTurnMemoryBackend().also { it.failAppend = true }
    val flakyRoot = Files.createTempDirectory("m7x-flaky-").toFile()
    VN97TurnMemoryWriter(flakyBackend, embedder, flakyRoot).use { writer ->
        val session = VN97AssistantSession(
            coordinator = coordinator("delivered"),
            turnMemoryWriter = writer,
        )
        val delivered = session.startTurn(
            userMessage = "preserve answer",
            principal = "runtime.user",
            createdNs = 20L,
            nowNs = 200L,
        )
        check(delivered.state == VN97AssistantTurnState.COMPLETED)
        check(delivered.finalResponse == "delivered")
        check(delivered.memoryCommit?.state == VN97AssistantMemoryCommitState.RETRY_REQUIRED)
        check(delivered.memoryCommit?.recordId == null)
        check(delivered.memoryCommit?.failureType?.contains("IllegalStateException") == true)
        check(writer.pendingTurnKey != null)
        check(!session.hasActiveTurn)
        check(flakyBackend.appendCalls == 1)

        flakyBackend.failAppend = false
        val recovered = session.retryCompletedTurnMemoryCommit(
            update = delivered,
            timestampNs = 201L,
        )
        check(recovered.state == VN97AssistantTurnState.COMPLETED)
        check(recovered.finalResponse == "delivered")
        check(recovered.memoryCommit?.state == VN97AssistantMemoryCommitState.COMMITTED)
        check(recovered.memoryCommit?.recordId == 1L)
        check(writer.pendingTurnKey == null)
        check(flakyBackend.appendCalls == 2)
    }

    println("M7X_COMPLETED_TURN_MEMORY_ORCHESTRATION_PASS")
}
