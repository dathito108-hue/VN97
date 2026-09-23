import ai.vn97.runtime.*

private class ScriptedInference(
    outputs: List<String>,
    private val embedding: FloatArray = floatArrayOf(3f, 4f),
) : NativeCognitionInference {
    private val queue = ArrayDeque(outputs)
    val calls = mutableListOf<Pair<NativeCognitionOperation, String>>()

    override fun generateOperation(
        operation: NativeCognitionOperation,
        requestJson: String,
    ): String {
        calls += operation to requestJson
        check(queue.isNotEmpty()) { "unexpected cognition call: $operation" }
        return queue.removeFirst()
    }

    override fun embedText(text: String, vectorDim: Int): FloatArray {
        check(vectorDim == embedding.size)
        return embedding.copyOf()
    }
}

private class FakeMemory : NativeMemoryRetriever {
    override val vectorDim: Int = 2
    var calls = 0

    override fun retrieve(
        query: NativeMemoryQuery,
        topK: Int,
    ): List<NativeMemoryContextItem> {
        calls += 1
        check(topK == 2)
        check(query.vector.contentEquals(floatArrayOf(3f, 4f)))
        return listOf(
            NativeMemoryContextItem(
                recordId = 42L,
                content = "memory fact",
                source = "unit",
                score = 1.0,
                semanticScore = 1.0,
                recencyScore = 0.5,
                importanceScore = 0.8,
            )
        )
    }
}

private inline fun expectContract(block: () -> Unit) {
    var failed = false
    try {
        block()
    } catch (_: NativeCognitionContractException) {
        failed = true
    }
    check(failed)
}

fun main() {
    val inference = ScriptedInference(
        listOf(
            """{"steps":[{"kind":"REASON","objective":"analyze","dependencies":[],"requires_verification":false,"min_confidence":0.0},{"kind":"RETRIEVE","objective":"retrieve","dependencies":[1],"requires_verification":false,"min_confidence":0.0},{"kind":"RESPOND","objective":"answer","dependencies":[1,2],"requires_verification":false,"min_confidence":0.0}]}""",
            """{"result":"reasoned","confidence":0.9}""",
            """{"query":"memory evidence","top_k":2,"kinds":["SEMANTIC"],"semantic_weight":1.0,"recency_weight":0.0,"importance_weight":0.0,"recency_half_life_ns":100}""",
            """{"result":"retrieved","confidence":0.95}""",
            """{"result":"final answer","confidence":1.0}""",
        )
    )
    val loop = NativeCognitionLoop(
        NativeTypedCognitionAdapter(inference)
    )
    val controller = loop.buildPlan("answer", createdNs = 1L)
    val memory = FakeMemory()
    val run = loop.runUntilBoundary(controller, memory)
    check(run.boundary == NativeCognitionBoundary.COMPLETED)
    check(run.planStatus == NativePlanStatus.COMPLETED)
    check(run.cycles == 3)
    check(run.finalResponse == "final answer")
    check(memory.calls == 1)
    check(controller.plan.step(2).evidenceRecordIds == listOf(42L))
    check(controller.plan.step(3).evidenceRecordIds == listOf(42L))
    check(inference.calls.map { it.first } == listOf(
        NativeCognitionOperation.PLAN,
        NativeCognitionOperation.STEP,
        NativeCognitionOperation.MEMORY_QUERY,
        NativeCognitionOperation.STEP,
        NativeCognitionOperation.STEP,
    ))

    val verifyInference = ScriptedInference(
        listOf(
            """{"steps":[{"kind":"REASON","objective":"candidate","dependencies":[],"requires_verification":true,"min_confidence":0.0},{"kind":"RESPOND","objective":"respond","dependencies":[1],"requires_verification":false,"min_confidence":0.0}]}""",
            """{"result":"candidate","confidence":0.8}""",
            """{"passed":true,"note":"verified"}""",
            """{"result":"verified response","confidence":1.0}""",
        )
    )
    val verifyLoop = NativeCognitionLoop(
        NativeTypedCognitionAdapter(verifyInference)
    )
    val verifyController = verifyLoop.buildPlan("verify", createdNs = 2L)
    val yielded = verifyLoop.runUntilBoundary(
        verifyController,
        maxCycles = 1,
    )
    check(yielded.boundary == NativeCognitionBoundary.YIELDED)
    check(
        verifyController.plan.step(1).status ==
            NativeStepStatus.WAITING_VERIFICATION
    )
    val verified = verifyLoop.runUntilBoundary(verifyController)
    check(verified.boundary == NativeCognitionBoundary.COMPLETED)
    check(verified.cycles == 2)
    check(verified.finalResponse == "verified response")

    val externalInference = ScriptedInference(
        listOf(
            """{"steps":[{"kind":"EXTERNAL","objective":"write","dependencies":[],"requires_verification":false,"min_confidence":0.0},{"kind":"RESPOND","objective":"respond","dependencies":[1],"requires_verification":false,"min_confidence":0.0}]}"""
        )
    )
    val externalLoop = NativeCognitionLoop(
        NativeTypedCognitionAdapter(externalInference)
    )
    val externalController = externalLoop.buildPlan(
        "external",
        createdNs = 3L,
    )
    val waiting = externalLoop.runUntilBoundary(externalController)
    check(waiting.boundary == NativeCognitionBoundary.WAITING_EXTERNAL)
    check(waiting.cycles == 1)
    check(
        externalController.plan.step(1).status ==
            NativeStepStatus.WAITING_EXTERNAL
    )

    val gameIntentInference = ScriptedInference(
        listOf(
            """{"capability_id":"device.game.tap","scope":{"package":"com.example.game"},"payload":{"duration_ms":80,"x_bps":5000,"y_bps":5000}}"""
        )
    )
    val gameIntentAdapter =
        NativeTypedCognitionAdapter(gameIntentInference)
    val gameIntent = gameIntentAdapter.proposeExternalIntent(
        NativeExternalIntentRequest(
            planId = "aa".repeat(32),
            goal = "play the current game",
            stepId = 1,
            objective = "tap the visible primary control",
            capabilities = listOf(
                NativeExternalCapabilityView(
                    capabilityId = "device.game.tap",
                    requiredScopeKeys = listOf("package"),
                    optionalScopeKeys = emptyList(),
                    approvalRequired = false,
                    maxPayloadUtf8Bytes = 96,
                    payloadSchemaJson =
                        "{\"duration_ms\":80,\"x_bps\":5000,\"y_bps\":5000}",
                )
            ),
        )
    )
    check(gameIntent.capabilityId == "device.game.tap")
    check(gameIntent.scope == mapOf("package" to "com.example.game"))
    check(
        gameIntent.payloadJson ==
            "{\"duration_ms\":80,\"x_bps\":5000,\"y_bps\":5000}"
    )
    val gameIntentRequestJson =
        gameIntentInference.calls.single().second
    check(
        "\"payload_schema\":{\"duration_ms\":80,\"x_bps\":5000,\"y_bps\":5000}" in
            gameIntentRequestJson
    )

    val invalidInference = ScriptedInference(
        listOf(
            """{"steps":[{"kind":"REASON","objective":"no final response","dependencies":[],"requires_verification":false,"min_confidence":0.0}]}"""
        )
    )
    expectContract {
        NativeCognitionLoop(
            NativeTypedCognitionAdapter(invalidInference)
        ).buildPlan("invalid", createdNs = 4L)
    }

    val failedController = NativePlanController.create(
        goal = "recover with a new plan",
        specs = listOf(
            NativePlanStepSpec(
                kind = NativeStepKind.REASON,
                objective = "old approach",
            ),
            NativePlanStepSpec(
                kind = NativeStepKind.RESPOND,
                objective = "old response",
                dependencies = listOf(1),
            ),
        ),
        createdNs = 5L,
    )
    failedController.beginStep(1)
    failedController.failStep(
        id = 1,
        reason = "old approach failed",
        retryable = false,
    )
    check(failedController.plan.status == NativePlanStatus.FAILED)

    val replanInference = ScriptedInference(
        listOf(
            """{"steps":[{"kind":"REASON","objective":"materially different approach","dependencies":[],"requires_verification":false,"min_confidence":0.0},{"kind":"RESPOND","objective":"new response","dependencies":[1],"requires_verification":false,"min_confidence":0.0}]}"""
        )
    )
    val replanLoop = NativeCognitionLoop(
        NativeTypedCognitionAdapter(replanInference)
    )
    val revision = replanLoop.replanTerminalPlan(
        previousPlan = failedController.plan,
        feedback = "failure evidence: old approach failed",
        createdNs = 6L,
    )
    check(revision.previousPlanId == failedController.plan.planId)
    check(revision.controller.plan.goal == failedController.plan.goal)
    check(revision.controller.plan.planId != failedController.plan.planId)
    check(revision.controller.plan.status == NativePlanStatus.READY)
    check(revision.controller.plan.transitionsUsed == 0)
    check(revision.controller.plan.memoryQueriesUsed == 0)
    check(
        revision.controller.plan.budget ==
            failedController.plan.budget
    )

    var nonterminalRejected = false
    try {
        replanLoop.replanTerminalPlan(
            previousPlan = revision.controller.plan,
            feedback = "not terminal",
            createdNs = 7L,
        )
    } catch (_: IllegalArgumentException) {
        nonterminalRejected = true
    }
    check(nonterminalRejected)

    println("M7N_COGNITION_LOOP_PASS")
}
