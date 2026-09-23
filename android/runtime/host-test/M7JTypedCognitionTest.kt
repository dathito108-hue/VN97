import ai.vn97.runtime.*

private class ScriptedInference(
    private val outputs: MutableList<String>,
    private val embedding: FloatArray = floatArrayOf(3.0f, 4.0f),
) : NativeCognitionInference {
    val calls = mutableListOf<Pair<NativeCognitionOperation, String>>()
    val embeddings = mutableListOf<Pair<String, Int>>()

    override fun generateOperation(
        operation: NativeCognitionOperation,
        requestJson: String,
    ): String {
        calls += operation to requestJson
        check(outputs.isNotEmpty()) { "unexpected cognition generation" }
        return outputs.removeAt(0)
    }

    override fun embedText(text: String, vectorDim: Int): FloatArray {
        embeddings += text to vectorDim
        return embedding.copyOf()
    }
}

private inline fun expectContract(block: () -> Unit) {
    var failed = false
    try {
        block()
    } catch (_: NativeCognitionContractException) {
        failed = true
    }
    check(failed) { "expected NativeCognitionContractException" }
}

fun main() {
    val inference = ScriptedInference(
        mutableListOf(
            "{\"steps\":[{\"kind\":\"REASON\",\"objective\":\"think\",\"dependencies\":[],\"requires_verification\":true,\"min_confidence\":0.75}]}",
            "{\"query\":\"memory evidence\",\"top_k\":2,\"kinds\":[\"SEMANTIC\"],\"semantic_weight\":1.0,\"recency_weight\":0.0,\"importance_weight\":0.0,\"recency_half_life_ns\":100}",
            "{\"result\":\"candidate\",\"confidence\":0.8}",
            "{\"passed\":true,\"note\":\"supported\"}",
            "{\"capability_id\":\"file.write\",\"scope\":{\"root\":\"app\",\"path\":\"note.txt\"},\"payload\":{\"z\":1,\"a\":\"x\"}}",
        )
    )
    val adapter = NativeTypedCognitionAdapter(inference)

    val plan = adapter.proposePlan(
        NativePlanDraftRequest(
            goal = "answer",
            maxSteps = 4,
        )
    )
    check(plan.steps.size == 1)
    check(plan.steps[0].kind == NativeStepKind.REASON)
    check(plan.steps[0].requiresVerification)
    check(plan.steps[0].minConfidence == 0.75)
    val planRequest = inference.calls[0].second
    check(planRequest == "{\"feedback\":\"\",\"goal\":\"answer\",\"max_steps\":4,\"previous_plan_id\":\"\",\"previous_steps\":[]}")

    val memory = adapter.memoryQuery(
        NativeMemoryQueryRequest(
            planId = "p1",
            goal = "answer",
            stepId = 1,
            objective = "retrieve",
            attempt = 1,
            previousFailure = "",
            dependencies = emptyList(),
            vectorDim = 2,
        )
    )
    check(memory.topK == 2)
    check(memory.kinds == listOf(NativeMemoryKind.SEMANTIC))
    check(memory.vector.contentEquals(floatArrayOf(3.0f, 4.0f)))
    check(inference.embeddings == listOf("memory evidence" to 2))

    val dependency = NativeDependencyResult(
        stepId = 1,
        kind = NativeStepKind.RETRIEVE,
        objective = "retrieve",
        result = "fact",
        confidence = 0.9,
        evidenceRecordIds = listOf(7L),
    )
    val proposal = adapter.proposeStep(
        NativeStepReasoningRequest(
            planId = "p1",
            goal = "answer",
            stepId = 2,
            kind = NativeStepKind.REASON,
            objective = "reason",
            attempt = 1,
            previousFailure = "",
            dependencies = listOf(dependency),
            memoryContext = listOf(
                NativeMemoryContextItem(7L, "fact", "unit", 1.0, 1.0, 0.5, 0.8)
            ),
            contextTruncated = false,
        )
    )
    check(proposal.result == "candidate" && proposal.confidence == 0.8)

    val decision = adapter.verifyStep(
        NativeVerificationRequest(
            planId = "p1",
            goal = "answer",
            stepId = 2,
            kind = NativeStepKind.REASON,
            objective = "reason",
            attempt = 1,
            candidate = proposal.result,
            confidence = proposal.confidence,
            dependencies = listOf(dependency),
            evidenceRecordIds = listOf(7L),
        )
    )
    check(decision.passed && decision.note == "supported")

    val intent = adapter.proposeExternalIntent(
        NativeExternalIntentRequest(
            planId = "p1",
            goal = "write note",
            stepId = 3,
            objective = "write exact note",
            capabilities = listOf(
                NativeExternalCapabilityView(
                    capabilityId = "file.write",
                    requiredScopeKeys = listOf("root", "path"),
                    optionalScopeKeys = emptyList(),
                    approvalRequired = true,
                    maxPayloadUtf8Bytes = 4096,
                )
            ),
        )
    )
    check(intent.capabilityId == "file.write")
    check(intent.scope.keys.toList() == listOf("path", "root"))
    check(intent.payloadJson == "{\"a\":\"x\",\"z\":1}")

    fun adapterWith(output: String, embedding: FloatArray = floatArrayOf(1.0f, 0.0f)) =
        NativeTypedCognitionAdapter(ScriptedInference(mutableListOf(output), embedding))

    expectContract {
        adapterWith("{\"result\":\"x\",\"result\":\"y\",\"confidence\":1.0}").proposeStep(
            NativeStepReasoningRequest("p", "g", 1, NativeStepKind.REASON, "o", 1, "", emptyList(), emptyList(), false)
        )
    }
    expectContract {
        adapterWith("{\"result\":\"x\",\"confidence\":NaN}").proposeStep(
            NativeStepReasoningRequest("p", "g", 1, NativeStepKind.REASON, "o", 1, "", emptyList(), emptyList(), false)
        )
    }
    expectContract {
        adapterWith("{\"passed\":true,\"note\":\"ok\"} trailing").verifyStep(
            NativeVerificationRequest("p", "g", 1, NativeStepKind.VERIFY, "o", 1, "x", 1.0, emptyList(), emptyList())
        )
    }
    expectContract {
        adapterWith("{\"result\":\"\\ud800\",\"confidence\":1.0}").proposeStep(
            NativeStepReasoningRequest("p", "g", 1, NativeStepKind.REASON, "o", 1, "", emptyList(), emptyList(), false)
        )
    }
    expectContract {
        adapterWith("{\"query\":\"q\",\"top_k\":2.0,\"kinds\":null,\"semantic_weight\":1.0,\"recency_weight\":0.0,\"importance_weight\":0.0,\"recency_half_life_ns\":1}").memoryQuery(
            NativeMemoryQueryRequest("p", "g", 1, "o", 1, "", emptyList(), 2)
        )
    }
    expectContract {
        adapterWith("{\"query\":\"q\",\"top_k\":1,\"kinds\":null,\"semantic_weight\":1.0,\"recency_weight\":0.0,\"importance_weight\":0.0,\"recency_half_life_ns\":1}", floatArrayOf(0.0f, 0.0f)).memoryQuery(
            NativeMemoryQueryRequest("p", "g", 1, "o", 1, "", emptyList(), 2)
        )
    }
    expectContract {
        adapterWith("{\"steps\":[],\"extra\":1}").proposePlan(NativePlanDraftRequest("g", 1))
    }

    check(inference.calls.map { it.first } == listOf(
        NativeCognitionOperation.PLAN,
        NativeCognitionOperation.MEMORY_QUERY,
        NativeCognitionOperation.STEP,
        NativeCognitionOperation.VERIFY,
        NativeCognitionOperation.EXTERNAL_INTENT,
    ))
    println("M7J_TYPED_COGNITION_PASS")
}
