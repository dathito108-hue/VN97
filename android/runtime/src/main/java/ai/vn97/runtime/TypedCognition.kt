package ai.vn97.runtime

import java.math.BigInteger
import kotlin.math.sqrt

enum class NativeStepKind { REASON, RETRIEVE, VERIFY, RESPOND, EXTERNAL }
enum class NativeMemoryKind { EPISODIC, SEMANTIC }

data class NativePlanStepSpec(
    val kind: NativeStepKind,
    val objective: String,
    val dependencies: List<Int> = emptyList(),
    val requiresVerification: Boolean = false,
    val minConfidence: Double = 0.0,
) {
    init {
        require(objective.isNotBlank()) { "step objective must not be empty" }
        require(dependencies.all { it > 0 }) { "step dependencies must be positive IDs" }
        require(dependencies.distinct().size == dependencies.size) { "step dependencies must be unique" }
        require(minConfidence.isFinite() && minConfidence in 0.0..1.0) {
            "minConfidence must be finite and in [0, 1]"
        }
    }
}

data class NativeDependencyResult(
    val stepId: Int,
    val kind: NativeStepKind,
    val objective: String,
    val result: String,
    val confidence: Double,
    val evidenceRecordIds: List<Long> = emptyList(),
) {
    init {
        require(stepId > 0) { "stepId must be positive" }
        require(confidence.isFinite() && confidence in 0.0..1.0) {
            "confidence must be finite and in [0, 1]"
        }
        require(evidenceRecordIds.all { it > 0L }) { "evidence record IDs must be positive" }
    }
}

data class NativePlanDraftRequest(
    val goal: String,
    val maxSteps: Int,
    val previousPlanId: String = "",
    val previousSteps: List<NativePlanStepSpec> = emptyList(),
    val feedback: String = "",
) {
    init {
        require(goal.isNotBlank()) { "goal must not be blank" }
        require(maxSteps > 0) { "maxSteps must be positive" }
    }
}

data class NativePlanDraft(val steps: List<NativePlanStepSpec>) {
    init { require(steps.isNotEmpty()) { "plan draft requires at least one step" } }
}

data class NativeMemoryQueryRequest(
    val planId: String,
    val goal: String,
    val stepId: Int,
    val objective: String,
    val attempt: Int,
    val previousFailure: String,
    val dependencies: List<NativeDependencyResult>,
    val vectorDim: Int,
) {
    init {
        require(stepId > 0) { "stepId must be positive" }
        require(attempt > 0) { "attempt must be positive" }
        require(vectorDim > 0) { "vectorDim must be positive" }
    }
}

data class NativeMemoryQuery(
    val vector: FloatArray,
    val topK: Int,
    val kinds: List<NativeMemoryKind>?,
    val semanticWeight: Double,
    val recencyWeight: Double,
    val importanceWeight: Double,
    val recencyHalfLifeNs: Long,
) {
    init {
        require(vector.isNotEmpty() && vector.all { it.isFinite() }) { "memory query vector must be finite and non-empty" }
        require(topK > 0) { "topK must be positive" }
        val weights = listOf(semanticWeight, recencyWeight, importanceWeight)
        require(weights.all { it.isFinite() && it >= 0.0 } && weights.sum() > 0.0) {
            "memory query weights must be finite, non-negative and contain a positive value"
        }
        require(recencyHalfLifeNs > 0L) { "recencyHalfLifeNs must be positive" }
        var normSquared = 0.0
        for (value in vector) {
            val v = value.toDouble()
            normSquared += v * v
        }
        require(sqrt(normSquared).let { it.isFinite() && it > 0.0 }) { "memory query vector norm must be positive" }
    }
}

data class NativeMemoryContextItem(
    val recordId: Long,
    val content: String,
    val source: String,
    val score: Double,
    val semanticScore: Double,
    val recencyScore: Double,
    val importanceScore: Double,
) {
    init {
        require(recordId > 0L) { "recordId must be positive" }
        require(listOf(score, semanticScore, recencyScore, importanceScore).all { it.isFinite() }) {
            "memory context scores must be finite"
        }
    }
}

data class NativeStepReasoningRequest(
    val planId: String,
    val goal: String,
    val stepId: Int,
    val kind: NativeStepKind,
    val objective: String,
    val attempt: Int,
    val previousFailure: String,
    val dependencies: List<NativeDependencyResult>,
    val memoryContext: List<NativeMemoryContextItem>,
    val contextTruncated: Boolean,
) {
    init {
        require(stepId > 0) { "stepId must be positive" }
        require(attempt > 0) { "attempt must be positive" }
    }
}

data class NativeStepProposal(val result: String, val confidence: Double) {
    init {
        require(result.isNotEmpty()) { "proposal result must not be empty" }
        require(confidence.isFinite() && confidence in 0.0..1.0) {
            "proposal confidence must be finite and in [0, 1]"
        }
    }
}

data class NativeVerificationRequest(
    val planId: String,
    val goal: String,
    val stepId: Int,
    val kind: NativeStepKind,
    val objective: String,
    val attempt: Int,
    val candidate: String,
    val confidence: Double,
    val dependencies: List<NativeDependencyResult>,
    val evidenceRecordIds: List<Long>,
) {
    init {
        require(stepId > 0) { "stepId must be positive" }
        require(attempt > 0) { "attempt must be positive" }
        require(confidence.isFinite() && confidence in 0.0..1.0) { "confidence must be finite and in [0, 1]" }
        require(evidenceRecordIds.all { it > 0L }) { "evidence record IDs must be positive" }
    }
}

data class NativeVerificationDecision(val passed: Boolean, val note: String)

data class NativeExternalCapabilityView(
    val capabilityId: String,
    val requiredScopeKeys: List<String>,
    val optionalScopeKeys: List<String>,
    val approvalRequired: Boolean,
    val maxPayloadUtf8Bytes: Int,
) {
    init {
        require(capabilityId.isNotEmpty()) { "capabilityId must not be empty" }
        require(requiredScopeKeys.all { it.isNotEmpty() } && optionalScopeKeys.all { it.isNotEmpty() }) {
            "scope keys must not be empty"
        }
        require((requiredScopeKeys + optionalScopeKeys).distinct().size == requiredScopeKeys.size + optionalScopeKeys.size) {
            "scope keys must be unique"
        }
        require(maxPayloadUtf8Bytes > 0) { "maxPayloadUtf8Bytes must be positive" }
    }
}

data class NativeExternalIntentRequest(
    val planId: String,
    val goal: String,
    val stepId: Int,
    val objective: String,
    val capabilities: List<NativeExternalCapabilityView>,
) {
    init {
        require(stepId > 0) { "stepId must be positive" }
        require(capabilities.isNotEmpty()) { "capabilities must not be empty" }
    }
}

data class NativeExternalIntent(
    val capabilityId: String,
    val scope: Map<String, String>,
    val payloadJson: String,
) {
    init {
        require(capabilityId.isNotEmpty()) { "capabilityId must not be empty" }
        require(scope.all { (key, value) -> key.isNotEmpty() && value.isNotEmpty() }) {
            "scope keys and values must be non-empty"
        }
        val parsed = VnStrictJson.parseObject(payloadJson)
        require(VnStrictJson.canonical(parsed) == payloadJson) { "payloadJson must be canonical strict JSON" }
    }
}

class NativeTypedCognitionAdapter(
    private val inference: NativeCognitionInference,
) {
    fun proposePlan(request: NativePlanDraftRequest): NativePlanDraft {
        val output = generate(
            NativeCognitionOperation.PLAN,
            VnStrictJson.objectOf(
                "goal" to VnStrictJson.string(request.goal),
                "max_steps" to VnStrictJson.int(request.maxSteps),
                "previous_plan_id" to VnStrictJson.string(request.previousPlanId),
                "previous_steps" to VnStrictJson.array(request.previousSteps.map(::stepSpecJson)),
                "feedback" to VnStrictJson.string(request.feedback),
            ),
        )
        exactKeys(output, setOf("steps"), "plan")
        val raw = output.array("steps", "plan")
        if (raw.values.isEmpty()) fail("plan steps must be a non-empty array")
        val steps = raw.values.mapIndexed { index, item -> parsePlanStep(item, index + 1) }
        if (steps.size > request.maxSteps) fail("plan steps exceed request maxSteps")
        return NativePlanDraft(steps)
    }

    fun memoryQuery(request: NativeMemoryQueryRequest): NativeMemoryQuery {
        val output = generate(
            NativeCognitionOperation.MEMORY_QUERY,
            VnStrictJson.objectOf(
                "plan_id" to VnStrictJson.string(request.planId),
                "goal" to VnStrictJson.string(request.goal),
                "step_id" to VnStrictJson.int(request.stepId),
                "objective" to VnStrictJson.string(request.objective),
                "attempt" to VnStrictJson.int(request.attempt),
                "previous_failure" to VnStrictJson.string(request.previousFailure),
                "dependencies" to dependenciesJson(request.dependencies),
                "vector_dim" to VnStrictJson.int(request.vectorDim),
            ),
        )
        exactKeys(
            output,
            setOf("query", "top_k", "kinds", "semantic_weight", "recency_weight", "importance_weight", "recency_half_life_ns"),
            "memory query",
        )
        val query = output.string("query", "memory query")
        if (query.isBlank()) fail("memory query text must be a non-empty string")
        val topK = output.int("top_k", "memory query", minimum = 1)
        val kinds = when (val rawKinds = output.values["kinds"]) {
            VnJsonNull -> null
            is VnJsonArray -> rawKinds.values.mapIndexed { index, item ->
                val name = (item as? VnJsonString)?.value
                    ?: fail("memory query kind ${index + 1} must be a string")
                try { NativeMemoryKind.valueOf(name) } catch (_: IllegalArgumentException) {
                    fail("memory query contains unknown kind")
                }
            }
            else -> fail("memory query kinds must be null or an array")
        }
        val semantic = output.nonnegativeDouble("semantic_weight", "memory query")
        val recency = output.nonnegativeDouble("recency_weight", "memory query")
        val importance = output.nonnegativeDouble("importance_weight", "memory query")
        if (semantic + recency + importance <= 0.0) fail("memory query weights must contain a positive value")
        val halfLife = output.long("recency_half_life_ns", "memory query", minimum = 1L)
        val vector = inference.embedText(query, request.vectorDim)
        if (vector.size != request.vectorDim) fail("retrieval embedding dimension mismatch")
        if (vector.any { !it.isFinite() }) fail("retrieval embedding contains non-finite values")
        var normSquared = 0.0
        for (value in vector) { val v = value.toDouble(); normSquared += v * v }
        val norm = sqrt(normSquared)
        if (!norm.isFinite() || norm == 0.0) fail("retrieval embedding has zero/invalid norm")
        return NativeMemoryQuery(vector, topK, kinds, semantic, recency, importance, halfLife)
    }

    fun proposeStep(request: NativeStepReasoningRequest): NativeStepProposal {
        val output = generate(
            NativeCognitionOperation.STEP,
            VnStrictJson.objectOf(
                "plan_id" to VnStrictJson.string(request.planId),
                "goal" to VnStrictJson.string(request.goal),
                "step_id" to VnStrictJson.int(request.stepId),
                "kind" to VnStrictJson.string(request.kind.name),
                "objective" to VnStrictJson.string(request.objective),
                "attempt" to VnStrictJson.int(request.attempt),
                "previous_failure" to VnStrictJson.string(request.previousFailure),
                "dependencies" to dependenciesJson(request.dependencies),
                "memory_context" to VnStrictJson.array(request.memoryContext.map(::memoryContextJson)),
                "context_truncated" to VnStrictJson.bool(request.contextTruncated),
            ),
        )
        exactKeys(output, setOf("result", "confidence"), "step proposal")
        val result = output.string("result", "step proposal")
        if (result.isEmpty()) fail("step proposal result must be a non-empty string")
        return NativeStepProposal(result, output.double01("confidence", "step proposal"))
    }

    fun verifyStep(request: NativeVerificationRequest): NativeVerificationDecision {
        val output = generate(
            NativeCognitionOperation.VERIFY,
            VnStrictJson.objectOf(
                "plan_id" to VnStrictJson.string(request.planId),
                "goal" to VnStrictJson.string(request.goal),
                "step_id" to VnStrictJson.int(request.stepId),
                "kind" to VnStrictJson.string(request.kind.name),
                "objective" to VnStrictJson.string(request.objective),
                "attempt" to VnStrictJson.int(request.attempt),
                "candidate" to VnStrictJson.string(request.candidate),
                "confidence" to VnStrictJson.double(request.confidence),
                "dependencies" to dependenciesJson(request.dependencies),
                "evidence_record_ids" to VnStrictJson.array(request.evidenceRecordIds.map(VnStrictJson::long)),
            ),
        )
        exactKeys(output, setOf("passed", "note"), "verification")
        return NativeVerificationDecision(
            passed = output.bool("passed", "verification"),
            note = output.string("note", "verification"),
        )
    }

    fun proposeExternalIntent(request: NativeExternalIntentRequest): NativeExternalIntent {
        val capabilityCatalog = VnStrictJson.array(request.capabilities.map { item ->
            VnStrictJson.objectOf(
                "capability_id" to VnStrictJson.string(item.capabilityId),
                "required_scope_keys" to VnStrictJson.array(item.requiredScopeKeys.map(VnStrictJson::string)),
                "optional_scope_keys" to VnStrictJson.array(item.optionalScopeKeys.map(VnStrictJson::string)),
                "approval_required" to VnStrictJson.bool(item.approvalRequired),
                "max_payload_utf8_bytes" to VnStrictJson.int(item.maxPayloadUtf8Bytes),
            )
        })
        val output = generate(
            NativeCognitionOperation.EXTERNAL_INTENT,
            VnStrictJson.objectOf(
                "plan_id" to VnStrictJson.string(request.planId),
                "goal" to VnStrictJson.string(request.goal),
                "step_id" to VnStrictJson.int(request.stepId),
                "objective" to VnStrictJson.string(request.objective),
                "capabilities" to capabilityCatalog,
            ),
        )
        exactKeys(output, setOf("capability_id", "scope", "payload"), "external intent")
        val capabilityId = output.string("capability_id", "external intent")
        if (capabilityId.isEmpty()) fail("external intent capability_id must be non-empty")
        val scopeObject = output.obj("scope", "external intent")
        val scope = linkedMapOf<String, String>()
        for (key in scopeObject.values.keys.sorted()) {
            if (key.isEmpty()) fail("external intent scope keys must be non-empty")
            val value = (scopeObject.values[key] as? VnJsonString)?.value
                ?: fail("external intent scope values must be strings")
            if (value.isEmpty()) fail("external intent scope values must be non-empty")
            scope[key] = value
        }
        val payload = output.values["payload"] as? VnJsonObject
            ?: fail("external intent payload must be an object")
        return NativeExternalIntent(
            capabilityId = capabilityId,
            scope = scope,
            payloadJson = VnStrictJson.canonical(payload),
         )
    }

    private fun generate(operation: NativeCognitionOperation, request: VnJsonObject): VnJsonObject {
        val requestJson = VnStrictJson.canonical(request)
        val output = inference.generateOperation(operation, requestJson)
        return VnStrictJson.parseObject(output)
    }
}

private fun stepSpecJson(spec: NativePlanStepSpec): VnJsonObject = VnStrictJson.objectOf(
    "kind" to VnStrictJson.string(spec.kind.name),
    "objective" to VnStrictJson.string(spec.objective),
    "dependencies" to VnStrictJson.array(spec.dependencies.map(VnStrictJson::int)),
    "requires_verification" to VnStrictJson.bool(spec.requiresVerification),
    "min_confidence" to VnStrictJson.double(spec.minConfidence),
)

private fun dependenciesJson(items: List<NativeDependencyResult>): VnJsonArray = VnStrictJson.array(items.map { item ->
    VnStrictJson.objectOf(
        "step_id" to VnStrictJson.int(item.stepId),
        "kind" to VnStrictJson.string(item.kind.name),
        "objective" to VnStrictJson.string(item.objective),
        "result" to VnStrictJson.string(item.result),
        "confidence" to VnStrictJson.double(item.confidence),
        "evidence_record_ids" to VnStrictJson.array(item.evidenceRecordIds.map(VnStrictJson::long)),
    )
})

private fun memoryContextJson(item: NativeMemoryContextItem): VnJsonObject = VnStrictJson.objectOf(
    "record_id" to VnStrictJson.long(item.recordId),
    "content" to VnStrictJson.string(item.content),
    "source" to VnStrictJson.string(item.source),
    "score" to VnStrictJson.double(item.score),
    "semantic_score" to VnStrictJson.double(item.semanticScore),
    "recency_score" to VnStrictJson.double(item.recencyScore),
    "importance_score" to VnStrictJson.double(item.importanceScore),
)

private fun parsePlanStep(value: VnJsonValue, index: Int): NativePlanStepSpec {
    val obj = value as? VnJsonObject ?: fail("plan step $index must be an object")
    exactKeys(obj, setOf("kind", "objective", "dependencies", "requires_verification", "min_confidence"), "plan step $index")
    val kindName = obj.string("kind", "plan step $index")
    val kind = try { NativeStepKind.valueOf(kindName) } catch (_: IllegalArgumentException) {
        fail("plan step $index has unknown kind")
    }
    val dependencies = obj.array("dependencies", "plan step $index").values.map { raw ->
        strictInt(raw, "plan step $index dependency", 1)
    }
    return try {
        NativePlanStepSpec(
            kind = kind,
            objective = obj.string("objective", "plan step $index"),
            dependencies = dependencies,
            requiresVerification = obj.bool("requires_verification", "plan step $index"),
            minConfidence = obj.double01("min_confidence", "plan step $index"),
        )
    } catch (exc: IllegalArgumentException) {
        throw NativeCognitionContractException("plan step $index is invalid", exc)
    }
}

private fun exactKeys(value: VnJsonObject, expected: Set<String>, label: String) {
    val actual = value.values.keys
    if (actual != expected) {
        val missing = (expected - actual).sorted()
        val extra = (actual - expected).sorted()
        fail("$label keys mismatch; missing=$missing, extra=$extra")
    }
}

private fun VnJsonObject.string(key: String, label: String): String =
    (values[key] as? VnJsonString)?.value ?: fail("$label $key must be a string")

private fun VnJsonObject.bool(key: String, label: String): Boolean =
    (values[key] as? VnJsonBoolean)?.value ?: fail("$label $key must be bool")

private fun VnJsonObject.array(key: String, label: String): VnJsonArray =
    values[key] as? VnJsonArray ?: fail("$label $key must be an array")

private fun VnJsonObject.obj(key: String, label: String): VnJsonObject =
    values[key] as? VnJsonObject ?: fail("$label $key must be an object")

private fun VnJsonObject.int(key: String, label: String, minimum: Int? = null): Int =
    strictInt(values[key] ?: fail("$label missing $key"), "$label $key", minimum)

private fun VnJsonObject.long(key: String, label: String, minimum: Long? = null): Long =
    strictLong(values[key] ?: fail("$label missing $key"), "$label $key", minimum)

private fun VnJsonObject.double01(key: String, label: String): Double {
    val value = strictDouble(values[key] ?: fail("$label missing $key"), "$label $key")
    if (value !in 0.0..1.0) fail("$label $key must be in [0, 1]")
    return value
}

private fun VnJsonObject.nonnegativeDouble(key: String, label: String): Double {
    val value = strictDouble(values[key] ?: fail("$label missing $key"), "$label $key")
    if (value < 0.0) fail("$label $key must be non-negative")
    return value
}

private fun strictInt(value: VnJsonValue, label: String, minimum: Int? = null): Int {
    val raw = (value as? VnJsonNumber)?.canonical ?: fail("$label must be an integer")
    if (raw.indexOfAny(charArrayOf('.', 'e', 'E')) >= 0) fail("$label must be an integer")
    val result = try { BigInteger(raw).intValueExact() } catch (_: ArithmeticException) {
        fail("$label is outside Int range")
    }
    if (minimum != null && result < minimum) fail("$label must be >= $minimum")
    return result
}

private fun strictLong(value: VnJsonValue, label: String, minimum: Long? = null): Long {
    val raw = (value as? VnJsonNumber)?.canonical ?: fail("$label must be an integer")
    if (raw.indexOfAny(charArrayOf('.', 'e', 'E')) >= 0) fail("$label must be an integer")
    val result = try { BigInteger(raw).longValueExact() } catch (_: ArithmeticException) {
        fail("$label is outside Long range")
    }
    if (minimum != null && result < minimum) fail("$label must be >= $minimum")
    return result
}

private fun strictDouble(value: VnJsonValue, label: String): Double {
    val raw = (value as? VnJsonNumber)?.canonical ?: fail("$label must be numeric")
    val result = raw.toDoubleOrNull() ?: fail("$label must be numeric")
    if (!result.isFinite()) fail("$label must be finite")
    return result
}

private fun fail(message: String): Nothing = throw NativeCognitionContractException(message)
