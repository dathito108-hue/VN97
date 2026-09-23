package ai.vn97.runtime

import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.charset.CodingErrorAction
import java.nio.charset.StandardCharsets
import java.security.MessageDigest

private val PLN_MAGIC = "VN97PLN1".toByteArray(StandardCharsets.US_ASCII)
private const val PLN_VERSION = 1
private const val PLN_HEADER_BYTES = 48
private const val PLN_MAX_PAYLOAD = 16 shl 20

class NativePlannerCheckpointException(message: String, cause: Throwable? = null) : IllegalArgumentException(message, cause)

object NativePlannerCheckpoint {
    fun encode(plan: NativePlan): ByteArray {
        validate(plan)
        if (plan.steps.any { it.status == NativeStepStatus.RUNNING }) {
            throw NativePlannerCheckpointException("checkpoint requires a safe point; pause an in-flight internal step first")
        }
        val payload = VnStrictJson.canonical(payload(plan)).toByteArray(StandardCharsets.UTF_8)
        if (payload.size > PLN_MAX_PAYLOAD) throw NativePlannerCheckpointException("checkpoint payload exceeds format limit")
        val digest = MessageDigest.getInstance("SHA-256").digest(payload)
        return ByteBuffer.allocate(PLN_HEADER_BYTES + payload.size).order(ByteOrder.LITTLE_ENDIAN).apply {
            put(PLN_MAGIC)
            putInt(PLN_VERSION)
            putInt(payload.size)
            put(digest)
            put(payload)
        }.array()
    }

    fun decode(blob: ByteArray): NativePlanController {
        if (blob.size < PLN_HEADER_BYTES) corrupt("checkpoint is shorter than header")
        val header = ByteBuffer.wrap(blob, 0, PLN_HEADER_BYTES).order(ByteOrder.LITTLE_ENDIAN)
        val magic = ByteArray(8).also(header::get)
        if (!magic.contentEquals(PLN_MAGIC)) corrupt("bad VN97PLN1 magic")
        val version = header.int
        if (version != PLN_VERSION) corrupt("unsupported VN97PLN1 version: $version")
        val payloadSize = header.int
        if (payloadSize < 0 || payloadSize > PLN_MAX_PAYLOAD) corrupt("checkpoint payload exceeds format limit")
        if (blob.size != PLN_HEADER_BYTES + payloadSize) corrupt("checkpoint length does not match header")
        val expected = ByteArray(32).also(header::get)
        val payload = blob.copyOfRange(PLN_HEADER_BYTES, blob.size)
        val actual = MessageDigest.getInstance("SHA-256").digest(payload)
        if (!actual.contentEquals(expected)) corrupt("checkpoint SHA-256 mismatch")
        val text = strictUtf8(payload)
        val root = try {
            VnStrictJson.parseObject(
                text,
                VnJsonLimits(
                    maxInputUtf8Bytes = PLN_MAX_PAYLOAD,
                    maxDepth = 64,
                    maxNodes = 256 * 1024,
                    maxStringUtf8Bytes = PLN_MAX_PAYLOAD,
                ),
            )
        } catch (exc: RuntimeException) {
            throw NativePlannerCheckpointException("checkpoint payload is not canonical JSON", exc)
        }
        val plan = try {
            parsePlan(root).also(::validate)
        } catch (exc: NativePlannerCheckpointException) {
            throw exc
        } catch (exc: RuntimeException) {
            throw NativePlannerCheckpointException("checkpoint payload fields are invalid", exc)
        }
        if (VnStrictJson.canonical(payload(plan)) != text) corrupt("checkpoint JSON is not canonical")
        return NativePlanController.restoreValidated(plan)
    }

    private fun payload(plan: NativePlan): VnJsonObject = VnStrictJson.objectOf(
        "plan_id" to VnStrictJson.string(plan.planId),
        "goal" to VnStrictJson.string(plan.goal),
        "created_ns" to VnStrictJson.long(plan.createdNs),
        "status" to VnStrictJson.int(planStatusCode(plan.status)),
        "transitions_used" to VnStrictJson.int(plan.transitionsUsed),
        "memory_queries_used" to VnStrictJson.int(plan.memoryQueriesUsed),
        "paused_reason" to VnStrictJson.string(plan.pausedReason),
        "terminal_reason" to VnStrictJson.string(plan.terminalReason),
        "budget" to VnStrictJson.objectOf(
            "max_transitions" to VnStrictJson.int(plan.budget.maxTransitions),
            "max_retries_per_step" to VnStrictJson.int(plan.budget.maxRetriesPerStep),
            "max_memory_queries" to VnStrictJson.int(plan.budget.maxMemoryQueries),
            "max_memory_hits" to VnStrictJson.int(plan.budget.maxMemoryHits),
        ),
        "steps" to VnStrictJson.array(plan.steps.map(::stepPayload)),
    )

    private fun stepPayload(step: NativePlannerStep): VnJsonObject = VnStrictJson.objectOf(
        "step_id" to VnStrictJson.int(step.stepId),
        "spec" to VnStrictJson.objectOf(
            "kind" to VnStrictJson.int(kindCode(step.spec.kind)),
            "objective" to VnStrictJson.string(step.spec.objective),
            "dependencies" to VnStrictJson.array(step.spec.dependencies.map(VnStrictJson::int)),
            "requires_verification" to VnStrictJson.bool(step.spec.requiresVerification),
            "min_confidence" to VnStrictJson.double(step.spec.minConfidence),
        ),
        "status" to VnStrictJson.int(stepStatusCode(step.status)),
        "attempts" to VnStrictJson.int(step.attempts),
        "result" to VnStrictJson.string(step.result),
        "confidence" to (step.confidence?.let(VnStrictJson::double) ?: VnJsonNull),
        "evidence_record_ids" to VnStrictJson.array(step.evidenceRecordIds.map(VnStrictJson::long)),
        "verification_note" to VnStrictJson.string(step.verificationNote),
        "failure_reason" to VnStrictJson.string(step.failureReason),
    )

    private fun parsePlan(root: VnJsonObject): NativePlan {
        exact(root, setOf("plan_id","goal","created_ns","status","transitions_used","memory_queries_used","paused_reason","terminal_reason","budget","steps"), "plan")
        val budgetObj = root.obj("budget")
        exact(budgetObj, setOf("max_transitions","max_retries_per_step","max_memory_queries","max_memory_hits"), "budget")
        val budget = NativeReasoningBudget(
            budgetObj.int("max_transitions", 1),
            budgetObj.int("max_retries_per_step", 0),
            budgetObj.int("max_memory_queries", 0),
            budgetObj.int("max_memory_hits", 1),
        )
        val stepValues = root.array("steps").values
        if (stepValues.isEmpty()) corrupt("plan requires at least one step")
        val steps = stepValues.mapIndexed { index, raw -> parseStep(raw, index + 1) }
        val plan = NativePlan(
            root.string("plan_id"), root.string("goal"), budget, steps, root.long("created_ns", 0L),
        )
        plan.status = planStatus(root.int("status", 1))
        plan.transitionsUsed = root.int("transitions_used", 0)
        plan.memoryQueriesUsed = root.int("memory_queries_used", 0)
        plan.pausedReason = root.string("paused_reason")
        plan.terminalReason = root.string("terminal_reason")
        return plan
    }

    private fun parseStep(raw: VnJsonValue, expectedId: Int): NativePlannerStep {
        val obj = raw as? VnJsonObject ?: corrupt("plan step must be an object")
        exact(obj, setOf("step_id","spec","status","attempts","result","confidence","evidence_record_ids","verification_note","failure_reason"), "step")
        val id = obj.int("step_id", 1)
        if (id != expectedId) corrupt("step IDs must be dense and ordered")
        val specObj = obj.obj("spec")
        exact(specObj, setOf("kind","objective","dependencies","requires_verification","min_confidence"), "step spec")
        val spec = NativePlanStepSpec(
            kind = kind(specObj.int("kind", 1)),
            objective = specObj.string("objective"),
            dependencies = specObj.array("dependencies").values.map { strictInt(it, "dependency", 1) },
            requiresVerification = specObj.bool("requires_verification"),
            minConfidence = specObj.double("min_confidence"),
        )
        return NativePlannerStep(id, spec).also { step ->
            step.status = stepStatus(obj.int("status", 1))
            step.attempts = obj.int("attempts", 0)
            step.result = obj.string("result")
            step.confidence = when (val value = obj.values["confidence"]) {
                VnJsonNull -> null
                null -> corrupt("step confidence missing")
                else -> strictDouble(value, "confidence")
            }
            step.evidenceRecordIds = obj.array("evidence_record_ids").values.map { strictLong(it, "evidence_record_id", 1L) }
            step.verificationNote = obj.string("verification_note")
            step.failureReason = obj.string("failure_reason")
        }
    }

    private fun validate(plan: NativePlan) {
        if (plan.createdNs < 0) corrupt("created_ns must be non-negative")
        if (plan.steps.isEmpty()) corrupt("plan requires at least one step")
        if (plan.transitionsUsed !in 0..plan.budget.maxTransitions) corrupt("transitions_used is outside budget")
        if (plan.memoryQueriesUsed !in 0..plan.budget.maxMemoryQueries) corrupt("memory_queries_used is outside budget")
        if (plan.planId != NativePlanIdentity.compute(plan.goal, plan.steps.map { it.spec }, plan.budget)) corrupt("plan_id does not match immutable plan definition")
        val running = plan.steps.count { it.status == NativeStepStatus.RUNNING }
        val waitingExternal = plan.steps.count { it.status == NativeStepStatus.WAITING_EXTERNAL }
        if (running > 1 || waitingExternal > 1) corrupt("plan has multiple active steps")
        if (plan.status == NativePlanStatus.COMPLETED && plan.steps.any { it.status != NativeStepStatus.SUCCEEDED }) corrupt("completed plan contains unfinished steps")
        if (plan.status == NativePlanStatus.FAILED && plan.steps.none { it.status == NativeStepStatus.FAILED }) corrupt("failed plan has no failed step")
        if (plan.status == NativePlanStatus.WAITING_EXTERNAL && waitingExternal != 1) corrupt("WAITING_EXTERNAL plan must have exactly one waiting step")
        if (plan.status == NativePlanStatus.PAUSED && running != 0) corrupt("paused plan cannot contain running step")
        plan.steps.forEachIndexed { index, s ->
            if (s.stepId != index + 1) corrupt("step IDs must be dense and ordered")
            if (s.spec.dependencies.any { it !in 1..index }) corrupt("step dependencies must reference earlier steps")
            if (s.attempts < 0) corrupt("step attempts must be non-negative")
            s.confidence?.let { if (!it.isFinite() || it !in 0.0..1.0) corrupt("step confidence is invalid") }
            if (s.status == NativeStepStatus.WAITING_EXTERNAL && s.spec.kind != NativeStepKind.EXTERNAL) corrupt("only EXTERNAL steps may wait externally")
            if (s.status in setOf(NativeStepStatus.WAITING_VERIFICATION, NativeStepStatus.SUCCEEDED) && (s.result.isEmpty() || s.confidence == null)) corrupt("completed candidate requires result and confidence")
            if (s.evidenceRecordIds.any { it <= 0 } || s.evidenceRecordIds.distinct().size != s.evidenceRecordIds.size) corrupt("evidence record IDs are invalid")
        }
    }

    private fun exact(obj: VnJsonObject, keys: Set<String>, label: String) {
        if (obj.values.keys != keys) corrupt("$label keys mismatch")
    }

    private fun VnJsonObject.string(key: String) = (values[key] as? VnJsonString)?.value ?: corrupt("$key must be string")
    private fun VnJsonObject.bool(key: String) = (values[key] as? VnJsonBoolean)?.value ?: corrupt("$key must be bool")
    private fun VnJsonObject.obj(key: String) = values[key] as? VnJsonObject ?: corrupt("$key must be object")
    private fun VnJsonObject.array(key: String) = values[key] as? VnJsonArray ?: corrupt("$key must be array")
    private fun VnJsonObject.int(key: String, minimum: Int) = strictInt(values[key] ?: corrupt("missing $key"), key, minimum)
    private fun VnJsonObject.long(key: String, minimum: Long) = strictLong(values[key] ?: corrupt("missing $key"), key, minimum)
    private fun VnJsonObject.double(key: String) = strictDouble(values[key] ?: corrupt("missing $key"), key)

    private fun strictInt(value: VnJsonValue, label: String, minimum: Int): Int {
        val raw = (value as? VnJsonNumber)?.canonical ?: corrupt("$label must be integer")
        if (raw.any { it == '.' || it == 'e' || it == 'E' }) corrupt("$label must be integer")
        val parsed = raw.toIntOrNull() ?: corrupt("$label is outside Int range")
        if (parsed < minimum) corrupt("$label is below minimum")
        return parsed
    }

    private fun strictLong(value: VnJsonValue, label: String, minimum: Long): Long {
        val raw = (value as? VnJsonNumber)?.canonical ?: corrupt("$label must be integer")
        if (raw.any { it == '.' || it == 'e' || it == 'E' }) corrupt("$label must be integer")
        val parsed = raw.toLongOrNull() ?: corrupt("$label is outside Long range")
        if (parsed < minimum) corrupt("$label is below minimum")
        return parsed
    }

    private fun strictDouble(value: VnJsonValue, label: String): Double {
        val raw = (value as? VnJsonNumber)?.canonical ?: corrupt("$label must be numeric")
        val parsed = raw.toDoubleOrNull() ?: corrupt("$label must be numeric")
        if (!parsed.isFinite()) corrupt("$label must be finite")
        return parsed
    }

    private fun stepStatusCode(v: NativeStepStatus) = v.ordinal + 1
    private fun planStatusCode(v: NativePlanStatus) = v.ordinal + 1
    private fun kindCode(v: NativeStepKind) = v.ordinal + 1
    private fun stepStatus(code: Int) = NativeStepStatus.values().getOrNull(code - 1) ?: corrupt("invalid step status")
    private fun planStatus(code: Int) = NativePlanStatus.values().getOrNull(code - 1) ?: corrupt("invalid plan status")
    private fun kind(code: Int) = NativeStepKind.values().getOrNull(code - 1) ?: corrupt("invalid step kind")

    private fun strictUtf8(bytes: ByteArray): String = try {
        StandardCharsets.UTF_8.newDecoder().onMalformedInput(CodingErrorAction.REPORT).onUnmappableCharacter(CodingErrorAction.REPORT)
            .decode(ByteBuffer.wrap(bytes)).toString()
    } catch (exc: Exception) {
        throw NativePlannerCheckpointException("checkpoint payload is not UTF-8", exc)
    }

    private fun corrupt(message: String): Nothing = throw NativePlannerCheckpointException(message)
}
