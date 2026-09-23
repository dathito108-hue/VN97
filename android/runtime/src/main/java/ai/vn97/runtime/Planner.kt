package ai.vn97.runtime

import java.nio.charset.StandardCharsets
import java.security.MessageDigest
import kotlin.math.min

enum class NativeStepStatus { PENDING, RUNNING, WAITING_VERIFICATION, WAITING_EXTERNAL, SUCCEEDED, FAILED, CANCELLED }
enum class NativePlanStatus { READY, RUNNING, WAITING_EXTERNAL, PAUSED, COMPLETED, FAILED, CANCELLED, BUDGET_EXHAUSTED }

data class NativeReasoningBudget(
    val maxTransitions: Int = 64,
    val maxRetriesPerStep: Int = 2,
    val maxMemoryQueries: Int = 8,
    val maxMemoryHits: Int = 8,
) {
    init {
        require(maxTransitions > 0)
        require(maxRetriesPerStep >= 0)
        require(maxMemoryQueries >= 0)
        require(maxMemoryHits > 0)
    }
}

class NativePlannerException(message: String) : IllegalStateException(message)

data class NativeStepDirective(val stepId: Int, val kind: NativeStepKind, val objective: String, val externalRequired: Boolean)

class NativePlannerStep internal constructor(val stepId: Int, val spec: NativePlanStepSpec) {
    var status = NativeStepStatus.PENDING; internal set
    var attempts = 0; internal set
    var result = ""; internal set
    var confidence: Double? = null; internal set
    var evidenceRecordIds: List<Long> = emptyList(); internal set
    var verificationNote = ""; internal set
    var failureReason = ""; internal set
}

class NativePlan internal constructor(
    val planId: String,
    val goal: String,
    val budget: NativeReasoningBudget,
    val steps: List<NativePlannerStep>,
    val createdNs: Long,
) {
    var status = NativePlanStatus.READY; internal set
    var transitionsUsed = 0; internal set
    var memoryQueriesUsed = 0; internal set
    var pausedReason = ""; internal set
    var terminalReason = ""; internal set

    fun step(id: Int): NativePlannerStep {
        if (id !in 1..steps.size) throw NativePlannerException("unknown stepId: $id")
        return steps[id - 1].also { if (it.stepId != id) throw NativePlannerException("step ordering invariant broken") }
    }

    fun isTerminal() = status in setOf(
        NativePlanStatus.COMPLETED, NativePlanStatus.FAILED,
        NativePlanStatus.CANCELLED, NativePlanStatus.BUDGET_EXHAUSTED,
    )
}

object NativePlanIdentity {
    fun compute(goal: String, specs: List<NativePlanStepSpec>, budget: NativeReasoningBudget = NativeReasoningBudget()): String {
        require(goal.isNotBlank())
        validateSpecs(specs)
        val payload = VnStrictJson.objectOf(
            "goal" to VnStrictJson.string(goal),
            "budget" to VnStrictJson.objectOf(
                "max_transitions" to VnStrictJson.int(budget.maxTransitions),
                "max_retries_per_step" to VnStrictJson.int(budget.maxRetriesPerStep),
                "max_memory_queries" to VnStrictJson.int(budget.maxMemoryQueries),
                "max_memory_hits" to VnStrictJson.int(budget.maxMemoryHits),
            ),
            "steps" to VnStrictJson.array(specs.map { s ->
                VnStrictJson.objectOf(
                    "kind" to VnStrictJson.int(kindCode(s.kind)),
                    "objective" to VnStrictJson.string(s.objective),
                    "dependencies" to VnStrictJson.array(s.dependencies.map(VnStrictJson::int)),
                    "requires_verification" to VnStrictJson.bool(s.requiresVerification),
                    "min_confidence" to VnStrictJson.double(s.minConfidence),
                )
            }),
        )
        val digest = MessageDigest.getInstance("SHA-256").digest(
            VnStrictJson.canonical(payload).toByteArray(StandardCharsets.UTF_8)
        )
        return digest.joinToString("") { "%02x".format(it.toInt() and 0xff) }
    }
}

class NativePlanController private constructor(val plan: NativePlan) {
    companion object {
        fun create(
            goal: String,
            specs: List<NativePlanStepSpec>,
            budget: NativeReasoningBudget = NativeReasoningBudget(),
            createdNs: Long = System.currentTimeMillis() * 1_000_000L,
        ): NativePlanController {
            require(goal.isNotBlank())
            require(createdNs >= 0)
            validateSpecs(specs)
            return NativePlanController(
                NativePlan(
                    NativePlanIdentity.compute(goal, specs, budget), goal, budget,
                    specs.mapIndexed { i, s -> NativePlannerStep(i + 1, s) }, createdNs,
                )
            )
        }
    }

    fun readySteps(): List<NativePlannerStep> {
        if (plan.isTerminal() || plan.status == NativePlanStatus.PAUSED || active()) return emptyList()
        return plan.steps.filter { it.status == NativeStepStatus.PENDING && depsOk(it) }
    }

    fun nextDirective(): NativeStepDirective? = readySteps().firstOrNull()?.let(::directive).also { if (it == null) refresh() }

    fun beginStep(id: Int): NativeStepDirective {
        mutable(); if (active()) fail("another step is already active")
        val s = plan.step(id)
        if (s.status != NativeStepStatus.PENDING) fail("step is not pending")
        if (!depsOk(s)) fail("step dependencies are not satisfied")
        transition(); s.attempts++; clearCandidate(s)
        if (s.spec.kind == NativeStepKind.EXTERNAL) {
            s.status = NativeStepStatus.WAITING_EXTERNAL; plan.status = NativePlanStatus.WAITING_EXTERNAL
        } else {
            s.status = NativeStepStatus.RUNNING; plan.status = NativePlanStatus.RUNNING
        }
        return directive(s)
    }

    fun completeStep(id: Int, result: String, confidence: Double, evidenceRecordIds: List<Long> = emptyList()) {
        mutable(); val s = plan.step(id)
        if (s.status != NativeStepStatus.RUNNING) fail("step is not running")
        transition(); finish(s, result, confidence, evidenceRecordIds)
    }

    fun recordExternalResult(id: Int, result: String, confidence: Double, evidenceRecordIds: List<Long> = emptyList()) {
        mutable(); val s = plan.step(id)
        if (s.status != NativeStepStatus.WAITING_EXTERNAL) fail("step is not waiting for external result")
        transition(); finish(s, result, confidence, evidenceRecordIds)
    }

    fun verifyStep(id: Int, passed: Boolean, note: String = "") {
        mutable(); val s = plan.step(id)
        if (s.status != NativeStepStatus.WAITING_VERIFICATION) fail("step is not waiting for verification")
        transition(); s.verificationNote = note
        if (passed) { s.status = NativeStepStatus.SUCCEEDED; refresh() }
        else retryOrFail(s, note.ifEmpty { "verification failed" })
    }

    fun failStep(id: Int, reason: String, retryable: Boolean = true) {
        mutable(); require(reason.isNotEmpty()); val s = plan.step(id)
        if (s.status !in setOf(NativeStepStatus.RUNNING, NativeStepStatus.WAITING_EXTERNAL, NativeStepStatus.WAITING_VERIFICATION)) fail("step is not active")
        transition()
        if (retryable) retryOrFail(s, reason) else {
            s.status = NativeStepStatus.FAILED; s.failureReason = reason
            plan.status = NativePlanStatus.FAILED; plan.terminalReason = "step ${s.stepId} failed: $reason"
        }
    }

    fun consumeMemoryQuery(topK: Int): Int {
        mutable(); require(topK > 0)
        if (plan.memoryQueriesUsed >= plan.budget.maxMemoryQueries) fail("memory query budget exhausted")
        plan.memoryQueriesUsed++
        return min(topK, plan.budget.maxMemoryHits)
    }

    fun pause(reason: String) {
        if (plan.isTerminal()) fail("terminal plan cannot be paused"); require(reason.isNotEmpty())
        if (plan.status == NativePlanStatus.PAUSED) return
        plan.steps.filter { it.status == NativeStepStatus.RUNNING }.forEach {
            it.status = NativeStepStatus.PENDING; clearCandidate(it); it.failureReason = "interrupted before durable result"
        }
        plan.pausedReason = reason; plan.status = NativePlanStatus.PAUSED
    }

    fun resume() {
        if (plan.isTerminal()) fail("terminal plan cannot be resumed")
        if (plan.status != NativePlanStatus.PAUSED) fail("plan is not paused")
        plan.pausedReason = ""; plan.status = NativePlanStatus.READY; refresh()
    }

    fun cancel(reason: String = "cancelled") {
        if (plan.isTerminal()) fail("plan is already terminal")
        plan.steps.filter { it.status !in setOf(NativeStepStatus.SUCCEEDED, NativeStepStatus.FAILED) }
            .forEach { it.status = NativeStepStatus.CANCELLED }
        plan.status = NativePlanStatus.CANCELLED; plan.terminalReason = reason
    }

    private fun finish(s: NativePlannerStep, result: String, confidence: Double, evidence: List<Long>) {
        require(result.isNotEmpty()); require(confidence.isFinite() && confidence in 0.0..1.0)
        require(evidence.all { it > 0 } && evidence.distinct().size == evidence.size)
        s.result = result; s.confidence = confidence; s.evidenceRecordIds = evidence.toList()
        s.status = if (s.spec.requiresVerification || confidence < s.spec.minConfidence) NativeStepStatus.WAITING_VERIFICATION else NativeStepStatus.SUCCEEDED
        refresh()
    }

    private fun retryOrFail(s: NativePlannerStep, reason: String) {
        if (maxOf(0, s.attempts - 1) < plan.budget.maxRetriesPerStep) {
            s.status = NativeStepStatus.PENDING; clearCandidate(s); s.failureReason = reason
        } else {
            s.status = NativeStepStatus.FAILED; s.failureReason = reason
            plan.status = NativePlanStatus.FAILED; plan.terminalReason = "step ${s.stepId} exhausted retry budget: $reason"
        }
        refresh()
    }

    private fun transition() {
        if (plan.transitionsUsed >= plan.budget.maxTransitions) {
            val reason = "reasoning transition budget exhausted"
            plan.steps.filter { it.status in setOf(NativeStepStatus.RUNNING, NativeStepStatus.WAITING_VERIFICATION, NativeStepStatus.WAITING_EXTERNAL) }
                .forEach { it.status = NativeStepStatus.CANCELLED; it.failureReason = reason }
            plan.status = NativePlanStatus.BUDGET_EXHAUSTED; plan.terminalReason = reason; fail(reason)
        }
        plan.transitionsUsed++
    }

    private fun refresh() {
        if (plan.isTerminal() || plan.status == NativePlanStatus.PAUSED) return
        plan.status = when {
            plan.steps.any { it.status == NativeStepStatus.FAILED } -> NativePlanStatus.FAILED
            plan.steps.all { it.status == NativeStepStatus.SUCCEEDED } -> NativePlanStatus.COMPLETED
            plan.steps.any { it.status == NativeStepStatus.WAITING_EXTERNAL } -> NativePlanStatus.WAITING_EXTERNAL
            active() -> NativePlanStatus.RUNNING
            else -> NativePlanStatus.READY
        }
        if (plan.status == NativePlanStatus.FAILED && plan.terminalReason.isEmpty()) plan.terminalReason = "a plan step failed"
    }

    private fun mutable() {
        if (plan.isTerminal()) fail("plan is terminal: ${plan.status.name}")
        if (plan.status == NativePlanStatus.PAUSED) fail("plan is paused")
    }

    private fun active() = plan.steps.any { it.status in setOf(NativeStepStatus.RUNNING, NativeStepStatus.WAITING_VERIFICATION, NativeStepStatus.WAITING_EXTERNAL) }
    private fun depsOk(s: NativePlannerStep) = s.spec.dependencies.all { plan.step(it).status == NativeStepStatus.SUCCEEDED }
    private fun directive(s: NativePlannerStep) = NativeStepDirective(s.stepId, s.spec.kind, s.spec.objective, s.spec.kind == NativeStepKind.EXTERNAL)
    private fun clearCandidate(s: NativePlannerStep) { s.result = ""; s.confidence = null; s.evidenceRecordIds = emptyList(); s.verificationNote = ""; s.failureReason = "" }
    private fun fail(message: String): Nothing = throw NativePlannerException(message)
}

private fun validateSpecs(specs: List<NativePlanStepSpec>) {
    require(specs.isNotEmpty())
    specs.forEachIndexed { i, s ->
        require(s.dependencies.all { it in 1..i }) { "step ${i + 1} dependencies must reference earlier steps" }
    }
}

private fun kindCode(kind: NativeStepKind) = when (kind) {
    NativeStepKind.REASON -> 1; NativeStepKind.RETRIEVE -> 2; NativeStepKind.VERIFY -> 3
    NativeStepKind.RESPOND -> 4; NativeStepKind.EXTERNAL -> 5
}
