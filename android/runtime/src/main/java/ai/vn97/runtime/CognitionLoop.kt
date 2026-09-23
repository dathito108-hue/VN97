package ai.vn97.runtime

import java.nio.ByteBuffer
import java.nio.charset.CodingErrorAction
import java.nio.charset.StandardCharsets

data class NativeCognitionLimits(
    val maxPlanSteps: Int = 16,
    val maxExternalSteps: Int = 4,
    val maxObjectiveUtf8Bytes: Int = 4096,
    val maxResultUtf8Bytes: Int = 64 * 1024,
    val maxNoteUtf8Bytes: Int = 8192,
    val maxDependencyContextUtf8Bytes: Int = 32 * 1024,
    val maxMemoryContextUtf8Bytes: Int = 64 * 1024,
    val maxCyclesPerRun: Int = 64,
    val requireFinalResponse: Boolean = true,
) {
    init {
        val numeric = listOf(
            maxPlanSteps,
            maxExternalSteps,
            maxObjectiveUtf8Bytes,
            maxResultUtf8Bytes,
            maxNoteUtf8Bytes,
            maxDependencyContextUtf8Bytes,
            maxMemoryContextUtf8Bytes,
            maxCyclesPerRun,
        )
        require(numeric.all { it > 0 }) { "all cognition limits must be positive" }
        require(maxExternalSteps <= maxPlanSteps) {
            "maxExternalSteps cannot exceed maxPlanSteps"
        }
    }
}

enum class NativeCognitionBoundary {
    COMPLETED,
    WAITING_EXTERNAL,
    PAUSED,
    FAILED,
    CANCELLED,
    BUDGET_EXHAUSTED,
    YIELDED,
    STALLED,
}

data class NativeCognitionRunResult(
    val boundary: NativeCognitionBoundary,
    val planStatus: NativePlanStatus,
    val cycles: Int,
    val finalResponse: String = "",
)

data class NativePlanRevision(
    val previousPlanId: String,
    val controller: NativePlanController,
)

class NativeCognitionLoopException(
    message: String,
    cause: Throwable? = null,
) : IllegalStateException(message, cause)

interface NativeMemoryRetriever {
    val vectorDim: Int

    fun retrieve(
        query: NativeMemoryQuery,
        topK: Int,
    ): List<NativeMemoryContextItem>
}

class NativeCognitionLoop(
    private val backend: NativeTypedCognitionAdapter,
    private val limits: NativeCognitionLimits = NativeCognitionLimits(),
) {
    fun buildPlan(
        goal: String,
        budget: NativeReasoningBudget = NativeReasoningBudget(),
        createdNs: Long = System.currentTimeMillis() * 1_000_000L,
    ): NativePlanController {
        require(goal.isNotBlank()) { "goal must not be empty" }
        val draft = call("backend proposePlan") {
            backend.proposePlan(
                NativePlanDraftRequest(
                    goal = goal,
                    maxSteps = limits.maxPlanSteps,
                )
            )
        }
        validateDraft(draft)
        return try {
            NativePlanController.create(
                goal = goal,
                specs = draft.steps,
                budget = budget,
                createdNs = createdNs,
            )
        } catch (exc: IllegalArgumentException) {
            throw NativeCognitionContractException(
                "backend plan draft is structurally invalid",
                exc,
            )
        }
    }

    fun refineUnstartedPlan(
        controller: NativePlanController,
        feedback: String,
        createdNs: Long = System.currentTimeMillis() * 1_000_000L,
    ): NativePlanRevision {
        val plan = controller.plan
        if (
            plan.status != NativePlanStatus.READY ||
            plan.transitionsUsed != 0 ||
            plan.memoryQueriesUsed != 0 ||
            plan.steps.any { it.status != NativeStepStatus.PENDING }
        ) {
            throw NativeCognitionLoopException(
                "only an unstarted READY plan can be refined"
            )
        }
        val boundedFeedback = truncateUtf8(feedback, limits.maxNoteUtf8Bytes).first
        val draft = call("backend proposePlan") {
            backend.proposePlan(
                NativePlanDraftRequest(
                    goal = plan.goal,
                    maxSteps = limits.maxPlanSteps,
                    previousPlanId = plan.planId,
                    previousSteps = plan.steps.map { it.spec },
                    feedback = boundedFeedback,
                )
            )
        }
        validateDraft(draft)
        val replacement = try {
            NativePlanController.create(
                goal = plan.goal,
                specs = draft.steps,
                budget = plan.budget,
                createdNs = createdNs,
            )
        } catch (exc: IllegalArgumentException) {
            throw NativeCognitionContractException(
                "refined plan draft is structurally invalid",
                exc,
            )
        }
        return NativePlanRevision(plan.planId, replacement)
    }

    fun replanTerminalPlan(
        controller: NativePlanController,
        feedback: String,
        createdNs: Long = System.currentTimeMillis() * 1_000_000L,
    ): NativePlanRevision {
        val plan = controller.plan
        require(plan.isTerminal()) {
            "only a terminal plan can start a new autonomous generation"
        }
        require(createdNs >= 0L) {
            "replan createdNs must be non-negative"
        }
        val boundedFeedback =
            truncateUtf8(feedback, limits.maxNoteUtf8Bytes).first
        require(boundedFeedback.isNotBlank()) {
            "terminal replan feedback must not be blank"
        }
        val draft = call("backend proposePlan") {
            backend.proposePlan(
                NativePlanDraftRequest(
                    goal = plan.goal,
                    maxSteps = limits.maxPlanSteps,
                    previousPlanId = plan.planId,
                    previousSteps = plan.steps.map { it.spec },
                    feedback = boundedFeedback,
                )
            )
        }
        validateDraft(draft)
        val replacement = try {
            NativePlanController.create(
                goal = plan.goal,
                specs = draft.steps,
                budget = plan.budget,
                createdNs = createdNs,
            )
        } catch (exc: IllegalArgumentException) {
            throw NativeCognitionContractException(
                "terminal replan draft is structurally invalid",
                exc,
            )
        }
        if (replacement.plan.planId == plan.planId) {
            throw NativeCognitionLoopException(
                "terminal replan did not change plan identity"
            )
        }
        return NativePlanRevision(
            previousPlanId = plan.planId,
            controller = replacement,
        )
    }

    fun finalResponse(controller: NativePlanController): String =
        controller.plan.steps.asReversed().firstOrNull {
            it.spec.kind == NativeStepKind.RESPOND &&
                it.status == NativeStepStatus.SUCCEEDED
        }?.result.orEmpty()

    fun runUntilBoundary(
        controller: NativePlanController,
        memory: NativeMemoryRetriever? = null,
        maxCycles: Int = limits.maxCyclesPerRun,
    ): NativeCognitionRunResult {
        require(maxCycles > 0) { "maxCycles must be positive" }
        var cycles = 0
        while (cycles < maxCycles) {
            if (controller.plan.isTerminal()) break
            if (
                controller.plan.status == NativePlanStatus.PAUSED ||
                controller.plan.status == NativePlanStatus.WAITING_EXTERNAL
            ) {
                break
            }

            val verification = controller.plan.steps.firstOrNull {
                it.status == NativeStepStatus.WAITING_VERIFICATION
            }
            if (verification != null) {
                try {
                    driveVerification(controller, verification)
                } catch (exc: NativePlannerException) {
                    if (
                        controller.plan.status !=
                        NativePlanStatus.BUDGET_EXHAUSTED
                    ) {
                        throw exc
                    }
                }
                cycles += 1
                continue
            }

            val directive = controller.nextDirective() ?: break
            val step = controller.plan.step(directive.stepId)
            try {
                driveStep(controller, step, memory)
            } catch (exc: NativePlannerException) {
                if (
                    controller.plan.status !=
                    NativePlanStatus.BUDGET_EXHAUSTED
                ) {
                    throw exc
                }
            }
            cycles += 1
        }

        val boundary = if (
            cycles >= maxCycles &&
            !controller.plan.isTerminal() &&
            controller.plan.status != NativePlanStatus.PAUSED &&
            controller.plan.status != NativePlanStatus.WAITING_EXTERNAL
        ) {
            NativeCognitionBoundary.YIELDED
        } else {
            boundary(controller.plan.status)
        }
        return NativeCognitionRunResult(
            boundary = boundary,
            planStatus = controller.plan.status,
            cycles = cycles,
            finalResponse = finalResponse(controller),
        )
    }

    private fun validateDraft(draft: NativePlanDraft) {
        if (draft.steps.size > limits.maxPlanSteps) {
            throw NativeCognitionContractException(
                "backend plan exceeds maxPlanSteps"
            )
        }
        val external = draft.steps.count { it.kind == NativeStepKind.EXTERNAL }
        if (external > limits.maxExternalSteps) {
            throw NativeCognitionContractException(
                "backend plan exceeds maxExternalSteps"
            )
        }
        for (step in draft.steps) {
            if (utf8Size(step.objective) > limits.maxObjectiveUtf8Bytes) {
                throw NativeCognitionContractException(
                    "backend step objective exceeds byte limit"
                )
            }
        }
        if (
            limits.requireFinalResponse &&
            draft.steps.last().kind != NativeStepKind.RESPOND
        ) {
            throw NativeCognitionContractException(
                "backend plan must end with RESPOND"
            )
        }
    }

    private fun driveVerification(
        controller: NativePlanController,
        step: NativePlannerStep,
    ) {
        val dependencies = dependencyResults(controller, step).first
        val confidence = step.confidence
            ?: throw NativeCognitionLoopException(
                "verification candidate lacks confidence"
            )
        val request = NativeVerificationRequest(
            planId = controller.plan.planId,
            goal = controller.plan.goal,
            stepId = step.stepId,
            kind = step.spec.kind,
            objective = step.spec.objective,
            attempt = step.attempts,
            candidate = step.result,
            confidence = confidence,
            dependencies = dependencies,
            evidenceRecordIds = step.evidenceRecordIds,
        )
        val decision = try {
            backend.verifyStep(request)
        } catch (exc: Exception) {
            controller.verifyStep(
                step.stepId,
                passed = false,
                note = "verification backend failure: ${exc::class.java.simpleName}",
            )
            return
        }

        if (utf8Size(decision.note) > limits.maxNoteUtf8Bytes) {
            safeFail(
                controller,
                step.stepId,
                "backend verification note exceeds byte limit",
                retryable = false,
            )
            return
        }
        controller.verifyStep(
            step.stepId,
            passed = decision.passed,
            note = decision.note,
        )
    }

    private fun driveStep(
        controller: NativePlanController,
        step: NativePlannerStep,
        memory: NativeMemoryRetriever?,
    ) {
        val previousFailure = step.failureReason
        controller.beginStep(step.stepId)
        if (step.spec.kind == NativeStepKind.EXTERNAL) return

        val (dependencies, dependencyTruncated) =
            dependencyResults(controller, step)
        var memoryContext = emptyList<NativeMemoryContextItem>()
        var memoryTruncated = false

        if (step.spec.kind == NativeStepKind.RETRIEVE) {
            if (memory == null) {
                safeFail(
                    controller,
                    step.stepId,
                    "retrieval step requires a memory retriever",
                    retryable = false,
                )
                return
            }
            val query: NativeMemoryQuery
            val retrieved: List<NativeMemoryContextItem>
            val topK: Int
            try {
                query = backend.memoryQuery(
                    NativeMemoryQueryRequest(
                        planId = controller.plan.planId,
                        goal = controller.plan.goal,
                        stepId = step.stepId,
                        objective = step.spec.objective,
                        attempt = step.attempts,
                        previousFailure = previousFailure,
                        dependencies = dependencies,
                        vectorDim = memory.vectorDim,
                    )
                )
                if (query.vector.size != memory.vectorDim) {
                    safeFail(
                        controller,
                        step.stepId,
                        "backend memory query vector has wrong dimension",
                        retryable = false,
                    )
                    return
                }
                topK = controller.consumeMemoryQuery(query.topK)
                retrieved = memory.retrieve(query, topK)
            } catch (exc: NativePlannerException) {
                throw exc
            } catch (exc: Exception) {
                safeFail(
                    controller,
                    step.stepId,
                    "memory cognition failure: ${exc::class.java.simpleName}",
                    retryable = true,
                )
                return
            }
            if (retrieved.size > topK) {
                safeFail(
                    controller,
                    step.stepId,
                    "memory retriever returned more than bounded topK",
                    retryable = false,
                )
                return
            }
            val bounded = boundedMemoryContext(retrieved)
            memoryContext = bounded.first
            memoryTruncated = bounded.second
        }

        val proposal = try {
            backend.proposeStep(
                NativeStepReasoningRequest(
                    planId = controller.plan.planId,
                    goal = controller.plan.goal,
                    stepId = step.stepId,
                    kind = step.spec.kind,
                    objective = step.spec.objective,
                    attempt = step.attempts,
                    previousFailure = previousFailure,
                    dependencies = dependencies,
                    memoryContext = memoryContext,
                    contextTruncated =
                        dependencyTruncated || memoryTruncated,
                )
            )
        } catch (exc: Exception) {
            safeFail(
                controller,
                step.stepId,
                "cognition backend failure: ${exc::class.java.simpleName}",
                retryable = true,
            )
            return
        }

        if (utf8Size(proposal.result) > limits.maxResultUtf8Bytes) {
            safeFail(
                controller,
                step.stepId,
                "backend proposal result exceeds byte limit",
                retryable = false,
            )
            return
        }

        controller.completeStep(
            id = step.stepId,
            result = proposal.result,
            confidence = proposal.confidence,
            evidenceRecordIds =
                inheritedEvidence(dependencies, memoryContext),
        )
    }

    private fun dependencyResults(
        controller: NativePlanController,
        step: NativePlannerStep,
    ): Pair<List<NativeDependencyResult>, Boolean> {
        var remaining = limits.maxDependencyContextUtf8Bytes
        var truncated = false
        val results = ArrayList<NativeDependencyResult>()
        step.spec.dependencies.forEachIndexed { index, dependencyId ->
            val dependency = controller.plan.step(dependencyId)
            if (dependency.status != NativeStepStatus.SUCCEEDED) {
                throw NativeCognitionLoopException(
                    "dependency is not succeeded"
                )
            }
            val boundedObjective = truncateUtf8(
                dependency.spec.objective,
                minOf(remaining, limits.maxObjectiveUtf8Bytes),
            )
            remaining = (remaining - utf8Size(boundedObjective.first))
                .coerceAtLeast(0)
            val boundedResult = truncateUtf8(
                dependency.result,
                remaining,
            )
            remaining = (remaining - utf8Size(boundedResult.first))
                .coerceAtLeast(0)
            truncated = truncated ||
                boundedObjective.second ||
                boundedResult.second
            val confidence = dependency.confidence
                ?: throw NativeCognitionLoopException(
                    "succeeded dependency lacks confidence"
                )
            results += NativeDependencyResult(
                stepId = dependency.stepId,
                kind = dependency.spec.kind,
                objective = boundedObjective.first,
                result = boundedResult.first,
                confidence = confidence,
                evidenceRecordIds = dependency.evidenceRecordIds,
            )
            if (
                remaining == 0 &&
                index != step.spec.dependencies.lastIndex
            ) {
                truncated = true
            }
        }
        return results to truncated
    }

    private fun boundedMemoryContext(
        context: List<NativeMemoryContextItem>,
    ): Pair<List<NativeMemoryContextItem>, Boolean> {
        var remaining = limits.maxMemoryContextUtf8Bytes
        var truncated = false
        val items = ArrayList<NativeMemoryContextItem>()
        context.forEachIndexed { index, item ->
            val source = truncateUtf8(item.source, remaining)
            remaining = (remaining - utf8Size(source.first)).coerceAtLeast(0)
            val content = truncateUtf8(item.content, remaining)
            remaining = (remaining - utf8Size(content.first)).coerceAtLeast(0)
            truncated = truncated || source.second || content.second
            items += item.copy(
                source = source.first,
                content = content.first,
            )
            if (remaining == 0 && index != context.lastIndex) {
                truncated = true
            }
        }
        return items to truncated
    }

    private fun inheritedEvidence(
        dependencies: List<NativeDependencyResult>,
        memoryContext: List<NativeMemoryContextItem>,
    ): List<Long> {
        val seen = LinkedHashSet<Long>()
        dependencies.forEach { seen.addAll(it.evidenceRecordIds) }
        memoryContext.forEach { seen.add(it.recordId) }
        return seen.toList()
    }

    private fun safeFail(
        controller: NativePlanController,
        stepId: Int,
        reason: String,
        retryable: Boolean,
    ) {
        try {
            controller.failStep(
                id = stepId,
                reason = reason,
                retryable = retryable,
            )
        } catch (exc: NativePlannerException) {
            if (
                controller.plan.status !=
                NativePlanStatus.BUDGET_EXHAUSTED
            ) {
                throw exc
            }
        }
    }

    private fun boundary(status: NativePlanStatus): NativeCognitionBoundary =
        when (status) {
            NativePlanStatus.COMPLETED -> NativeCognitionBoundary.COMPLETED
            NativePlanStatus.WAITING_EXTERNAL ->
                NativeCognitionBoundary.WAITING_EXTERNAL
            NativePlanStatus.PAUSED -> NativeCognitionBoundary.PAUSED
            NativePlanStatus.FAILED -> NativeCognitionBoundary.FAILED
            NativePlanStatus.CANCELLED -> NativeCognitionBoundary.CANCELLED
            NativePlanStatus.BUDGET_EXHAUSTED ->
                NativeCognitionBoundary.BUDGET_EXHAUSTED
            else -> NativeCognitionBoundary.STALLED
        }

    private fun <T> call(label: String, block: () -> T): T = try {
        block()
    } catch (exc: Exception) {
        throw NativeCognitionLoopException(
            "$label raised ${exc::class.java.simpleName}",
            exc,
        )
    }
}

private fun utf8Size(text: String): Int =
    text.toByteArray(StandardCharsets.UTF_8).size

private fun truncateUtf8(
    text: String,
    maxBytes: Int,
): Pair<String, Boolean> {
    require(maxBytes >= 0)
    val bytes = text.toByteArray(StandardCharsets.UTF_8)
    if (bytes.size <= maxBytes) return text to false
    var end = maxBytes
    while (end > 0) {
        val decoder = StandardCharsets.UTF_8.newDecoder()
            .onMalformedInput(CodingErrorAction.REPORT)
            .onUnmappableCharacter(CodingErrorAction.REPORT)
        try {
            val decoded = decoder.decode(
                ByteBuffer.wrap(bytes, 0, end)
            ).toString()
            return decoded to true
        } catch (_: Exception) {
            end -= 1
        }
    }
    return "" to true
}
