package ai.vn97.platform

import ai.vn97.runtime.NativeCognitionBoundary
import ai.vn97.runtime.NativeCognitionLimits
import ai.vn97.runtime.NativeCognitionRunResult
import ai.vn97.runtime.NativeMemoryRetriever
import ai.vn97.runtime.NativePlanController
import ai.vn97.runtime.NativeReasoningBudget
import ai.vn97.runtime.NativeTypedCognitionAdapter
import ai.vn97.runtime.NativeCognitionLoop

class M6ExternalCoordinatorException(
    message: String,
    cause: Throwable? = null,
) : IllegalStateException(message, cause)

enum class M6ExternalCoordinatorEvent {
    NONE,
    APPROVAL_REQUIRED,
    EXECUTED,
    APPROVAL_REJECTED,
}

class M6PendingExternalApproval internal constructor(
    internal val request: M6ExternalActionRequest,
    val principal: String,
    val prompt: M6ApprovalPrompt,
) {
    val planId: String get() = request.planId
    val stepId: Int get() = request.stepId
    val capabilityId: String get() = request.capabilityId
    val scopeDigest: String get() = request.scope.digest
    val requestDigest: String get() = request.requestDigest
    val presentationJson: String get() = prompt.presentationJson
}

data class M6ExternalCoordinatorResult(
    val event: M6ExternalCoordinatorEvent,
    val cognition: NativeCognitionRunResult,
    val execution: M6ExecutionResult? = null,
    val pendingApproval: M6PendingExternalApproval? = null,
) {
    init {
        when (event) {
            M6ExternalCoordinatorEvent.NONE,
            M6ExternalCoordinatorEvent.APPROVAL_REJECTED,
            -> require(execution == null && pendingApproval == null)

            M6ExternalCoordinatorEvent.APPROVAL_REQUIRED ->
                require(execution == null && pendingApproval != null)

            M6ExternalCoordinatorEvent.EXECUTED ->
                require(execution != null && pendingApproval == null)
        }
    }
}

class M6EndToEndExternalCoordinator(
    cognition: NativeTypedCognitionAdapter,
    private val binder: M6ExternalIntentBinder,
    private val approvals: M6ExternalApprovalHandoff,
    private val executionFabric: M6ExternalExecutionFabric,
    cognitionLimits: NativeCognitionLimits = NativeCognitionLimits(),
) {
    private val cognitionAdapter = cognition
    private val cognitionLoop = NativeCognitionLoop(cognition, cognitionLimits)

    fun buildPlan(
        goal: String,
        budget: NativeReasoningBudget = NativeReasoningBudget(),
        createdNs: Long = System.currentTimeMillis() * 1_000_000L,
    ): NativePlanController = cognitionLoop.buildPlan(
        goal = goal,
        budget = budget,
        createdNs = createdNs,
    )

    fun advance(
        controller: NativePlanController,
        principal: String,
        memory: NativeMemoryRetriever? = null,
        maxCycles: Int = 64,
        promptTtlNs: Long = 300_000_000_000L,
        nowNs: Long,
    ): M6ExternalCoordinatorResult {
        require(maxCycles > 0) { "maxCycles must be positive" }
        require(promptTtlNs > 0L) { "promptTtlNs must be positive" }
        require(nowNs >= 0L) { "nowNs must be non-negative" }

        val boundary = cognitionLoop.runUntilBoundary(
            controller = controller,
            memory = memory,
            maxCycles = maxCycles,
        )
        if (boundary.boundary != NativeCognitionBoundary.WAITING_EXTERNAL) {
            return M6ExternalCoordinatorResult(
                event = M6ExternalCoordinatorEvent.NONE,
                cognition = boundary,
            )
        }

        val request = binder.proposeAndBind(controller, cognitionAdapter)
        return try {
            val executed = executionFabric.executeWaiting(
                controller = controller,
                request = request,
                principal = principal,
                nowNs = nowNs,
            )
            M6ExternalCoordinatorResult(
                event = M6ExternalCoordinatorEvent.EXECUTED,
                cognition = cognitionLoop.runUntilBoundary(
                    controller = controller,
                    memory = memory,
                    maxCycles = maxCycles,
                ),
                execution = executed,
            )
        } catch (_: M6ApprovalRequiredException) {
            val prompt = approvals.createPrompt(
                request = request,
                principal = principal,
                promptTtlNs = promptTtlNs,
                nowNs = nowNs,
            )
            M6ExternalCoordinatorResult(
                event = M6ExternalCoordinatorEvent.APPROVAL_REQUIRED,
                cognition = boundary,
                pendingApproval = M6PendingExternalApproval(
                    request = request,
                    principal = principal,
                    prompt = prompt,
                ),
            )
        }
    }

    fun resolveApproval(
        controller: NativePlanController,
        pending: M6PendingExternalApproval,
        approved: Boolean,
        memory: NativeMemoryRetriever? = null,
        maxCycles: Int = 64,
        approvalTtlNs: Long = 60_000_000_000L,
        nowNs: Long,
    ): M6ExternalCoordinatorResult {
        require(maxCycles > 0) { "maxCycles must be positive" }
        require(approvalTtlNs > 0L) { "approvalTtlNs must be positive" }
        require(nowNs >= 0L) { "nowNs must be non-negative" }
        validatePending(controller, pending)

        val token = approvals.resolve(
            prompt = pending.prompt,
            approved = approved,
            approvalTtlNs = approvalTtlNs,
            nowNs = nowNs,
        )
        if (token == null) {
            controller.failStep(
                pending.stepId,
                "external action approval rejected",
                retryable = false,
            )
            return M6ExternalCoordinatorResult(
                event = M6ExternalCoordinatorEvent.APPROVAL_REJECTED,
                cognition = cognitionLoop.runUntilBoundary(
                    controller = controller,
                    memory = memory,
                    maxCycles = maxCycles,
                ),
            )
        }

        val executed = executionFabric.executeWaiting(
            controller = controller,
            request = pending.request,
            principal = pending.principal,
            approval = token,
            nowNs = nowNs,
        )
        return M6ExternalCoordinatorResult(
            event = M6ExternalCoordinatorEvent.EXECUTED,
            cognition = cognitionLoop.runUntilBoundary(
                controller = controller,
                memory = memory,
                maxCycles = maxCycles,
            ),
            execution = executed,
        )
    }

    private fun validatePending(
        controller: NativePlanController,
        pending: M6PendingExternalApproval,
    ) {
        if (pending.prompt.principal != pending.principal) {
            throw M6ExternalCoordinatorException(
                "pending approval principal binding is invalid"
            )
        }
        if (pending.prompt.requestDigest != pending.requestDigest) {
            throw M6ExternalCoordinatorException(
                "pending approval request binding is invalid"
            )
        }
        val snapshot = try {
            binder.requestForBackend(controller)
        } catch (exc: Exception) {
            throw M6ExternalCoordinatorException(
                "controller is no longer at the pending external boundary",
                exc,
            )
        }
        if (
            snapshot.planId != pending.planId ||
            snapshot.stepId != pending.stepId ||
            snapshot.objective != pending.request.objective
        ) {
            throw M6ExternalCoordinatorException(
                "pending approval no longer matches the immutable waiting step"
            )
        }
    }
}
