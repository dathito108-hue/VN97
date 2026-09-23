package ai.vn97.platform

import ai.vn97.runtime.NativeCognitionBoundary
import ai.vn97.runtime.NativeMemoryRetriever
import ai.vn97.runtime.NativePlanController
import ai.vn97.runtime.NativeReasoningBudget
import java.nio.charset.StandardCharsets

/** Bounded host-facing turn orchestration above the canonical M7S coordinator. */
data class VN97AssistantSessionLimits(
    val maxUserTurnUtf8Bytes: Int = 32 * 1024,
    val maxCyclesPerAdvance: Int = 64,
    val maxExternalHandoffsPerAdvance: Int = 4,
    val promptTtlNs: Long = 300_000_000_000L,
    val approvalTtlNs: Long = 60_000_000_000L,
) {
    init {
        require(maxUserTurnUtf8Bytes > 0) { "maxUserTurnUtf8Bytes must be positive" }
        require(maxCyclesPerAdvance > 0) { "maxCyclesPerAdvance must be positive" }
        require(maxExternalHandoffsPerAdvance > 0) {
            "maxExternalHandoffsPerAdvance must be positive"
        }
        require(promptTtlNs > 0L) { "promptTtlNs must be positive" }
        require(approvalTtlNs > 0L) { "approvalTtlNs must be positive" }
    }
}

enum class VN97AssistantTurnState {
    COMPLETED,
    APPROVAL_REQUIRED,
    APPROVAL_REJECTED,
    YIELDED,
    PAUSED,
    FAILED,
    CANCELLED,
    BUDGET_EXHAUSTED,
    STALLED,
}

class VN97AssistantTurn internal constructor(
    val turnId: Long,
    internal val controller: NativePlanController,
    val principal: String,
) {
    val planId: String get() = controller.plan.planId
    val goal: String get() = controller.plan.goal
}

class VN97AssistantApproval internal constructor(
    internal val pending: M6PendingExternalApproval,
) {
    val planId: String get() = pending.planId
    val stepId: Int get() = pending.stepId
    val capabilityId: String get() = pending.capabilityId
    val scopeDigest: String get() = pending.scopeDigest
    val requestDigest: String get() = pending.requestDigest
    val presentationJson: String get() = pending.presentationJson
    val expiresNs: Long get() = pending.prompt.expiresNs
}

data class VN97AssistantTurnUpdate(
    val turn: VN97AssistantTurn,
    val state: VN97AssistantTurnState,
    val finalResponse: String = "",
    val approval: VN97AssistantApproval? = null,
    val executions: List<M6ExecutionResult> = emptyList(),
) {
    init {
        require(
            (state == VN97AssistantTurnState.APPROVAL_REQUIRED) == (approval != null)
        ) { "approval is present only for APPROVAL_REQUIRED" }
        if (state != VN97AssistantTurnState.COMPLETED) {
            require(finalResponse.isEmpty()) {
                "finalResponse is present only for COMPLETED turns"
            }
        }
    }
}

/**
 * Production-facing assistant turn lifecycle.
 *
 * This class owns no model/backend, authority, approval secret, capability registry or side-effect
 * handler. Those remain below M7S/M6. It only drives one canonical planner turn at a time and
 * exposes user-visible boundaries.
 */
class VN97AssistantSession(
    private val coordinator: M6EndToEndExternalCoordinator,
    val limits: VN97AssistantSessionLimits = VN97AssistantSessionLimits(),
    private val defaultMemory: NativeMemoryRetriever? = null,
) {
    private var active: VN97AssistantTurn? = null
    private var activeMemory: NativeMemoryRetriever? = null
    private var pendingApproval: M6PendingExternalApproval? = null
    private var nextTurnId = 1L

    val hasActiveTurn: Boolean
        @Synchronized get() = active != null

    @Synchronized
    fun startTurn(
        userMessage: String,
        principal: String,
        memory: NativeMemoryRetriever? = null,
        budget: NativeReasoningBudget = NativeReasoningBudget(),
        createdNs: Long = System.currentTimeMillis() * 1_000_000L,
        nowNs: Long,
    ): VN97AssistantTurnUpdate {
        check(active == null) { "an assistant turn is already active" }
        validateUserMessage(userMessage)
        validatePrincipal(principal)
        require(createdNs >= 0L) { "createdNs must be non-negative" }
        require(nowNs >= 0L) { "nowNs must be non-negative" }

        val turnMemory = memoryForNewTurn(memory)
        val controller = coordinator.buildPlan(
            goal = userMessage,
            budget = budget,
            createdNs = createdNs,
        )
        val turn = createActiveTurn(controller, principal, turnMemory)
        return try {
            drive(turn, turnMemory, nowNs, seed = null)
        } catch (exc: Throwable) {
            clearIfActive(turn)
            throw exc
        }
    }

    /**
     * Reattach a VN97PLN1/M7M-restored planner to the same production turn lifecycle.
     * Any prior approval prompt is intentionally not restored. WAITING_EXTERNAL is rebound through
     * M7O/M7P: a successful M7Q receipt replays, otherwise authority requires a fresh approval.
     */
    @Synchronized
    fun resumeRestoredTurn(
        controller: NativePlanController,
        principal: String,
        memory: NativeMemoryRetriever? = null,
        nowNs: Long,
    ): VN97AssistantTurnUpdate {
        check(active == null) { "an assistant turn is already active" }
        validatePrincipal(principal)
        require(nowNs >= 0L) { "nowNs must be non-negative" }
        require(!controller.plan.isTerminal()) { "restored planner is already terminal" }

        val turnMemory = memoryForNewTurn(memory)
        val turn = createActiveTurn(controller, principal, turnMemory)
        return try {
            drive(turn, turnMemory, nowNs, seed = null)
        } catch (exc: Throwable) {
            clearIfActive(turn)
            throw exc
        }
    }

    @Synchronized
    fun continueTurn(
        turn: VN97AssistantTurn,
        memory: NativeMemoryRetriever? = null,
        nowNs: Long,
    ): VN97AssistantTurnUpdate {
        requireActive(turn)
        check(pendingApproval == null) {
            "the active turn is waiting for approval resolution"
        }
        require(nowNs >= 0L) { "nowNs must be non-negative" }
        val turnMemory = memoryForActiveTurn(memory)
        return try {
            drive(turn, turnMemory, nowNs, seed = null)
        } catch (exc: Throwable) {
            clearIfActive(turn)
            throw exc
        }
    }

    @Synchronized
    fun resolveApproval(
        turn: VN97AssistantTurn,
        approval: VN97AssistantApproval,
        approved: Boolean,
        memory: NativeMemoryRetriever? = null,
        nowNs: Long,
    ): VN97AssistantTurnUpdate {
        requireActive(turn)
        require(nowNs >= 0L) { "nowNs must be non-negative" }
        val turnMemory = memoryForActiveTurn(memory)
        val expected = checkNotNull(pendingApproval) {
            "the active turn is not waiting for approval"
        }
        check(expected === approval.pending) {
            "approval does not belong to the active turn"
        }
        pendingApproval = null

        return try {
            val resolved = coordinator.resolveApproval(
                controller = turn.controller,
                pending = expected,
                approved = approved,
                memory = turnMemory,
                maxCycles = limits.maxCyclesPerAdvance,
                approvalTtlNs = limits.approvalTtlNs,
                nowNs = nowNs,
            )
            drive(turn, turnMemory, nowNs, seed = resolved)
        } catch (exc: Throwable) {
            clearIfActive(turn)
            throw exc
        }
    }

    private fun createActiveTurn(
        controller: NativePlanController,
        principal: String,
        memory: NativeMemoryRetriever?,
    ): VN97AssistantTurn {
        if (nextTurnId == Long.MAX_VALUE) {
            throw IllegalStateException("assistant turn sequence exhausted")
        }
        val turn = VN97AssistantTurn(nextTurnId++, controller, principal)
        active = turn
        activeMemory = memory
        pendingApproval = null
        return turn
    }

    private fun drive(
        turn: VN97AssistantTurn,
        memory: NativeMemoryRetriever?,
        nowNs: Long,
        seed: M6ExternalCoordinatorResult?,
    ): VN97AssistantTurnUpdate {
        val executions = ArrayList<M6ExecutionResult>()
        var result = seed ?: coordinator.advance(
            controller = turn.controller,
            principal = turn.principal,
            memory = memory,
            maxCycles = limits.maxCyclesPerAdvance,
            promptTtlNs = limits.promptTtlNs,
            nowNs = nowNs,
        )
        var externalHandoffs = 0

        while (true) {
            result.execution?.let {
                executions += it
                externalHandoffs += 1
            }

            when (result.event) {
                M6ExternalCoordinatorEvent.APPROVAL_REQUIRED -> {
                    val pending = checkNotNull(result.pendingApproval)
                    pendingApproval = pending
                    return VN97AssistantTurnUpdate(
                        turn = turn,
                        state = VN97AssistantTurnState.APPROVAL_REQUIRED,
                        approval = VN97AssistantApproval(pending),
                        executions = executions.toList(),
                    )
                }

                M6ExternalCoordinatorEvent.APPROVAL_REJECTED -> {
                    clearIfActive(turn)
                    return VN97AssistantTurnUpdate(
                        turn = turn,
                        state = VN97AssistantTurnState.APPROVAL_REJECTED,
                        executions = executions.toList(),
                    )
                }

                M6ExternalCoordinatorEvent.NONE,
                M6ExternalCoordinatorEvent.EXECUTED,
                -> Unit
            }

            if (result.cognition.boundary != NativeCognitionBoundary.WAITING_EXTERNAL) {
                return boundaryUpdate(turn, result, executions)
            }
            if (externalHandoffs >= limits.maxExternalHandoffsPerAdvance) {
                return VN97AssistantTurnUpdate(
                    turn = turn,
                    state = VN97AssistantTurnState.YIELDED,
                    executions = executions.toList(),
                )
            }

            result = coordinator.advance(
                controller = turn.controller,
                principal = turn.principal,
                memory = memory,
                maxCycles = limits.maxCyclesPerAdvance,
                promptTtlNs = limits.promptTtlNs,
                nowNs = nowNs,
            )
        }
    }

    private fun boundaryUpdate(
        turn: VN97AssistantTurn,
        result: M6ExternalCoordinatorResult,
        executions: List<M6ExecutionResult>,
    ): VN97AssistantTurnUpdate {
        val state = when (result.cognition.boundary) {
            NativeCognitionBoundary.COMPLETED -> VN97AssistantTurnState.COMPLETED
            NativeCognitionBoundary.WAITING_EXTERNAL,
            NativeCognitionBoundary.YIELDED,
            -> VN97AssistantTurnState.YIELDED
            NativeCognitionBoundary.PAUSED -> VN97AssistantTurnState.PAUSED
            NativeCognitionBoundary.FAILED -> VN97AssistantTurnState.FAILED
            NativeCognitionBoundary.CANCELLED -> VN97AssistantTurnState.CANCELLED
            NativeCognitionBoundary.BUDGET_EXHAUSTED -> VN97AssistantTurnState.BUDGET_EXHAUSTED
            NativeCognitionBoundary.STALLED -> VN97AssistantTurnState.STALLED
        }
        val terminalForSession = state in setOf(
            VN97AssistantTurnState.COMPLETED,
            VN97AssistantTurnState.FAILED,
            VN97AssistantTurnState.CANCELLED,
            VN97AssistantTurnState.BUDGET_EXHAUSTED,
            VN97AssistantTurnState.STALLED,
        )
        if (terminalForSession) clearIfActive(turn)
        return VN97AssistantTurnUpdate(
            turn = turn,
            state = state,
            finalResponse = if (state == VN97AssistantTurnState.COMPLETED) {
                result.cognition.finalResponse
            } else {
                ""
            },
            executions = executions.toList(),
        )
    }

    private fun memoryForNewTurn(
        memory: NativeMemoryRetriever?,
    ): NativeMemoryRetriever? {
        val bound = defaultMemory
        if (bound != null) {
            check(memory == null || memory === bound) {
                "cannot override the session-bound sovereign memory"
            }
            return bound
        }
        return memory
    }

    private fun memoryForActiveTurn(
        memory: NativeMemoryRetriever?,
    ): NativeMemoryRetriever? {
        val bound = activeMemory
        check(memory == null || memory === bound) {
            "cannot change sovereign memory during an active assistant turn"
        }
        return bound
    }

    private fun requireActive(turn: VN97AssistantTurn) {
        check(active === turn) { "turn is not the active assistant turn" }
    }

    private fun clearIfActive(turn: VN97AssistantTurn) {
        if (active === turn) {
            active = null
            activeMemory = null
            pendingApproval = null
        }
    }

    private fun validateUserMessage(value: String) {
        require(value.isNotBlank()) { "userMessage must not be blank" }
        require(value.toByteArray(StandardCharsets.UTF_8).size <= limits.maxUserTurnUtf8Bytes) {
            "userMessage exceeds maxUserTurnUtf8Bytes"
        }
    }

    private fun validatePrincipal(value: String) {
        val allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._:-/".toSet()
        require(value.isNotEmpty()) { "principal must not be empty" }
        require(value.toByteArray(StandardCharsets.UTF_8).size <= 256) {
            "principal exceeds byte bound"
        }
        require(value.all { it in allowed }) {
            "principal contains unsupported characters"
        }
    }
}
