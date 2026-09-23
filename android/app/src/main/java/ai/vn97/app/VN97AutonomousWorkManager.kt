package ai.vn97.app

import ai.vn97.platform.AndroidContinuationScheduler
import ai.vn97.platform.ContinuationOutcome
import ai.vn97.platform.ComputeMode
import ai.vn97.platform.VN97AssistantContinuationContext
import ai.vn97.platform.VN97AssistantContinuationSpec
import ai.vn97.platform.VN97AssistantContinuationWork
import ai.vn97.platform.VN97AssistantSessionLimits
import ai.vn97.platform.VN97AssistantTurnState
import ai.vn97.platform.VN97ForegroundAssistantContinuationSession
import ai.vn97.platform.createVN97AutonomousReplanSeed
import ai.vn97.platform.openVN97ForegroundAssistantContinuation
import ai.vn97.runtime.NativeActivatedInventoryModelLoader
import ai.vn97.runtime.NativeBackend
import ai.vn97.runtime.NativePlan
import ai.vn97.runtime.NativePlanStatus
import ai.vn97.runtime.NativeStepKind
import ai.vn97.runtime.NativeStepStatus
import ai.vn97.runtime.NativeRuntimeConfig
import android.os.SystemClock
import java.io.File

data class VN97AutonomousApprovalRequest(
    val jobId: Int,
    val generation: Int,
    val capabilityId: String,
    val requestDigest: String,
    val scopeDigest: String,
    val presentationJson: String,
    val expiresNs: Long,
)

class VN97AutonomousWorkManager(
    private val application: VN97Application,
) {
    private val store = VN97AutonomousGoalStore(
        File(
            application.noBackupFilesDir,
            "vn97-autonomous-goals",
        )
    )
    private val scheduler =
        AndroidContinuationScheduler(application)
    private var foregroundApproval:
        VN97ForegroundAssistantContinuationSession? = null
    private var foregroundApprovalJobId: Int? = null
    private var reopenForegroundAssistantAfterApproval = false

    @Synchronized
    fun startGoal(goal: String): VN97AutonomousGoalRecord {
        require(goal.isNotBlank()) {
            "autonomous goal must not be blank"
        }
        require(
            store.list().count { !it.terminal } < MAX_ACTIVE_GOALS
        ) {
            "active autonomous goal bound exceeded"
        }

        val seed = application.assistant.createAutonomousSeed(goal)
        val jobId = allocateJobId(seed.plan.planId)
        val binding = scheduler.persistAssistant(
            jobId = jobId,
            snapshot = seed.snapshot,
            plan = seed.plan,
            principal = VN97AppAssistant.APP_PRINCIPAL,
        )
        val createdNs = seed.plan.createdNs
        var record = VN97AutonomousGoalRecord(
            jobId = jobId,
            planId = binding.planId,
            modelIdHex = binding.modelIdHex,
            principal = binding.principal,
            goal = seed.plan.goal,
            state = VN97AutonomousGoalState.SCHEDULED,
            createdNs = createdNs,
            updatedNs = createdNs,
        )
        try {
            store.save(record)
        } catch (exc: Throwable) {
            scheduler.cancelAssistantAndDelete(jobId)
            throw exc
        }

        val schedulingFailure = runCatching {
            scheduler.scheduleAssistant(
                VN97AssistantContinuationSpec(
                    jobId = jobId,
                    runtimeConfig = seed.runtimeConfig,
                    binding = binding,
                    minimumLatencyMillis = 0L,
                )
            )
        }.fold(
            onSuccess = { result ->
                if (result > 0) null
                else "Android JobScheduler rejected initial scheduling"
            },
            onFailure = { exc ->
                "Android JobScheduler scheduling failed: " +
                    exc::class.java.simpleName
            },
        )
        if (schedulingFailure != null) {
            record = record.copy(
                state = VN97AutonomousGoalState.PAUSED,
                updatedNs = wallNowNs(),
                terminalReason = schedulingFailure,
            )
            store.save(record)
        }
        return record
    }

    @Synchronized
    fun listGoals(): List<VN97AutonomousGoalRecord> =
        store.list().sortedByDescending { it.createdNs }

    @Synchronized
    fun goal(jobId: Int): VN97AutonomousGoalRecord? =
        store.loadOrNull(jobId)

    @Synchronized
    fun pendingApprovalRequest(): VN97AutonomousApprovalRequest? {
        val waiting = store.list()
            .asSequence()
            .filter {
                it.state == VN97AutonomousGoalState.WAITING_APPROVAL
            }
            .maxWithOrNull(
                compareBy<VN97AutonomousGoalRecord> {
                    it.generation
                }.thenBy { it.createdNs }
            )
            ?: run {
                closeForegroundApprovalLocked()
                return null
            }

        val existing = foregroundApproval
        if (
            existing != null &&
            foregroundApprovalJobId == waiting.jobId
        ) {
            return approvalRequest(waiting, existing)
        }

        closeForegroundApprovalLocked()
        return application.withSovereignExecution {
            scheduler.cancel(waiting.jobId)
            val reopen =
                application.assistant.releaseForBackgroundContinuation()
            var model: ai.vn97.runtime.NativeActivatedModel? = null
            try {
                val openedModel = openActivatedModel()
                model = openedModel
                requireModelIdentity(
                    waiting,
                    openedModel.info.modelId,
                )
                val session =
                    openVN97ForegroundAssistantContinuation(
                        context = application,
                        platformRuntime = application.platformRuntime,
                        jobId = waiting.jobId,
                        model = openedModel,
                        grants = VN97ProductionAuthority.grants(
                            application,
                            waiting.principal,
                        ),
                        runtimeConfig =
                            runtimeConfigFor(openedModel),
                        sessionLimits = VN97AssistantSessionLimits(
                            maxCyclesPerAdvance = 4,
                            maxExternalHandoffsPerAdvance = 1,
                        ),
                        auditFileName =
                            "m13-actions-" +
                                waiting.jobId +
                                ".jsonl",
                        nowNs = SystemClock.elapsedRealtimeNanos(),
                    )
                model = null
                val update = session.currentUpdate
                if (
                    update.state ==
                        VN97AssistantTurnState.APPROVAL_REQUIRED
                ) {
                    foregroundApproval = session
                    foregroundApprovalJobId = waiting.jobId
                    reopenForegroundAssistantAfterApproval = reopen
                    return@withSovereignExecution approvalRequest(
                        waiting,
                        session,
                    )
                }

                val committed = session.commit()
                val updated = updateRecordFromForeground(
                    waiting,
                    committed,
                    approved = null,
                )
                store.save(updated)
                if (
                    !updated.terminal &&
                    updated.state ==
                        VN97AutonomousGoalState.SCHEDULED
                ) {
                    reschedule(updated.jobId)
                }
                if (reopen) {
                    application.assistant.openIfActivated()
                }
                null
            } catch (exc: Throwable) {
                model?.close()
                if (reopen) {
                    runCatching {
                        application.assistant.openIfActivated()
                    }
                }
                throw exc
            }
        }
    }

    @Synchronized
    fun resolvePendingApproval(
        approved: Boolean,
    ): VN97AutonomousGoalRecord {
        val session = checkNotNull(foregroundApproval) {
            "no autonomous approval session is open"
        }
        val jobId = checkNotNull(foregroundApprovalJobId) {
            "autonomous approval job identity is missing"
        }
        val record = checkNotNull(store.loadOrNull(jobId)) {
            "autonomous approval goal does not exist"
        }
        check(
            record.state ==
                VN97AutonomousGoalState.WAITING_APPROVAL
        ) {
            "autonomous goal is not waiting for approval"
        }

        return application.withSovereignExecution {
            try {
                val resolved = session.resolveApproval(
                    approved = approved,
                    nowNs = SystemClock.elapsedRealtimeNanos(),
                )
                session.commit()
                foregroundApproval = null
                foregroundApprovalJobId = null

                val updated = updateRecordFromForeground(
                    record,
                    resolved,
                    approved = approved,
                )
                store.save(updated)

                val result = if (
                    !updated.terminal &&
                    updated.state ==
                        VN97AutonomousGoalState.SCHEDULED
                ) {
                    reschedule(updated.jobId)
                } else {
                    updated
                }
                reopenForegroundAssistantLocked()
                result
            } catch (exc: Throwable) {
                foregroundApproval = null
                foregroundApprovalJobId = null
                runCatching { session.close() }
                reopenForegroundAssistantLocked()
                throw exc
            }
        }
    }

    @Synchronized
    fun releaseForegroundApprovalSession() {
        closeForegroundApprovalLocked()
    }

    @Synchronized
    fun reschedule(jobId: Int): VN97AutonomousGoalRecord {
        val record = checkNotNull(store.loadOrNull(jobId)) {
            "autonomous goal does not exist"
        }
        check(!record.terminal) {
            "terminal autonomous goal cannot be rescheduled"
        }
        val model = openActivatedModel()
        model.use {
            requireModelIdentity(record, model.info.modelId)
            val runtimeConfig = NativeRuntimeConfig(
                layers = model.info.layers,
                batch = 1,
                dModel = model.info.dModel,
                dState = model.info.dState,
                recurrentBackend = NativeBackend.AUTO,
                packedBackend = NativeBackend.AUTO,
            )
            val binding =
                ai.vn97.platform.VN97AssistantContinuationBinding(
                    principal = record.principal,
                    planId = record.planId,
                    modelIdHex = record.modelIdHex,
                )
            val result = scheduler.scheduleAssistant(
                VN97AssistantContinuationSpec(
                    jobId = record.jobId,
                    runtimeConfig = runtimeConfig,
                    binding = binding,
                    minimumLatencyMillis = 0L,
                )
            )
            check(result > 0) {
                "Android JobScheduler rejected autonomous goal"
            }
        }
        return record.copy(
            state = VN97AutonomousGoalState.SCHEDULED,
            updatedNs = wallNowNs(),
            terminalReason = "",
        ).also(store::save)
    }

    @Synchronized
    fun cancel(jobId: Int): VN97AutonomousGoalRecord {
        val record = checkNotNull(store.loadOrNull(jobId)) {
            "autonomous goal does not exist"
        }
        if (record.terminal) return record
        if (foregroundApprovalJobId == jobId) {
            closeForegroundApprovalLocked()
        }
        scheduler.cancelAssistantAndDelete(jobId)
        return record.copy(
            state = VN97AutonomousGoalState.CANCELLED,
            updatedNs = wallNowNs(),
            terminalReason = "cancelled by user",
        ).also(store::save)
    }

    fun createContinuationWork(): VN97AssistantContinuationWork =
        VN97AssistantContinuationWork { context ->
            runContinuation(context)
        }

    private fun runContinuation(
        context: VN97AssistantContinuationContext,
    ): ContinuationOutcome {
        val initial = checkNotNull(
            store.loadOrNull(context.jobId)
        ) {
            "autonomous continuation has no VN97GOA1 record"
        }
        requireContextIdentity(initial, context)
        if (initial.terminal) return ContinuationOutcome.COMPLETE

        if (initial.wakeCount >= MAX_WAKE_COUNT) {
            if (!context.controller.plan.isTerminal()) {
                context.controller.cancel(
                    "autonomous wake budget exhausted"
                )
            }
            val exhausted = initial.copy(
                state = VN97AutonomousGoalState.FAILED,
                updatedNs = wallNowNs(),
                terminalReason =
                    "autonomous wake budget exhausted",
            )
            store.save(exhausted)
            return ContinuationOutcome.COMPLETE
        }

        val running = initial.copy(
            state = VN97AutonomousGoalState.RUNNING,
            updatedNs = wallNowNs(),
            wakeCount = initial.wakeCount + 1,
            terminalReason = "",
        )
        store.save(running)

        return application.withSovereignExecution {
            val reopenForeground =
                application.assistant
                    .releaseForBackgroundContinuation()
            try {
                openActivatedModel().use { model ->
                context.requireActivatedModel(model)
                requireModelIdentity(running, model.info.modelId)

                val update =
                    application.platformRuntime
                        .resumeProductionAssistantContinuation(
                            context = context,
                            model = model,
                            grants = VN97ProductionAuthority.grants(
                                application,
                                running.principal,
                            ),
                            sessionLimits =
                                limitsFor(context.budget.mode),
                            auditFileName =
                                "m13-actions-" +
                                    running.jobId +
                                    ".jsonl",
                            nowNs =
                                SystemClock.elapsedRealtimeNanos(),
                        )

                val plan = context.controller.plan
                val now = wallNowNs()
                when (update.state) {
                    ai.vn97.platform.VN97AssistantTurnState.COMPLETED -> {
                        store.save(
                            running.copy(
                                state =
                                    VN97AutonomousGoalState.COMPLETED,
                                updatedNs = now,
                                finalResponse =
                                    update.finalResponse,
                                terminalReason = "",
                            )
                        )
                        ContinuationOutcome.COMPLETE
                    }

                    ai.vn97.platform.VN97AssistantTurnState.APPROVAL_REQUIRED -> {
                        store.save(
                            running.copy(
                                state =
                                    VN97AutonomousGoalState.WAITING_APPROVAL,
                                updatedNs = now,
                                terminalReason =
                                    "foreground approval required",
                            )
                        )
                        ContinuationOutcome.COMPLETE
                    }

                    ai.vn97.platform.VN97AssistantTurnState.PAUSED -> {
                        store.save(
                            running.copy(
                                state =
                                    VN97AutonomousGoalState.PAUSED,
                                updatedNs = now,
                                terminalReason =
                                    plan.pausedReason.ifBlank {
                                        "planner paused"
                                    },
                            )
                        )
                        ContinuationOutcome.COMPLETE
                    }

                    ai.vn97.platform.VN97AssistantTurnState.FAILED -> {
                        maybeStartReplanGeneration(
                            current = running,
                            terminalPlan = plan,
                            model = model,
                            terminalState =
                                VN97AutonomousGoalState.FAILED,
                            reason =
                                plan.terminalReason.ifBlank {
                                    "planner failed"
                                },
                        )
                        ContinuationOutcome.COMPLETE
                    }

                    ai.vn97.platform.VN97AssistantTurnState.CANCELLED -> {
                        store.save(
                            running.copy(
                                state =
                                    VN97AutonomousGoalState.CANCELLED,
                                updatedNs = now,
                                terminalReason =
                                    plan.terminalReason.ifBlank {
                                        "planner cancelled"
                                    },
                            )
                        )
                        ContinuationOutcome.COMPLETE
                    }

                    ai.vn97.platform.VN97AssistantTurnState.BUDGET_EXHAUSTED -> {
                        maybeStartReplanGeneration(
                            current = running,
                            terminalPlan = plan,
                            model = model,
                            terminalState =
                                VN97AutonomousGoalState.BUDGET_EXHAUSTED,
                            reason =
                                plan.terminalReason.ifBlank {
                                    "planner budget exhausted"
                                },
                        )
                        ContinuationOutcome.COMPLETE
                    }

                    ai.vn97.platform.VN97AssistantTurnState.APPROVAL_REJECTED -> {
                        store.save(
                            running.copy(
                                state =
                                    VN97AutonomousGoalState.FAILED,
                                updatedNs = now,
                                terminalReason =
                                    plan.terminalReason.ifBlank {
                                        "external approval rejected"
                                    },
                            )
                        )
                        ContinuationOutcome.COMPLETE
                    }

                    ai.vn97.platform.VN97AssistantTurnState.YIELDED -> {
                        store.save(
                            running.copy(
                                state =
                                    VN97AutonomousGoalState.SCHEDULED,
                                updatedNs = now,
                                terminalReason = "",
                            )
                        )
                        ContinuationOutcome.RESCHEDULE
                    }

                    ai.vn97.platform.VN97AssistantTurnState.STALLED -> {
                        if (!plan.isTerminal()) {
                            context.controller.cancel(
                                "stalled plan superseded by autonomous replan"
                            )
                        }
                        maybeStartReplanGeneration(
                            current = running,
                            terminalPlan = context.controller.plan,
                            model = model,
                            terminalState =
                                VN97AutonomousGoalState.FAILED,
                            reason =
                                "planner stalled without a runnable directive",
                        )
                        ContinuationOutcome.COMPLETE
                    }
                }
            }
        } catch (exc: Throwable) {
            val now = wallNowNs()
            val plan = context.controller.plan
            if (plan.isTerminal()) {
                store.save(
                    terminalRecordFromPlan(
                        running,
                        plan,
                        now,
                    )
                )
                ContinuationOutcome.COMPLETE
            } else if (
                plan.status == NativePlanStatus.WAITING_EXTERNAL
            ) {
                store.save(
                    running.copy(
                        state = VN97AutonomousGoalState.PAUSED,
                        updatedNs = now,
                        terminalReason =
                            "external boundary blocked: " +
                                exc::class.java.simpleName,
                    )
                )
                ContinuationOutcome.COMPLETE
            } else if (
                plan.status == NativePlanStatus.PAUSED
            ) {
                store.save(
                    running.copy(
                        state = VN97AutonomousGoalState.PAUSED,
                        updatedNs = now,
                        terminalReason =
                            plan.pausedReason.ifBlank {
                                "planner paused after wake failure"
                            },
                    )
                )
                ContinuationOutcome.COMPLETE
            } else if (running.wakeCount >= MAX_WAKE_COUNT) {
                store.save(
                    running.copy(
                        state = VN97AutonomousGoalState.FAILED,
                        updatedNs = now,
                        terminalReason =
                            "autonomous wake failed after retry budget: " +
                                exc::class.java.simpleName,
                    )
                )
                ContinuationOutcome.COMPLETE
            } else {
                store.save(
                    running.copy(
                        state =
                            VN97AutonomousGoalState.SCHEDULED,
                        updatedNs = now,
                        terminalReason =
                            "last wake failed: " +
                                exc::class.java.simpleName,
                    )
                )
                ContinuationOutcome.RESCHEDULE
            }
            } finally {
                if (reopenForeground) {
                    runCatching {
                        application.assistant.openIfActivated()
                    }
                }
            }
        }
    }

    private fun maybeStartReplanGeneration(
        current: VN97AutonomousGoalRecord,
        terminalPlan: NativePlan,
        model: ai.vn97.runtime.NativeActivatedModel,
        terminalState: VN97AutonomousGoalState,
        reason: String,
    ): VN97AutonomousGoalRecord? {
        require(terminalPlan.isTerminal()) {
            "autonomous replan requires terminal planner"
        }
        require(
            terminalState == VN97AutonomousGoalState.FAILED ||
                terminalState ==
                    VN97AutonomousGoalState.BUDGET_EXHAUSTED
        ) {
            "only failed/exhausted autonomous generations replan"
        }
        val now = wallNowNs()
        val terminalRecord = current.copy(
            state = terminalState,
            updatedNs = now,
            terminalReason = reason,
        )
        store.save(terminalRecord)

        if (current.generation >= MAX_REPLAN_GENERATIONS) {
            return null
        }

        val seed = createVN97AutonomousReplanSeed(
            model = model,
            previousPlan = terminalPlan,
            feedback = buildReplanFeedback(
                terminalPlan,
                reason,
            ),
            createdNs = now,
        )
        val jobId = allocateJobId(seed.plan.planId)
        val binding = scheduler.persistAssistant(
            jobId = jobId,
            snapshot = seed.snapshot,
            plan = seed.plan,
            principal = current.principal,
        )
        var successor = VN97AutonomousGoalRecord(
            jobId = jobId,
            planId = binding.planId,
            modelIdHex = binding.modelIdHex,
            principal = binding.principal,
            goal = seed.plan.goal,
            state = VN97AutonomousGoalState.SCHEDULED,
            rootJobId = current.rootJobId,
            generation = current.generation + 1,
            previousJobId = current.jobId,
            createdNs = seed.plan.createdNs,
            updatedNs = seed.plan.createdNs,
        )
        try {
            store.save(successor)
        } catch (exc: Throwable) {
            scheduler.cancelAssistantAndDelete(jobId)
            throw exc
        }

        val schedulingFailure = runCatching {
            scheduler.scheduleAssistant(
                VN97AssistantContinuationSpec(
                    jobId = jobId,
                    runtimeConfig = seed.runtimeConfig,
                    binding = binding,
                    minimumLatencyMillis = 0L,
                )
            )
        }.fold(
            onSuccess = { result ->
                if (result > 0) null
                else "Android JobScheduler rejected replan generation"
            },
            onFailure = { exc ->
                "replan scheduling failed: " +
                    exc::class.java.simpleName
            },
        )
        if (schedulingFailure != null) {
            successor = successor.copy(
                state = VN97AutonomousGoalState.PAUSED,
                updatedNs = wallNowNs(),
                terminalReason = schedulingFailure,
            )
            store.save(successor)
        }
        store.save(
            terminalRecord.copy(
                updatedNs = maxOf(
                    terminalRecord.updatedNs,
                    successor.updatedNs,
                ),
                terminalReason =
                    reason.take(12 * 1024) +
                        " | successor_job=" +
                        successor.jobId +
                        " generation=" +
                        successor.generation,
            )
        )
        return successor
    }

    private fun buildReplanFeedback(
        plan: NativePlan,
        reason: String,
    ): String = buildString {
        append("Previous autonomous generation ended without satisfying ")
        append("the same user goal. Create a materially revised plan. ")
        append("Do not repeat failed steps unchanged.\n")
        append("terminal_status=")
        append(plan.status.name)
        append("\nterminal_reason=")
        append(reason.take(1024))
        plan.steps.take(24).forEach { step ->
            append("\nstep=")
            append(step.stepId)
            append(" kind=")
            append(step.spec.kind.name)
            append(" status=")
            append(step.status.name)
            append(" objective=")
            append(step.spec.objective.replace('\n', ' ').take(384))
            if (step.result.isNotBlank()) {
                append(" result=")
                append(step.result.replace('\n', ' ').take(512))
            }
            if (step.failureReason.isNotBlank()) {
                append(" failure=")
                append(
                    step.failureReason
                        .replace('\n', ' ')
                        .take(512)
                )
            }
        }
    }.take(MAX_REPLAN_FEEDBACK_CHARS)

    private fun approvalRequest(
        record: VN97AutonomousGoalRecord,
        session: VN97ForegroundAssistantContinuationSession,
    ): VN97AutonomousApprovalRequest {
        val approval = checkNotNull(
            session.currentUpdate.approval
        ) {
            "foreground autonomous continuation lacks approval"
        }
        check(approval.planId == record.planId) {
            "approval plan does not match VN97GOA1"
        }
        return VN97AutonomousApprovalRequest(
            jobId = record.jobId,
            generation = record.generation,
            capabilityId = approval.capabilityId,
            requestDigest = approval.requestDigest,
            scopeDigest = approval.scopeDigest,
            presentationJson = approval.presentationJson,
            expiresNs = approval.expiresNs,
        )
    }

    private fun updateRecordFromForeground(
        record: VN97AutonomousGoalRecord,
        update: ai.vn97.platform.VN97AssistantTurnUpdate,
        approved: Boolean?,
    ): VN97AutonomousGoalRecord {
        val now = wallNowNs()
        return when (update.state) {
            VN97AssistantTurnState.COMPLETED ->
                record.copy(
                    state = VN97AutonomousGoalState.COMPLETED,
                    updatedNs = now,
                    finalResponse = update.finalResponse,
                    terminalReason = "",
                )

            VN97AssistantTurnState.APPROVAL_REQUIRED ->
                record.copy(
                    state = VN97AutonomousGoalState.WAITING_APPROVAL,
                    updatedNs = now,
                    terminalReason =
                        "additional foreground approval required",
                )

            VN97AssistantTurnState.APPROVAL_REJECTED ->
                record.copy(
                    state = VN97AutonomousGoalState.FAILED,
                    updatedNs = now,
                    terminalReason =
                        if (approved == false) {
                            "external action rejected by user"
                        } else {
                            "external approval was rejected"
                        },
                )

            VN97AssistantTurnState.YIELDED ->
                record.copy(
                    state = VN97AutonomousGoalState.SCHEDULED,
                    updatedNs = now,
                    terminalReason = "",
                )

            VN97AssistantTurnState.PAUSED ->
                record.copy(
                    state = VN97AutonomousGoalState.PAUSED,
                    updatedNs = now,
                    terminalReason = "planner paused after approval",
                )

            VN97AssistantTurnState.FAILED ->
                record.copy(
                    state = VN97AutonomousGoalState.FAILED,
                    updatedNs = now,
                    terminalReason =
                        "planner failed after approval resolution",
                )

            VN97AssistantTurnState.CANCELLED ->
                record.copy(
                    state = VN97AutonomousGoalState.CANCELLED,
                    updatedNs = now,
                    terminalReason =
                        "planner cancelled after approval resolution",
                )

            VN97AssistantTurnState.BUDGET_EXHAUSTED ->
                record.copy(
                    state =
                        VN97AutonomousGoalState.BUDGET_EXHAUSTED,
                    updatedNs = now,
                    terminalReason =
                        "planner budget exhausted after approval",
                )

            VN97AssistantTurnState.STALLED ->
                record.copy(
                    state = VN97AutonomousGoalState.PAUSED,
                    updatedNs = now,
                    terminalReason =
                        "planner stalled after approval resolution",
                )
        }
    }

    private fun closeForegroundApprovalLocked() {
        val session = foregroundApproval
        foregroundApproval = null
        foregroundApprovalJobId = null
        if (session != null) {
            runCatching { session.close() }
        }
        reopenForegroundAssistantLocked()
    }

    private fun reopenForegroundAssistantLocked() {
        if (reopenForegroundAssistantAfterApproval) {
            reopenForegroundAssistantAfterApproval = false
            runCatching {
                application.assistant.openIfActivated()
            }
        }
    }

    private fun runtimeConfigFor(
        model: ai.vn97.runtime.NativeActivatedModel,
    ): NativeRuntimeConfig =
        NativeRuntimeConfig(
            layers = model.info.layers,
            batch = 1,
            dModel = model.info.dModel,
            dState = model.info.dState,
            recurrentBackend = NativeBackend.AUTO,
            packedBackend = NativeBackend.AUTO,
        )

    private fun terminalRecordFromPlan(
        running: VN97AutonomousGoalRecord,
        plan: NativePlan,
        nowNs: Long,
    ): VN97AutonomousGoalRecord {
        check(plan.isTerminal()) {
            "terminal ledger recovery requires terminal planner"
        }
        return when (plan.status) {
            NativePlanStatus.COMPLETED -> {
                val response = plan.steps
                    .asReversed()
                    .firstOrNull {
                        it.spec.kind == NativeStepKind.RESPOND &&
                            it.status == NativeStepStatus.SUCCEEDED
                    }
                    ?.result
                    .orEmpty()
                if (response.isBlank()) {
                    running.copy(
                        state = VN97AutonomousGoalState.FAILED,
                        updatedNs = nowNs,
                        terminalReason =
                            "completed planner is missing final response",
                    )
                } else {
                    running.copy(
                        state = VN97AutonomousGoalState.COMPLETED,
                        updatedNs = nowNs,
                        finalResponse = response,
                        terminalReason = "",
                    )
                }
            }

            NativePlanStatus.FAILED ->
                running.copy(
                    state = VN97AutonomousGoalState.FAILED,
                    updatedNs = nowNs,
                    terminalReason =
                        plan.terminalReason.ifBlank {
                            "planner failed"
                        },
                )

            NativePlanStatus.CANCELLED ->
                running.copy(
                    state = VN97AutonomousGoalState.CANCELLED,
                    updatedNs = nowNs,
                    terminalReason =
                        plan.terminalReason.ifBlank {
                            "planner cancelled"
                        },
                )

            NativePlanStatus.BUDGET_EXHAUSTED ->
                running.copy(
                    state =
                        VN97AutonomousGoalState.BUDGET_EXHAUSTED,
                    updatedNs = nowNs,
                    terminalReason =
                        plan.terminalReason.ifBlank {
                            "planner budget exhausted"
                        },
                )

            else -> throw IllegalStateException(
                "unexpected non-terminal planner state"
            )
        }
    }

    private fun limitsFor(
        mode: ComputeMode,
    ): VN97AssistantSessionLimits {
        val cycles = when (mode) {
            ComputeMode.BLOCKED -> 1
            ComputeMode.LOW_POWER -> 2
            ComputeMode.BALANCED -> 4
            ComputeMode.PERFORMANCE -> 8
        }
        return VN97AssistantSessionLimits(
            maxCyclesPerAdvance = cycles,
            maxExternalHandoffsPerAdvance = 1,
        )
    }

    private fun openActivatedModel() =
        checkNotNull(
            NativeActivatedInventoryModelLoader.openOrNull(
                File(
                    application.noBackupFilesDir,
                    "vn97-capabilities",
                )
            )
        ) {
            "trusted VN97 model is not active"
        }

    private fun allocateJobId(planId: String): Int {
        val seed =
            planId.substring(0, 8).toLong(16).toInt() and JOB_MASK
        repeat(MAX_JOB_PROBES) { offset ->
            val low = (seed + offset) and JOB_MASK
            val candidate = JOB_PREFIX or low
            if (
                candidate > 0 &&
                !store.contains(candidate) &&
                !scheduler.hasPendingJob(candidate)
            ) {
                return candidate
            }
        }
        throw IllegalStateException(
            "could not allocate autonomous JobScheduler ID"
        )
    }

    private fun requireContextIdentity(
        record: VN97AutonomousGoalRecord,
        context: VN97AssistantContinuationContext,
    ) {
        check(record.planId == context.binding.planId) {
            "VN97GOA1 plan does not match continuation binding"
        }
        check(record.modelIdHex == context.binding.modelIdHex) {
            "VN97GOA1 model does not match continuation binding"
        }
        check(record.principal == context.binding.principal) {
            "VN97GOA1 principal does not match continuation binding"
        }
        check(record.planId == context.controller.plan.planId) {
            "VN97GOA1 plan does not match restored planner"
        }
        check(record.goal == context.controller.plan.goal) {
            "VN97GOA1 goal does not match restored planner"
        }
    }

    private fun requireModelIdentity(
        record: VN97AutonomousGoalRecord,
        modelId: ByteArray,
    ) {
        val actual = modelId.joinToString("") {
            "%02x".format(it.toInt() and 0xff)
        }
        check(actual == record.modelIdHex) {
            "activated model changed since autonomous goal creation"
        }
    }

    companion object {
        private const val MAX_ACTIVE_GOALS = 32
        private const val MAX_WAKE_COUNT = 256
        private const val MAX_REPLAN_GENERATIONS = 3
        private const val MAX_REPLAN_FEEDBACK_CHARS = 8 * 1024
        private const val MAX_JOB_PROBES = 128
        private const val JOB_PREFIX = 0x40000000
        private const val JOB_MASK = 0x3fffffff
    }
}

private fun wallNowNs(): Long =
    Math.multiplyExact(
        System.currentTimeMillis(),
        1_000_000L,
    )
