package ai.vn97.app

import ai.vn97.platform.AndroidContinuationScheduler
import ai.vn97.platform.ContinuationOutcome
import ai.vn97.platform.ComputeMode
import ai.vn97.platform.VN97AssistantContinuationContext
import ai.vn97.platform.VN97AssistantContinuationSpec
import ai.vn97.platform.VN97AssistantContinuationWork
import ai.vn97.platform.VN97AssistantSessionLimits
import ai.vn97.runtime.NativeActivatedInventoryModelLoader
import ai.vn97.runtime.NativeBackend
import ai.vn97.runtime.NativePlan
import ai.vn97.runtime.NativePlanStatus
import ai.vn97.runtime.NativeStepKind
import ai.vn97.runtime.NativeStepStatus
import ai.vn97.runtime.NativeRuntimeConfig
import android.os.SystemClock
import java.io.File

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
                        store.save(
                            running.copy(
                                state =
                                    VN97AutonomousGoalState.FAILED,
                                updatedNs = now,
                                terminalReason =
                                    plan.terminalReason.ifBlank {
                                        "planner failed"
                                    },
                            )
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
                        store.save(
                            running.copy(
                                state =
                                    VN97AutonomousGoalState.BUDGET_EXHAUSTED,
                                updatedNs = now,
                                terminalReason =
                                    plan.terminalReason.ifBlank {
                                        "planner budget exhausted"
                                    },
                            )
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

                    ai.vn97.platform.VN97AssistantTurnState.YIELDED,
                    ai.vn97.platform.VN97AssistantTurnState.STALLED,
                    -> {
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
