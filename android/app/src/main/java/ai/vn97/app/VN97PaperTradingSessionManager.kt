package ai.vn97.app

import ai.vn97.platform.VN97MarketDataPolicy
import ai.vn97.platform.VN97PaperPerformanceEvaluator
import ai.vn97.platform.VN97PaperTradingDecisionKind
import ai.vn97.runtime.NativeActivatedInventoryModelLoader
import ai.vn97.runtime.NativeCognitionInferenceEngine
import ai.vn97.runtime.NativeMemoryKind
import android.app.job.JobInfo
import android.app.job.JobScheduler
import android.content.ComponentName
import android.os.PersistableBundle
import android.os.SystemClock
import java.io.File
import java.nio.charset.StandardCharsets
import java.security.MessageDigest
import java.util.concurrent.atomic.AtomicBoolean

data class VN97PaperTradingSessionConfig(
    val endpoint: String,
    val sourceId: String,
    val symbols: Set<String>,
    val userGoal: String,
    val intervalMillis: Long = 60_000L,
    val deadlineWallTimeMillis: Long = 0L,
    val maxEpisodes: Int = 256,
    val requiresBatteryNotLow: Boolean = true,
    val requiresCharging: Boolean = false,
) {
    init {
        require(userGoal.isNotBlank()) {
            "paper trading session goal must not be blank"
        }
        require(
            userGoal.toByteArray(StandardCharsets.UTF_8).size <=
                VN97PaperTradingSessionRecord.MAX_GOAL_BYTES
        ) {
            "paper trading session goal exceeds byte bound"
        }
        require(
            intervalMillis in
                VN97PaperTradingSessionRecord.MIN_INTERVAL_MILLIS..
                    VN97PaperTradingSessionRecord.MAX_INTERVAL_MILLIS
        ) {
            "paper trading session interval is outside bounds"
        }
        require(maxEpisodes in 1..VN97PaperTradingSessionRecord.MAX_EPISODES) {
            "paper trading session maxEpisodes is outside bounds"
        }
        require(deadlineWallTimeMillis >= 0L) {
            "paper trading session deadline must be non-negative"
        }
        VN97MarketDataPolicy(
            sourceId = sourceId,
            allowedSymbols = symbols,
        )
    }
}

data class VN97PaperTradingSessionReport(
    val jobId: Int,
    val sessionId: String,
    val state: VN97PaperTradingSessionState,
    val episodesAttempted: Int,
    val maxEpisodes: Int,
    val wakeCount: Int,
    val lastDecision: String,
    val lastOutcome: String,
    val terminalReason: String,
    val nextRunWallTimeMillis: Long,
)

class VN97PaperTradingSessionManager(
    private val application: VN97Application,
) {
    private val store = VN97PaperTradingSessionStore(
        File(
            application.noBackupFilesDir,
            "vn97-paper-trading-sessions",
        )
    )
    private val scheduler = checkNotNull(
        application.getSystemService(JobScheduler::class.java)
    )
    private val performanceEvidence =
        VN97PaperPerformanceEvidenceStore(
            File(
                application.noBackupFilesDir,
                "vn97-paper-performance",
            )
        )

    fun startSession(
        config: VN97PaperTradingSessionConfig,
    ): VN97PaperTradingSessionRecord =
        application.withSovereignExecution {
            val now = System.currentTimeMillis()
            require(
                config.deadlineWallTimeMillis == 0L ||
                    config.deadlineWallTimeMillis > now
            ) {
                "paper trading session deadline must be in the future"
            }
            check(
                store.list().count { !it.terminal } < MAX_ACTIVE_SESSIONS
            ) {
                "paper trading active-session bound reached"
            }

            val reopenForeground =
                application.assistant.releaseForBackgroundContinuation()
            val modelId = try {
                openActivatedModel().use { model ->
                    model.info.modelId.hex()
                }
            } finally {
                if (reopenForeground) {
                    runCatching { application.assistant.openIfActivated() }
                }
            }
            val symbols = config.symbols.toList().sorted()
            val sessionId = sessionId(
                modelId = modelId,
                endpoint = config.endpoint,
                sourceId = config.sourceId,
                symbols = symbols,
                userGoal = config.userGoal,
                createdWallTimeMillis = now,
            )
            val jobId = allocateJobId(sessionId)
            val record = VN97PaperTradingSessionRecord(
                jobId = jobId,
                sessionId = sessionId,
                modelIdHex = modelId,
                endpoint = config.endpoint,
                sourceId = config.sourceId,
                symbols = symbols,
                userGoal = config.userGoal,
                state = VN97PaperTradingSessionState.SCHEDULED,
                createdWallTimeMillis = now,
                updatedWallTimeMillis = now,
                intervalMillis = config.intervalMillis,
                nextRunWallTimeMillis = now,
                deadlineWallTimeMillis =
                    config.deadlineWallTimeMillis,
                maxEpisodes = config.maxEpisodes,
                requiresBatteryNotLow =
                    config.requiresBatteryNotLow,
                requiresCharging = config.requiresCharging,
            )
            store.save(record)
            schedule(record)
        }

    fun pause(jobId: Int): VN97PaperTradingSessionRecord =
        application.withSovereignExecution {
            val record = requireSession(jobId)
            check(!record.terminal) {
                "terminal paper trading session cannot be paused"
            }
            val paused = record.copy(
                state = VN97PaperTradingSessionState.PAUSED,
                updatedWallTimeMillis = monotonicNow(record),
                terminalReason = "",
            )
            store.save(paused)
            scheduler.cancel(jobId)
            paused
        }

    fun resume(jobId: Int): VN97PaperTradingSessionRecord =
        application.withSovereignExecution {
            val record = requireSession(jobId)
            check(record.state == VN97PaperTradingSessionState.PAUSED) {
                "only paused paper trading session can resume"
            }
            val now = monotonicNow(record)
            val resumed = record.copy(
                state = VN97PaperTradingSessionState.SCHEDULED,
                updatedWallTimeMillis = now,
                nextRunWallTimeMillis = now,
                terminalReason = "",
            )
            store.save(resumed)
            schedule(resumed)
        }

    fun stop(jobId: Int): VN97PaperTradingSessionRecord =
        application.withSovereignExecution {
            val record = requireSession(jobId)
            if (record.terminal) return@withSovereignExecution record
            val stopped = record.copy(
                state = VN97PaperTradingSessionState.STOPPED,
                updatedWallTimeMillis = monotonicNow(record),
                terminalReason = "paper session stopped by user",
            )
            store.save(stopped)
            scheduler.cancel(jobId)
            appendTerminalMemoryBestEffort(stopped)
            stopped
        }

    fun report(jobId: Int): VN97PaperTradingSessionReport =
        reportOf(requireSession(jobId))

    fun listReports(): List<VN97PaperTradingSessionReport> =
        store.list().map(::reportOf)

    fun performanceEvidence(): VN97PaperPerformanceAggregate =
        performanceEvidence.aggregate()

    fun reconcileAfterSystemRestart(): List<VN97PaperTradingSessionRecord> =
        application.withSovereignExecution {
            store.list().map { original ->
                if (original.terminal ||
                    original.state == VN97PaperTradingSessionState.PAUSED
                ) {
                    scheduler.cancel(original.jobId)
                    return@map original
                }

                val now = System.currentTimeMillis()
                if (deadlineExpired(original, now)) {
                    scheduler.cancel(original.jobId)
                    return@map finish(
                        original,
                        VN97PaperTradingSessionState.COMPLETED,
                        "paper session deadline reached",
                    )
                }
                if (original.episodesAttempted >= original.maxEpisodes) {
                    scheduler.cancel(original.jobId)
                    return@map finish(
                        original,
                        VN97PaperTradingSessionState.COMPLETED,
                        "paper session episode budget completed",
                    )
                }
                if (original.wakeCount >= MAX_WAKE_COUNT) {
                    scheduler.cancel(original.jobId)
                    return@map finish(
                        original,
                        VN97PaperTradingSessionState.FAILED,
                        "paper session wake bound exhausted",
                    )
                }

                val reset =
                    if (original.state ==
                        VN97PaperTradingSessionState.RUNNING
                    ) {
                        original.copy(
                            state =
                                VN97PaperTradingSessionState.SCHEDULED,
                            updatedWallTimeMillis = monotonicNow(original),
                            nextRunWallTimeMillis =
                                maxOf(
                                    original.nextRunWallTimeMillis,
                                    now,
                                ),
                            lastOutcome =
                                boundedOutcome(
                                    "process/reboot reconciliation; " +
                                        "consumed snapshots are not replayed"
                                ),
                        ).also(store::save)
                    } else {
                        original
                    }
                if (scheduler.getPendingJob(reset.jobId) != null) {
                    reset
                } else {
                    schedule(reset)
                }
            }
        }

    internal fun runScheduledEpisode(
        jobId: Int,
        expectedSessionId: String,
        stopped: AtomicBoolean,
    ) {
        application.withSovereignExecution {
            var record = store.loadOrNull(jobId) ?: return@withSovereignExecution
            if (record.sessionId != expectedSessionId) {
                return@withSovereignExecution
            }
            if (
                record.terminal ||
                record.state == VN97PaperTradingSessionState.PAUSED
            ) {
                return@withSovereignExecution
            }

            val nowMs = System.currentTimeMillis()
            if (deadlineExpired(record, nowMs)) {
                finish(
                    record,
                    VN97PaperTradingSessionState.COMPLETED,
                    "paper session deadline reached",
                )
                return@withSovereignExecution
            }
            if (record.episodesAttempted >= record.maxEpisodes) {
                finish(
                    record,
                    VN97PaperTradingSessionState.COMPLETED,
                    "paper session episode budget completed",
                )
                return@withSovereignExecution
            }
            if (record.wakeCount >= MAX_WAKE_COUNT) {
                finish(
                    record,
                    VN97PaperTradingSessionState.FAILED,
                    "paper session wake bound exhausted",
                )
                return@withSovereignExecution
            }
            if (stopped.get()) return@withSovereignExecution

            record = record.copy(
                state = VN97PaperTradingSessionState.RUNNING,
                updatedWallTimeMillis = monotonicNow(record),
                wakeCount = record.wakeCount + 1,
                terminalReason = "",
            )
            store.save(record)

            val handoff = runCatching {
                application.assistant.releaseForBackgroundContinuation()
            }
            if (handoff.isFailure) {
                val retry = record.copy(
                    state = VN97PaperTradingSessionState.SCHEDULED,
                    updatedWallTimeMillis = monotonicNow(record),
                    nextRunWallTimeMillis =
                        nextRun(record, System.currentTimeMillis()),
                    lastOutcome = boundedOutcome(
                        "VN97_BUSY " +
                            (handoff.exceptionOrNull()
                                ?.javaClass?.simpleName ?: "unknown")
                    ),
                    terminalReason = "",
                )
                store.save(retry)
                if (!stopped.get()) schedule(retry)
                return@withSovereignExecution
            }
            val reopenForeground = handoff.getOrThrow()
            try {
                executeOneWake(record, stopped)
            } finally {
                if (reopenForeground) {
                    runCatching { application.assistant.openIfActivated() }
                }
            }
        }
    }

    private fun executeOneWake(
        running: VN97PaperTradingSessionRecord,
        stopped: AtomicBoolean,
    ) {
        var record = running
        val nowMs = System.currentTimeMillis()
        val nowNs = wallNowNs(nowMs)
        try {
            openActivatedModel().use { model ->
                requireModelIdentity(record, model.info.modelId)
                val source =
                    application.platformRuntime
                        .createProductionReadOnlyMarketDataSource(
                            endpoint = record.endpoint,
                            sourceId = record.sourceId,
                            allowedSymbols = record.symbols.toSet(),
                        )
                val snapshot = source.fetch(nowNs)
                if (
                    snapshot.snapshotId == record.lastSnapshotId ||
                    snapshot.observedNs <= record.lastObservedNs
                ) {
                    val waiting = record.copy(
                        state =
                            VN97PaperTradingSessionState.SCHEDULED,
                        updatedWallTimeMillis = monotonicNow(record),
                        nextRunWallTimeMillis =
                            nextRun(record, nowMs),
                        lastOutcome =
                            boundedOutcome(
                                "NO_FRESH_SNAPSHOT observed_ns=" +
                                    snapshot.observedNs
                            ),
                        terminalReason = "",
                    )
                    store.save(waiting)
                    if (!stopped.get()) schedule(waiting)
                    return
                }

                val account =
                    application.platformRuntime
                        .openProductionPaperTradingAccount(
                            fileName =
                                "paper-" +
                                    record.sessionId.take(16) +
                                    ".vn97trd1"
                        )

                // M15F preflight: a fresh snapshot must mark every currently
                // held paper position before VN97 may reason or execute.
                VN97PaperPerformanceEvaluator.evaluate(
                    account = account.snapshot(),
                    market = snapshot,
                )

                record = record.copy(
                    updatedWallTimeMillis = monotonicNow(record),
                    episodesAttempted =
                        record.episodesAttempted + 1,
                    lastSnapshotId = snapshot.snapshotId,
                    lastObservedNs = snapshot.observedNs,
                    lastOutcome =
                        boundedOutcome(
                            "SNAPSHOT_CONSUMED id=" +
                                snapshot.snapshotId
                        ),
                )
                store.save(record)
                if (stopped.get()) return

                application.platformRuntime
                    .openOrCreateProductionMemory(model)
                    .use { memory ->
                        val agent =
                            application.platformRuntime
                                .createProductionPaperTradingAgent(
                                    model = model,
                                    account = account,
                                    memory = memory,
                                )
                        val result = agent.evaluate(
                            userGoal = record.userGoal,
                            snapshot = snapshot,
                            nowNs = nowNs,
                        )
                        val outcome = resultOutcome(result)
                        val performance =
                            VN97PaperPerformanceEvaluator.evaluate(
                                account = account.snapshot(),
                                market = snapshot,
                            )
                        val evidence =
                            VN97PaperPerformanceEvidence.from(
                                sessionId = record.sessionId,
                                jobId = record.jobId,
                                episode = record.episodesAttempted,
                                recordedWallTimeMillis = nowMs,
                                performance = performance,
                                decision = result.finalResponse,
                                outcome = outcome,
                            )
                        performanceEvidence.append(evidence)
                        appendPerformanceMemory(
                            model = model,
                            memory = memory,
                            record = record,
                            evidence = evidence,
                            nowNs = nowNs,
                        )
                        appendEpisodeMemory(
                            model = model,
                            memory = memory,
                            record = record,
                            planId = result.planId,
                            decision = result.finalResponse,
                            outcome = outcome,
                            nowNs = nowNs,
                        )

                        val finished =
                            record.episodesAttempted >=
                                record.maxEpisodes ||
                                deadlineExpired(record, nowMs)
                        val next =
                            if (finished) {
                                record.copy(
                                    state =
                                        VN97PaperTradingSessionState.COMPLETED,
                                    updatedWallTimeMillis =
                                        monotonicNow(record),
                                    lastPlanId = result.planId,
                                    lastDecision =
                                        boundedDecision(
                                            result.finalResponse
                                        ),
                                    lastOutcome =
                                        boundedOutcome(outcome),
                                    terminalReason =
                                        if (
                                            record.episodesAttempted >=
                                                record.maxEpisodes
                                        ) {
                                            "paper session episode budget completed"
                                        } else {
                                            "paper session deadline reached"
                                        },
                                )
                            } else {
                                record.copy(
                                    state =
                                        VN97PaperTradingSessionState.SCHEDULED,
                                    updatedWallTimeMillis =
                                        monotonicNow(record),
                                    nextRunWallTimeMillis =
                                        nextRun(record, nowMs),
                                    lastPlanId = result.planId,
                                    lastDecision =
                                        boundedDecision(
                                            result.finalResponse
                                        ),
                                    lastOutcome =
                                        boundedOutcome(outcome),
                                    terminalReason = "",
                                )
                            }
                        store.save(next)
                        if (next.terminal) {
                            appendTerminalMemory(
                                model = model,
                                memory = memory,
                                record = next,
                                nowNs = nowNs,
                            )
                        } else if (!stopped.get()) {
                            schedule(next)
                        }
                    }
            }
        } catch (exc: Throwable) {
            if (isModelIdentityFailure(exc)) {
                finish(
                    record,
                    VN97PaperTradingSessionState.FAILED,
                    "activated VN97 model changed during paper session",
                )
                return
            }
            val latest = store.loadOrNull(record.jobId)
                ?: return
            if (
                latest.terminal ||
                latest.state == VN97PaperTradingSessionState.PAUSED
            ) {
                return
            }
            if (latest.wakeCount >= MAX_WAKE_COUNT) {
                finish(
                    latest,
                    VN97PaperTradingSessionState.FAILED,
                    "paper session wake bound exhausted after error",
                )
                return
            }
            val retry = latest.copy(
                state = VN97PaperTradingSessionState.SCHEDULED,
                updatedWallTimeMillis = monotonicNow(latest),
                nextRunWallTimeMillis =
                    nextRun(latest, System.currentTimeMillis()),
                lastOutcome = boundedOutcome(
                    "WAKE_ERROR " +
                        exc.javaClass.simpleName +
                        ": " +
                        (exc.message ?: "unspecified")
                ),
                terminalReason = "",
            )
            store.save(retry)
            if (!stopped.get()) schedule(retry)
        }
    }

    private fun schedule(
        record: VN97PaperTradingSessionRecord,
    ): VN97PaperTradingSessionRecord {
        check(!record.terminal)
        check(record.state == VN97PaperTradingSessionState.SCHEDULED)
        if (
            record.scheduleAttemptCount >=
                VN97PaperTradingSessionRecord.MAX_SCHEDULE_ATTEMPTS
        ) {
            return finish(
                record,
                VN97PaperTradingSessionState.FAILED,
                "paper session scheduling bound exhausted",
            )
        }
        val now = System.currentTimeMillis()
        val delay = maxOf(
            0L,
            record.nextRunWallTimeMillis - now,
        )
        val extras = PersistableBundle().apply {
            putString(EXTRA_SESSION_ID, record.sessionId)
        }
        val builder = JobInfo.Builder(
            record.jobId,
            ComponentName(
                application,
                VN97PaperTradingJobService::class.java,
            ),
        )
            .setPersisted(true)
            .setMinimumLatency(delay)
            .setBackoffCriteria(
                BACKOFF_MILLIS,
                JobInfo.BACKOFF_POLICY_EXPONENTIAL,
            )
            .setRequiresBatteryNotLow(record.requiresBatteryNotLow)
            .setRequiresCharging(record.requiresCharging)
            .setExtras(extras)

        val scheduled = scheduler.schedule(builder.build())
        val updated = record.copy(
            updatedWallTimeMillis = monotonicNow(record),
            scheduleAttemptCount = record.scheduleAttemptCount + 1,
            lastScheduledWallTimeMillis =
                maxOf(record.lastScheduledWallTimeMillis, now),
            lastOutcome =
                if (scheduled > 0) {
                    record.lastOutcome
                } else {
                    boundedOutcome(
                        "JobScheduler rejected paper session"
                    )
                },
        )
        store.save(updated)
        return updated
    }

    private fun finish(
        record: VN97PaperTradingSessionRecord,
        state: VN97PaperTradingSessionState,
        reason: String,
    ): VN97PaperTradingSessionRecord {
        require(
            state == VN97PaperTradingSessionState.STOPPED ||
                state == VN97PaperTradingSessionState.COMPLETED ||
                state == VN97PaperTradingSessionState.FAILED
        )
        val terminal = record.copy(
            state = state,
            updatedWallTimeMillis = monotonicNow(record),
            terminalReason =
                reason.take(
                    VN97PaperTradingSessionRecord
                        .MAX_TERMINAL_REASON_BYTES
                ),
        )
        store.save(terminal)
        scheduler.cancel(record.jobId)
        appendTerminalMemoryBestEffort(terminal)
        return terminal
    }

    private fun appendPerformanceMemory(
        model: ai.vn97.runtime.NativeActivatedModel,
        memory: ai.vn97.runtime.NativeMemoryStore,
        record: VN97PaperTradingSessionRecord,
        evidence: VN97PaperPerformanceEvidence,
        nowNs: Long,
    ) {
        val content = buildString {
            append("VN97PAPERPERF1")
            append(10.toChar())
            append("session_id=")
            append(record.sessionId)
            append(10.toChar())
            append("episode=")
            append(evidence.episode)
            append(10.toChar())
            append("snapshot_id=")
            append(evidence.snapshotId)
            append(10.toChar())
            append("marked_equity_micros=")
            append(evidence.markedEquityMicros)
            append(10.toChar())
            append("total_pnl_micros=")
            append(evidence.totalPnlMicros)
            append(10.toChar())
            append("return_bps=")
            append(evidence.totalReturnBasisPoints)
            append(10.toChar())
            append("max_authority=paper_simulation_only")
        }.take(MAX_MEMORY_CONTENT_CHARS)
        val engine = NativeCognitionInferenceEngine(model)
        memory.append(
            kind = NativeMemoryKind.EPISODIC,
            timestampNs = nowNs,
            importance = 0.9f,
            source = PAPER_PERFORMANCE_MEMORY_SOURCE,
            content = content,
            vector = engine.embedText(content, memory.vectorDim),
            parentId = 0L,
            durable = true,
        )
    }

    private fun appendEpisodeMemory(
        model: ai.vn97.runtime.NativeActivatedModel,
        memory: ai.vn97.runtime.NativeMemoryStore,
        record: VN97PaperTradingSessionRecord,
        planId: String,
        decision: String,
        outcome: String,
        nowNs: Long,
    ) {
        val content = buildString {
            append("VN97PAPERTRADEEP1")
            append(10.toChar())
            append("session_id=")
            append(record.sessionId)
            append(10.toChar())
            append("episode=")
            append(record.episodesAttempted)
            append(10.toChar())
            append("snapshot_id=")
            append(record.lastSnapshotId)
            append(10.toChar())
            append("plan_id=")
            append(planId)
            append(10.toChar())
            append("decision=")
            append(decision.replace(10.toChar(), ' ').take(4096))
            append(10.toChar())
            append("outcome=")
            append(outcome.replace(10.toChar(), ' ').take(4096))
        }.take(MAX_MEMORY_CONTENT_CHARS)
        val engine = NativeCognitionInferenceEngine(model)
        memory.append(
            kind = NativeMemoryKind.EPISODIC,
            timestampNs = nowNs,
            importance = 0.9f,
            source = PAPER_EPISODE_MEMORY_SOURCE,
            content = content,
            vector = engine.embedText(content, memory.vectorDim),
            parentId = 0L,
            durable = true,
        )
    }

    private fun appendTerminalMemory(
        model: ai.vn97.runtime.NativeActivatedModel,
        memory: ai.vn97.runtime.NativeMemoryStore,
        record: VN97PaperTradingSessionRecord,
        nowNs: Long,
    ) {
        val content = buildString {
            append("VN97PAPERTRADETERM1")
            append(10.toChar())
            append("session_id=")
            append(record.sessionId)
            append(10.toChar())
            append("state=")
            append(record.state.name)
            append(10.toChar())
            append("episodes=")
            append(record.episodesAttempted)
            append('/')
            append(record.maxEpisodes)
            append(10.toChar())
            append("reason=")
            append(record.terminalReason.take(4096))
        }.take(MAX_MEMORY_CONTENT_CHARS)
        val engine = NativeCognitionInferenceEngine(model)
        memory.append(
            kind = NativeMemoryKind.EPISODIC,
            timestampNs = nowNs,
            importance = 0.85f,
            source = PAPER_TERMINAL_MEMORY_SOURCE,
            content = content,
            vector = engine.embedText(content, memory.vectorDim),
            parentId = 0L,
            durable = true,
        )
    }

    private fun appendTerminalMemoryBestEffort(
        record: VN97PaperTradingSessionRecord,
    ) {
        var reopenForeground = false
        runCatching {
            reopenForeground =
                application.assistant
                    .releaseForBackgroundContinuation()
            openActivatedModel().use { model ->
                requireModelIdentity(record, model.info.modelId)
                application.platformRuntime
                    .openOrCreateProductionMemory(model)
                    .use { memory ->
                        appendTerminalMemory(
                            model = model,
                            memory = memory,
                            record = record,
                            nowNs = wallNowNs(
                                System.currentTimeMillis()
                            ),
                        )
                    }
            }
        }
        if (reopenForeground) {
            runCatching { application.assistant.openIfActivated() }
        }
    }

    internal fun shouldSystemRetry(jobId: Int): Boolean {
        val record = store.loadOrNull(jobId) ?: return false
        return !record.terminal &&
            record.state != VN97PaperTradingSessionState.PAUSED
    }

    private fun resultOutcome(
        result: ai.vn97.platform.VN97PaperTradingAgentResult,
    ): String {
        val paper = result.paperResult
        return if (
            paper.decision.kind == VN97PaperTradingDecisionKind.HOLD
        ) {
            "HOLD snapshot=" + paper.snapshotId
        } else {
            val fill = checkNotNull(paper.fill)
            buildString {
                append("PAPER_FILL sequence=")
                append(fill.sequence)
                append(" symbol=")
                append(fill.symbol)
                append(" side=")
                append(fill.side.name)
                append(" quantity_microunits=")
                append(fill.quantityMicrounits)
                append(" fill_price_micros=")
                append(fill.fillPriceMicros)
                append(" cash_after_micros=")
                append(fill.cashAfterMicros)
            }
        }
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

    private fun requireSession(jobId: Int) =
        checkNotNull(store.loadOrNull(jobId)) {
            "paper trading session does not exist"
        }

    private fun requireModelIdentity(
        record: VN97PaperTradingSessionRecord,
        modelId: ByteArray,
    ) {
        check(modelId.hex() == record.modelIdHex) {
            "paper trading activated model identity changed"
        }
    }

    private fun isModelIdentityFailure(exc: Throwable): Boolean =
        exc is IllegalStateException &&
            exc.message?.contains(
                "model identity changed",
                ignoreCase = true,
            ) == true

    private fun deadlineExpired(
        record: VN97PaperTradingSessionRecord,
        nowMillis: Long,
    ): Boolean =
        record.deadlineWallTimeMillis > 0L &&
            nowMillis >= record.deadlineWallTimeMillis

    private fun nextRun(
        record: VN97PaperTradingSessionRecord,
        nowMillis: Long,
    ): Long =
        Math.addExact(
            maxOf(nowMillis, record.nextRunWallTimeMillis),
            record.intervalMillis,
        )

    private fun monotonicNow(
        record: VN97PaperTradingSessionRecord,
    ): Long =
        maxOf(
            record.updatedWallTimeMillis,
            System.currentTimeMillis(),
        )

    private fun reportOf(
        record: VN97PaperTradingSessionRecord,
    ) = VN97PaperTradingSessionReport(
        jobId = record.jobId,
        sessionId = record.sessionId,
        state = record.state,
        episodesAttempted = record.episodesAttempted,
        maxEpisodes = record.maxEpisodes,
        wakeCount = record.wakeCount,
        lastDecision = record.lastDecision,
        lastOutcome = record.lastOutcome,
        terminalReason = record.terminalReason,
        nextRunWallTimeMillis = record.nextRunWallTimeMillis,
    )

    private fun allocateJobId(sessionId: String): Int {
        val seed = sessionId.take(8).toLong(16).toInt() and JOB_MASK
        repeat(MAX_JOB_PROBES) { offset ->
            val candidate =
                JOB_PREFIX or ((seed + offset) and JOB_MASK)
            if (
                candidate > 0 &&
                !store.contains(candidate) &&
                scheduler.getPendingJob(candidate) == null
            ) {
                return candidate
            }
        }
        error("could not allocate paper trading JobScheduler ID")
    }

    companion object {
        const val EXTRA_SESSION_ID = "vn97_paper_session_id"
        private const val BACKOFF_MILLIS = 5L * 60L * 1000L
        private const val MAX_ACTIVE_SESSIONS = 8
        private const val MAX_WAKE_COUNT =
            VN97PaperTradingSessionRecord.MAX_WAKE_COUNT
        private const val MAX_JOB_PROBES = 128
        private const val JOB_PREFIX = 0x60000000
        private const val JOB_MASK = 0x0fffffff
        private const val MAX_MEMORY_CONTENT_CHARS = 12 * 1024
        private const val PAPER_EPISODE_MEMORY_SOURCE =
            "vn97.trading.paper.episode"
        private const val PAPER_TERMINAL_MEMORY_SOURCE =
            "vn97.trading.paper.terminal"
        private const val PAPER_PERFORMANCE_MEMORY_SOURCE =
            "vn97.trading.paper.performance"
    }
}

private fun sessionId(
    modelId: String,
    endpoint: String,
    sourceId: String,
    symbols: List<String>,
    userGoal: String,
    createdWallTimeMillis: Long,
): String {
    val canonical = buildString {
        append("VN97PAPERSESSION1")
        append(10.toChar())
        append(modelId)
        append(10.toChar())
        append(endpoint)
        append(10.toChar())
        append(sourceId)
        append(10.toChar())
        append(symbols.joinToString(","))
        append(10.toChar())
        append(userGoal)
        append(10.toChar())
        append(createdWallTimeMillis)
    }
    return MessageDigest.getInstance("SHA-256")
        .digest(canonical.toByteArray(StandardCharsets.UTF_8))
        .hex()
}

private fun boundedDecision(value: String): String =
    value.take(VN97PaperTradingSessionRecord.MAX_DECISION_BYTES)

private fun boundedOutcome(value: String): String =
    value.take(VN97PaperTradingSessionRecord.MAX_OUTCOME_BYTES)

private fun wallNowNs(nowMillis: Long): Long =
    Math.multiplyExact(nowMillis, 1_000_000L)

private fun ByteArray.hex(): String =
    joinToString("") { "%02x".format(it.toInt() and 0xff) }
