package ai.vn97.app

import ai.vn97.platform.M6AndroidProductionCapabilities
import ai.vn97.platform.M6ExternalCoordinatorEvent
import ai.vn97.platform.M6ExternalCoordinatorResult
import ai.vn97.platform.M6ReceiptStatus
import ai.vn97.platform.VN97GameAccessibilityController
import ai.vn97.platform.VN97GameEpisodeMemory
import ai.vn97.platform.VN97GameEpisodeTerminal
import ai.vn97.runtime.NativeActivatedInventoryModelLoader
import ai.vn97.runtime.NativeCognitionBoundary
import ai.vn97.runtime.NativeCognitionInferenceEngine
import ai.vn97.runtime.NativePlanController
import ai.vn97.runtime.NativeTypedCognitionAdapter
import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.os.Build
import android.os.IBinder
import android.os.SystemClock
import java.io.File
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

enum class VN97GameAgentState {
    IDLE,
    WAITING_FOR_GAME,
    RUNNING,
    COMPLETED,
    FAILED,
    CANCELLED,
}

data class VN97GameAgentSnapshot(
    val state: VN97GameAgentState,
    val actionCount: Int = 0,
    val detail: String = "",
) {
    init {
        require(actionCount >= 0)
        require(detail.toByteArray(Charsets.UTF_8).size <= 4096)
    }
}

class VN97GameAgentService : Service() {
    private val executor = Executors.newSingleThreadExecutor()
    private val cancelled = AtomicBoolean(false)
    private val episodeActive = AtomicBoolean(false)

    override fun onCreate() {
        super.onCreate()
        ensureChannel()
        startForegroundCompat(
            buildOngoingNotification("Game agent preparing")
        )
    }

    override fun onStartCommand(
        intent: Intent?,
        flags: Int,
        startId: Int,
    ): Int {
        if (intent?.action == ACTION_STOP) {
            cancelled.set(true)
            publishSnapshot(
                VN97GameAgentSnapshot(
                    VN97GameAgentState.CANCELLED,
                    snapshot().actionCount,
                    "cancel requested by user",
                )
            )
            if (!episodeActive.get()) {
                stopForeground(STOP_FOREGROUND_REMOVE)
                stopSelf()
            }
            return START_NOT_STICKY
        }

        val goal = intent?.getStringExtra(EXTRA_GOAL)?.trim().orEmpty()
        if (goal.isEmpty()) {
            publishFailure("game episode goal is missing")
            stopSelf()
            return START_NOT_STICKY
        }
        if (
            goal.toByteArray(Charsets.UTF_8).size >
                MAX_GOAL_UTF8_BYTES
        ) {
            publishFailure("game episode goal exceeds byte bound")
            stopSelf()
            return START_NOT_STICKY
        }
        if (!episodeActive.compareAndSet(false, true)) {
            return START_NOT_STICKY
        }

        cancelled.set(false)
        executor.execute {
            val final = try {
                runEpisode(goal)
            } catch (exc: Throwable) {
                VN97GameAgentSnapshot(
                    state =
                        if (cancelled.get()) {
                            VN97GameAgentState.CANCELLED
                        } else {
                            VN97GameAgentState.FAILED
                        },
                    actionCount = snapshot().actionCount,
                    detail =
                        if (cancelled.get()) {
                            "game episode cancelled"
                        } else {
                            "game episode failed: " +
                                exc::class.java.simpleName
                        },
                )
            }
            publishSnapshot(final)
            publishTerminalNotification(final)
            episodeActive.set(false)
            stopForeground(STOP_FOREGROUND_REMOVE)
            stopSelfResult(startId)
        }
        return START_NOT_STICKY
    }

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onDestroy() {
        cancelled.set(true)
        executor.shutdownNow()
        super.onDestroy()
    }

    private fun runEpisode(goal: String): VN97GameAgentSnapshot {
        val app = application as VN97Application
        val session =
            app.platformRuntime.gameControlPolicy.activeSessionOrNull()
                ?: error("no active game-control authorization")
        check(VN97GameAccessibilityController.isConnected()) {
            "VN97 game accessibility service is not connected"
        }
        check(app.screenCaptureBroker.isActive()) {
            "user-approved screen sharing is not active"
        }

        publishSnapshot(
            VN97GameAgentSnapshot(
                VN97GameAgentState.WAITING_FOR_GAME,
                detail = "open the authorized game to begin",
            )
        )
        waitForForeground(
            packageName = session.packageName,
            timeoutMillis = FOREGROUND_WAIT_MILLIS,
        )

        return app.withSovereignExecution {
            val reopenForeground =
                app.assistant.releaseForBackgroundContinuation()
            try {
                val currentSession =
                    app.platformRuntime.gameControlPolicy
                        .activeSessionOrNull()
                        ?: error("game-control authorization expired")
                check(currentSession.packageName == session.packageName) {
                    "game-control authorization changed before episode start"
                }

                val model = checkNotNull(
                    NativeActivatedInventoryModelLoader.openOrNull(
                        File(
                            app.noBackupFilesDir,
                            "vn97-capabilities",
                        )
                    )
                ) {
                    "trusted VN97 model is not active"
                }
                model.use {
                    check(model.info.hasVisionProjection) {
                        "activated VN97 model has no production vision weights"
                    }
                    val engine = NativeCognitionInferenceEngine(model)
                    val cognition = NativeTypedCognitionAdapter(engine)
                    val grants =
                        M6AndroidProductionCapabilities
                            .userApprovedGameControlGrants(
                                VN97AppAssistant.APP_PRINCIPAL,
                                session.packageName,
                            )
                    val coordinator =
                        app.platformRuntime
                            .createProductionGameExternalCoordinator(
                                cognition = cognition,
                                grants = grants,
                                auditFileName =
                                    "m14-game-actions.jsonl",
                            )
                    app.platformRuntime
                        .openOrCreateProductionMemory(model)
                        .use { memory ->
                            val episodeMemory =
                                VN97GameEpisodeMemory.production(
                                    engine = engine,
                                    memory = memory,
                                    packageName = session.packageName,
                                    userGoal = goal,
                                    startedNs = wallNowNs(),
                                )
                            val memoryContext = episodeMemory.begin()
                            try {
                                val result = runClosedLoop(
                                    app = app,
                                    goal = goal,
                                    packageName = session.packageName,
                                    engine = engine,
                                    coordinator = coordinator,
                                    memory = memory,
                                    episodeMemory = episodeMemory,
                                    priorStrategyContext =
                                        memoryContext.priorStrategyContext,
                                )
                                val learned = episodeMemory.finish(
                                    terminal = when (result.state) {
                                        VN97GameAgentState.COMPLETED ->
                                            VN97GameEpisodeTerminal.COMPLETED
                                        VN97GameAgentState.CANCELLED ->
                                            VN97GameEpisodeTerminal.CANCELLED
                                        else ->
                                            VN97GameEpisodeTerminal.FAILED
                                    },
                                    detail =
                                        result.detail.ifBlank {
                                            result.state.name.lowercase()
                                        },
                                    timestampNs = wallNowNs(),
                                )
                                result.copy(
                                    detail = buildString {
                                        append(result.detail)
                                        if (
                                            learned.learnedStrategy.isNotBlank()
                                        ) {
                                            if (isNotEmpty()) append("\n")
                                            append("Learned: ")
                                            append(
                                                learned.learnedStrategy.take(
                                                    MAX_DETAIL_CHARS / 2
                                                )
                                            )
                                        }
                                    }.take(MAX_DETAIL_CHARS)
                                )
                            } catch (exc: Throwable) {
                                val terminal =
                                    if (cancelled.get()) {
                                        VN97GameEpisodeTerminal.CANCELLED
                                    } else {
                                        VN97GameEpisodeTerminal.FAILED
                                    }
                                runCatching {
                                    episodeMemory.finish(
                                        terminal = terminal,
                                        detail =
                                            if (cancelled.get()) {
                                                "game episode cancelled"
                                            } else {
                                                "game episode failed: " +
                                                    exc::class.java.simpleName
                                            },
                                        timestampNs = wallNowNs(),
                                    )
                                }.exceptionOrNull()?.let(exc::addSuppressed)
                                throw exc
                            }
                        }
                }
            } finally {
                if (reopenForeground) {
                    runCatching {
                        app.assistant.openIfActivated()
                    }
                }
            }
        }
    }

    private fun runClosedLoop(
        app: VN97Application,
        goal: String,
        packageName: String,
        engine: NativeCognitionInferenceEngine,
        coordinator: ai.vn97.platform.M6EndToEndExternalCoordinator,
        memory: ai.vn97.runtime.NativeMemoryStore,
        episodeMemory: VN97GameEpisodeMemory,
        priorStrategyContext: String,
    ): VN97GameAgentSnapshot {
        val started = SystemClock.elapsedRealtime()
        var actionCount = 0
        var previousVerification = ""
        var afterElapsedNs = SystemClock.elapsedRealtimeNanos()
        var frame =
            app.screenCaptureBroker.awaitFreshFrame(
                afterElapsedRealtimeNs = afterElapsedNs,
            )
        var observation = engine.perceiveVision(frame.prepared)

        while (true) {
            checkNotCancelled()
            checkEpisodeBudget(started, actionCount)
            requireGameStillAuthorized(app, packageName)
            check(
                VN97GameAccessibilityController
                    .currentPackageName() == packageName
            ) {
                "authorized game is no longer foreground"
            }

            publishSnapshot(
                VN97GameAgentSnapshot(
                    state = VN97GameAgentState.RUNNING,
                    actionCount = actionCount,
                    detail = "fresh frame ${actionCount + 1}",
                )
            )

            val controller = coordinator.buildPlan(
                goal = buildFrameGoal(
                    userGoal = goal,
                    packageName = packageName,
                    actionCount = actionCount,
                    observation = observation,
                    previousVerification = previousVerification,
                    priorStrategyContext = priorStrategyContext,
                ),
                createdNs = wallNowNs(),
            )
            val decision = advanceUntilActionOrTerminal(
                coordinator = coordinator,
                controller = controller,
                memory = memory,
            )

            val execution = decision.execution
            if (execution == null) {
                check(
                    decision.cognition.boundary ==
                        NativeCognitionBoundary.COMPLETED
                ) {
                    "game decision ended without action or completion"
                }
                val finalResponse =
                    decision.cognition.finalResponse.trim()
                check(finalResponse.isNotEmpty()) {
                    "game episode completed without final response"
                }
                return VN97GameAgentSnapshot(
                    VN97GameAgentState.COMPLETED,
                    actionCount,
                    finalResponse.take(MAX_DETAIL_CHARS),
                )
            }

            val receipt = execution.receipt
            check(receipt.status == M6ReceiptStatus.SUCCEEDED) {
                "game action did not succeed"
            }
            check(receipt.capabilityId in GAME_CAPABILITIES) {
                "non-game capability escaped game-only coordinator"
            }
            val actionIndex = actionCount

            afterElapsedNs = SystemClock.elapsedRealtimeNanos()
            frame = app.screenCaptureBroker.awaitFreshFrame(
                afterElapsedRealtimeNs = afterElapsedNs,
            )
            val afterObservation =
                engine.perceiveVision(frame.prepared)
            previousVerification = verifyOutcome(
                engine = engine,
                goal = goal,
                beforeObservation = observation,
                afterObservation = afterObservation,
                capabilityId = receipt.capabilityId,
                actionResult = receipt.result,
            )
            episodeMemory.recordAction(
                actionIndex = actionIndex,
                capabilityId = receipt.capabilityId,
                actionResult = receipt.result,
                beforeObservation = observation,
                afterObservation = afterObservation,
                verification = previousVerification,
                timestampNs = wallNowNs(),
            )
            actionCount += 1
            observation = afterObservation
        }
    }

    private fun advanceUntilActionOrTerminal(
        coordinator: ai.vn97.platform.M6EndToEndExternalCoordinator,
        controller: NativePlanController,
        memory: ai.vn97.runtime.NativeMemoryStore,
    ): M6ExternalCoordinatorResult {
        repeat(MAX_DECISION_ADVANCES) {
            checkNotCancelled()
            val result = coordinator.advance(
                controller = controller,
                principal = VN97AppAssistant.APP_PRINCIPAL,
                memory = memory,
                maxCycles = MAX_COGNITION_CYCLES_PER_ADVANCE,
                nowNs = SystemClock.elapsedRealtimeNanos(),
            )
            if (result.execution != null) {
                return result
            }
            when (result.event) {
                M6ExternalCoordinatorEvent.APPROVAL_REQUIRED ->
                    error("game-only capability unexpectedly requested approval")

                M6ExternalCoordinatorEvent.APPROVAL_REJECTED ->
                    error("game-only capability approval was rejected")

                M6ExternalCoordinatorEvent.EXECUTED ->
                    error("executed event is missing execution receipt")

                M6ExternalCoordinatorEvent.NONE -> Unit
            }
            when (result.cognition.boundary) {
                NativeCognitionBoundary.COMPLETED ->
                    return result

                NativeCognitionBoundary.YIELDED ->
                    Unit

                NativeCognitionBoundary.WAITING_EXTERNAL ->
                    error("game action remained unresolved")

                NativeCognitionBoundary.PAUSED,
                NativeCognitionBoundary.FAILED,
                NativeCognitionBoundary.CANCELLED,
                NativeCognitionBoundary.BUDGET_EXHAUSTED,
                NativeCognitionBoundary.STALLED,
                -> error(
                    "game decision ended at " +
                        result.cognition.boundary.name
                )
            }
        }
        error("game decision reasoning advance bound exhausted")
    }

    private fun buildFrameGoal(
        userGoal: String,
        packageName: String,
        actionCount: Int,
        observation: String,
        previousVerification: String,
        priorStrategyContext: String,
    ): String = buildString {
        append("VN97GAME2\n")
        append("Operate only the currently authorized Android game package. ")
        append("This is one closed-loop frame. Decide from the supplied fresh ")
        append("visual observation. If an action is needed, plan exactly one ")
        append("EXTERNAL game action (tap, swipe, or back) before any response. ")
        append("Do not plan app launch, clipboard, network, file, or other tools. ")
        append("If the user's episode goal is already satisfied or no safe game ")
        append("action is needed, respond without an EXTERNAL step.\n")
        append("package=")
        append(packageName)
        append("\naction_index=")
        append(actionCount)
        append("\nuser_goal=")
        append(userGoal.take(MAX_PROMPT_FIELD_CHARS))
        append("\nvisual_observation=")
        append(
            observation
                .replace('\n', ' ')
                .take(MAX_PROMPT_FIELD_CHARS)
        )
        if (priorStrategyContext.isNotBlank()) {
            append("\nrecalled_strategy=")
            append(
                priorStrategyContext
                    .replace('\n', ' ')
                    .take(MAX_PRIOR_STRATEGY_CHARS)
            )
        }
        if (previousVerification.isNotBlank()) {
            append("\nprevious_outcome=")
            append(
                previousVerification
                    .replace('\n', ' ')
                    .take(MAX_VERIFICATION_CHARS)
            )
        }
        append("\ndecision=")
    }

    private fun verifyOutcome(
        engine: NativeCognitionInferenceEngine,
        goal: String,
        beforeObservation: String,
        afterObservation: String,
        capabilityId: String,
        actionResult: String,
    ): String {
        val prompt = buildString {
            append("VN97GAMEVERIFY2\n")
            append("Use only the supplied before/after visual observations ")
            append("and the recorded governed game action. Describe visible ")
            append("progress toward the user goal and remaining uncertainty. ")
            append("Do not request or execute tools.\n")
            append("goal=")
            append(goal.take(MAX_PROMPT_FIELD_CHARS))
            append("\naction=")
            append(capabilityId)
            append("\naction_result=")
            append(actionResult.take(512))
            append("\nbefore=")
            append(
                beforeObservation
                    .replace('\n', ' ')
                    .take(MAX_PROMPT_FIELD_CHARS)
            )
            append("\nafter=")
            append(
                afterObservation
                    .replace('\n', ' ')
                    .take(MAX_PROMPT_FIELD_CHARS)
            )
            append("\nverification=")
        }
        return engine.generateText(
            prompt,
            MAX_VERIFY_TOKENS,
        ).trim().ifEmpty {
            "No textual verification was produced; use the fresh after-frame."
        }.take(MAX_VERIFICATION_CHARS)
    }

    private fun waitForForeground(
        packageName: String,
        timeoutMillis: Long,
    ) {
        val deadline =
            SystemClock.elapsedRealtime() + timeoutMillis
        while (
            VN97GameAccessibilityController
                .currentPackageName() != packageName
        ) {
            checkNotCancelled()
            if (SystemClock.elapsedRealtime() >= deadline) {
                error("timed out waiting for authorized game foreground")
            }
            Thread.sleep(FOREGROUND_POLL_MILLIS)
        }
    }

    private fun requireGameStillAuthorized(
        app: VN97Application,
        packageName: String,
    ) {
        val session =
            app.platformRuntime.gameControlPolicy
                .activeSessionOrNull()
                ?: error("game-control authorization expired or was revoked")
        check(session.packageName == packageName) {
            "game-control authorization changed during episode"
        }
        check(app.screenCaptureBroker.isActive()) {
            "screen sharing stopped during game episode"
        }
        check(VN97GameAccessibilityController.isConnected()) {
            "game accessibility service disconnected"
        }
    }

    private fun checkEpisodeBudget(
        startedElapsedMillis: Long,
        actionCount: Int,
    ) {
        check(actionCount < MAX_ACTIONS) {
            "game episode action budget exhausted"
        }
        check(
            SystemClock.elapsedRealtime() - startedElapsedMillis <
                MAX_EPISODE_MILLIS
        ) {
            "game episode time budget exhausted"
        }
    }

    private fun checkNotCancelled() {
        check(!cancelled.get()) {
            "game episode cancelled"
        }
    }

    private fun ensureChannel() {
        val manager = checkNotNull(
            getSystemService(NotificationManager::class.java)
        )
        manager.createNotificationChannel(
            NotificationChannel(
                NOTIFICATION_CHANNEL,
                "VN97 game agent",
                NotificationManager.IMPORTANCE_LOW,
            ).apply {
                description =
                    "Shows the user-controlled VN97 closed-loop game session."
                setShowBadge(false)
            }
        )
    }

    private fun startForegroundCompat(notification: Notification) {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.UPSIDE_DOWN_CAKE) {
            startForeground(
                NOTIFICATION_ID,
                notification,
                ServiceInfo.FOREGROUND_SERVICE_TYPE_SPECIAL_USE,
            )
        } else {
            startForeground(NOTIFICATION_ID, notification)
        }
    }

    private fun buildOngoingNotification(text: String): Notification {
        val openApp = PendingIntent.getActivity(
            this,
            0,
            Intent(this, VN97MainActivity::class.java),
            PendingIntent.FLAG_UPDATE_CURRENT or
                PendingIntent.FLAG_IMMUTABLE,
        )
        val stop = PendingIntent.getService(
            this,
            1,
            Intent(this, VN97GameAgentService::class.java)
                .setAction(ACTION_STOP),
            PendingIntent.FLAG_UPDATE_CURRENT or
                PendingIntent.FLAG_IMMUTABLE,
        )
        return Notification.Builder(this, NOTIFICATION_CHANNEL)
            .setSmallIcon(R.drawable.ic_vn97_assistant)
            .setContentTitle("VN97 game agent")
            .setContentText(text.take(120))
            .setContentIntent(openApp)
            .setOngoing(true)
            .setCategory(Notification.CATEGORY_SERVICE)
            .addAction(
                Notification.Action.Builder(
                    R.drawable.ic_vn97_assistant,
                    "Stop",
                    stop,
                ).build()
            )
            .build()
    }

    private fun publishSnapshot(value: VN97GameAgentSnapshot) {
        setSnapshot(value)
        if (
            value.state == VN97GameAgentState.RUNNING ||
            value.state == VN97GameAgentState.WAITING_FOR_GAME
        ) {
            val text =
                if (value.state == VN97GameAgentState.WAITING_FOR_GAME) {
                    "Open the authorized game"
                } else {
                    "Closed-loop actions: ${value.actionCount}/$MAX_ACTIONS"
                }
            val manager = checkNotNull(
                getSystemService(NotificationManager::class.java)
            )
            manager.notify(
                NOTIFICATION_ID,
                buildOngoingNotification(text),
            )
        }
    }

    private fun publishFailure(message: String) {
        publishSnapshot(
            VN97GameAgentSnapshot(
                VN97GameAgentState.FAILED,
                detail = message.take(MAX_DETAIL_CHARS),
            )
        )
    }

    private fun publishTerminalNotification(
        value: VN97GameAgentSnapshot,
    ) {
        val manager = checkNotNull(
            getSystemService(NotificationManager::class.java)
        )
        val title = when (value.state) {
            VN97GameAgentState.COMPLETED ->
                "VN97 game episode completed"
            VN97GameAgentState.CANCELLED ->
                "VN97 game episode stopped"
            else ->
                "VN97 game episode ended"
        }
        val text = when (value.state) {
            VN97GameAgentState.COMPLETED ->
                "Completed after ${value.actionCount} governed actions."
            VN97GameAgentState.CANCELLED ->
                "Stopped after ${value.actionCount} governed actions."
            else ->
                "Ended after ${value.actionCount} governed actions."
        }
        manager.notify(
            TERMINAL_NOTIFICATION_ID,
            Notification.Builder(this, NOTIFICATION_CHANNEL)
                .setSmallIcon(R.drawable.ic_vn97_assistant)
                .setContentTitle(title)
                .setContentText(text)
                .setAutoCancel(true)
                .build(),
        )
    }

    companion object {
        const val ACTION_STOP =
            "ai.vn97.app.action.STOP_GAME_AGENT"
        private const val ACTION_START =
            "ai.vn97.app.action.START_GAME_AGENT"
        private const val EXTRA_GOAL = "goal"
        private const val NOTIFICATION_CHANNEL = "vn97-game-agent"
        private const val NOTIFICATION_ID = 9714
        private const val TERMINAL_NOTIFICATION_ID = 9715

        private const val MAX_GOAL_UTF8_BYTES = 32 * 1024
        private const val MAX_DETAIL_CHARS = 4096
        private const val MAX_PROMPT_FIELD_CHARS = 4096
        private const val MAX_VERIFICATION_CHARS = 2048
        private const val MAX_PRIOR_STRATEGY_CHARS = 4096
        private const val MAX_VERIFY_TOKENS = 256
        private const val MAX_ACTIONS = 64
        private const val MAX_DECISION_ADVANCES = 4
        private const val MAX_COGNITION_CYCLES_PER_ADVANCE = 8
        private const val MAX_EPISODE_MILLIS = 10 * 60 * 1000L
        private const val FOREGROUND_WAIT_MILLIS = 30_000L
        private const val FOREGROUND_POLL_MILLIS = 100L

        private val GAME_CAPABILITIES = setOf(
            M6AndroidProductionCapabilities.GAME_TAP_CAPABILITY,
            M6AndroidProductionCapabilities.GAME_SWIPE_CAPABILITY,
            M6AndroidProductionCapabilities.GAME_BACK_CAPABILITY,
        )

        @Volatile
        private var latest =
            VN97GameAgentSnapshot(VN97GameAgentState.IDLE)

        fun snapshot(): VN97GameAgentSnapshot = latest

        private fun setSnapshot(value: VN97GameAgentSnapshot) {
            latest = value
        }

        fun start(
            context: Context,
            goal: String,
        ) {
            require(goal.isNotBlank()) {
                "game episode goal must not be blank"
            }
            context.startForegroundService(
                Intent(context, VN97GameAgentService::class.java)
                    .setAction(ACTION_START)
                    .putExtra(EXTRA_GOAL, goal)
            )
        }

        fun stop(context: Context) {
            context.startService(
                Intent(context, VN97GameAgentService::class.java)
                    .setAction(ACTION_STOP)
            )
        }
    }
}

private fun wallNowNs(): Long =
    Math.multiplyExact(
        System.currentTimeMillis(),
        1_000_000L,
    )
