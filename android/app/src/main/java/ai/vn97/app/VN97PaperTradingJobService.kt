package ai.vn97.app

import ai.vn97.platform.AndroidComputeGovernor
import ai.vn97.platform.VN97ExecutionHealthDomain
import ai.vn97.platform.VN97ExecutionHealthLease
import ai.vn97.platform.VN97ExecutionHealthState
import ai.vn97.platform.VN97ExecutionHealthStore
import ai.vn97.platform.VN97ExecutionWatchdog
import android.app.job.JobParameters
import android.app.job.JobService
import java.io.File
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

class VN97PaperTradingJobService : JobService() {
    private val executor =
        Executors.newSingleThreadExecutor()
    private val healthStore by lazy(
        LazyThreadSafetyMode.SYNCHRONIZED
    ) {
        VN97ExecutionHealthStore(
            File(
                noBackupFilesDir,
                "vn97-execution-health",
            )
        )
    }

    @Volatile
    private var stopped: AtomicBoolean? = null

    @Volatile
    private var systemStopped = false

    override fun onStartJob(
        params: JobParameters,
    ): Boolean {
        val sessionId =
            params.extras.getString(
                VN97PaperTradingSessionManager
                    .EXTRA_SESSION_ID
            ) ?: return false
        val cancellation =
            AtomicBoolean(false)
        stopped = cancellation
        systemStopped = false
        executor.execute {
            var retry = false
            var lease:
                VN97ExecutionHealthLease? = null
            var watchdog:
                VN97ExecutionWatchdog? = null
            var failure: Throwable? = null
            try {
                val app =
                    application as?
                        VN97Application
                        ?: return@execute
                val budget =
                    AndroidComputeGovernor(this)
                        .budget()
                if (!budget.runnable) {
                    retry =
                        app.paperTrading
                            .shouldSystemRetry(
                                params.jobId
                            )
                    return@execute
                }
                val begin =
                    healthStore.begin(
                        domain =
                            VN97ExecutionHealthDomain
                                .PAPER_TRADING_JOB,
                        key =
                            params.jobId
                                .toString(),
                        nowWallTimeMillis =
                            System
                                .currentTimeMillis(),
                        maxRunMillis =
                            maxOf(
                                1L,
                                budget.maxRunMillis,
                            ),
                    )
                if (begin.suppressed) {
                    retry =
                        app.paperTrading
                            .shouldSystemRetry(
                                params.jobId
                            )
                    return@execute
                }
                lease =
                    checkNotNull(
                        begin.lease
                    )
                watchdog =
                    VN97ExecutionWatchdog(
                        store = healthStore,
                        lease = lease,
                        cancellation =
                            cancellation,
                    )
                app.mobileRecovery
                    .recover(
                        VN97MobileRecoveryTrigger
                            .EXECUTION_ENTRY
                    )
                    .requireActivationReady()
                app.runtimeResources
                    .requireRunnable(
                        VN97RuntimeExecutionClass
                            .BACKGROUND
                    )
                app.paperTrading
                    .runScheduledEpisode(
                        jobId = params.jobId,
                        expectedSessionId =
                            sessionId,
                        stopped =
                            cancellation,
                    )
                retry =
                    if (cancellation.get()) {
                        app.paperTrading
                            .shouldSystemRetry(
                                params.jobId
                            )
                    } else {
                        false
                    }
            } catch (exc: RuntimeException) {
                failure = exc
                val app =
                    application as?
                        VN97Application
                retry =
                    app?.paperTrading
                        ?.shouldSystemRetry(
                            params.jobId
                        ) == true
            } finally {
                watchdog?.close()
                lease?.let {
                    currentLease ->
                    runCatching {
                        healthStore.finish(
                            lease =
                                currentLease,
                            state =
                                if (
                                    failure == null &&
                                    !cancellation.get()
                                ) {
                                    VN97ExecutionHealthState
                                        .SUCCEEDED
                                } else {
                                    VN97ExecutionHealthState
                                        .FAILED
                                },
                            nowWallTimeMillis =
                                System
                                    .currentTimeMillis(),
                            detail =
                                failure
                                    ?.javaClass
                                    ?.simpleName
                                    ?: if (
                                        cancellation.get()
                                    ) {
                                        "paper execution cancelled or stopped"
                                    } else {
                                        ""
                                    },
                        )
                    }
                }
                if (!systemStopped) {
                    jobFinished(
                        params,
                        retry ||
                            cancellation.get(),
                    )
                }
            }
        }
        return true
    }

    override fun onStopJob(
        params: JobParameters,
    ): Boolean {
        systemStopped = true
        stopped?.set(true)
        val app =
            application as?
                VN97Application
                ?: return false
        return app.paperTrading
            .shouldSystemRetry(
                params.jobId
            )
    }

    override fun onDestroy() {
        systemStopped = true
        stopped?.set(true)
        executor.shutdownNow()
        super.onDestroy()
    }
}

