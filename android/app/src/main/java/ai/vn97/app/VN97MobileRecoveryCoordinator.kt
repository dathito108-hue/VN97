package ai.vn97.app

import ai.vn97.platform.VN97ExecutionHealthDomain
import ai.vn97.platform.VN97ExecutionHealthState
import ai.vn97.platform.VN97ExecutionHealthStore
import ai.vn97.platform.VN97ExecutionWatchdog
import java.io.File
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.locks.ReentrantLock

class VN97MobileRecoveryCoordinator(
    private val application: VN97Application,
) {
    private val recoveryLock =
        ReentrantLock(true)
    private val processStartScheduled =
        AtomicBoolean(false)
    private val worker =
        Executors.newSingleThreadExecutor { task ->
            Thread(
                task,
                "vn97-mobile-recovery",
            ).apply {
                isDaemon = true
            }
        }
    private val healthStore by lazy(
        LazyThreadSafetyMode.SYNCHRONIZED
    ) {
        VN97ExecutionHealthStore(
            File(
                application.noBackupFilesDir,
                "vn97-execution-health",
            )
        )
    }

    @Volatile
    private var latest:
        VN97MobileRecoveryReport? = null

    fun scheduleProcessStartRecovery() {
        if (
            !processStartScheduled
                .compareAndSet(false, true)
        ) {
            return
        }
        worker.execute {
            val cancellation =
                AtomicBoolean(false)
            val begin =
                runCatching {
                    healthStore.begin(
                        domain =
                            VN97ExecutionHealthDomain
                                .MOBILE_RECOVERY,
                        key = "process-start",
                        nowWallTimeMillis =
                            System
                                .currentTimeMillis(),
                        maxRunMillis =
                            PROCESS_RECOVERY_WATCHDOG_MILLIS,
                    )
                }.getOrNull()
            if (
                begin == null ||
                begin.suppressed
            ) {
                processStartScheduled.set(false)
                return@execute
            }
            val lease =
                checkNotNull(begin.lease)
            val watchdog =
                VN97ExecutionWatchdog(
                    store = healthStore,
                    lease = lease,
                    cancellation = cancellation,
                )
            var failure:
                RuntimeException? = null
            try {
                recover(
                    VN97MobileRecoveryTrigger
                        .PROCESS_START
                )
            } catch (exc: RuntimeException) {
                failure = exc
            } finally {
                watchdog.close()
                runCatching {
                    healthStore.finish(
                        lease = lease,
                        state =
                            if (failure == null) {
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
                                .orEmpty(),
                    )
                }
                processStartScheduled.set(false)
            }
        }
    }

    fun latestReport():
        VN97MobileRecoveryReport? =
        latest

    fun recover(
        trigger: VN97MobileRecoveryTrigger,
    ): VN97MobileRecoveryReport {
        val acquired =
            try {
                recoveryLock.tryLock(
                    RECOVERY_LOCK_TIMEOUT_MILLIS,
                    TimeUnit.MILLISECONDS,
                )
            } catch (exc: InterruptedException) {
                Thread.currentThread()
                    .interrupt()
                throw IllegalStateException(
                    "VN97 recovery lock wait interrupted",
                    exc,
                )
            }
        check(acquired) {
            "VN97 recovery lock timed out"
        }
        try {
            val report =
                application
                    .withSovereignExecutionBounded(
                        SOVEREIGN_LOCK_TIMEOUT_MILLIS
                    ) {
                        executeVN97MobileRecovery(
                            trigger = trigger,
                        ) { domain ->
                            recoverDomain(domain)
                        }
                    }
            latest = report
            return report
        } finally {
            recoveryLock.unlock()
        }
    }

    private fun recoverDomain(
        domain: VN97MobileRecoveryDomain,
    ) {
        when (domain) {
            VN97MobileRecoveryDomain.ACTIVATION -> {
                application.provisioner
                    .recoverPending()
                if (
                    BuildConfig
                        .VN97_TURNKEY_REQUIRED
                ) {
                    application.bundledBootstrap
                        .activateIfPresent(
                            required = true
                        )
                }
            }

            VN97MobileRecoveryDomain.AUTONOMOUS ->
                application.autonomousWork
                    .reconcileAfterSystemRestart()

            VN97MobileRecoveryDomain.PAPER_TRADING ->
                application.paperTrading
                    .reconcileAfterSystemRestart()
        }
    }

    companion object {
        private const val RECOVERY_LOCK_TIMEOUT_MILLIS =
            15_000L
        private const val SOVEREIGN_LOCK_TIMEOUT_MILLIS =
            15_000L
        private const val PROCESS_RECOVERY_WATCHDOG_MILLIS =
            60_000L
    }
}

