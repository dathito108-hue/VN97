package ai.vn97.app

import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.locks.ReentrantLock
import kotlin.concurrent.withLock

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
            try {
                recover(
                    VN97MobileRecoveryTrigger
                        .PROCESS_START
                )
            } catch (_: RuntimeException) {
                // Domain failures are represented in the report.
                // A coordinator-level runtime failure must not
                // crash a cold-starting Android process.
            } finally {
                processStartScheduled.set(false)
            }
        }
    }

    fun latestReport():
        VN97MobileRecoveryReport? =
        latest

    fun recover(
        trigger: VN97MobileRecoveryTrigger,
    ): VN97MobileRecoveryReport =
        recoveryLock.withLock {
            val report =
                application
                    .withSovereignExecution {
                        executeVN97MobileRecovery(
                            trigger = trigger,
                        ) { domain ->
                            recoverDomain(domain)
                        }
                    }
            latest = report
            report
        }

    private fun recoverDomain(
        domain: VN97MobileRecoveryDomain,
    ) {
        when (domain) {
            VN97MobileRecoveryDomain.ACTIVATION ->
                application.provisioner
                    .recoverPending()

            VN97MobileRecoveryDomain.AUTONOMOUS ->
                application.autonomousWork
                    .reconcileAfterSystemRestart()

            VN97MobileRecoveryDomain.PAPER_TRADING ->
                application.paperTrading
                    .reconcileAfterSystemRestart()
        }
    }
}
