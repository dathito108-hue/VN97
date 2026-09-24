package ai.vn97.platform

import java.util.concurrent.Executors
import java.util.concurrent.ScheduledFuture
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean

class VN97ExecutionWatchdog(
    private val store: VN97ExecutionHealthStore,
    private val lease: VN97ExecutionHealthLease,
    private val cancellation: AtomicBoolean,
    private val nowWallTimeMillis: () -> Long =
        System::currentTimeMillis,
) : AutoCloseable {
    private val executor =
        Executors.newSingleThreadScheduledExecutor {
            task ->
            Thread(
                task,
                "vn97-execution-watchdog",
            ).apply {
                isDaemon = true
            }
        }

    private val future: ScheduledFuture<*>

    init {
        val delay =
            maxOf(
                0L,
                lease.deadlineWallTimeMillis -
                    nowWallTimeMillis(),
            )
        future =
            executor.schedule(
                {
                    cancellation.set(true)
                    runCatching {
                        store.finish(
                            lease = lease,
                            state =
                                VN97ExecutionHealthState
                                    .TIMED_OUT,
                            nowWallTimeMillis =
                                maxOf(
                                    nowWallTimeMillis(),
                                    lease
                                        .deadlineWallTimeMillis,
                                ),
                            detail =
                                "execution watchdog deadline exceeded",
                        )
                    }
                },
                delay,
                TimeUnit.MILLISECONDS,
            )
    }

    override fun close() {
        future.cancel(false)
        executor.shutdownNow()
    }
}
