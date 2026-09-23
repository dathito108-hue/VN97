package ai.vn97.app

import android.app.job.JobParameters
import android.app.job.JobService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

class VN97PaperTradingJobService : JobService() {
    private val executor = Executors.newSingleThreadExecutor()
    @Volatile
    private var stopped: AtomicBoolean? = null

    override fun onStartJob(params: JobParameters): Boolean {
        val sessionId =
            params.extras.getString(
                VN97PaperTradingSessionManager.EXTRA_SESSION_ID
            ) ?: return false
        val cancellation = AtomicBoolean(false)
        stopped = cancellation
        executor.execute {
            try {
                val app = application as? VN97Application
                    ?: return@execute
                app.paperTrading.runScheduledEpisode(
                    jobId = params.jobId,
                    expectedSessionId = sessionId,
                    stopped = cancellation,
                )
            } finally {
                if (!cancellation.get()) {
                    jobFinished(params, false)
                }
            }
        }
        return true
    }

    override fun onStopJob(params: JobParameters): Boolean {
        stopped?.set(true)
        val app = application as? VN97Application
            ?: return false
        return app.paperTrading.shouldSystemRetry(params.jobId)
    }

    override fun onDestroy() {
        stopped?.set(true)
        executor.shutdownNow()
        super.onDestroy()
    }
}
