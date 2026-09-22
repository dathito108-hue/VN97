package ai.vn97.platform

import ai.vn97.runtime.AtomicCheckpointStore
import ai.vn97.runtime.NativeBackend
import ai.vn97.runtime.NativeRuntimeConfig
import ai.vn97.runtime.NativeRuntimeOwner
import ai.vn97.runtime.RuntimeLifecycle
import android.app.job.JobInfo
import android.app.job.JobParameters
import android.app.job.JobScheduler
import android.app.job.JobService
import android.content.ComponentName
import android.content.Context
import android.os.PersistableBundle
import java.io.File
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

data class ContinuationSpec(
    val jobId: Int,
    val runtimeConfig: NativeRuntimeConfig,
    val minimumLatencyMillis: Long = 0,
) {
    init {
        require(jobId > 0) { "jobId must be positive" }
        require(minimumLatencyMillis >= 0) { "minimumLatencyMillis must be non-negative" }
    }
}

enum class ContinuationOutcome {
    COMPLETE,
    RESCHEDULE,
}

class ContinuationContext internal constructor(
    val runtime: NativeRuntimeOwner,
    val budget: ComputeBudget,
    private val stopped: AtomicBoolean,
) {
    fun isStopped(): Boolean = stopped.get()
}

fun interface ContinuationWork {
    fun run(context: ContinuationContext): ContinuationOutcome
}

interface ContinuationWorkProvider {
    fun createVN97ContinuationWork(): ContinuationWork
}

class AndroidContinuationScheduler(private val context: Context) {
    private val scheduler: JobScheduler =
        checkNotNull(context.getSystemService(JobScheduler::class.java))

    fun schedule(spec: ContinuationSpec): Int {
        val extras = PersistableBundle().apply {
            putInt("layers", spec.runtimeConfig.layers)
            putInt("batch", spec.runtimeConfig.batch)
            putInt("d_model", spec.runtimeConfig.dModel)
            putInt("d_state", spec.runtimeConfig.dState)
            putInt("recurrent_backend", spec.runtimeConfig.recurrentBackend.code)
            putInt("packed_backend", spec.runtimeConfig.packedBackend.code)
        }
        val component = ComponentName(context, VN97ContinuationJobService::class.java)
        val job = JobInfo.Builder(spec.jobId, component)
            .setPersisted(true)
            .setMinimumLatency(spec.minimumLatencyMillis)
            .setBackoffCriteria(5 * 60 * 1000L, JobInfo.BACKOFF_POLICY_EXPONENTIAL)
            .setExtras(extras)
            .build()
        return scheduler.schedule(job)
    }

    fun cancel(jobId: Int) = scheduler.cancel(jobId)
}

class VN97ContinuationJobService : JobService() {
    private val executor = Executors.newSingleThreadExecutor()
    @Volatile private var stopped: AtomicBoolean? = null

    override fun onStartJob(params: JobParameters): Boolean {
        val cancellation = AtomicBoolean(false)
        stopped = cancellation
        executor.execute {
            var reschedule = true
            try {
                val provider = application as? ContinuationWorkProvider
                    ?: error("Application must implement ContinuationWorkProvider")
                val budget = AndroidComputeGovernor(this).budget()
                if (budget.runnable && !cancellation.get()) {
                    val config = params.extras.toRuntimeConfig()
                    val root = File(noBackupFilesDir, "vn97-continuity/${params.jobId}")
                    val store = AtomicCheckpointStore(root)
                    NativeRuntimeOwner(store).use { owner ->
                        val info = owner.restoreOrCreate(config)
                        when (info.lifecycle) {
                            RuntimeLifecycle.CREATED -> owner.activate()
                            RuntimeLifecycle.SUSPENDED -> owner.resume()
                            RuntimeLifecycle.ACTIVE -> Unit
                        }
                        val outcome = provider.createVN97ContinuationWork().run(
                            ContinuationContext(owner, budget, cancellation)
                        )
                        if (!cancellation.get()) {
                            owner.suspendAndPersist()
                        }
                        reschedule = outcome == ContinuationOutcome.RESCHEDULE
                    }
                }
            } catch (_: Throwable) {
                reschedule = true
            } finally {
                if (!cancellation.get()) {
                    jobFinished(params, reschedule)
                }
            }
        }
        return true
    }

    override fun onStopJob(params: JobParameters): Boolean {
        stopped?.set(true)
        return true
    }

    override fun onDestroy() {
        stopped?.set(true)
        executor.shutdownNow()
        super.onDestroy()
    }
}

private fun PersistableBundle.toRuntimeConfig(): NativeRuntimeConfig {
    return NativeRuntimeConfig(
        layers = getInt("layers"),
        batch = getInt("batch"),
        dModel = getInt("d_model"),
        dState = getInt("d_state"),
        recurrentBackend = backendFromCode(getInt("recurrent_backend")),
        packedBackend = backendFromCode(getInt("packed_backend")),
    )
}

private fun backendFromCode(code: Int): NativeBackend =
    NativeBackend.entries.firstOrNull { it.code == code }
        ?: error("invalid native backend code: $code")
