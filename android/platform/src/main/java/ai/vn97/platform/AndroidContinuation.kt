package ai.vn97.platform

import ai.vn97.runtime.AtomicCheckpointStore
import ai.vn97.runtime.AtomicCompositeContinuityStore
import ai.vn97.runtime.NativeActivatedModel
import ai.vn97.runtime.NativeBackend
import ai.vn97.runtime.NativePlan
import ai.vn97.runtime.NativePlanController
import ai.vn97.runtime.NativePlanStatus
import ai.vn97.runtime.NativeRuntimeCheckpointSnapshot
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

private const val CONTINUATION_MODE_KEY = "vn97_mode"
private const val CONTINUATION_MODE_RUNTIME = "runtime"
private const val CONTINUATION_MODE_ASSISTANT = "assistant"
private const val ASSISTANT_PRINCIPAL_KEY = "assistant_principal"
private const val ASSISTANT_PLAN_ID_KEY = "assistant_plan_id"
private const val ASSISTANT_MODEL_ID_KEY = "assistant_model_id"

private const val CONTINUATION_BACKOFF_MILLIS = 5 * 60 * 1000L

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

data class VN97AssistantContinuationSpec(
    val jobId: Int,
    val runtimeConfig: NativeRuntimeConfig,
    val binding: VN97AssistantContinuationBinding,
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

class VN97AssistantContinuationContext internal constructor(
    val jobId: Int,
    val runtime: NativeRuntimeOwner,
    val controller: NativePlanController,
    val binding: VN97AssistantContinuationBinding,
    val epoch: Long,
    val budget: ComputeBudget,
    private val stopped: AtomicBoolean,
) {
    val principal: String get() = binding.principal

    fun isStopped(): Boolean = stopped.get()

    fun requireActivatedModel(model: NativeActivatedModel) {
        binding.requireModelId(model.info.modelId)
    }
}

fun interface ContinuationWork {
    fun run(context: ContinuationContext): ContinuationOutcome
}

fun interface VN97AssistantContinuationWork {
    fun run(context: VN97AssistantContinuationContext): ContinuationOutcome
}

interface ContinuationWorkProvider {
    fun createVN97ContinuationWork(): ContinuationWork
}

interface VN97AssistantContinuationWorkProvider {
    fun createVN97AssistantContinuationWork(): VN97AssistantContinuationWork
}

class AndroidContinuationScheduler(private val context: Context) {
    private val scheduler: JobScheduler =
        checkNotNull(context.getSystemService(JobScheduler::class.java))

    fun schedule(spec: ContinuationSpec): Int = scheduleJob(
        jobId = spec.jobId,
        minimumLatencyMillis = spec.minimumLatencyMillis,
        extras = runtimeExtras(spec.runtimeConfig).apply {
            putString(CONTINUATION_MODE_KEY, CONTINUATION_MODE_RUNTIME)
        },
    )

    /** Persist one safe M7M epoch and freeze its M6 principal/model identity before scheduling. */
    fun persistAssistant(
        jobId: Int,
        snapshot: NativeRuntimeCheckpointSnapshot,
        plan: NativePlan,
        principal: String,
    ): VN97AssistantContinuationBinding {
        require(jobId > 0) { "jobId must be positive" }
        require(!plan.isTerminal()) {
            "terminal plan must not be persisted as assistant continuation"
        }
        val root = context.continuationRoot(jobId)
        val bindingStore = VN97AssistantContinuationBindingStore(root)
        val existing = bindingStore.loadOrNull()
        if (existing != null) {
            check(existing.principal == principal && existing.planId == plan.planId) {
                "assistant continuation principal/plan identity is immutable"
            }
            check(snapshot.modelBinding.bound) {
                "assistant continuation requires a model-bound runtime snapshot"
            }
            existing.requireModelId(snapshot.modelBinding.modelId)
        }
        val manifest = AtomicCompositeContinuityStore(root).save(snapshot, plan)
        val binding = VN97AssistantContinuationBinding(
            principal = principal,
            planId = manifest.planId,
            modelIdHex = manifest.modelId.lowerHex(),
        )
        return bindingStore.bind(binding)
    }

    /**
     * Schedule one persisted, reboot-surviving assistant continuation bound to the exact
     * durable VN97ACB1 principal/plan/model identity returned by persistAssistant(...).
     */
    fun scheduleAssistant(spec: VN97AssistantContinuationSpec): Int {
        val root = context.continuationRoot(spec.jobId)
        VN97AssistantContinuationBindingStore(root).require(spec.binding)
        val continuity = AtomicCompositeContinuityStore(root).loadOrNull()
            ?: error("assistant continuation VN97CNT1 is missing")
        restoreAssistantContinuation(spec.binding, continuity)

        return scheduleJob(
            jobId = spec.jobId,
            minimumLatencyMillis = spec.minimumLatencyMillis,
            extras = runtimeExtras(spec.runtimeConfig).apply {
                putString(CONTINUATION_MODE_KEY, CONTINUATION_MODE_ASSISTANT)
                putString(ASSISTANT_PRINCIPAL_KEY, spec.binding.principal)
                putString(ASSISTANT_PLAN_ID_KEY, spec.binding.planId)
                putString(ASSISTANT_MODEL_ID_KEY, spec.binding.modelIdHex)
            },
        )
    }

    fun cancel(jobId: Int) = scheduler.cancel(jobId)

    fun hasPendingJob(jobId: Int): Boolean {
        require(jobId > 0) { "jobId must be positive" }
        return scheduler.getPendingJob(jobId) != null
    }

    fun cancelAssistantAndDelete(jobId: Int) {
        require(jobId > 0) { "jobId must be positive" }
        scheduler.cancel(jobId)
        val root = context.continuationRoot(jobId)
        AtomicCompositeContinuityStore(root).delete()
        VN97AssistantContinuationBindingStore(root).delete()
    }

    private fun scheduleJob(
        jobId: Int,
        minimumLatencyMillis: Long,
        extras: PersistableBundle,
    ): Int {
        val component = ComponentName(context, VN97ContinuationJobService::class.java)
        val job = JobInfo.Builder(jobId, component)
            .setPersisted(true)
            .setMinimumLatency(minimumLatencyMillis)
            .setBackoffCriteria(
                CONTINUATION_BACKOFF_MILLIS,
                JobInfo.BACKOFF_POLICY_EXPONENTIAL,
            )
            .setExtras(extras)
            .build()
        return scheduler.schedule(job)
    }
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
                val budget = AndroidComputeGovernor(this).budget()
                if (budget.runnable && !cancellation.get()) {
                    reschedule = when (
                        params.extras.getString(CONTINUATION_MODE_KEY)
                            ?: CONTINUATION_MODE_RUNTIME
                    ) {
                        CONTINUATION_MODE_RUNTIME ->
                            runRuntimeContinuation(params, budget, cancellation)
                        CONTINUATION_MODE_ASSISTANT ->
                            runAssistantContinuation(params, budget, cancellation)
                        else -> error("unknown VN97 continuation mode")
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

    private fun runRuntimeContinuation(
        params: JobParameters,
        budget: ComputeBudget,
        cancellation: AtomicBoolean,
    ): Boolean {
        val provider = application as? ContinuationWorkProvider
            ?: error("Application must implement ContinuationWorkProvider")
        val config = params.extras.toRuntimeConfig()
        val root = continuationRoot(params.jobId)
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
            return outcome == ContinuationOutcome.RESCHEDULE
        }
    }

    private fun runAssistantContinuation(
        params: JobParameters,
        budget: ComputeBudget,
        cancellation: AtomicBoolean,
    ): Boolean {
        val provider = application as? VN97AssistantContinuationWorkProvider
            ?: error("Application must implement VN97AssistantContinuationWorkProvider")
        val config = params.extras.toRuntimeConfig()
        val binding = params.extras.toAssistantBinding()
        val root = continuationRoot(params.jobId)
        val bindingStore = VN97AssistantContinuationBindingStore(root)
        bindingStore.require(binding)
        val compositeStore = AtomicCompositeContinuityStore(root)
        val continuity = compositeStore.loadOrNull()
        if (continuity == null) {
            bindingStore.delete()
            return false
        }
        val restored = restoreAssistantContinuation(binding, continuity)

        val ownerCheckpoint = AtomicCheckpointStore(
            File(root, "owner-runtime"),
            fileName = "runtime.vn97run1",
        )
        NativeRuntimeOwner(ownerCheckpoint).use { owner ->
            val info = owner.restoreComposite(config, restored.continuity)
            check(info.lifecycle == RuntimeLifecycle.SUSPENDED) {
                "assistant composite runtime must restore SUSPENDED"
            }
            owner.resume()

            val outcome = provider.createVN97AssistantContinuationWork().run(
                VN97AssistantContinuationContext(
                    jobId = params.jobId,
                    runtime = owner,
                    controller = restored.controller,
                    binding = restored.binding,
                    epoch = restored.epoch,
                    budget = budget,
                    stopped = cancellation,
                )
            )
            if (cancellation.get()) return true

            val plan = restored.controller.plan
            if (plan.isTerminal()) {
                owner.suspendAndSnapshot()
                compositeStore.delete()
                bindingStore.delete()
                return false
            }

            val snapshot = owner.suspendAndSnapshot()
            check(snapshot.modelBinding.bound) {
                "assistant continuation lost runtime model binding"
            }
            binding.requireModelId(snapshot.modelBinding.modelId)
            val manifest = compositeStore.save(snapshot, plan)
            check(manifest.planId == binding.planId) {
                "assistant continuation plan identity changed across persisted epoch"
            }
            binding.requireModelId(manifest.modelId)

            if (
                plan.status == NativePlanStatus.WAITING_EXTERNAL ||
                plan.status == NativePlanStatus.PAUSED
            ) {
                return false
            }
            return outcome == ContinuationOutcome.RESCHEDULE
        }
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

private fun Context.continuationRoot(jobId: Int): File =
    File(noBackupFilesDir, "vn97-continuity/$jobId")

private fun ByteArray.lowerHex(): String =
    joinToString("") { "%02x".format(it.toInt() and 0xff) }

private fun runtimeExtras(config: NativeRuntimeConfig): PersistableBundle =
    PersistableBundle().apply {
        putInt("layers", config.layers)
        putInt("batch", config.batch)
        putInt("d_model", config.dModel)
        putInt("d_state", config.dState)
        putInt("recurrent_backend", config.recurrentBackend.code)
        putInt("packed_backend", config.packedBackend.code)
    }

private fun PersistableBundle.toRuntimeConfig(): NativeRuntimeConfig =
    NativeRuntimeConfig(
        layers = getInt("layers"),
        batch = getInt("batch"),
        dModel = getInt("d_model"),
        dState = getInt("d_state"),
        recurrentBackend = backendFromCode(getInt("recurrent_backend")),
        packedBackend = backendFromCode(getInt("packed_backend")),
    )

private fun PersistableBundle.toAssistantBinding(): VN97AssistantContinuationBinding =
    VN97AssistantContinuationBinding(
        principal = requirePersistedString(ASSISTANT_PRINCIPAL_KEY),
        planId = requirePersistedString(ASSISTANT_PLAN_ID_KEY),
        modelIdHex = requirePersistedString(ASSISTANT_MODEL_ID_KEY),
    )

private fun PersistableBundle.requirePersistedString(key: String): String =
    getString(key)?.takeIf { it.isNotEmpty() }
        ?: error("missing persisted VN97 continuation field: $key")

private fun backendFromCode(code: Int): NativeBackend =
    NativeBackend.entries.firstOrNull { it.code == code }
        ?: error("invalid native backend code: $code")
