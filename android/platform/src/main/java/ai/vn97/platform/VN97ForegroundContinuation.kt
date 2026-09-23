package ai.vn97.platform

import ai.vn97.runtime.AtomicCheckpointStore
import ai.vn97.runtime.AtomicCompositeContinuityStore
import ai.vn97.runtime.NativeActivatedModel
import ai.vn97.runtime.NativePlan
import ai.vn97.runtime.NativeRuntimeConfig
import ai.vn97.runtime.NativeRuntimeOwner
import ai.vn97.runtime.RuntimeLifecycle
import android.content.Context
import java.io.File

class VN97ForegroundAssistantContinuationSession internal constructor(
    val jobId: Int,
    val binding: VN97AssistantContinuationBinding,
    private val owner: NativeRuntimeOwner,
    private val resources: VN97ProductionAssistantResources,
    private val compositeStore: AtomicCompositeContinuityStore,
    private val bindingStore: VN97AssistantContinuationBindingStore,
    private var update: VN97AssistantTurnUpdate,
) : AutoCloseable {
    private var closed = false
    private var committed = false

    val currentUpdate: VN97AssistantTurnUpdate
        @Synchronized get() = update

    val plan: NativePlan
        @Synchronized get() = update.turn.controller.plan

    @Synchronized
    fun resolveApproval(
        approved: Boolean,
        nowNs: Long,
    ): VN97AssistantTurnUpdate {
        check(!closed) {
            "foreground continuation session is closed"
        }
        check(!committed) {
            "foreground continuation session is already committed"
        }
        require(nowNs >= 0L) {
            "nowNs must be non-negative"
        }
        val approval = checkNotNull(update.approval) {
            "foreground continuation is not waiting for approval"
        }
        check(update.state == VN97AssistantTurnState.APPROVAL_REQUIRED) {
            "foreground continuation state is not APPROVAL_REQUIRED"
        }
        update = resources.session.resolveApproval(
            turn = update.turn,
            approval = approval,
            approved = approved,
            nowNs = nowNs,
        )
        return update
    }

    @Synchronized
    fun commit(): VN97AssistantTurnUpdate {
        check(!closed) {
            "foreground continuation session is closed"
        }
        check(!committed) {
            "foreground continuation session is already committed"
        }
        val controller = update.turn.controller
        val plan = controller.plan
        if (plan.isTerminal()) {
            owner.suspendAndSnapshot()
            compositeStore.delete()
            bindingStore.delete()
        } else {
            val snapshot = owner.suspendAndSnapshot()
            check(snapshot.modelBinding.bound) {
                "foreground continuation lost runtime model binding"
            }
            binding.requireModelId(snapshot.modelBinding.modelId)
            val manifest = compositeStore.save(snapshot, plan)
            check(manifest.planId == binding.planId) {
                "foreground continuation plan identity changed"
            }
            binding.requireModelId(manifest.modelId)
        }
        committed = true
        closeLocked()
        return update
    }

    @Synchronized
    override fun close() {
        if (!closed) {
            closeLocked()
        }
    }

    private fun closeLocked() {
        var failure: Throwable? = null
        try {
            resources.close()
        } catch (exc: Throwable) {
            failure = exc
        }
        try {
            owner.close()
        } catch (exc: Throwable) {
            val first = failure
            if (first == null) failure = exc
            else first.addSuppressed(exc)
        }
        closed = true
        failure?.let { throw it }
    }
}

fun openVN97ForegroundAssistantContinuation(
    context: Context,
    platformRuntime: AndroidPlatformRuntime,
    jobId: Int,
    model: NativeActivatedModel,
    grants: List<M6PolicyGrant>,
    runtimeConfig: NativeRuntimeConfig,
    cognitionRuntimeConfig: ai.vn97.runtime.NativeCognitionRuntimeConfig =
        ai.vn97.runtime.NativeCognitionRuntimeConfig(),
    cognitionLimits: ai.vn97.runtime.NativeCognitionLimits =
        ai.vn97.runtime.NativeCognitionLimits(),
    sessionLimits: VN97AssistantSessionLimits =
        VN97AssistantSessionLimits(),
    auditFileName: String = "m6-actions.jsonl",
    nowNs: Long,
): VN97ForegroundAssistantContinuationSession {
    require(jobId > 0) {
        "jobId must be positive"
    }
    require(nowNs >= 0L) {
        "nowNs must be non-negative"
    }
    val appContext = context.applicationContext
    val root = appContext.continuationRoot(jobId)
    val bindingStore = VN97AssistantContinuationBindingStore(root)
    val binding = bindingStore.loadOrNull()
        ?: error("assistant continuation binding is missing")
    binding.requireModelId(model.info.modelId)

    val compositeStore = AtomicCompositeContinuityStore(root)
    val continuity = compositeStore.loadOrNull()
        ?: error("assistant continuation VN97CNT1 is missing")
    val restored = restoreAssistantContinuation(
        expected = binding,
        continuity = continuity,
    )

    val ownerCheckpoint = AtomicCheckpointStore(
        File(root, "owner-runtime"),
        fileName = "runtime.vn97run1",
    )
    val owner = NativeRuntimeOwner(ownerCheckpoint)
    try {
        val info = owner.restoreComposite(
            config = runtimeConfig,
            continuity = restored.continuity,
        )
        check(info.lifecycle == RuntimeLifecycle.SUSPENDED) {
            "foreground continuation must restore SUSPENDED"
        }
        owner.resume()

        val resources =
            platformRuntime.createProductionMemoryBackedAssistant(
                model = model,
                grants = grants,
                cognitionRuntimeConfig = cognitionRuntimeConfig,
                cognitionLimits = cognitionLimits,
                sessionLimits = sessionLimits,
                auditFileName = auditFileName,
            )
        try {
            val update = resources.session.resumeRestoredTurn(
                controller = restored.controller,
                principal = restored.binding.principal,
                nowNs = nowNs,
            )
            return VN97ForegroundAssistantContinuationSession(
                jobId = jobId,
                binding = binding,
                owner = owner,
                resources = resources,
                compositeStore = compositeStore,
                bindingStore = bindingStore,
                update = update,
            )
        } catch (exc: Throwable) {
            resources.close()
            throw exc
        }
    } catch (exc: Throwable) {
        owner.close()
        throw exc
    }
}
