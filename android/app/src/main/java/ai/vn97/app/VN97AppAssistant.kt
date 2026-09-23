package ai.vn97.app

import ai.vn97.platform.M6AndroidProductionCapabilities
import ai.vn97.platform.VN97AssistantTurnState
import ai.vn97.platform.VN97AssistantTurnUpdate
import ai.vn97.platform.VN97MobileEvidenceConfig
import ai.vn97.platform.VN97MobileEvidenceRecord
import ai.vn97.platform.VN97ProductionAssistantResources
import ai.vn97.runtime.NativeActivatedInventoryModelLoader
import ai.vn97.runtime.NativeActivatedModel
import ai.vn97.runtime.NativeCognitionInferenceEngine
import ai.vn97.runtime.NativePreparedAudio
import ai.vn97.runtime.NativePreparedVision
import android.content.Intent
import android.os.SystemClock
import java.io.File

data class VN97AppApprovalRequest(
    val capabilityId: String,
    val requestDigest: String,
    val scopeDigest: String,
    val presentationJson: String,
    val expiresNs: Long,
)

data class VN97AppTurnResult(
    val userMessage: String,
    val update: VN97AssistantTurnUpdate,
)

class VN97AppAssistant(
    private val application: VN97Application,
) : AutoCloseable {
    private val lock = Any()
    private var model: NativeActivatedModel? = null
    private var resources: VN97ProductionAssistantResources? = null
    private var pendingResult: VN97AppTurnResult? = null

    fun openIfActivated(): Boolean = synchronized(lock) {
        if (model != null && resources != null) return true

        val opened = NativeActivatedInventoryModelLoader.openOrNull(
            File(application.noBackupFilesDir, "vn97-capabilities")
        ) ?: return false
        val assistant = try {
            application.platformRuntime.createProductionMemoryBackedAssistant(
                model = opened,
                grants = productionGrants(),
            )
        } catch (exc: Throwable) {
            opened.close()
            throw exc
        }
        model = opened
        resources = assistant
        pendingResult = null
        true
    }

    fun runTurn(
        userMessage: String,
        maxAdvances: Int = 8,
    ): VN97AppTurnResult = synchronized(lock) {
        require(userMessage.isNotBlank()) { "userMessage must not be blank" }
        require(maxAdvances > 0) { "maxAdvances must be positive" }
        check(pendingResult == null) {
            "an assistant turn is already waiting for approval"
        }
        val session = checkNotNull(resources) {
            "trusted VN97 model is not active"
        }.session

        val first = session.startTurn(
            userMessage = userMessage,
            principal = APP_PRINCIPAL,
            nowNs = SystemClock.elapsedRealtimeNanos(),
        )
        rememberPending(
            VN97AppTurnResult(
                userMessage = userMessage,
                update = advanceYielded(session, first, maxAdvances),
            )
        )
    }

    fun hasProductionVoice(): Boolean = synchronized(lock) {
        model?.info?.hasAudioProjection == true
    }

    fun hasProductionVision(): Boolean = synchronized(lock) {
        model?.info?.hasVisionProjection == true
    }

    fun perceiveVision(
        preparedVision: NativePreparedVision,
    ): String = synchronized(lock) {
        check(pendingResult == null) {
            "cannot run perception while approval is pending"
        }
        val activeResources = checkNotNull(resources) {
            "trusted VN97 model is not active"
        }
        check(!activeResources.session.hasActiveTurn) {
            "cannot run perception while an assistant turn is active"
        }
        val activeModel = checkNotNull(model) {
            "trusted VN97 model is not active"
        }
        check(activeModel.info.hasVisionProjection) {
            "activated VN97 model has no production vision weights"
        }
        NativeCognitionInferenceEngine(activeModel)
            .perceiveVision(preparedVision)
    }

    fun verifyVisualOutcome(
        goal: String,
        beforeObservation: String,
        afterObservation: String,
        maxNewTokens: Int = 384,
    ): String = synchronized(lock) {
        require(goal.isNotBlank()) { "visual verification goal must not be blank" }
        require(beforeObservation.isNotBlank()) {
            "beforeObservation must not be blank"
        }
        require(afterObservation.isNotBlank()) {
            "afterObservation must not be blank"
        }
        require(maxNewTokens in 1..1024) {
            "visual verification token budget is invalid"
        }
        check(pendingResult == null) {
            "cannot verify visual outcome while approval is pending"
        }
        val activeResources = checkNotNull(resources) {
            "trusted VN97 model is not active"
        }
        check(!activeResources.session.hasActiveTurn) {
            "cannot verify visual outcome while a turn is active"
        }
        val activeModel = checkNotNull(model) {
            "trusted VN97 model is not active"
        }
        check(activeModel.info.hasVisionProjection) {
            "activated VN97 model has no production vision weights"
        }
        val prompt = buildString {
            append("VN97VISVERIFY1\n")
            append("Use only the supplied before/after visual observations. ")
            append("State whether the user goal is visibly satisfied and what ")
            append("remains uncertain. Do not request or execute tools.\n")
            append("goal=")
            append(goal.take(MAX_VISUAL_VERIFY_FIELD_CHARS))
            append("\nbefore=")
            append(beforeObservation.take(MAX_VISUAL_VERIFY_FIELD_CHARS))
            append("\nafter=")
            append(afterObservation.take(MAX_VISUAL_VERIFY_FIELD_CHARS))
            append("\nverification=")
        }
        NativeCognitionInferenceEngine(activeModel)
            .generateText(prompt, maxNewTokens)
            .trim()
            .ifEmpty {
                throw IllegalStateException(
                    "VN97 visual verification returned empty output"
                )
            }
    }

    fun hasActiveTurn(): Boolean = synchronized(lock) {
        resources?.session?.hasActiveTurn == true
    }

    fun runVoiceTurn(
        preparedAudio: NativePreparedAudio,
        maxAdvances: Int = 8,
    ): VN97AppTurnResult = synchronized(lock) {
        require(maxAdvances > 0) {
            "maxAdvances must be positive"
        }
        check(pendingResult == null) {
            "an assistant turn is already waiting for approval"
        }
        check(resources != null) {
            "trusted VN97 model is not active"
        }
        val activeModel = checkNotNull(model) {
            "trusted VN97 model is not active"
        }
        check(activeModel.info.hasAudioProjection) {
            "activated VN97 model has no production speech weights"
        }

        val transcript =
            NativeCognitionInferenceEngine(activeModel)
                .transcribeAudio(preparedAudio)
        runTurn(
            userMessage = transcript,
            maxAdvances = maxAdvances,
        )
    }

    fun collectMobileEvidence(
        config: VN97MobileEvidenceConfig = VN97MobileEvidenceConfig(),
    ): VN97MobileEvidenceRecord = synchronized(lock) {
        check(pendingResult == null) {
            "cannot benchmark while approval is pending"
        }
        val activeResources = checkNotNull(resources) {
            "trusted VN97 model is not active"
        }
        check(!activeResources.session.hasActiveTurn) {
            "cannot benchmark while an assistant turn is active"
        }
        val activeModel = checkNotNull(model) {
            "trusted VN97 model is not active"
        }
        application.platformRuntime.collectProductionMobileEvidence(
            model = activeModel,
            config = config,
        )
    }

    fun pendingApproval(): VN97AppApprovalRequest? = synchronized(lock) {
        val result = pendingResult ?: return null
        val approval = checkNotNull(result.update.approval) {
            "pending app result lacks canonical M6 approval"
        }
        VN97AppApprovalRequest(
            capabilityId = approval.capabilityId,
            requestDigest = approval.requestDigest,
            scopeDigest = approval.scopeDigest,
            presentationJson = approval.presentationJson,
            expiresNs = approval.expiresNs,
        )
    }

    fun resolvePendingApproval(
        approved: Boolean,
        maxAdvances: Int = 8,
    ): VN97AppTurnResult = synchronized(lock) {
        require(maxAdvances > 0) { "maxAdvances must be positive" }
        val current = checkNotNull(pendingResult) {
            "no assistant approval is pending"
        }
        val approval = checkNotNull(current.update.approval) {
            "pending app result lacks canonical M6 approval"
        }
        val session = checkNotNull(resources) {
            "trusted VN97 model is not active"
        }.session
        pendingResult = null

        val resolved = session.resolveApproval(
            turn = current.update.turn,
            approval = approval,
            approved = approved,
            nowNs = SystemClock.elapsedRealtimeNanos(),
        )
        rememberPending(
            VN97AppTurnResult(
                userMessage = current.userMessage,
                update = advanceYielded(session, resolved, maxAdvances),
            )
        )
    }

    private fun advanceYielded(
        session: ai.vn97.platform.VN97AssistantSession,
        initial: VN97AssistantTurnUpdate,
        maxAdvances: Int,
    ): VN97AssistantTurnUpdate {
        var update = initial
        var advances = 1
        while (
            update.state == VN97AssistantTurnState.YIELDED &&
            advances < maxAdvances
        ) {
            update = session.continueTurn(
                turn = update.turn,
                nowNs = SystemClock.elapsedRealtimeNanos(),
            )
            advances += 1
        }
        return update
    }

    private fun rememberPending(result: VN97AppTurnResult): VN97AppTurnResult {
        if (result.update.state == VN97AssistantTurnState.APPROVAL_REQUIRED) {
            checkNotNull(result.update.approval) {
                "APPROVAL_REQUIRED update lacks approval"
            }
            pendingResult = result
        } else {
            pendingResult = null
        }
        return result
    }

    fun reloadActivatedModel(): Boolean = synchronized(lock) {
        check(pendingResult == null) {
            "cannot replace model while approval is pending"
        }
        val activeResources = resources
        check(activeResources == null || !activeResources.session.hasActiveTurn) {
            "cannot replace model while a turn is active"
        }
        closeLocked()
        openIfActivated()
    }

    override fun close() {
        synchronized(lock) {
            pendingResult = null
            closeLocked()
        }
    }

    private fun productionGrants() =
        buildList {
            add(
                M6AndroidProductionCapabilities.userApprovedClipboardGrant(
                    APP_PRINCIPAL
                )
            )
            val launcher = Intent(Intent.ACTION_MAIN).apply {
                addCategory(Intent.CATEGORY_LAUNCHER)
            }
            application.packageManager
                .queryIntentActivities(launcher, 0)
                .asSequence()
                .mapNotNull { it.activityInfo?.packageName }
                .filter { it.isNotBlank() }
                .distinct()
                .sorted()
                .take(MAX_LAUNCHABLE_APP_GRANTS)
                .forEach { packageName ->
                    runCatching {
                        M6AndroidProductionCapabilities
                            .userApprovedAppLaunchGrant(
                                APP_PRINCIPAL,
                                packageName,
                            )
                    }.getOrNull()?.let(::add)
                }
        }

    private fun closeLocked() {
        var failure: Throwable? = null
        try {
            resources?.close()
        } catch (exc: Throwable) {
            failure = exc
        } finally {
            resources = null
            try {
                model?.close()
            } catch (exc: Throwable) {
                val firstFailure = failure
                if (firstFailure == null) {
                    failure = exc
                } else {
                    firstFailure.addSuppressed(exc)
                }
            } finally {
                model = null
            }
        }
        failure?.let { throw it }
    }

    companion object {
        const val APP_PRINCIPAL = "runtime.user"
        private const val MAX_LAUNCHABLE_APP_GRANTS = 512
        private const val MAX_VISUAL_VERIFY_FIELD_CHARS = 16 * 1024
    }
}
