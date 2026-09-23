package ai.vn97.app

import ai.vn97.platform.M6AndroidProductionCapabilities
import ai.vn97.platform.VN97AssistantTurnState
import ai.vn97.platform.VN97AssistantTurnUpdate
import ai.vn97.platform.VN97ProductionAssistantResources
import ai.vn97.runtime.NativeActivatedInventoryModelLoader
import ai.vn97.runtime.NativeActivatedModel
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
                grants = listOf(
                    M6AndroidProductionCapabilities.userApprovedClipboardGrant(
                        APP_PRINCIPAL
                    )
                ),
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
    }
}
