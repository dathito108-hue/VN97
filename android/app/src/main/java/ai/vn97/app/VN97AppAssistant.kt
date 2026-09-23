package ai.vn97.app

import ai.vn97.platform.M6PolicyGrant
import ai.vn97.platform.VN97AssistantTurnState
import ai.vn97.platform.VN97AssistantTurnUpdate
import ai.vn97.platform.VN97ProductionAssistantResources
import ai.vn97.runtime.NativeActivatedInventoryModelLoader
import ai.vn97.runtime.NativeActivatedModel
import android.os.SystemClock
import java.io.File

class VN97AppAssistant(
    private val application: VN97Application,
) : AutoCloseable {
    private val lock = Any()
    private var model: NativeActivatedModel? = null
    private var resources: VN97ProductionAssistantResources? = null

    fun openIfActivated(): Boolean = synchronized(lock) {
        if (model != null && resources != null) return true

        val opened = NativeActivatedInventoryModelLoader.openOrNull(
            File(application.noBackupFilesDir, "vn97-capabilities")
        ) ?: return false
        val assistant = try {
            application.platformRuntime.createProductionMemoryBackedAssistant(
                model = opened,
                grants = emptyList<M6PolicyGrant>(),
            )
        } catch (exc: Throwable) {
            opened.close()
            throw exc
        }
        model = opened
        resources = assistant
        true
    }

    fun runTurn(
        userMessage: String,
        principal: String = "runtime.user",
        maxAdvances: Int = 8,
    ): VN97AssistantTurnUpdate = synchronized(lock) {
        require(userMessage.isNotBlank()) { "userMessage must not be blank" }
        require(maxAdvances > 0) { "maxAdvances must be positive" }
        val session = checkNotNull(resources) {
            "trusted VN97 model is not active"
        }.session

        var update = session.startTurn(
            userMessage = userMessage,
            principal = principal,
            nowNs = SystemClock.elapsedRealtimeNanos(),
        )
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

    override fun close() = synchronized(lock) {
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
}
