package ai.vn97.app

import ai.vn97.platform.VN97RemoteCapabilityFetchApproval
import ai.vn97.platform.VN97RemoteCapabilityFetchCoordinator
import ai.vn97.platform.VN97RemoteCapabilityFetchResult
import android.os.SystemClock

class VN97AppRemoteCapabilityFetch(
    private val application: VN97Application,
) {
    private val coordinator:
        VN97RemoteCapabilityFetchCoordinator by lazy(
            LazyThreadSafetyMode.SYNCHRONIZED
        ) {
            application.platformRuntime
                .createProductionRemoteCapabilityFetchCoordinator(
                    principal = PRINCIPAL
                )
        }

    fun pending():
        VN97RemoteCapabilityFetchApproval? =
        coordinator.pendingApproval()

    fun prepare(
        url: String,
    ): VN97RemoteCapabilityFetchApproval =
        coordinator.prepare(
            url = url,
            nowNs =
                SystemClock.elapsedRealtimeNanos(),
        )

    fun resolve(
        approved: Boolean,
    ): VN97RemoteCapabilityFetchResult {
        val result = coordinator.resolve(
            approved = approved,
            nowNs =
                SystemClock.elapsedRealtimeNanos(),
        )
        if (result.approved) {
            application.knowledgeAcquisition
                .noteRemoteFetch(result)
        }
        return result
    }

    fun clearPending() {
        coordinator.clearPending()
    }

    companion object {
        private const val PRINCIPAL =
            "vn97.user.capability-acquisition"
    }
}
