package ai.vn97.app

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent

class VN97FloatingAssistantBootReceiver :
    BroadcastReceiver() {
    override fun onReceive(
        context: Context,
        intent: Intent,
    ) {
        val trigger =
            when (intent.action) {
                Intent.ACTION_BOOT_COMPLETED ->
                    VN97MobileRecoveryTrigger
                        .SYSTEM_RESTART

                Intent.ACTION_MY_PACKAGE_REPLACED ->
                    VN97MobileRecoveryTrigger
                        .PACKAGE_REPLACED

                else -> return
            }

        VN97FloatingAssistantService
            .startIfEnabled(context)

        val pending = goAsync()
        val app =
            context.applicationContext as?
                VN97Application
        if (app == null) {
            pending.finish()
            return
        }

        Thread(
            {
                try {
                    app.mobileRecovery
                        .recover(trigger)
                } finally {
                    pending.finish()
                }
            },
            "vn97-continuity-reconcile",
        ).start()
    }
}
