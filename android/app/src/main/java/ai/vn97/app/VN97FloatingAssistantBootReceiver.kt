package ai.vn97.app

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent

class VN97FloatingAssistantBootReceiver : BroadcastReceiver() {
    override fun onReceive(context: Context, intent: Intent) {
        when (intent.action) {
            Intent.ACTION_BOOT_COMPLETED,
            Intent.ACTION_MY_PACKAGE_REPLACED,
            -> VN97FloatingAssistantService.startIfEnabled(context)
        }
    }
}
