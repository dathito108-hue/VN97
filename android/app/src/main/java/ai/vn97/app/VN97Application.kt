package ai.vn97.app

import ai.vn97.platform.AndroidPlatformRuntime
import android.app.Application

class VN97Application : Application() {
    val platformRuntime: AndroidPlatformRuntime by lazy(LazyThreadSafetyMode.SYNCHRONIZED) {
        AndroidPlatformRuntime(applicationContext)
    }

    val assistant: VN97AppAssistant by lazy(LazyThreadSafetyMode.SYNCHRONIZED) {
        VN97AppAssistant(this)
    }

    val provisioner: VN97AppProvisioner by lazy(LazyThreadSafetyMode.SYNCHRONIZED) {
        VN97AppProvisioner(this)
    }
}
