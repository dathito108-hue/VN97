package ai.vn97.app

import ai.vn97.platform.AndroidPlatformRuntime
import android.app.Application

class VN97Application : Application() {
    val platformRuntime: AndroidPlatformRuntime by lazy(LazyThreadSafetyMode.SYNCHRONIZED) {
        AndroidPlatformRuntime(applicationContext)
    }
}
