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

    val screenCaptureBroker: VN97ScreenCaptureBroker by lazy(
        LazyThreadSafetyMode.SYNCHRONIZED
    ) {
        VN97ScreenCaptureBroker()
    }

    val visualActions: VN97VisualActionCoordinator by lazy(
        LazyThreadSafetyMode.SYNCHRONIZED
    ) {
        VN97VisualActionCoordinator(
            assistant = assistant,
            screenCaptureBroker = screenCaptureBroker,
        )
    }

    val provisioner: VN97AppProvisioner by lazy(LazyThreadSafetyMode.SYNCHRONIZED) {
        VN97AppProvisioner(this)
    }

    val bundledBootstrap: VN97BundledBootstrap by lazy(LazyThreadSafetyMode.SYNCHRONIZED) {
        VN97BundledBootstrap(this, provisioner)
    }
}
