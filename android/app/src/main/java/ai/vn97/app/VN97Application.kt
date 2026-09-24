package ai.vn97.app

import ai.vn97.platform.AndroidPlatformRuntime
import ai.vn97.platform.VN97AssistantContinuationWork
import ai.vn97.platform.VN97AssistantContinuationWorkProvider
import android.app.Application
import java.util.concurrent.locks.ReentrantLock
import kotlin.concurrent.withLock

class VN97Application :
    Application(),
    VN97AssistantContinuationWorkProvider {
    @PublishedApi
    internal val sovereignExecutionLock =
        ReentrantLock(true)

    val platformRuntime: AndroidPlatformRuntime by lazy(LazyThreadSafetyMode.SYNCHRONIZED) {
        AndroidPlatformRuntime(applicationContext)
    }

    val assistant: VN97AppAssistant by lazy(LazyThreadSafetyMode.SYNCHRONIZED) {
        VN97AppAssistant(this)
    }

    val autonomousWork: VN97AutonomousWorkManager by lazy(
        LazyThreadSafetyMode.SYNCHRONIZED
    ) {
        VN97AutonomousWorkManager(this)
    }

    val paperTrading: VN97PaperTradingSessionManager by lazy(
        LazyThreadSafetyMode.SYNCHRONIZED
    ) {
        VN97PaperTradingSessionManager(this)
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

    val knowledgeAcquisition: VN97AppKnowledgeAcquisition by lazy(
        LazyThreadSafetyMode.SYNCHRONIZED
    ) {
        VN97AppKnowledgeAcquisition(this)
    }

    val remoteCapabilityFetch: VN97AppRemoteCapabilityFetch by lazy(
        LazyThreadSafetyMode.SYNCHRONIZED
    ) {
        VN97AppRemoteCapabilityFetch(this)
    }

    val mobileRecovery: VN97MobileRecoveryCoordinator by lazy(
        LazyThreadSafetyMode.SYNCHRONIZED
    ) {
        VN97MobileRecoveryCoordinator(this)
    }

    val bundledBootstrap: VN97BundledBootstrap by lazy(LazyThreadSafetyMode.SYNCHRONIZED) {
        VN97BundledBootstrap(this, provisioner)
    }

    override fun onCreate() {
        super.onCreate()
        mobileRecovery
            .scheduleProcessStartRecovery()
    }

    override fun createVN97AssistantContinuationWork():
        VN97AssistantContinuationWork {
        mobileRecovery
            .recover(
                VN97MobileRecoveryTrigger
                    .EXECUTION_ENTRY
            )
            .requireActivationReady()
        return autonomousWork
            .createContinuationWork()
    }

    inline fun <T> withSovereignExecution(
        block: () -> T,
    ): T = sovereignExecutionLock.withLock(block)
}
