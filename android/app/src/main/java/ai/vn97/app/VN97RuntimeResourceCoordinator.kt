package ai.vn97.app

import ai.vn97.platform.AndroidComputeGovernor
import android.app.Activity
import android.app.Application
import android.os.Bundle
import java.util.concurrent.atomic.AtomicInteger

class VN97RuntimeResourceCoordinator(
    private val application: VN97Application,
) : Application.ActivityLifecycleCallbacks {
    private val governor =
        AndroidComputeGovernor(application)
    private val visibleActivities =
        AtomicInteger(0)

    init {
        application
            .registerActivityLifecycleCallbacks(
                this
            )
    }

    fun decision(
        executionClass:
            VN97RuntimeExecutionClass,
    ): VN97RuntimeResourceDecision =
        VN97RuntimeResourcePolicy.evaluate(
            signals = governor.signals(),
            executionClass = executionClass,
        )

    fun requireRunnable(
        executionClass:
            VN97RuntimeExecutionClass,
    ): VN97RuntimeResourceDecision {
        val current = decision(executionClass)
        check(current.runnable) {
            current.reason
        }
        return current
    }

    fun onTrimMemory(
        level: Int,
    ): Boolean {
        if (
            visibleActivities.get() > 0 ||
            !VN97RuntimeResourcePolicy
                .shouldReleaseIdleAssistant(
                    level
                )
        ) {
            return false
        }
        return application.assistant
            .releaseForResourcePressure()
    }

    fun onLowMemory(): Boolean {
        if (visibleActivities.get() > 0) {
            return false
        }
        return application.assistant
            .releaseForResourcePressure()
    }

    override fun onActivityStarted(
        activity: Activity,
    ) {
        visibleActivities.incrementAndGet()
    }

    override fun onActivityStopped(
        activity: Activity,
    ) {
        visibleActivities
            .updateAndGet {
                current ->
                (current - 1)
                    .coerceAtLeast(0)
            }
    }

    override fun onActivityCreated(
        activity: Activity,
        savedInstanceState: Bundle?,
    ) = Unit

    override fun onActivityResumed(
        activity: Activity,
    ) = Unit

    override fun onActivityPaused(
        activity: Activity,
    ) = Unit

    override fun onActivitySaveInstanceState(
        activity: Activity,
        outState: Bundle,
    ) = Unit

    override fun onActivityDestroyed(
        activity: Activity,
    ) = Unit
}
