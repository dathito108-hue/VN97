package ai.vn97.app

import ai.vn97.platform.AndroidComputeGovernor

class VN97RuntimeResourceCoordinator(
    private val application: VN97Application,
) {
    private val governor =
        AndroidComputeGovernor(application)

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

    fun onLowMemory(): Boolean =
        application.assistant
            .releaseForResourcePressure()
}
