package ai.vn97.app

import ai.vn97.platform.ComputeMode
import ai.vn97.platform.ComputeSignals
import ai.vn97.platform.VN97ComputePolicy

enum class VN97RuntimeExecutionClass {
    INTERACTIVE,
    BACKGROUND,
    HEAVY,
}

data class VN97RuntimeResourceDecision(
    val runnable: Boolean,
    val maxInteractiveAdvances: Int,
    val reason: String,
) {
    init {
        require(
            maxInteractiveAdvances in 0..16
        )
        if (runnable) {
            require(
                maxInteractiveAdvances > 0
            )
            require(reason.isEmpty())
        } else {
            require(
                maxInteractiveAdvances == 0
            )
            require(reason.isNotBlank())
        }
    }
}

object VN97RuntimeResourcePolicy {
    fun evaluate(
        signals: ComputeSignals,
        executionClass:
            VN97RuntimeExecutionClass,
    ): VN97RuntimeResourceDecision {
        val budget =
            VN97ComputePolicy.evaluate(signals)

        return when (executionClass) {
            VN97RuntimeExecutionClass
                .INTERACTIVE -> {
                when {
                    signals.memoryPressure >=
                        VN97ComputePolicy
                            .MEMORY_CRITICAL ->
                        blocked(
                            "Android memory pressure is critical"
                        )

                    signals.thermalStatus >=
                        VN97ComputePolicy
                            .THERMAL_SEVERE ->
                        blocked(
                            "device thermal state is severe"
                        )

                    signals.memoryPressure >=
                        VN97ComputePolicy
                            .MEMORY_MODERATE ||
                        signals.thermalStatus >=
                            VN97ComputePolicy
                                .THERMAL_MODERATE ||
                        (
                            !signals.charging &&
                                signals
                                    .batteryPercent <=
                                20
                        ) ->
                        runnable(
                            maxAdvances = 4
                        )

                    else ->
                        runnable(
                            maxAdvances = 8
                        )
                }
            }

            VN97RuntimeExecutionClass
                .BACKGROUND -> {
                if (budget.runnable) {
                    runnable(
                        maxAdvances =
                            when (budget.mode) {
                                ComputeMode
                                    .LOW_POWER -> 2
                                ComputeMode
                                    .BALANCED -> 4
                                ComputeMode
                                    .PERFORMANCE -> 8
                                ComputeMode
                                    .BLOCKED -> 0
                            }
                    )
                } else {
                    blocked(
                        "background VN97 compute is blocked by the mobile resource governor"
                    )
                }
            }

            VN97RuntimeExecutionClass
                .HEAVY -> {
                if (
                    budget.runnable &&
                    budget.mode !=
                        ComputeMode.LOW_POWER &&
                    signals.memoryPressure ==
                        VN97ComputePolicy
                            .MEMORY_NORMAL
                ) {
                    runnable(
                        maxAdvances =
                            if (
                                budget.mode ==
                                    ComputeMode
                                        .PERFORMANCE
                            ) {
                                8
                            } else {
                                4
                            }
                    )
                } else {
                    blocked(
                        "heavy VN97 compute requires non-low-power thermal, battery, and memory conditions"
                    )
                }
            }
        }
    }

    fun shouldReleaseIdleAssistant(
        trimLevel: Int,
    ): Boolean =
        trimLevel >=
            TRIM_MEMORY_RUNNING_LOW

    private fun runnable(
        maxAdvances: Int,
    ) =
        VN97RuntimeResourceDecision(
            runnable = true,
            maxInteractiveAdvances =
                maxAdvances,
            reason = "",
        )

    private fun blocked(
        reason: String,
    ) =
        VN97RuntimeResourceDecision(
            runnable = false,
            maxInteractiveAdvances = 0,
            reason = reason,
        )

    // Android ComponentCallbacks2 value. Kept here
    // as a pure policy constant so host tests do not
    // require android.jar.
    private const val TRIM_MEMORY_RUNNING_LOW =
        10
}
