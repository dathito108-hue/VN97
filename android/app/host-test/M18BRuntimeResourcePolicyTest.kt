package ai.vn97.app

import ai.vn97.platform.ComputeMode
import ai.vn97.platform.ComputeSignals
import ai.vn97.platform.VN97ComputePolicy

private inline fun expectM18BFailure(
    label: String,
    block: () -> Unit,
) {
    check(runCatching(block).isFailure) {
        "expected M18B failure: $label"
    }
}

fun main() {
    val normal =
        ComputeSignals(
            batteryPercent = 80,
            charging = true,
            thermalStatus = 0,
            memoryPressure =
                VN97ComputePolicy
                    .MEMORY_NORMAL,
        )
    val normalBudget =
        VN97ComputePolicy.evaluate(normal)
    check(
        normalBudget.mode ==
            ComputeMode.PERFORMANCE
    )

    val moderateMemory =
        normal.copy(
            charging = false,
            memoryPressure =
                VN97ComputePolicy
                    .MEMORY_MODERATE,
        )
    val moderateBudget =
        VN97ComputePolicy.evaluate(
            moderateMemory
        )
    check(
        moderateBudget.mode ==
            ComputeMode.LOW_POWER
    )
    check(moderateBudget.runnable)

    val criticalMemory =
        normal.copy(
            memoryPressure =
                VN97ComputePolicy
                    .MEMORY_CRITICAL,
        )
    val criticalBudget =
        VN97ComputePolicy.evaluate(
            criticalMemory
        )
    check(
        criticalBudget.mode ==
            ComputeMode.BLOCKED
    )
    check(!criticalBudget.runnable)

    val interactive =
        VN97RuntimeResourcePolicy
            .evaluate(
                signals = normal,
                executionClass =
                    VN97RuntimeExecutionClass
                        .INTERACTIVE,
            )
    check(interactive.runnable)
    check(
        interactive
            .maxInteractiveAdvances == 8
    )

    val moderateInteractive =
        VN97RuntimeResourcePolicy
            .evaluate(
                signals = moderateMemory,
                executionClass =
                    VN97RuntimeExecutionClass
                        .INTERACTIVE,
            )
    check(moderateInteractive.runnable)
    check(
        moderateInteractive
            .maxInteractiveAdvances == 4
    )

    val lowBatteryInteractive =
        VN97RuntimeResourcePolicy
            .evaluate(
                signals =
                    ComputeSignals(
                        batteryPercent = 7,
                        charging = false,
                        thermalStatus = 0,
                    ),
                executionClass =
                    VN97RuntimeExecutionClass
                        .INTERACTIVE,
            )
    check(lowBatteryInteractive.runnable)
    check(
        lowBatteryInteractive
            .maxInteractiveAdvances == 4
    )

    val lowBatteryBackground =
        VN97RuntimeResourcePolicy
            .evaluate(
                signals =
                    ComputeSignals(
                        batteryPercent = 7,
                        charging = false,
                        thermalStatus = 0,
                    ),
                executionClass =
                    VN97RuntimeExecutionClass
                        .BACKGROUND,
            )
    check(!lowBatteryBackground.runnable)

    val criticalInteractive =
        VN97RuntimeResourcePolicy
            .evaluate(
                signals = criticalMemory,
                executionClass =
                    VN97RuntimeExecutionClass
                        .INTERACTIVE,
            )
    check(!criticalInteractive.runnable)
    check(
        criticalInteractive
            .maxInteractiveAdvances == 0
    )

    val thermalInteractive =
        VN97RuntimeResourcePolicy
            .evaluate(
                signals =
                    normal.copy(
                        thermalStatus =
                            VN97ComputePolicy
                                .THERMAL_SEVERE
                    ),
                executionClass =
                    VN97RuntimeExecutionClass
                        .INTERACTIVE,
            )
    check(!thermalInteractive.runnable)

    val heavyNormal =
        VN97RuntimeResourcePolicy
            .evaluate(
                signals = normal,
                executionClass =
                    VN97RuntimeExecutionClass
                        .HEAVY,
            )
    check(heavyNormal.runnable)
    check(
        heavyNormal
            .maxInteractiveAdvances == 8
    )

    val heavyModerateMemory =
        VN97RuntimeResourcePolicy
            .evaluate(
                signals = moderateMemory,
                executionClass =
                    VN97RuntimeExecutionClass
                        .HEAVY,
            )
    check(!heavyModerateMemory.runnable)

    val backgroundModerate =
        VN97RuntimeResourcePolicy
            .evaluate(
                signals = moderateMemory,
                executionClass =
                    VN97RuntimeExecutionClass
                        .BACKGROUND,
            )
    check(backgroundModerate.runnable)
    check(
        backgroundModerate
            .maxInteractiveAdvances == 2
    )

    check(
        !VN97RuntimeResourcePolicy
            .shouldReleaseIdleAssistant(5)
    )
    check(
        VN97RuntimeResourcePolicy
            .shouldReleaseIdleAssistant(10)
    )
    check(
        VN97RuntimeResourcePolicy
            .shouldReleaseIdleAssistant(20)
    )

    expectM18BFailure(
        "invalid memory pressure"
    ) {
        ComputeSignals(
            batteryPercent = 80,
            charging = false,
            thermalStatus = 0,
            memoryPressure = 3,
        )
    }

    println(
        "M18B runtime resource policy contracts: PASS"
    )
}
