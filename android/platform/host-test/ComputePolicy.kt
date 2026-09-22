import ai.vn97.platform.*

fun main() {
    check(
        VN97ComputePolicy.evaluate(ComputeSignals(80, true, 0)).mode ==
            ComputeMode.PERFORMANCE
    )
    check(
        VN97ComputePolicy.evaluate(ComputeSignals(70, false, 0)).mode ==
            ComputeMode.BALANCED
    )
    val low = VN97ComputePolicy.evaluate(ComputeSignals(20, false, 0))
    check(low.mode == ComputeMode.LOW_POWER)
    check(low.maxTokensPerWake == 64L)
    val thermal = VN97ComputePolicy.evaluate(ComputeSignals(90, true, 2))
    check(thermal.mode == ComputeMode.LOW_POWER)
    val severe = VN97ComputePolicy.evaluate(ComputeSignals(100, true, 3))
    check(severe.mode == ComputeMode.BLOCKED)
    check(!severe.runnable)
    val criticalBattery = VN97ComputePolicy.evaluate(ComputeSignals(10, false, 0))
    check(criticalBattery.mode == ComputeMode.BLOCKED)

    try {
        ComputeSignals(101, false, 0)
        error("invalid battery must fail")
    } catch (_: IllegalArgumentException) {
    }

    println("M7C_COMPUTE_POLICY_PASS")
}
