package ai.vn97.platform

enum class ComputeMode {
    BLOCKED,
    LOW_POWER,
    BALANCED,
    PERFORMANCE,
}

data class ComputeSignals(
    val batteryPercent: Int,
    val charging: Boolean,
    val thermalStatus: Int,
    val memoryPressure: Int =
        VN97ComputePolicy.MEMORY_NORMAL,
) {
    init {
        require(batteryPercent in 0..100) {
            "batteryPercent must be in 0..100"
        }
        require(thermalStatus >= 0) {
            "thermalStatus must be non-negative"
        }
        require(
            memoryPressure in
                VN97ComputePolicy.MEMORY_NORMAL..
                    VN97ComputePolicy.MEMORY_CRITICAL
        ) {
            "memoryPressure is invalid"
        }
    }
}

data class ComputeBudget(
    val mode: ComputeMode,
    val maxTokensPerWake: Long,
    val maxRunMillis: Long,
    val retryDelayMillis: Long,
) {
    init {
        require(maxTokensPerWake >= 0)
        require(maxRunMillis >= 0)
        require(retryDelayMillis > 0)
    }

    val runnable: Boolean get() = mode != ComputeMode.BLOCKED
}

object VN97ComputePolicy {
    const val THERMAL_LIGHT = 1
    const val THERMAL_MODERATE = 2
    const val THERMAL_SEVERE = 3

    const val MEMORY_NORMAL = 0
    const val MEMORY_MODERATE = 1
    const val MEMORY_CRITICAL = 2

    fun evaluate(signals: ComputeSignals): ComputeBudget {
        if (
            signals.memoryPressure >=
                MEMORY_CRITICAL ||
            signals.thermalStatus >= THERMAL_SEVERE ||
            (!signals.charging &&
                signals.batteryPercent <= 10)
        ) {
            return ComputeBudget(
                ComputeMode.BLOCKED,
                0,
                0,
                30 * 60 * 1000L,
            )
        }
        if (
            signals.memoryPressure >=
                MEMORY_MODERATE ||
            signals.thermalStatus >= THERMAL_MODERATE ||
            (!signals.charging &&
                signals.batteryPercent <= 20)
        ) {
            return ComputeBudget(
                ComputeMode.LOW_POWER,
                64,
                15_000,
                10 * 60 * 1000L,
            )
        }
        if (signals.charging && signals.thermalStatus <= THERMAL_LIGHT) {
            return ComputeBudget(ComputeMode.PERFORMANCE, 512, 60_000, 5 * 60 * 1000L)
        }
        return ComputeBudget(ComputeMode.BALANCED, 256, 30_000, 10 * 60 * 1000L)
    }
}
