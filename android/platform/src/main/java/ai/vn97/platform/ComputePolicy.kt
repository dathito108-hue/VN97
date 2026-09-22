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
) {
    init {
        require(batteryPercent in 0..100) { "batteryPercent must be in 0..100" }
        require(thermalStatus >= 0) { "thermalStatus must be non-negative" }
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

    fun evaluate(signals: ComputeSignals): ComputeBudget {
        if (
            signals.thermalStatus >= THERMAL_SEVERE ||
            (!signals.charging && signals.batteryPercent <= 10)
        ) {
            return ComputeBudget(ComputeMode.BLOCKED, 0, 0, 30 * 60 * 1000L)
        }
        if (
            signals.thermalStatus >= THERMAL_MODERATE ||
            (!signals.charging && signals.batteryPercent <= 20)
        ) {
            return ComputeBudget(ComputeMode.LOW_POWER, 64, 15_000, 10 * 60 * 1000L)
        }
        if (signals.charging && signals.thermalStatus <= THERMAL_LIGHT) {
            return ComputeBudget(ComputeMode.PERFORMANCE, 512, 60_000, 5 * 60 * 1000L)
        }
        return ComputeBudget(ComputeMode.BALANCED, 256, 30_000, 10 * 60 * 1000L)
    }
}
