package ai.vn97.platform

import android.app.ActivityManager
import android.content.Context
import android.os.BatteryManager
import android.os.Build
import android.os.PowerManager

class AndroidComputeGovernor(context: Context) {
    private val battery =
        context.getSystemService(
            BatteryManager::class.java
        )
    private val power =
        context.getSystemService(
            PowerManager::class.java
        )
    private val activity =
        context.getSystemService(
            ActivityManager::class.java
        )

    fun signals(): ComputeSignals {
        val capacity = battery?.getIntProperty(BatteryManager.BATTERY_PROPERTY_CAPACITY) ?: -1
        val boundedCapacity = if (capacity in 0..100) capacity else 100
        val charging = battery?.isCharging ?: false
        val thermal = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            power?.currentThermalStatus ?: PowerManager.THERMAL_STATUS_NONE
        } else {
            VN97ComputePolicy.THERMAL_LIGHT
        }
        val memoryPressure =
            ActivityManager.MemoryInfo()
                .also { info ->
                    activity?.getMemoryInfo(info)
                }
                .let { info ->
                    when {
                        activity == null ->
                            VN97ComputePolicy
                                .MEMORY_MODERATE

                        info.lowMemory ->
                            VN97ComputePolicy
                                .MEMORY_CRITICAL

                        info.threshold > 0L &&
                            info.availMem.toDouble() /
                                info.threshold.toDouble() <=
                                MODERATE_MEMORY_RATIO ->
                            VN97ComputePolicy
                                .MEMORY_MODERATE

                        else ->
                            VN97ComputePolicy
                                .MEMORY_NORMAL
                    }
                }
        return ComputeSignals(
            batteryPercent = boundedCapacity,
            charging = charging,
            thermalStatus = thermal,
            memoryPressure = memoryPressure,
        )
    }

    fun budget(): ComputeBudget =
        VN97ComputePolicy.evaluate(signals())

    companion object {
        private const val MODERATE_MEMORY_RATIO =
            1.75
    }
}
