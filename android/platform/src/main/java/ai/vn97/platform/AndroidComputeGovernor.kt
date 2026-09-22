package ai.vn97.platform

import android.content.Context
import android.os.BatteryManager
import android.os.Build
import android.os.PowerManager

class AndroidComputeGovernor(context: Context) {
    private val battery = context.getSystemService(BatteryManager::class.java)
    private val power = context.getSystemService(PowerManager::class.java)

    fun signals(): ComputeSignals {
        val capacity = battery?.getIntProperty(BatteryManager.BATTERY_PROPERTY_CAPACITY) ?: -1
        val boundedCapacity = if (capacity in 0..100) capacity else 100
        val charging = battery?.isCharging ?: false
        val thermal = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            power?.currentThermalStatus ?: PowerManager.THERMAL_STATUS_NONE
        } else {
            VN97ComputePolicy.THERMAL_LIGHT
        }
        return ComputeSignals(
            batteryPercent = boundedCapacity,
            charging = charging,
            thermalStatus = thermal,
        )
    }

    fun budget(): ComputeBudget = VN97ComputePolicy.evaluate(signals())
}
