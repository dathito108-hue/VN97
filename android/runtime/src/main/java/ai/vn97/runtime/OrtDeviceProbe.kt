package ai.vn97.runtime

import android.app.ActivityManager
import android.content.Context
import android.os.Build
import android.os.PowerManager

object OrtDeviceProbe {
    fun inspect(
        context: Context,
        qnnBackendPath: String? = null,
    ): OrtDeviceCapabilities {
        val activity = context.getSystemService(ActivityManager::class.java)
            ?: error("ActivityManager unavailable")
        val memory = ActivityManager.MemoryInfo()
        activity.getMemoryInfo(memory)

        val power = context.getSystemService(PowerManager::class.java)
        val thermal = if (Build.VERSION.SDK_INT >= 29 && power != null) {
            power.currentThermalStatus.coerceIn(0, 6)
        } else {
            0
        }

        val socManufacturer = if (Build.VERSION.SDK_INT >= 31) {
            Build.SOC_MANUFACTURER.orEmpty()
        } else {
            ""
        }
        val socModel = if (Build.VERSION.SDK_INT >= 31) {
            Build.SOC_MODEL.orEmpty()
        } else {
            ""
        }

        val hardware = Build.HARDWARE.orEmpty()
        val qualcommFingerprint = (
            "$hardware $socManufacturer $socModel"
        ).lowercase()
        val qualcomm = listOf(
            "qualcomm",
            "snapdragon",
            "qcom",
            "sm8350",
        ).any { qualcommFingerprint.contains(it) }

        return OrtDeviceCapabilities(
            sdkInt = Build.VERSION.SDK_INT,
            logicalCores = Runtime.getRuntime().availableProcessors()
                .coerceAtLeast(1),
            availableMemoryBytes = memory.availMem.coerceAtLeast(1L),
            thermalStatus = thermal,
            hardware = hardware,
            socManufacturer = socManufacturer,
            socModel = socModel,
            nnapiAvailable = Build.VERSION.SDK_INT >= 27,
            xnnpackAvailable = true,
            qnnAvailable = qualcomm && qnnBackendPath != null,
            qnnBackendPath = qnnBackendPath,
        )
    }
}
