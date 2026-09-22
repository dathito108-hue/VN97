package ai.vn97.platform

import android.content.Context
import android.content.pm.PackageManager

class AndroidPermissionDeniedException(
    val capabilityId: String,
    val missingPermissions: Set<String>,
) : SecurityException(
    "Android permissions missing for $capabilityId: ${missingPermissions.sorted().joinToString()}"
)

class AndroidPermissionBroker(
    context: Context,
    requirements: Map<String, Set<String>> = emptyMap(),
) {
    private val appContext = context.applicationContext
    private val requiredByCapability = requirements.mapValues { (_, values) -> values.toSet() }.toMap()

    fun missingPermissions(capabilityId: String): Set<String> =
        requiredByCapability[capabilityId].orEmpty().filterTo(linkedSetOf()) { permission ->
            appContext.checkSelfPermission(permission) != PackageManager.PERMISSION_GRANTED
        }

    fun requireGranted(capabilityId: String) {
        val missing = missingPermissions(capabilityId)
        if (missing.isNotEmpty()) throw AndroidPermissionDeniedException(capabilityId, missing)
    }
}
