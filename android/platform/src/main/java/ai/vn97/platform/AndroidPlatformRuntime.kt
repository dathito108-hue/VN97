package ai.vn97.platform

import android.content.Context

class AndroidPlatformRuntime(
    context: Context,
    approvalKeyAlias: String = "vn97.approval.v1",
    approvalIssuer: String = "local-user",
    permissionRequirements: Map<String, Set<String>> = emptyMap(),
) {
    val approvals: AndroidApprovalController = AndroidApprovalController(
        keyAlias = approvalKeyAlias,
        issuer = approvalIssuer,
    )

    internal val permissionBroker = AndroidPermissionBroker(context, permissionRequirements)
    internal val appDeviceAdapter = AndroidAppDeviceAdapter(context, permissionBroker)
}
