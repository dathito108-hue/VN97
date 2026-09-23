package ai.vn97.platform

import android.content.Context
import java.io.File

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

    val externalApprovals: M6ExternalApprovalHandoff =
        M6ExternalApprovalHandoff(AndroidApprovalControllerPort(approvals))

    fun createExternalExecutionFabric(
        registry: M6TypedCapabilityRegistry,
        grants: List<M6PolicyGrant>,
        audit: M6ActionAudit = M6InMemoryActionAudit(),
    ): M6ExternalExecutionFabric {
        val authority = M6DenyByDefaultAuthorityGate(
            grants = grants,
            approvals = AndroidApprovalControllerPort(approvals),
        )
        return M6ExternalExecutionFabric(registry, authority, audit)
    }

    fun createDurableExternalExecutionFabric(
        registry: M6TypedCapabilityRegistry,
        grants: List<M6PolicyGrant>,
        auditRoot: File,
        auditFileName: String = "m6-actions.jsonl",
    ): M6ExternalExecutionFabric = createExternalExecutionFabric(
        registry = registry,
        grants = grants,
        audit = M6DurableActionAudit(
            root = auditRoot,
            fileName = auditFileName,
        ),
    )

    internal val permissionBroker = AndroidPermissionBroker(context, permissionRequirements)
    internal val appDeviceAdapter = AndroidAppDeviceAdapter(context, permissionBroker)
}
