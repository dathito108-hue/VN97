package ai.vn97.platform

import ai.vn97.runtime.NativeCognitionLimits
import ai.vn97.runtime.NativeTypedCognitionAdapter
import android.content.Context
import java.io.File

class AndroidPlatformRuntime(
    context: Context,
    approvalKeyAlias: String = "vn97.approval.v1",
    approvalIssuer: String = "local-user",
    permissionRequirements: Map<String, Set<String>> = emptyMap(),
) {
    private val appContext = context.applicationContext

    val approvals: AndroidApprovalController = AndroidApprovalController(
        keyAlias = approvalKeyAlias,
        issuer = approvalIssuer,
    )

    val externalApprovals: M6ExternalApprovalHandoff =
        M6ExternalApprovalHandoff(AndroidApprovalControllerPort(approvals))

    internal val permissionBroker = AndroidPermissionBroker(appContext, permissionRequirements)
    internal val appDeviceAdapter = AndroidAppDeviceAdapter(appContext, permissionBroker)
    private val productionCapabilities = M6AndroidProductionCapabilities(appDeviceAdapter)

    val externalIntentBinder: M6ExternalIntentBinder = productionCapabilities.intentBinder

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

    fun createProductionExternalExecutionFabric(
        grants: List<M6PolicyGrant>,
        auditFileName: String = "m6-actions.jsonl",
    ): M6ExternalExecutionFabric = createDurableExternalExecutionFabric(
        registry = productionCapabilities.createSealedRegistry(),
        grants = grants,
        auditRoot = appContext.noBackupFilesDir,
        auditFileName = auditFileName,
    )

    fun createProductionExternalCoordinator(
        cognition: NativeTypedCognitionAdapter,
        grants: List<M6PolicyGrant>,
        cognitionLimits: NativeCognitionLimits = NativeCognitionLimits(),
        auditFileName: String = "m6-actions.jsonl",
    ): M6EndToEndExternalCoordinator = M6EndToEndExternalCoordinator(
        cognition = cognition,
        binder = externalIntentBinder,
        approvals = externalApprovals,
        executionFabric = createProductionExternalExecutionFabric(
            grants = grants,
            auditFileName = auditFileName,
        ),
        cognitionLimits = cognitionLimits,
    )
}
