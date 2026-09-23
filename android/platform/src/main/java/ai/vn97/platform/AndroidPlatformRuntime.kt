package ai.vn97.platform

import ai.vn97.runtime.NativeActivatedModel
import ai.vn97.runtime.NativeCognitionInferenceEngine
import ai.vn97.runtime.NativeCognitionLimits
import ai.vn97.runtime.NativeCognitionRuntimeConfig
import ai.vn97.runtime.NativeMemoryStore
import ai.vn97.runtime.NativeTypedCognitionAdapter
import android.content.Context
import java.io.File
import java.nio.file.Files

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

    /**
     * Production assistant path rooted directly in one activated VN97 model.
     * No generic inference/backend parameter is accepted here.
     */
    fun createProductionAssistantSession(
        model: NativeActivatedModel,
        grants: List<M6PolicyGrant>,
        cognitionRuntimeConfig: NativeCognitionRuntimeConfig = NativeCognitionRuntimeConfig(),
        cognitionLimits: NativeCognitionLimits = NativeCognitionLimits(),
        sessionLimits: VN97AssistantSessionLimits = VN97AssistantSessionLimits(),
        auditFileName: String = "m6-actions.jsonl",
    ): VN97AssistantSession {
        val cognition = NativeTypedCognitionAdapter(
            NativeCognitionInferenceEngine(
                model = model,
                config = cognitionRuntimeConfig,
            )
        )
        return VN97AssistantSession(
            coordinator = createProductionExternalCoordinator(
                cognition = cognition,
                grants = grants,
                cognitionLimits = cognitionLimits,
                auditFileName = auditFileName,
            ),
            limits = sessionLimits,
        )
    }

    /** Open the canonical native VN97MEM1 store from app-private no-backup storage. */
    fun openOrCreateProductionMemory(
        model: NativeActivatedModel,
        fileName: String = "memory.vn97mem1",
        recoverTornTail: Boolean = true,
    ): NativeMemoryStore {
        require(isSafeMemoryFileName(fileName)) {
            "memory fileName must be one bounded app-private file component"
        }
        val root = File(appContext.noBackupFilesDir, "vn97-memory")
        if (root.exists()) {
            require(root.isDirectory && !Files.isSymbolicLink(root.toPath())) {
                "VN97 memory root must be a non-symlink directory"
            }
        } else {
            check(root.mkdirs()) { "failed to create VN97 memory root" }
        }

        val file = File(root, fileName)
        if (file.exists()) {
            require(file.isFile && !Files.isSymbolicLink(file.toPath())) {
                "VN97 memory target must be a non-symlink regular file"
            }
            val opened = NativeMemoryStore.open(
                file = file,
                recoverTornTail = recoverTornTail,
            )
            return try {
                require(opened.vectorDim == model.info.dModel) {
                    "VN97MEM1 vector dimension does not match activated model dModel"
                }
                opened
            } catch (exc: Throwable) {
                opened.close()
                throw exc
            }
        }
        return NativeMemoryStore.create(
            file = file,
            vectorDim = model.info.dModel,
            overwrite = false,
        )
    }
}

private fun isSafeMemoryFileName(value: String): Boolean =
    value.isNotEmpty() &&
        value.length <= 128 &&
        value != "." &&
        value != ".." &&
        value.all { ch ->
            ch in 'a'..'z' || ch in 'A'..'Z' || ch in '0'..'9' ||
                ch == '.' || ch == '_' || ch == '-'
        }
