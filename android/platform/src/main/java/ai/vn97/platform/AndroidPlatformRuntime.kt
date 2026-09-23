package ai.vn97.platform

import ai.vn97.runtime.NativeActivatedModel
import ai.vn97.runtime.NativeCognitionInferenceEngine
import ai.vn97.runtime.NativeCognitionLimits
import ai.vn97.runtime.NativeCognitionRuntimeConfig
import ai.vn97.runtime.NativeMemoryRetriever
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

    internal val permissionBroker =
        AndroidPermissionBroker(appContext, permissionRequirements)

    val gameControlPolicy =
        VN97GameControlPolicy(appContext)

    internal val appDeviceAdapter =
        AndroidAppDeviceAdapter(
            appContext,
            permissionBroker,
            gameControlPolicy,
        )

    private val productionCapabilities =
        M6AndroidProductionCapabilities(appDeviceAdapter)

    val externalIntentBinder: M6ExternalIntentBinder =
        productionCapabilities.intentBinder

    val gameIntentBinder: M6ExternalIntentBinder =
        productionCapabilities.gameIntentBinder

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

    fun collectProductionMobileEvidence(
        model: NativeActivatedModel,
        config: VN97MobileEvidenceConfig = VN97MobileEvidenceConfig(),
    ): VN97MobileEvidenceRecord =
        VN97OnDeviceEvidenceCollector(appContext).collect(
            model = model,
            config = config,
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

    fun createProductionGameExternalExecutionFabric(
        grants: List<M6PolicyGrant>,
        auditFileName: String = "m14-game-actions.jsonl",
    ): M6ExternalExecutionFabric = createDurableExternalExecutionFabric(
        registry = productionCapabilities.createSealedGameRegistry(),
        grants = grants,
        auditRoot = appContext.noBackupFilesDir,
        auditFileName = auditFileName,
    )

    fun createProductionGameExternalCoordinator(
        cognition: NativeTypedCognitionAdapter,
        grants: List<M6PolicyGrant>,
        cognitionLimits: NativeCognitionLimits = NativeCognitionLimits(),
        auditFileName: String = "m14-game-actions.jsonl",
    ): M6EndToEndExternalCoordinator = M6EndToEndExternalCoordinator(
        cognition = cognition,
        binder = gameIntentBinder,
        approvals = externalApprovals,
        executionFabric = createProductionGameExternalExecutionFabric(
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
    ): VN97AssistantSession = assembleProductionAssistantSession(
        model = model,
        grants = grants,
        cognitionRuntimeConfig = cognitionRuntimeConfig,
        cognitionLimits = cognitionLimits,
        sessionLimits = sessionLimits,
        auditFileName = auditFileName,
        memory = null,
        turnMemoryWriter = null,
        turnMemoryRecovery = null,
    )

    /**
     * Open the canonical app-private VN97MEM1 store and bind it to the existing M7T
     * assistant session for the complete turn lifecycle.
     */
    fun createProductionMemoryBackedAssistant(
        model: NativeActivatedModel,
        grants: List<M6PolicyGrant>,
        cognitionRuntimeConfig: NativeCognitionRuntimeConfig = NativeCognitionRuntimeConfig(),
        cognitionLimits: NativeCognitionLimits = NativeCognitionLimits(),
        sessionLimits: VN97AssistantSessionLimits = VN97AssistantSessionLimits(),
        auditFileName: String = "m6-actions.jsonl",
        memoryFileName: String = "memory.vn97mem1",
        recoverTornTail: Boolean = true,
        turnMemoryJournalFileName: String = "turn-memory.vn97twj1",
        recoverTurnMemoryJournalTornTail: Boolean = true,
        maxTurnMemoryJournalEntries: Int = 200_000,
        turnMemoryImportance: Float = 0.75f,
        turnMemoryRecoveryFileName: String = "turn-memory-recovery.vn97tmr1",
    ): VN97ProductionAssistantResources {
        val memory = openOrCreateProductionMemory(
            model = model,
            fileName = memoryFileName,
            recoverTornTail = recoverTornTail,
        )
        val turnMemoryWriter = try {
            createProductionTurnMemoryWriter(
                model = model,
                memory = memory,
                cognitionRuntimeConfig = cognitionRuntimeConfig,
                journalFileName = turnMemoryJournalFileName,
                recoverTornTail = recoverTurnMemoryJournalTornTail,
                maxJournalEntries = maxTurnMemoryJournalEntries,
                importance = turnMemoryImportance,
            )
        } catch (exc: Throwable) {
            memory.close()
            throw exc
        }
        val turnMemoryRecovery = try {
            VN97TurnMemoryRecovery(
                writer = turnMemoryWriter,
                root = File(appContext.noBackupFilesDir, "vn97-memory"),
                fileName = turnMemoryRecoveryFileName,
            )
        } catch (exc: Throwable) {
            try {
                turnMemoryWriter.close()
            } finally {
                memory.close()
            }
            throw exc
        }
        val recoveredTurnMemoryCommit = try {
            turnMemoryRecovery.recoverPendingOrNull()
        } catch (exc: Throwable) {
            try {
                turnMemoryWriter.close()
            } finally {
                memory.close()
            }
            throw exc
        }
        return try {
            VN97ProductionAssistantResources(
                session = assembleProductionAssistantSession(
                    model = model,
                    grants = grants,
                    cognitionRuntimeConfig = cognitionRuntimeConfig,
                    cognitionLimits = cognitionLimits,
                    sessionLimits = sessionLimits,
                    auditFileName = auditFileName,
                    memory = memory,
                    turnMemoryWriter = turnMemoryWriter,
                    turnMemoryRecovery = turnMemoryRecovery,
                ),
                memory = memory,
                turnMemoryWriter = turnMemoryWriter,
                turnMemoryRecovery = turnMemoryRecovery,
                recoveredTurnMemoryCommit = recoveredTurnMemoryCommit,
            )
        } catch (exc: Throwable) {
            try {
                turnMemoryWriter.close()
            } finally {
                memory.close()
            }
            throw exc
        }
    }

    /**
     * Resume one M7Z-restored non-terminal planner through the exact production
     * M7T/M7V/M7Y assistant path. The context controller is mutated in place and
     * the JobService commits that same controller in the next VN97CNT1 epoch.
     */
    fun resumeProductionAssistantContinuation(
        context: VN97AssistantContinuationContext,
        model: NativeActivatedModel,
        grants: List<M6PolicyGrant>,
        cognitionRuntimeConfig: NativeCognitionRuntimeConfig =
            NativeCognitionRuntimeConfig(),
        cognitionLimits: NativeCognitionLimits = NativeCognitionLimits(),
        sessionLimits: VN97AssistantSessionLimits = VN97AssistantSessionLimits(),
        auditFileName: String = "m6-actions.jsonl",
        nowNs: Long,
    ): VN97AssistantTurnUpdate {
        require(nowNs >= 0L) { "nowNs must be non-negative" }
        check(!context.isStopped()) { "assistant continuation was stopped" }
        context.requireActivatedModel(model)

        return createProductionMemoryBackedAssistant(
            model = model,
            grants = grants,
            cognitionRuntimeConfig = cognitionRuntimeConfig,
            cognitionLimits = cognitionLimits,
            sessionLimits = sessionLimits,
            auditFileName = auditFileName,
        ).use { resources ->
            resources.session.resumeRestoredTurn(
                controller = context.controller,
                principal = context.principal,
                nowNs = nowNs,
            )
        }
    }

    private fun assembleProductionAssistantSession(
        model: NativeActivatedModel,
        grants: List<M6PolicyGrant>,
        cognitionRuntimeConfig: NativeCognitionRuntimeConfig,
        cognitionLimits: NativeCognitionLimits,
        sessionLimits: VN97AssistantSessionLimits,
        auditFileName: String,
        memory: NativeMemoryRetriever?,
        turnMemoryWriter: VN97TurnMemoryWriter?,
        turnMemoryRecovery: VN97TurnMemoryRecovery?,
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
            defaultMemory = memory,
            turnMemoryWriter = turnMemoryWriter,
            turnMemoryRecovery = turnMemoryRecovery,
        )
    }

    /**
     * Create the M7W idempotent write-back transaction layer over the exact native
     * VN97MEM1 store and activated VN97 cognition embedder.
     */
    fun createProductionTurnMemoryWriter(
        model: NativeActivatedModel,
        memory: NativeMemoryStore,
        cognitionRuntimeConfig: NativeCognitionRuntimeConfig = NativeCognitionRuntimeConfig(),
        journalFileName: String = "turn-memory.vn97twj1",
        recoverTornTail: Boolean = true,
        maxJournalEntries: Int = 200_000,
        importance: Float = 0.75f,
    ): VN97TurnMemoryWriter {
        require(memory.vectorDim == model.info.dModel) {
            "VN97MEM1 vector dimension does not match activated model dModel"
        }
        val engine = NativeCognitionInferenceEngine(
            model = model,
            config = cognitionRuntimeConfig,
        )
        val root = File(appContext.noBackupFilesDir, "vn97-memory")
        return VN97TurnMemoryWriter(
            backend = NativeVN97TurnMemoryBackend(memory),
            embedder = VN97TurnMemoryEmbedder { text, vectorDim ->
                engine.embedText(text, vectorDim)
            },
            journalRoot = root,
            journalFileName = journalFileName,
            recoverTornTail = recoverTornTail,
            maxJournalEntries = maxJournalEntries,
            importance = importance,
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

class VN97ProductionAssistantResources internal constructor(
    val session: VN97AssistantSession,
    val memory: NativeMemoryStore,
    val turnMemoryWriter: VN97TurnMemoryWriter,
    val turnMemoryRecovery: VN97TurnMemoryRecovery,
    val recoveredTurnMemoryCommit: VN97RecoveredTurnMemoryCommit?,
) : AutoCloseable {
    override fun close() {
        try {
            turnMemoryWriter.close()
        } finally {
            memory.close()
        }
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
