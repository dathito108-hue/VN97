package ai.vn97.runtime

import java.io.File
import java.nio.charset.StandardCharsets
import java.security.SecureRandom

enum class VN97BackendTransactionState {
    PREPARED,
    COMMITTED,
    ROLLED_BACK,
}

data class VN97BackendStatus(
    val state: VN97BackendTransactionState,
    val runtimeRevision: String? = null,
) {
    init {
        if (state == VN97BackendTransactionState.COMMITTED) {
            inventoryRequireRevision(
                runtimeRevision ?: throw IllegalArgumentException(
                    "committed status requires runtime_revision"
                ),
                "runtime_revision",
            )
        } else if (runtimeRevision != null) {
            throw IllegalArgumentException(
                "non-committed backend status must not contain runtime_revision"
            )
        }
    }
}

data class VN97PreparedActivation(
    val backendId: String,
    val token: String,
    val artifactSha256: String,
) {
    init {
        inventoryRequireId(backendId, "backend_id")
        inventoryRequireToken(token, "backend token")
        inventoryRequireSha(artifactSha256, "artifact_sha256")
    }
}

interface VN97CapabilityActivationBackend {
    val backendId: String

    fun prepare(
        verified: VN97VerifiedCapability,
        plan: VN97CompatibilityPlan,
    ): VN97PreparedActivation

    fun commit(token: String): String
    fun inspect(token: String): VN97BackendStatus
    fun rollback(token: String)
}

class VN97ActivationConflictException(message: String) :
    VN97CapabilityActivationException(message)

class VN97ActivationRecoveryRequired(message: String, cause: Throwable? = null) :
    VN97CapabilityActivationException(message, cause)

class VN97VersionTransitionException(message: String) :
    VN97CapabilityActivationException(message)

class VN97CapabilityActivationCoordinator(
    private val store: VN97CapabilityInventoryStore,
) {
    private val random = SecureRandom()

    fun inventory(): VN97InventorySnapshot = store.load()

    fun activate(
        verified: VN97VerifiedCapability,
        plan: VN97CompatibilityPlan,
        profile: VN97CompatibilityProfile,
        backend: VN97CapabilityActivationBackend,
        stageRoot: File,
        trustStore: VN97CapabilityTrustStore,
        adapters: List<VN97AdapterSpec> = emptyList(),
        allowLossy: Boolean = false,
        allowSameVersionReplace: Boolean = false,
        allowDowngrade: Boolean = false,
    ): VN97CapabilityInventoryItem {
        val fresh = VN97CapabilityTrustVerifier.verify(
            staged = verified.staged,
            stageRoot = stageRoot,
            trustStore = trustStore,
        )
        if (fresh != verified) {
            throw VN97ActivationConflictException(
                "verified capability changed before activation"
            )
        }
        val expected = VN97CapabilityCompatibility.plan(
            fresh,
            profile,
            adapters,
            allowLossy,
        )
        if (plan != expected) {
            throw VN97ActivationConflictException(
                "compatibility plan is stale or mismatched"
            )
        }
        inventoryRequireId(backend.backendId, "backend_id")
        val identity = planIdentity(plan, fresh)

        var existing: VN97CapabilityInventoryItem? = null
        val txId = store.locked { tx ->
            val state = tx.state
            if (state.pending != null) {
                throw VN97ActivationRecoveryRequired(
                    "inventory has unresolved transaction"
                )
            }
            val stack = state.stacks[plan.capabilityId]
            val current = stack?.lastOrNull()?.item
            if (current != null) {
                val exact =
                    current.packageSha256 == plan.packageSha256 &&
                        current.signatureSha256 == plan.signatureSha256 &&
                        current.profileSha256 == plan.profileSha256 &&
                        current.planSha256 == identity.planSha256 &&
                        current.backendId == backend.backendId
                if (exact) {
                    existing = current
                    return@locked ""
                }
                if (stack.size >= 64) {
                    throw VN97VersionTransitionException(
                        "activation stack depth limit reached"
                    )
                }
                if (plan.capabilityVersion < current.capabilityVersion &&
                    !allowDowngrade
                ) {
                    throw VN97VersionTransitionException(
                        "capability downgrade is denied"
                    )
                }
                if (plan.capabilityVersion == current.capabilityVersion &&
                    !allowSameVersionReplace
                ) {
                    throw VN97VersionTransitionException(
                        "same-version replacement is denied"
                    )
                }
            }

            val id = activationTransactionId(
                generation = state.generation,
                identity = identity,
            )
            state.pending = VN97PendingTransaction(
                txId = id,
                operation = "activate",
                phase = "reserved",
                capabilityId = plan.capabilityId,
                backendId = backend.backendId,
                generationBase = state.generation,
                identity = identity,
                backendToken = null,
                artifactSha256 = null,
                activationId = null,
            )
            tx.persist()
            id
        }
        existing?.let { return it }

        val prepared = try {
            backend.prepare(fresh, plan)
        } catch (exc: Exception) {
            clearReserved(txId)
            throw exc
        }
        if (prepared.backendId != backend.backendId) {
            clearReserved(txId)
            throw VN97ActivationConflictException(
                "backend prepare returned mismatched backend_id"
            )
        }

        store.locked { tx ->
            val pending = tx.state.pending
            if (pending == null ||
                pending.txId != txId ||
                pending.phase != "reserved"
            ) {
                try {
                    backend.rollback(prepared.token)
                } finally {
                    throw VN97ActivationRecoveryRequired(
                        "activation reservation changed before commit"
                    )
                }
            }
            val currentPending = checkNotNull(tx.state.pending)
            tx.state.pending = currentPending.copy(
                phase = "prepared",
                backendToken = prepared.token,
                artifactSha256 = prepared.artifactSha256,
            )
            tx.persist()
        }

        val revision = try {
            backend.commit(prepared.token).also {
                inventoryRequireRevision(it, "runtime_revision")
            }
        } catch (exc: Exception) {
            try {
                backend.rollback(prepared.token)
            } catch (rollbackExc: Exception) {
                throw VN97ActivationRecoveryRequired(
                    "activation needs recovery after commit failure",
                    rollbackExc,
                )
            }
            clearPrepared(txId, prepared.token)
            throw exc
        }

        return finalizeActivation(
            txId = txId,
            revision = revision,
            action = VN97InventoryAction.ACTIVATE,
        )
    }

    fun rollback(
        capabilityId: String,
        backend: VN97CapabilityActivationBackend,
    ): VN97CapabilityInventoryItem? {
        inventoryRequireId(capabilityId, "capability_id")
        inventoryRequireId(backend.backendId, "backend_id")
        lateinit var token: String
        val txId = store.locked { tx ->
            val state = tx.state
            if (state.pending != null) {
                throw VN97ActivationRecoveryRequired(
                    "inventory has unresolved transaction"
                )
            }
            val stack = state.stacks[capabilityId]
                ?: throw VN97CapabilityActivationException(
                    "capability is not active"
                )
            val current = stack.last()
            if (current.item.backendId != backend.backendId) {
                throw VN97ActivationConflictException(
                    "rollback backend mismatch"
                )
            }
            token = current.backendToken
            val id = rollbackTransactionId(
                generation = state.generation,
                activationId = current.item.activationId,
            )
            state.pending = VN97PendingTransaction(
                txId = id,
                operation = "rollback",
                phase = "prepared",
                capabilityId = capabilityId,
                backendId = backend.backendId,
                generationBase = state.generation,
                identity = null,
                backendToken = token,
                artifactSha256 = current.item.artifactSha256,
                activationId = current.item.activationId,
            )
            tx.persist()
            id
        }

        try {
            backend.rollback(token)
        } catch (exc: Exception) {
            throw VN97ActivationRecoveryRequired(
                "rollback requires recovery",
                exc,
            )
        }
        return finalizeRollback(txId, VN97InventoryAction.ROLLBACK)
    }

    fun recover(
        backends: Map<String, VN97CapabilityActivationBackend>,
    ): VN97InventorySnapshot {
        val pending = store.locked { tx ->
            val value = tx.state.pending ?: return@locked null
            if (value.phase == "reserved") {
                tx.state.pending = null
                tx.persist()
                return@locked null
            }
            value
        } ?: return inventory()

        val backend = backends[pending.backendId]
            ?: throw VN97ActivationRecoveryRequired(
                "required recovery backend is unavailable"
            )
        if (backend.backendId != pending.backendId) {
            throw VN97ActivationRecoveryRequired(
                "required recovery backend identity mismatch"
            )
        }
        val token = pending.backendToken
            ?: throw VN97ActivationRecoveryRequired(
                "prepared transaction lacks backend token"
            )
        val status = backend.inspect(token)

        if (pending.operation == "activate") {
            when (status.state) {
                VN97BackendTransactionState.COMMITTED -> finalizeActivation(
                    txId = pending.txId,
                    revision = checkNotNull(status.runtimeRevision),
                    action = VN97InventoryAction.RECOVER_ACTIVATE,
                )
                VN97BackendTransactionState.PREPARED -> {
                    try {
                        backend.rollback(token)
                    } catch (exc: Exception) {
                        throw VN97ActivationRecoveryRequired(
                            "prepared activation rollback failed",
                            exc,
                        )
                    }
                    clearPrepared(pending.txId, token)
                }
                VN97BackendTransactionState.ROLLED_BACK ->
                    clearPrepared(pending.txId, token)
            }
        } else {
            if (status.state != VN97BackendTransactionState.ROLLED_BACK) {
                try {
                    backend.rollback(token)
                } catch (exc: Exception) {
                    throw VN97ActivationRecoveryRequired(
                        "rollback recovery failed",
                        exc,
                    )
                }
            }
            finalizeRollback(
                pending.txId,
                VN97InventoryAction.RECOVER_ROLLBACK,
            )
        }
        return inventory()
    }

    private fun clearReserved(txId: String) {
        store.locked { tx ->
            val pending = tx.state.pending
            if (pending != null &&
                pending.txId == txId &&
                pending.phase == "reserved"
            ) {
                tx.state.pending = null
                tx.persist()
            }
        }
    }

    private fun clearPrepared(txId: String, token: String) {
        store.locked { tx ->
            val pending = tx.state.pending
            if (pending != null &&
                pending.txId == txId &&
                pending.backendToken == token
            ) {
                tx.state.pending = null
                tx.persist()
            }
        }
    }

    private fun finalizeActivation(
        txId: String,
        revision: String,
        action: VN97InventoryAction,
    ): VN97CapabilityInventoryItem {
        inventoryRequireRevision(revision, "runtime_revision")
        return store.locked { tx ->
            val state = tx.state
            val pending = state.pending
            if (pending == null ||
                pending.txId != txId ||
                pending.operation != "activate" ||
                pending.phase != "prepared"
            ) {
                throw VN97ActivationRecoveryRequired(
                    "activation transaction changed before finalization"
                )
            }
            val identity = pending.identity
                ?: throw VN97ActivationRecoveryRequired(
                    "prepared activation lacks identity"
                )
            val artifact = pending.artifactSha256
                ?: throw VN97ActivationRecoveryRequired(
                    "prepared activation lacks artifact"
                )
            val token = pending.backendToken
                ?: throw VN97ActivationRecoveryRequired(
                    "prepared activation lacks token"
                )
            val activationId = activationId(
                identity,
                pending.backendId,
                artifact,
                revision,
            )
            val item = VN97CapabilityInventoryItem(
                activationId = activationId,
                capabilityId = identity.capabilityId,
                capabilityVersion = identity.capabilityVersion,
                packageSha256 = identity.packageSha256,
                publisherKeyId = identity.publisherKeyId,
                signatureSha256 = identity.signatureSha256,
                profileId = identity.profileId,
                profileSha256 = identity.profileSha256,
                planSha256 = identity.planSha256,
                runtimeApiVersion = identity.runtimeApiVersion,
                backendId = pending.backendId,
                artifactSha256 = artifact,
                runtimeRevision = revision,
                sourceOrigin = identity.sourceOrigin,
                sourceSha256 = identity.sourceSha256,
                sourceLicense = identity.sourceLicense,
            )
            val stack = state.stacks.getOrPut(item.capabilityId) { mutableListOf() }
            if (stack.size >= 64) {
                throw VN97ActivationRecoveryRequired(
                    "activation stack became full"
                )
            }
            stack += VN97StoredActivation(item, token)
            state.generation += 1L
            state.history += VN97InventoryEvent(
                generation = state.generation,
                action = action,
                capabilityId = item.capabilityId,
                activationId = item.activationId,
                packageSha256 = item.packageSha256,
            )
            if (state.history.size > 4096) {
                repeat(state.history.size - 4096) { state.history.removeAt(0) }
            }
            state.pending = null
            tx.persist()
            item
        }
    }

    private fun finalizeRollback(
        txId: String,
        action: VN97InventoryAction,
    ): VN97CapabilityInventoryItem? = store.locked { tx ->
        val state = tx.state
        val pending = state.pending
        if (pending == null ||
            pending.txId != txId ||
            pending.operation != "rollback"
        ) {
            throw VN97ActivationRecoveryRequired(
                "rollback transaction changed before finalization"
            )
        }
        val stack = state.stacks[pending.capabilityId]
            ?: throw VN97ActivationConflictException(
                "active capability changed during rollback"
            )
        val expectedActivation = pending.activationId
            ?: throw VN97ActivationRecoveryRequired(
                "rollback transaction lacks activation_id"
            )
        if (stack.last().item.activationId != expectedActivation) {
            throw VN97ActivationConflictException(
                "active capability changed during rollback"
            )
        }
        val removed = stack.removeAt(stack.lastIndex)
        if (stack.isEmpty()) state.stacks.remove(removed.item.capabilityId)
        state.generation += 1L
        state.history += VN97InventoryEvent(
            generation = state.generation,
            action = action,
            capabilityId = removed.item.capabilityId,
            activationId = removed.item.activationId,
            packageSha256 = removed.item.packageSha256,
        )
        if (state.history.size > 4096) {
            repeat(state.history.size - 4096) { state.history.removeAt(0) }
        }
        state.pending = null
        tx.persist()
        stack.lastOrNull()?.item
    }

    private fun planIdentity(
        plan: VN97CompatibilityPlan,
        verified: VN97VerifiedCapability,
    ): VN97ActivationIdentity {
        val source = verified.parsed.manifest.source
        return VN97ActivationIdentity(
            capabilityId = plan.capabilityId,
            capabilityVersion = plan.capabilityVersion,
            packageSha256 = plan.packageSha256,
            planSha256 = plan.sha256(),
            profileId = plan.profileId,
            profileSha256 = plan.profileSha256,
            publisherKeyId = plan.publisherKeyId,
            runtimeApiVersion = plan.runtimeApiVersion,
            signatureSha256 = plan.signatureSha256,
            sourceLicense = source.license,
            sourceOrigin = source.origin,
            sourceSha256 = source.sourceSha256,
        )
    }

    private fun activationTransactionId(
        generation: Long,
        identity: VN97ActivationIdentity,
    ): String = hashCanonical(
        VnStrictJson.objectOf(
            "generation" to VnJsonNumber(generation.toString()),
            "identity" to identityToJson(identity),
            "nonce" to VnStrictJson.string(randomNonce()),
            "operation" to VnStrictJson.string("activate"),
        )
    )

    private fun rollbackTransactionId(
        generation: Long,
        activationId: String,
    ): String = hashCanonical(
        VnStrictJson.objectOf(
            "activation_id" to VnStrictJson.string(activationId),
            "generation" to VnJsonNumber(generation.toString()),
            "nonce" to VnStrictJson.string(randomNonce()),
            "operation" to VnStrictJson.string("rollback"),
        )
    )

    private fun activationId(
        identity: VN97ActivationIdentity,
        backendId: String,
        artifactSha256: String,
        revision: String,
    ): String {
        val fields = identityToJson(identity).values.toMutableMap()
        fields["artifact_sha256"] = VnStrictJson.string(artifactSha256)
        fields["backend_id"] = VnStrictJson.string(backendId)
        fields["runtime_revision"] = VnStrictJson.string(revision)
        return hashCanonical(VnJsonObject(fields))
    }

    private fun randomNonce(): String {
        val bytes = ByteArray(16)
        random.nextBytes(bytes)
        return bytes.joinToString("") { "%02x".format(it.toInt() and 0xff) }
    }

    private fun hashCanonical(value: VnJsonValue): String = sha256Hex(
        VnStrictJson.canonical(value).toByteArray(StandardCharsets.UTF_8)
    )
}
