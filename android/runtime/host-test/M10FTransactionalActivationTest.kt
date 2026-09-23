package ai.vn97.runtime

import java.io.File
import java.nio.file.Files

private fun sha(ch: String) = ch.repeat(64)

private fun verified(
    version: Long,
    packageChar: String,
): VN97VerifiedCapability {
    val manifest = VN97CapabilityManifest(
        capabilityId = "model.language",
        capabilityVersion = version,
        kind = "weights",
        source = VN97CapabilitySource("local", sha("a"), "test"),
        sections = listOf(
            VN97CapabilitySection(
                1,
                "model_image",
                "VN97MI1",
                8,
                sha("b"),
                256,
            )
        ),
    )
    return VN97VerifiedCapability(
        staged = VN97StagedCapability("stage-$version-$packageChar"),
        parsed = VN97ParsedCapabilityPackage(
            sha(packageChar),
            manifest,
            512,
        ),
        publisherKeyId = "owner",
        signatureSha256 = sha("c"),
    )
}

private class FakeBackend : VN97CapabilityActivationBackend {
    override val backendId = "vn97.model_image"
    var prepares = 0
    var commits = 0
    var rollbacks = 0
    var crashAfterCommit = false
    var crashDuringPrepare = false
    private var nextId = 1
    private val states = mutableMapOf<String, VN97BackendStatus>()

    override fun prepare(
        verified: VN97VerifiedCapability,
        plan: VN97CompatibilityPlan,
    ): VN97PreparedActivation {
        prepares++
        if (crashDuringPrepare) {
            throw AssertionError("simulated process death during prepare")
        }
        val token = "tok-${nextId++}"
        states[token] = VN97BackendStatus(
            VN97BackendTransactionState.PREPARED
        )
        return VN97PreparedActivation(
            backendId,
            token,
            sha("d"),
        )
    }

    override fun commit(token: String): String {
        commits++
        val revision = "rev-$commits"
        states[token] = VN97BackendStatus(
            VN97BackendTransactionState.COMMITTED,
            revision,
        )
        if (crashAfterCommit) {
            throw AssertionError("simulated process death after commit")
        }
        return revision
    }

    override fun inspect(token: String): VN97BackendStatus =
        checkNotNull(states[token])

    override fun rollback(token: String) {
        rollbacks++
        states[token] = VN97BackendStatus(
            VN97BackendTransactionState.ROLLED_BACK
        )
    }
}

private inline fun <reified T : Throwable> expectFailure(
    block: () -> Unit,
) {
    var hit = false
    try {
        block()
    } catch (error: Throwable) {
        if (error is T) {
            hit = true
        } else {
            throw error
        }
    }
    check(hit)
}

fun main() {
    val directProfile = VN97CompatibilityProfile(
        profileId = "mobile",
        runtimeApiVersion = 1,
        supportedKinds = setOf("weights"),
        minCapabilityVersion = 1,
        maxCapabilityVersion = 10,
        formatRules = listOf(
            VN97FormatRule(
                "model_image",
                listOf("VN97MI1"),
            )
        ),
    )
    val v1 = verified(1, "1")
    val p1 = VN97CapabilityCompatibility.plan(v1, directProfile)
    check(p1.disposition == VN97CompatibilityDisposition.DIRECT)
    check(p1.sections.single().adapterId == null)
    check(p1.profileSha256 == directProfile.fingerprint())
    check(p1.sha256().length == 64)

    val adaptProfile = directProfile.copy(
        profileId = "mobile-adapt",
        formatRules = listOf(
            VN97FormatRule(
                "model_image",
                listOf("VN97MI2"),
            )
        ),
    )
    val adapter = VN97AdapterSpec(
        "mi1-to-mi2",
        "model_image",
        "VN97MI1",
        "VN97MI2",
    )
    val adapted = VN97CapabilityCompatibility.plan(
        v1,
        adaptProfile,
        listOf(adapter),
    )
    check(adapted.disposition == VN97CompatibilityDisposition.ADAPT_REQUIRED)
    check(adapted.sections.single().adapterId == "mi1-to-mi2")
    expectFailure<VN97CompatibilityException> {
        VN97CapabilityCompatibility.plan(v1, adaptProfile)
    }
    expectFailure<VN97CompatibilityException> {
        VN97CapabilityCompatibility.plan(
            v1,
            adaptProfile,
            listOf(adapter.copy(lossy = true)),
            allowLossy = false,
        )
    }

    val root = Files.createTempDirectory("m10f-inventory-").toFile()
    val store = VN97CapabilityInventoryStore(root)
    val coordinator = VN97CapabilityActivationCoordinator(store)
    val backend = FakeBackend()
    val trust = VN97CapabilityTrustStore()
    val stageRoot = File(root, "stage").also { it.mkdir() }

    VN97CapabilityTrustVerifier.next = v1
    val item1 = coordinator.activate(
        v1,
        p1,
        directProfile,
        backend,
        stageRoot,
        trust,
    )
    check(item1.capabilityVersion == 1L)
    check(backend.prepares == 1 && backend.commits == 1)
    check(coordinator.inventory().generation == 1L)
    check(coordinator.inventory().pendingOperation == null)
    check(root.resolve(VN97_INVENTORY_FILE).isFile)

    val idempotent = coordinator.activate(
        v1,
        p1,
        directProfile,
        backend,
        stageRoot,
        trust,
    )
    check(idempotent.activationId == item1.activationId)
    check(backend.prepares == 1)

    val v2 = verified(2, "2")
    val p2 = VN97CapabilityCompatibility.plan(v2, directProfile)
    VN97CapabilityTrustVerifier.next = v2
    val item2 = coordinator.activate(
        v2,
        p2,
        directProfile,
        backend,
        stageRoot,
        trust,
    )
    check(item2.capabilityVersion == 2L)
    check(coordinator.inventory().generation == 2L)

    val restored = coordinator.rollback(
        "model.language",
        backend,
    )
    check(restored?.activationId == item1.activationId)
    check(coordinator.inventory().generation == 3L)

    val v1Replacement = verified(1, "3")
    val p1Replacement = VN97CapabilityCompatibility.plan(
        v1Replacement,
        directProfile,
    )
    VN97CapabilityTrustVerifier.next = v1Replacement
    expectFailure<VN97VersionTransitionException> {
        coordinator.activate(
            v1Replacement,
            p1Replacement,
            directProfile,
            backend,
            stageRoot,
            trust,
        )
    }

    VN97CapabilityTrustVerifier.next = v2
    backend.crashAfterCommit = true
    expectFailure<AssertionError> {
        coordinator.activate(
            v2,
            p2,
            directProfile,
            backend,
            stageRoot,
            trust,
        )
    }
    backend.crashAfterCommit = false
    check(coordinator.inventory().pendingOperation == "activate")
    val recovered = coordinator.recover(
        mapOf(backend.backendId to backend)
    )
    check(recovered.pendingOperation == null)
    check(
        recovered.current("model.language")
            ?.capabilityVersion == 2L
    )
    check(
        recovered.history.last().action ==
            VN97InventoryAction.RECOVER_ACTIVATE
    )

    val v3 = verified(3, "4")
    val p3 = VN97CapabilityCompatibility.plan(v3, directProfile)
    VN97CapabilityTrustVerifier.next = v3
    backend.crashDuringPrepare = true
    expectFailure<AssertionError> {
        coordinator.activate(
            v3,
            p3,
            directProfile,
            backend,
            stageRoot,
            trust,
        )
    }
    backend.crashDuringPrepare = false
    check(coordinator.inventory().pendingOperation == "activate")
    val afterReservedRecovery = coordinator.recover(
        mapOf(backend.backendId to backend)
    )
    check(afterReservedRecovery.pendingOperation == null)
    check(
        afterReservedRecovery.current("model.language")
            ?.capabilityVersion == 2L
    )

    println("M10F_TRANSACTIONAL_ACTIVATION_PASS")
}
