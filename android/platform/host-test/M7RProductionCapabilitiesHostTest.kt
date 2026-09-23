package ai.vn97.platform

import ai.vn97.runtime.NativeExternalIntent
import ai.vn97.runtime.NativePlan
import ai.vn97.runtime.NativePlanController
import ai.vn97.runtime.NativePlannerStep
import ai.vn97.runtime.NativePlanStatus
import ai.vn97.runtime.NativePlanStepSpec
import ai.vn97.runtime.NativeStepKind
import ai.vn97.runtime.NativeStepStatus

private data class RecordedTap(
    val packageName: String,
    val x: Int,
    val y: Int,
    val durationMillis: Long,
)

private data class RecordedSwipe(
    val packageName: String,
    val startX: Int,
    val startY: Int,
    val endX: Int,
    val endY: Int,
    val durationMillis: Long,
)

private class M7RRecordingActions : M6AndroidActionPort {
    val launched = mutableListOf<String>()
    val clipboard = mutableListOf<String>()
    val taps = mutableListOf<RecordedTap>()
    val swipes = mutableListOf<RecordedSwipe>()
    val backs = mutableListOf<String>()

    override fun launchPackage(packageName: String): String {
        launched += packageName
        return "launched:$packageName"
    }

    override fun writeClipboard(text: String): String {
        clipboard += text
        return "clipboard:written"
    }

    override fun gameTap(
        packageName: String,
        xBasisPoints: Int,
        yBasisPoints: Int,
        durationMillis: Long,
    ): String {
        taps += RecordedTap(
            packageName,
            xBasisPoints,
            yBasisPoints,
            durationMillis,
        )
        return "game:tap:$packageName"
    }

    override fun gameSwipe(
        packageName: String,
        startXBasisPoints: Int,
        startYBasisPoints: Int,
        endXBasisPoints: Int,
        endYBasisPoints: Int,
        durationMillis: Long,
    ): String {
        swipes += RecordedSwipe(
            packageName,
            startXBasisPoints,
            startYBasisPoints,
            endXBasisPoints,
            endYBasisPoints,
            durationMillis,
        )
        return "game:swipe:$packageName"
    }

    override fun gameBack(packageName: String): String {
        backs += packageName
        return "game:back:$packageName"
    }
}

private class M7RAcceptingApproval : M6ApprovalControllerPort {
    override fun createPrompt(
        requestDigest: String,
        principal: String,
        presentationJson: String,
        promptTtlNs: Long,
        nowNs: Long,
    ) = M6ApprovalPrompt(
        "00112233445566778899aabbccddeeff",
        principal,
        requestDigest,
        presentationJson,
        nowNs,
        nowNs + promptTtlNs,
    )

    override fun resolve(
        prompt: M6ApprovalPrompt,
        approved: Boolean,
        approvalTtlNs: Long,
        nowNs: Long,
    ): M6ApprovalToken? = if (approved) {
        token(prompt.requestDigest, prompt.principal, nowNs, nowNs + approvalTtlNs)
    } else {
        null
    }

    override fun verify(
        token: M6ApprovalToken,
        requestDigest: String,
        principal: String,
        nowNs: Long,
    ) {
        check(token.requestDigest == requestDigest)
        check(token.principal == principal)
        check(nowNs in token.issuedNs until token.expiresNs)
    }

    fun token(
        requestDigest: String,
        principal: String = "runtime.user",
        issuedNs: Long = 10L,
        expiresNs: Long = 100L,
    ) = M6ApprovalToken(
        approvalId = "ffeeddccbbaa99887766554433221100",
        issuer = "local-user",
        principal = principal,
        requestDigest = requestDigest,
        issuedNs = issuedNs,
        expiresNs = expiresNs,
        signature = "00".repeat(32),
    )
}

private fun waiting(planId: String, objective: String): NativePlanController =
    NativePlanController(
        NativePlan(
            planId,
            "external production action",
            listOf(
                NativePlannerStep(
                    1,
                    NativePlanStepSpec(NativeStepKind.EXTERNAL, objective),
                    NativeStepStatus.WAITING_EXTERNAL,
                )
            ),
            NativePlanStatus.WAITING_EXTERNAL,
        )
    )

private inline fun expectRejected(block: () -> Unit) {
    var rejected = false
    try {
        block()
    } catch (_: IllegalArgumentException) {
        rejected = true
    }
    check(rejected)
}

fun main() {
    val actions = M7RRecordingActions()
    val assembly = M6AndroidProductionCapabilities(actions)
    check(
        assembly.descriptors.map { it.capabilityId } ==
            listOf(
                "app.launch",
                "device.clipboard.write",
                "device.game.tap",
                "device.game.swipe",
                "device.game.back",
            )
    )
    check(
        assembly.intentBinder.capabilities.map { it.capabilityId } ==
            assembly.descriptors.map { it.capabilityId }
    )
    check(
        assembly.intentBinder.capabilities.map {
            it.payloadSchemaJson
        } == assembly.descriptors.map { it.payloadSchemaJson }
    )
    check(
        assembly.gameIntentBinder.capabilities.map {
            it.capabilityId
        } == listOf(
            M6AndroidProductionCapabilities.GAME_TAP_CAPABILITY,
            M6AndroidProductionCapabilities.GAME_SWIPE_CAPABILITY,
            M6AndroidProductionCapabilities.GAME_BACK_CAPABILITY,
        )
    )
    val gameOnlyRegistry = assembly.createSealedGameRegistry()
    check(gameOnlyRegistry.sealed)
    assembly.descriptors.take(2).forEach { descriptor ->
        check(descriptor.approvalRequired)
        check(descriptor.maxLeaseUses == 1)
        check(descriptor.maxLeaseNs == 30_000_000_000L)
    }
    assembly.descriptors.drop(2).forEach { descriptor ->
        check(!descriptor.approvalRequired)
        check(descriptor.maxLeaseUses == 1)
        check(descriptor.maxLeaseNs == 10_000_000_000L)
    }

    val registry = assembly.createSealedRegistry()
    check(registry.sealed)
    var sealRejected = false
    try {
        registry.register(
            assembly.descriptors.first(),
            M6CapabilityHandler { M6ActionOutcome(true, "unexpected") },
        )
    } catch (_: M6AuthorityException) {
        sealRejected = true
    }
    check(sealRejected)

    val planId = "ab".repeat(32)
    val appController = waiting(planId, "launch exact package")
    val appRequest = assembly.intentBinder.bind(
        appController,
        NativeExternalIntent(
            "app.launch",
            mapOf("package" to "com.example.vn97"),
            "{}",
        ),
    )
    registry.validate(appRequest)
    val approvals = M7RAcceptingApproval()
    val appAudit = M6InMemoryActionAudit()
    val appFabric = M6ExternalExecutionFabric(
        registry,
        M6DenyByDefaultAuthorityGate(
            listOf(
                M6PolicyGrant(
                    "runtime.user",
                    "app.launch",
                    appRequest.scope.digest,
                )
            ),
            approvals,
        ),
        appAudit,
    )
    val appResult = appFabric.executeWaiting(
        appController,
        appRequest,
        "runtime.user",
        approval = approvals.token(appRequest.requestDigest),
        nowNs = 20L,
    )
    check(!appResult.replayed)
    check(actions.launched == listOf("com.example.vn97"))
    check(appController.plan.status == NativePlanStatus.COMPLETED)
    check(appAudit.receipts.single().status == M6ReceiptStatus.SUCCEEDED)

    // M7P replay must still happen before approval/lease/handler when using M7R handlers.
    val restored = waiting(planId, "launch exact package")
    val replay = appFabric.executeWaiting(
        restored,
        appRequest,
        "runtime.user",
        nowNs = 21L,
    )
    check(replay.replayed)
    check(actions.launched.size == 1)

    val invalidApp = assembly.intentBinder.bind(
        waiting("cd".repeat(32), "launch package"),
        NativeExternalIntent(
            "app.launch",
            mapOf("package" to "bad package"),
            "{}",
        ),
    )
    expectRejected { registry.validate(invalidApp) }

    val productionAppGrant =
        M6AndroidProductionCapabilities.userApprovedAppLaunchGrant(
            "runtime.user",
            "com.example.vn97",
        )
    check(
        productionAppGrant.capabilityId ==
            M6AndroidProductionCapabilities.APP_LAUNCH_CAPABILITY
    )
    check(
        productionAppGrant.scopeDigest ==
            M6CapabilityScope.fromMap(
                mapOf(
                    M6AndroidProductionCapabilities.APP_PACKAGE_SCOPE to
                        "com.example.vn97"
                )
            ).digest
    )
    check(productionAppGrant.approvalRequired == true)
    check(productionAppGrant.maxLeaseUses == 1)

    val productionClipboardGrant =
        M6AndroidProductionCapabilities.userApprovedClipboardGrant(
            "runtime.user"
        )
    check(
        productionClipboardGrant.capabilityId ==
            M6AndroidProductionCapabilities.CLIPBOARD_WRITE_CAPABILITY
    )
    check(
        productionClipboardGrant.scopeDigest ==
            M6CapabilityScope.fromMap(
                mapOf(
                    M6AndroidProductionCapabilities.CLIPBOARD_CHANNEL_SCOPE to
                        M6AndroidProductionCapabilities.CLIPBOARD_CHANNEL_VALUE
                )
            ).digest
    )
    check(productionClipboardGrant.approvalRequired == true)
    check(productionClipboardGrant.maxLeaseUses == 1)
    check(productionClipboardGrant.maxLeaseNs == 30_000_000_000L)

    val clipboardText = "hello\n\"VN97\" \\ mobile 😀"
    val clipboardController = waiting("ef".repeat(32), "write clipboard")
    val clipboardRequest = assembly.intentBinder.bind(
        clipboardController,
        NativeExternalIntent(
            "device.clipboard.write",
            mapOf("channel" to "system-clipboard"),
            "{\"text\":\"hello\\n\\\"VN97\\\" \\\\ mobile 😀\"}",
        ),
    )
    registry.validate(clipboardRequest)
    val clipboardFabric = M6ExternalExecutionFabric(
        registry,
        M6DenyByDefaultAuthorityGate(
            listOf(
                M6PolicyGrant(
                    "runtime.user",
                    "device.clipboard.write",
                    clipboardRequest.scope.digest,
                )
            ),
            approvals,
        ),
        M6InMemoryActionAudit(),
    )
    clipboardFabric.executeWaiting(
        clipboardController,
        clipboardRequest,
        "runtime.user",
        approval = approvals.token(clipboardRequest.requestDigest),
        nowNs = 20L,
    )
    check(actions.clipboard == listOf(clipboardText))

    val extraPayload = assembly.intentBinder.bind(
        waiting("12".repeat(32), "write clipboard"),
        NativeExternalIntent(
            "device.clipboard.write",
            mapOf("channel" to "system-clipboard"),
            "{\"text\":\"ok\",\"extra\":true}",
        ),
    )
    expectRejected { registry.validate(extraPayload) }

    val nonCanonicalEscape = assembly.intentBinder.bind(
        waiting("34".repeat(32), "write clipboard"),
        NativeExternalIntent(
            "device.clipboard.write",
            mapOf("channel" to "system-clipboard"),
            "{\"text\":\"a\\/b\"}",
        ),
    )
    expectRejected { registry.validate(nonCanonicalEscape) }

    val tooLong = "a".repeat(16 * 1024 + 1)
    val oversized = assembly.intentBinder.bind(
        waiting("56".repeat(32), "write clipboard"),
        NativeExternalIntent(
            "device.clipboard.write",
            mapOf("channel" to "system-clipboard"),
            "{\"text\":\"$tooLong\"}",
        ),
    )
    expectRejected { registry.validate(oversized) }
    check(actions.clipboard.size == 1)

    val gamePackage = "com.example.game"
    val gameGrants =
        M6AndroidProductionCapabilities.userApprovedGameControlGrants(
            "runtime.user",
            gamePackage,
        )
    check(
        gameGrants.map { it.capabilityId } ==
            listOf(
                M6AndroidProductionCapabilities.GAME_TAP_CAPABILITY,
                M6AndroidProductionCapabilities.GAME_SWIPE_CAPABILITY,
                M6AndroidProductionCapabilities.GAME_BACK_CAPABILITY,
            )
    )
    gameGrants.forEach { grant ->
        check(grant.approvalRequired == false)
        check(grant.maxLeaseUses == 1)
        check(grant.maxLeaseNs == 10_000_000_000L)
        check(
            grant.scopeDigest ==
                M6CapabilityScope.fromMap(
                    mapOf(
                        M6AndroidProductionCapabilities.APP_PACKAGE_SCOPE to
                            gamePackage
                    )
                ).digest
        )
    }

    val gameFabric = M6ExternalExecutionFabric(
        registry,
        M6DenyByDefaultAuthorityGate(
            gameGrants,
            approvals,
        ),
        M6InMemoryActionAudit(),
    )

    val tapController =
        waiting("67".repeat(32), "tap the visible game control")
    val tapRequest = assembly.intentBinder.bind(
        tapController,
        NativeExternalIntent(
            M6AndroidProductionCapabilities.GAME_TAP_CAPABILITY,
            mapOf("package" to gamePackage),
            "{\"duration_ms\":80,\"x_bps\":5000,\"y_bps\":5000}",
        ),
    )
    registry.validate(tapRequest)
    gameFabric.executeWaiting(
        tapController,
        tapRequest,
        "runtime.user",
        nowNs = 30L,
    )
    check(
        actions.taps.single() ==
            RecordedTap(gamePackage, 5000, 5000, 80L)
    )

    val swipeController =
        waiting("68".repeat(32), "swipe in the game")
    val swipeRequest = assembly.intentBinder.bind(
        swipeController,
        NativeExternalIntent(
            M6AndroidProductionCapabilities.GAME_SWIPE_CAPABILITY,
            mapOf("package" to gamePackage),
            "{\"duration_ms\":300,\"end_x_bps\":8000,\"end_y_bps\":5000,\"start_x_bps\":2000,\"start_y_bps\":5000}",
        ),
    )
    registry.validate(swipeRequest)
    gameFabric.executeWaiting(
        swipeController,
        swipeRequest,
        "runtime.user",
        nowNs = 31L,
    )
    check(
        actions.swipes.single() ==
            RecordedSwipe(
                gamePackage,
                2000,
                5000,
                8000,
                5000,
                300L,
            )
    )

    val backController =
        waiting("69".repeat(32), "go back in the game")
    val backRequest = assembly.intentBinder.bind(
        backController,
        NativeExternalIntent(
            M6AndroidProductionCapabilities.GAME_BACK_CAPABILITY,
            mapOf("package" to gamePackage),
            "{}",
        ),
    )
    registry.validate(backRequest)
    gameFabric.executeWaiting(
        backController,
        backRequest,
        "runtime.user",
        nowNs = 32L,
    )
    check(actions.backs == listOf(gamePackage))

    val invalidTap = assembly.intentBinder.bind(
        waiting("70".repeat(32), "invalid tap"),
        NativeExternalIntent(
            M6AndroidProductionCapabilities.GAME_TAP_CAPABILITY,
            mapOf("package" to gamePackage),
            "{\"duration_ms\":80,\"x_bps\":10001,\"y_bps\":5000}",
        ),
    )
    expectRejected { registry.validate(invalidTap) }

    val invalidSwipe = assembly.intentBinder.bind(
        waiting("71".repeat(32), "invalid swipe"),
        NativeExternalIntent(
            M6AndroidProductionCapabilities.GAME_SWIPE_CAPABILITY,
            mapOf("package" to gamePackage),
            "{\"duration_ms\":300,\"end_x_bps\":5000,\"end_y_bps\":5000,\"start_x_bps\":5000,\"start_y_bps\":5000}",
        ),
    )
    expectRejected { registry.validate(invalidSwipe) }

    println("M7R_PRODUCTION_CAPABILITY_ASSEMBLY_PASS")
}
