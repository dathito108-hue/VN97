import ai.vn97.platform.*
import ai.vn97.runtime.*
import java.nio.file.Files
import java.security.MessageDigest

private inline fun expectIntegrity(block: () -> Unit) {
    var failed = false
    try { block() } catch (_: M6AuditIntegrityException) { failed = true }
    check(failed) { "expected M6AuditIntegrityException" }
}

private inline fun expectCapacity(block: () -> Unit) {
    var failed = false
    try { block() } catch (_: M6AuditCapacityException) { failed = true }
    check(failed) { "expected M6AuditCapacityException" }
}

private fun requestFixture(): Pair<M6ExternalActionRequest, M6CapabilityDescriptor> {
    val step = NativePlannerStep(
        1,
        NativePlanStepSpec(NativeStepKind.EXTERNAL, "echo"),
        NativeStepStatus.WAITING_EXTERNAL,
    )
    val controller = NativePlanController(
        NativePlan(
            "ab".repeat(32),
            "g",
            listOf(step),
            NativePlanStatus.WAITING_EXTERNAL,
        )
    )
    val descriptor = M6CapabilityDescriptor(
        "test.echo",
        setOf("target"),
        approvalRequired = true,
        maxLeaseUses = 1,
    )
    val binder = M6ExternalIntentBinder(listOf(descriptor))
    val request = binder.bind(
        controller,
        NativeExternalIntent(
            "test.echo",
            mapOf("target" to "alpha"),
            "{\"message\":\"hi\"}",
        ),
    )
    check(request.requestDigest == "e9a4311db6a33f19a118ce9c9cb9f7fcffde650594a562e93a5280f36e050d73")
    check(request.scope.digest == "1b22e7f4a338b62840bbd0d9361ae0eda309ead2df2b1e8b455ea0237eb5174e")
    return request to descriptor
}

private fun copyAudit(bytes: ByteArray): java.io.File {
    val root = Files.createTempDirectory("vn97-m7q-copy-").toFile()
    root.resolve("m6-actions.jsonl").writeBytes(bytes)
    return root
}

fun main() {
    val (request, descriptor) = requestFixture()
    val root = Files.createTempDirectory("vn97-m7q-").toFile()
    try {
        val audit = M6DurableActionAudit(root)
        check(audit.receipts.isEmpty())
        val receipt = audit.append(
            status = M6ReceiptStatus.SUCCEEDED,
            request = request,
            principal = "runtime.user",
            leaseId = "00112233445566778899aabbccddeeff",
            approvalId = "ffeeddccbbaa99887766554433221100",
            outcome = M6ActionOutcome(true, "external-ok", 0.9, listOf(7L)),
            timestampNs = 21,
        )
        check(receipt.receiptId == "e59095fc030b9fba6ab644534ff36bf10776db06a7ffd652c3284be4caf03f84")

        val file = root.resolve("m6-actions.jsonl")
        val firstBytes = file.readBytes()
        check(firstBytes.size == 654) { "unexpected fixture bytes: ${firstBytes.size}" }
        val fileSha = MessageDigest.getInstance("SHA-256")
            .digest(firstBytes)
            .joinToString("") { "%02x".format(it.toInt() and 0xff) }
        check(fileSha == "34ca083dace3b8ccaf62ab443446bf84f8a21f360a064201df433d06578e2536") { fileSha }

        val reopened = M6DurableActionAudit(root)
        check(reopened.receipts.single() == receipt)
        check(reopened.successfulReceipt(request.requestDigest) == receipt)

        // A process-restarted fabric replays the durable successful receipt before authority/handler execution.
        var calls = 0
        val registry = M6TypedCapabilityRegistry()
        registry.register(
            descriptor,
            M6CapabilityHandler {
                calls += 1
                M6ActionOutcome(true, "must-not-run")
            },
        )
        registry.seal()
        val authority = M6DenyByDefaultAuthorityGate(emptyList(), approvals = null)
        val fabric = M6ExternalExecutionFabric(registry, authority, reopened)
        val replayStep = NativePlannerStep(
            1,
            NativePlanStepSpec(NativeStepKind.EXTERNAL, "echo"),
            NativeStepStatus.WAITING_EXTERNAL,
        )
        val replayController = NativePlanController(
            NativePlan(
                request.planId,
                "g",
                listOf(replayStep),
                NativePlanStatus.WAITING_EXTERNAL,
            )
        )
        val replay = fabric.executeWaiting(
            replayController,
            request,
            "runtime.user",
            nowNs = 22,
        )
        check(replay.replayed)
        check(calls == 0)
        check(replayStep.result == "external-ok")
        check(replayStep.evidenceRecordIds == listOf(7L))

        val second = reopened.append(
            status = M6ReceiptStatus.DENIED,
            request = request,
            principal = "runtime.user",
            errorType = "M6AuthorizationDeniedException",
            timestampNs = 23,
        )
        check(second.sequence == 2)
        check(second.previousReceiptId == receipt.receiptId)
        val twice = M6DurableActionAudit(root)
        check(twice.receipts.size == 2)
        check(twice.receipts[1].previousReceiptId == twice.receipts[0].receiptId)
        check(twice.successfulReceipt(request.requestDigest)?.receiptId == receipt.receiptId)

        val tornRoot = copyAudit(firstBytes.copyOf(firstBytes.size - 1))
        try { expectIntegrity { M6DurableActionAudit(tornRoot) } } finally { tornRoot.deleteRecursively() }

        val tampered = firstBytes.copyOf()
        val marker = "external-ok".toByteArray()
        val replacement = "external-NO".toByteArray()
        val at = tampered.indices.firstOrNull { i ->
            i + marker.size <= tampered.size && tampered.copyOfRange(i, i + marker.size).contentEquals(marker)
        } ?: error("result marker missing")
        replacement.copyInto(tampered, at)
        val tamperedRoot = copyAudit(tampered)
        try { expectIntegrity { M6DurableActionAudit(tamperedRoot) } } finally { tamperedRoot.deleteRecursively() }

        val nonCanonicalText = firstBytes.toString(Charsets.UTF_8)
            .replace("\"confidence\":0.9", "\"confidence\":0.90")
        val nonCanonicalRoot = copyAudit(nonCanonicalText.toByteArray())
        try { expectIntegrity { M6DurableActionAudit(nonCanonicalRoot) } } finally { nonCanonicalRoot.deleteRecursively() }

        val boundedRoot = copyAudit(firstBytes)
        try {
            val bounded = M6DurableActionAudit(boundedRoot, maxRecords = 1)
            expectCapacity {
                bounded.append(
                    M6ReceiptStatus.FAILED,
                    request,
                    "runtime.user",
                    errorType = "x",
                    timestampNs = 24,
                )
            }
        } finally { boundedRoot.deleteRecursively() }

        val changedRoot = copyAudit(firstBytes)
        try {
            val changed = M6DurableActionAudit(changedRoot)
            Files.write(
                changedRoot.resolve("m6-actions.jsonl").toPath(),
                byteArrayOf('x'.code.toByte()),
                java.nio.file.StandardOpenOption.APPEND,
            )
            expectIntegrity {
                changed.append(
                    M6ReceiptStatus.FAILED,
                    request,
                    "runtime.user",
                    errorType = "x",
                    timestampNs = 25,
                )
            }
        } finally { changedRoot.deleteRecursively() }

        println("M7Q_DURABLE_AUDIT_PASS")
    } finally {
        root.deleteRecursively()
    }
}
