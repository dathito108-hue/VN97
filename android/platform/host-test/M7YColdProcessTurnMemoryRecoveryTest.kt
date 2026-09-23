import ai.vn97.platform.*
import ai.vn97.runtime.*
import java.nio.file.Files

private class RecoveryBackend : VN97TurnMemoryBackend {
    override val vectorDim = 4
    private val records = linkedMapOf<Long, NativeMemoryRecord>()
    var appendCalls = 0
    var failAppend = false
    var failNextRecordRead = false

    override fun stats() = NativeMemoryStoreStats(
        vectorDim = vectorDim,
        recordCount = records.size.toLong(),
        lastRecordId = records.keys.maxOrNull() ?: 0L,
        validBytes = 0L,
    )

    override fun append(
        kind: NativeMemoryKind,
        timestampNs: Long,
        importance: Float,
        source: String,
        content: String,
        vector: FloatArray,
    ): Long {
        appendCalls += 1
        if (failAppend) error("injected append failure")
        check(vector.size == vectorDim)
        val id = (records.keys.maxOrNull() ?: 0L) + 1L
        records[id] = NativeMemoryRecord(
            id,
            timestampNs,
            0L,
            kind,
            importance,
            source,
            content,
        )
        return id
    }

    override fun record(recordId: Long): NativeMemoryRecord {
        if (failNextRecordRead) {
            failNextRecordRead = false
            error("injected post-append crash window")
        }
        return records[recordId] ?: error("missing record")
    }

    fun recordTimestamp(recordId: Long): Long =
        checkNotNull(records[recordId]).timestampNs
}

private fun completedTurn(createdNs: Long): VN97AssistantTurn {
    val controller = NativePlanController.create(
        goal = "remember after cold process restart",
        specs = listOf(
            NativePlanStepSpec(
                kind = NativeStepKind.RESPOND,
                objective = "answer",
            )
        ),
        createdNs = createdNs,
    )
    controller.beginStep(1)
    controller.completeStep(1, "durable answer", 1.0)
    return VN97AssistantTurn(1L, controller, "runtime.user")
}

private inline fun expectFailure(block: () -> Unit) {
    var failed = false
    try {
        block()
    } catch (_: Exception) {
        failed = true
    }
    check(failed)
}

fun main() {
    val embedder = VN97TurnMemoryEmbedder { _, dim ->
        FloatArray(dim) { 1.0f }
    }

    // Crash after VN97MEM1 append but before M7W COMMITTED: recovery verifies the
    // expected existing record and commits the journal without a duplicate append.
    val root = Files.createTempDirectory("m7y-post-append-").toFile()
    val backend = RecoveryBackend().also { it.failNextRecordRead = true }
    val firstWriter = VN97TurnMemoryWriter(backend, embedder, root)
    val firstRecovery = VN97TurnMemoryRecovery(firstWriter, root)
    expectFailure {
        firstRecovery.commitCompletedTurn(
            turn = completedTurn(10L),
            finalResponse = "durable answer",
            timestampNs = 100L,
        )
    }
    check(firstWriter.pendingTurnKey != null)
    check(backend.appendCalls == 1)
    firstWriter.close()

    val secondWriter = VN97TurnMemoryWriter(backend, embedder, root)
    val secondRecovery = VN97TurnMemoryRecovery(secondWriter, root)
    val recovered = checkNotNull(secondRecovery.recoverPendingOrNull())
    check(recovered.recordId == 1L)
    check(recovered.planId == completedTurn(10L).planId)
    check(secondWriter.pendingTurnKey == null)
    check(backend.appendCalls == 1)
    check(!root.resolve("turn-memory-recovery.vn97tmr1").exists())
    secondWriter.close()

    // PREPARED before append: a later retry/restart preserves the original completion
    // timestamp from VN97TMR1 even when the caller supplies a newer retry timestamp.
    val root2 = Files.createTempDirectory("m7y-prepared-").toFile()
    val backend2 = RecoveryBackend().also { it.failAppend = true }
    val writer2 = VN97TurnMemoryWriter(backend2, embedder, root2)
    val recovery2 = VN97TurnMemoryRecovery(writer2, root2)
    val turn2 = completedTurn(20L)
    expectFailure {
        recovery2.commitCompletedTurn(turn2, "durable answer", 200L)
    }
    check(writer2.pendingTurnKey != null)
    backend2.failAppend = false
    check(
        recovery2.commitCompletedTurn(
            turn2,
            "durable answer",
            999L,
        ) == 1L
    )
    check(backend2.recordTimestamp(1L) == 200L)
    check(writer2.pendingTurnKey == null)
    writer2.close()

    // Fail closed if a pending VN97TWJ1 transaction exists without the recovery envelope.
    val root3 = Files.createTempDirectory("m7y-missing-envelope-").toFile()
    val backend3 = RecoveryBackend().also { it.failAppend = true }
    val writer3 = VN97TurnMemoryWriter(backend3, embedder, root3)
    val recovery3 = VN97TurnMemoryRecovery(writer3, root3)
    expectFailure {
        recovery3.commitCompletedTurn(
            completedTurn(30L),
            "durable answer",
            300L,
        )
    }
    writer3.close()
    check(root3.resolve("turn-memory-recovery.vn97tmr1").delete())
    backend3.failAppend = false
    VN97TurnMemoryWriter(backend3, embedder, root3).use { reopened ->
        val cold = VN97TurnMemoryRecovery(reopened, root3)
        expectFailure { cold.recoverPendingOrNull() }
        check(reopened.pendingTurnKey != null)
        check(backend3.appendCalls == 1)
    }

    // Recovery metadata is integrity checked independently from VN97TWJ1.
    val root4 = Files.createTempDirectory("m7y-corrupt-envelope-").toFile()
    val backend4 = RecoveryBackend().also { it.failAppend = true }
    val writer4 = VN97TurnMemoryWriter(backend4, embedder, root4)
    val recovery4 = VN97TurnMemoryRecovery(writer4, root4)
    expectFailure {
        recovery4.commitCompletedTurn(
            completedTurn(40L),
            "durable answer",
            400L,
        )
    }
    writer4.close()
    val envelope = root4.resolve("turn-memory-recovery.vn97tmr1")
    val corrupt = envelope.readBytes()
    corrupt[corrupt.lastIndex] = (corrupt.last().toInt() xor 1).toByte()
    envelope.writeBytes(corrupt)
    VN97TurnMemoryWriter(backend4, embedder, root4).use { reopened ->
        val cold = VN97TurnMemoryRecovery(reopened, root4)
        expectFailure { cold.recoverPendingOrNull() }
    }

    println("M7Y_COLD_PROCESS_TURN_MEMORY_RECOVERY_PASS")
}
