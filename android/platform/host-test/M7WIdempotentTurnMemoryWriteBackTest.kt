import ai.vn97.platform.*
import ai.vn97.runtime.*
import java.nio.file.Files

private class FakeTurnMemoryBackend : VN97TurnMemoryBackend {
    override val vectorDim = 4
    private val records = linkedMapOf<Long, NativeMemoryRecord>()
    var appendCalls = 0
    var failAppend = false

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
            id, timestampNs, 0L, kind, importance, source, content
        )
        return id
    }

    override fun record(recordId: Long): NativeMemoryRecord =
        records[recordId] ?: error("missing record")

    fun appendForeign(): Long {
        val id = (records.keys.maxOrNull() ?: 0L) + 1L
        records[id] = NativeMemoryRecord(
            id, 1L, 0L, NativeMemoryKind.EPISODIC, 0.5f, "foreign", "foreign"
        )
        return id
    }
}

private fun completedTurn(createdNs: Long = 10L): VN97AssistantTurn {
    val controller = NativePlanController.create(
        goal = "hello",
        specs = listOf(NativePlanStepSpec(NativeStepKind.RESPOND, "answer")),
        createdNs = createdNs,
    )
    controller.beginStep(1)
    controller.completeStep(1, "world", 1.0)
    return VN97AssistantTurn(1L, controller, "runtime.user")
}

private inline fun expectFailure(block: () -> Unit) {
    var failed = false
    try { block() } catch (_: Exception) { failed = true }
    check(failed)
}

fun main() {
    val embedder = VN97TurnMemoryEmbedder { _, dim -> FloatArray(dim) { 1.0f } }

    val root = Files.createTempDirectory("m7w-normal-").toFile()
    val backend = FakeTurnMemoryBackend()
    VN97TurnMemoryWriter(backend, embedder, root).use { writer ->
        check(writer.commitCompletedTurn(completedTurn(), "world", 100L) == 1L)
        check(writer.commitCompletedTurn(completedTurn(), "world", 101L) == 1L)
        check(backend.appendCalls == 1)
    }

    val root2 = Files.createTempDirectory("m7w-prepared-").toFile()
    val backend2 = FakeTurnMemoryBackend().also { it.failAppend = true }
    VN97TurnMemoryWriter(backend2, embedder, root2).use { writer ->
        expectFailure {
            writer.commitCompletedTurn(completedTurn(20L), "world", 200L)
        }
        check(writer.pendingTurnKey != null)
    }
    backend2.failAppend = false
    VN97TurnMemoryWriter(backend2, embedder, root2).use { writer ->
        check(writer.commitCompletedTurn(completedTurn(20L), "world", 201L) == 1L)
        check(writer.pendingTurnKey == null)
    }
    check(backend2.appendCalls == 2)

    val root3 = Files.createTempDirectory("m7w-conflict-").toFile()
    val backend3 = FakeTurnMemoryBackend().also { it.failAppend = true }
    VN97TurnMemoryWriter(backend3, embedder, root3).use { writer ->
        expectFailure {
            writer.commitCompletedTurn(completedTurn(30L), "world", 300L)
        }
    }
    backend3.failAppend = false
    backend3.appendForeign()
    val before = backend3.appendCalls
    VN97TurnMemoryWriter(backend3, embedder, root3).use { writer ->
        expectFailure {
            writer.commitCompletedTurn(completedTurn(30L), "world", 301L)
        }
    }
    check(backend3.appendCalls == before)

    val journal = root.resolve("turn-memory.vn97twj1")
    val exact = journal.readBytes()
    journal.appendBytes(byteArrayOf(1, 2, 3, 4, 5))
    VN97TurnMemoryWriter(backend, embedder, root).use { }
    check(journal.length() == exact.size.toLong())

    val corrupt = journal.readBytes()
    corrupt[20] = (corrupt[20].toInt() xor 1).toByte()
    journal.writeBytes(corrupt)
    expectFailure { VN97TurnMemoryWriter(backend, embedder, root).close() }

    println("M7W_IDEMPOTENT_TURN_MEMORY_WRITEBACK_PASS")
}
