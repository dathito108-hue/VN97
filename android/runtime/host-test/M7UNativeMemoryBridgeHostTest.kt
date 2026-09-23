import ai.vn97.runtime.*
import java.nio.file.Files

private inline fun expectArgument(block: () -> Unit) {
    var failed = false
    try { block() } catch (_: IllegalArgumentException) { failed = true }
    check(failed)
}

private inline fun expectState(block: () -> Unit) {
    var failed = false
    try { block() } catch (_: IllegalStateException) { failed = true }
    check(failed)
}


private class RetrievalInference(
    private val semanticId: Long,
) : NativeCognitionInference {
    private var stepCalls = 0

    override fun generateOperation(
        operation: NativeCognitionOperation,
        requestJson: String,
    ): String = when (operation) {
        NativeCognitionOperation.MEMORY_QUERY ->
            """{"query":"semantic","top_k":2,"kinds":["SEMANTIC"],"semantic_weight":1.0,"recency_weight":0.0,"importance_weight":0.0,"recency_half_life_ns":100}"""

        NativeCognitionOperation.STEP -> {
            stepCalls += 1
            if (stepCalls == 1) {
                check(requestJson.contains("semantic fact ✓"))
                check(requestJson.contains("\"record_id\":$semanticId"))
                """{"result":"retrieved","confidence":1.0}"""
            } else {
                check(requestJson.contains("\"evidence_record_ids\":[$semanticId]"))
                """{"result":"final response","confidence":1.0}"""
            }
        }

        else -> error("unexpected cognition operation: $operation")
    }

    override fun embedText(text: String, vectorDim: Int): FloatArray {
        check(text == "semantic")
        check(vectorDim == 2)
        return floatArrayOf(1f, 0f)
    }
}

fun main() {
    val root = Files.createTempDirectory("vn97-m7u-memory-").toFile()
    val file = root.resolve("memory.vn97mem1")

    val created = NativeMemoryStore.create(file, vectorDim = 2)
    val initial = created.stats()
    check(initial.vectorDim == 2)
    check(initial.recordCount == 0L)
    check(initial.lastRecordId == 0L)
    check(initial.validBytes > 0L)

    val semanticId = created.append(
        kind = NativeMemoryKind.SEMANTIC,
        timestampNs = 10L,
        importance = 0.9f,
        source = "unit:α",
        content = "semantic fact ✓",
        vector = floatArrayOf(1f, 0f),
    )
    val episodicId = created.append(
        kind = NativeMemoryKind.EPISODIC,
        timestampNs = 20L,
        importance = 0.4f,
        source = "unit:β",
        content = "episode two",
        vector = floatArrayOf(0f, 1f),
        parentId = semanticId,
    )
    check(semanticId == 1L && episodicId == 2L)
    check(created.stats().recordCount == 2L)
    val direct = created.record(semanticId)
    check(direct.recordId == semanticId)
    check(direct.kind == NativeMemoryKind.SEMANTIC)
    check(direct.timestampNs == 10L)
    check(direct.parentId == 0L)
    check(direct.importance == 0.9f)
    check(direct.source == "unit:α")
    check(direct.content == "semantic fact ✓")

    var locked = false
    try {
        NativeMemoryStore.open(file).close()
    } catch (exc: NativeMemoryException) {
        locked = exc.status == NativeMemoryStatus.LOCKED
    }
    check(locked) { "second native store must fail exclusive writer lock" }

    expectArgument {
        created.append(
            NativeMemoryKind.SEMANTIC,
            timestampNs = 30L,
            importance = 0.5f,
            source = "bad",
            content = "wrong dimension",
            vector = floatArrayOf(1f),
        )
    }
    created.close()

    val store = NativeMemoryStore.openForTest(
        file = file,
        recoverTornTail = true,
        clockNs = { 100L },
    )

    val semanticQuery = NativeMemoryQuery(
        vector = floatArrayOf(1f, 0f),
        topK = 2,
        kinds = listOf(NativeMemoryKind.SEMANTIC),
        semanticWeight = 1.0,
        recencyWeight = 0.0,
        importanceWeight = 0.0,
        recencyHalfLifeNs = 100L,
    )
    val semanticHits = store.retrieve(semanticQuery, topK = 2)
    check(semanticHits.size == 1)
    check(semanticHits.single().recordId == semanticId)
    check(semanticHits.single().source == "unit:α")
    check(semanticHits.single().content == "semantic fact ✓")
    check(semanticHits.single().semanticScore > 0.99)

    val cognition = NativeTypedCognitionAdapter(RetrievalInference(semanticId))
    val controller = NativePlanController.create(
        goal = "use memory",
        specs = listOf(
            NativePlanStepSpec(NativeStepKind.RETRIEVE, "retrieve fact"),
            NativePlanStepSpec(
                NativeStepKind.RESPOND,
                "answer",
                dependencies = listOf(1),
            ),
        ),
        createdNs = 50L,
    )
    val cognitionResult = NativeCognitionLoop(cognition).runUntilBoundary(
        controller = controller,
        memory = store,
    )
    check(cognitionResult.boundary == NativeCognitionBoundary.COMPLETED)
    check(cognitionResult.finalResponse == "final response")
    check(controller.plan.step(1).evidenceRecordIds == listOf(semanticId))
    check(controller.plan.step(2).evidenceRecordIds == listOf(semanticId))

    val importanceQuery = NativeMemoryQuery(
        vector = floatArrayOf(1f, 0f),
        topK = 2,
        kinds = null,
        semanticWeight = 0.0,
        recencyWeight = 0.0,
        importanceWeight = 1.0,
        recencyHalfLifeNs = 100L,
    )
    val ranked = store.retrieve(importanceQuery, topK = 2)
    check(ranked.map { it.recordId } == listOf(semanticId, episodicId))
    check(ranked[0].importanceScore > ranked[1].importanceScore)

    val wrongDimension = NativeMemoryQuery(
        vector = floatArrayOf(1f, 0f, 0f),
        topK = 1,
        kinds = null,
        semanticWeight = 1.0,
        recencyWeight = 0.0,
        importanceWeight = 0.0,
        recencyHalfLifeNs = 100L,
    )
    expectArgument { store.retrieve(wrongDimension, 1) }
    expectArgument { store.retrieve(semanticQuery, 1025) }

    val compacted = store.compact(
        NativeMemoryRetentionPolicy(maxRecords = 1L),
        nowNs = 100L,
    )
    check(compacted.retained >= 1L)
    check(compacted.removed >= 0L)
    check(store.stats().recordCount == compacted.retained)

    store.close()
    expectState { store.stats() }

    NativeMemoryStore.open(file).use { reopened ->
        check(reopened.vectorDim == 2)
        check(reopened.stats().recordCount == compacted.retained)
    }

    root.deleteRecursively()
    println("M7U_NATIVE_MEMORY_BRIDGE_PASS")
}
