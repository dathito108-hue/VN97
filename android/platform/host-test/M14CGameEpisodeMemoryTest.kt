package ai.vn97.platform

import ai.vn97.runtime.NativeMemoryContextItem
import ai.vn97.runtime.NativeMemoryKind
import ai.vn97.runtime.NativeMemoryQuery

private data class FakeGameMemoryRecord(
    val id: Long,
    val kind: NativeMemoryKind,
    val source: String,
    val content: String,
    val parentId: Long,
)

private class FakeGameMemory : VN97GameMemoryPort {
    override val vectorDim: Int = 2
    private var nextId = 10L
    val appended = mutableListOf<FakeGameMemoryRecord>()
    var retrieval = emptyList<NativeMemoryContextItem>()
    var lastQuery: NativeMemoryQuery? = null

    override fun append(
        kind: NativeMemoryKind,
        timestampNs: Long,
        importance: Float,
        source: String,
        content: String,
        vector: FloatArray,
        parentId: Long,
    ): Long {
        check(timestampNs >= 0L)
        check(importance in 0f..1f)
        check(vector.size == vectorDim)
        val id = nextId++
        appended += FakeGameMemoryRecord(
            id = id,
            kind = kind,
            source = source,
            content = content,
            parentId = parentId,
        )
        return id
    }

    override fun retrieve(
        query: NativeMemoryQuery,
        topK: Int,
    ): List<NativeMemoryContextItem> {
        check(topK == 16)
        lastQuery = query
        return retrieval
    }
}

private inline fun expectFailure(block: () -> Unit) {
    var failed = false
    try {
        block()
    } catch (_: IllegalArgumentException) {
        failed = true
    } catch (_: IllegalStateException) {
        failed = true
    }
    check(failed)
}

fun main() {
    val memory = FakeGameMemory().apply {
        retrieval = listOf(
            NativeMemoryContextItem(
                recordId = 1L,
                content =
                    """{"package":"com.example.game","strategy":"prefer verified center taps"}""",
                source = VN97GameEpisodeMemory.STRATEGY_SOURCE,
                score = 0.9,
                semanticScore = 0.9,
                recencyScore = 0.5,
                importanceScore = 0.9,
            ),
            NativeMemoryContextItem(
                recordId = 2L,
                content =
                    """{"package":"com.other.game","strategy":"irrelevant"}""",
                source = VN97GameEpisodeMemory.STRATEGY_SOURCE,
                score = 0.8,
                semanticScore = 0.8,
                recencyScore = 0.5,
                importanceScore = 0.8,
            ),
            NativeMemoryContextItem(
                recordId = 3L,
                content =
                    """{"package":"com.example.game","strategy":"wrong source"}""",
                source = "other.source",
                score = 0.7,
                semanticScore = 0.7,
                recencyScore = 0.5,
                importanceScore = 0.7,
            ),
        )
    }
    val embedded = mutableListOf<String>()
    val summarized = mutableListOf<String>()
    val episode = VN97GameEpisodeMemory(
        memory = memory,
        embedder = VN97GameMemoryEmbedder { text, vectorDim ->
            check(vectorDim == 2)
            embedded += text
            floatArrayOf(1f, 0f)
        },
        summarizer = VN97GameMemorySummarizer { prompt ->
            summarized += prompt
            "Prefer verified center taps and re-observe after every action."
        },
        packageName = "com.example.game",
        userGoal = "reach the next checkpoint",
        startedNs = 100L,
    )

    val context = episode.begin()
    check(context.rootRecordId == 10L)
    check(
        "prefer verified center taps" in
            context.priorStrategyContext
    )
    check("irrelevant" !in context.priorStrategyContext)
    check(memory.appended.single().source ==
        VN97GameEpisodeMemory.EPISODE_SOURCE)
    check(memory.lastQuery?.kinds ==
        listOf(NativeMemoryKind.SEMANTIC))

    val actionId = episode.recordAction(
        actionIndex = 0,
        capabilityId =
            M6AndroidProductionCapabilities.GAME_TAP_CAPABILITY,
        actionResult = "game:tap:com.example.game",
        beforeObservation = "button visible",
        afterObservation = "checkpoint visible",
        verification = "tap advanced the game",
        timestampNs = 200L,
    )
    check(actionId == 11L)
    check(episode.recordedActions == 1)
    check(memory.appended.last().parentId ==
        context.rootRecordId)
    check(memory.appended.last().source ==
        VN97GameEpisodeMemory.ACTION_SOURCE)

    expectFailure {
        episode.recordAction(
            actionIndex = 2,
            capabilityId =
                M6AndroidProductionCapabilities.GAME_BACK_CAPABILITY,
            actionResult = "game:back:com.example.game",
            beforeObservation = "before",
            afterObservation = "after",
            verification = "verified",
            timestampNs = 201L,
        )
    }

    val result = episode.finish(
        terminal = VN97GameEpisodeTerminal.COMPLETED,
        detail = "checkpoint reached",
        timestampNs = 300L,
    )
    check(result.terminalRecordId == 12L)
    check(result.strategyRecordId == 13L)
    check("verified center taps" in result.learnedStrategy)
    check(
        memory.appended.map { it.source } ==
            listOf(
                VN97GameEpisodeMemory.EPISODE_SOURCE,
                VN97GameEpisodeMemory.ACTION_SOURCE,
                VN97GameEpisodeMemory.TERMINAL_SOURCE,
                VN97GameEpisodeMemory.STRATEGY_SOURCE,
            )
    )
    check(memory.appended[2].parentId == context.rootRecordId)
    check(memory.appended[3].parentId == context.rootRecordId)
    check(summarized.single().contains("VN97GAMELEARN1"))
    check(summarized.single().contains("tap advanced the game"))
    check(embedded.isNotEmpty())

    expectFailure {
        episode.finish(
            terminal = VN97GameEpisodeTerminal.COMPLETED,
            detail = "duplicate",
            timestampNs = 301L,
        )
    }

    println("M14C_GAME_EPISODE_MEMORY_PASS")
}
