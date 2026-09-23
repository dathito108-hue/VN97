package ai.vn97.platform

import ai.vn97.runtime.NativeCognitionInferenceEngine
import ai.vn97.runtime.NativeMemoryContextItem
import ai.vn97.runtime.NativeMemoryKind
import ai.vn97.runtime.NativeMemoryQuery
import ai.vn97.runtime.NativeMemoryStore
import java.nio.charset.StandardCharsets
import java.security.MessageDigest

enum class VN97GameEpisodeTerminal {
    COMPLETED,
    FAILED,
    CANCELLED,
}

data class VN97GameEpisodeMemoryContext(
    val episodeKey: String,
    val rootRecordId: Long,
    val priorStrategyContext: String,
) {
    init {
        require(
            episodeKey.length == 64 &&
                episodeKey.all { it in "0123456789abcdef" }
        )
        require(rootRecordId > 0L)
        require(
            priorStrategyContext.toByteArray(StandardCharsets.UTF_8).size <=
                MAX_PRIOR_CONTEXT_BYTES
        )
    }

    companion object {
        const val MAX_PRIOR_CONTEXT_BYTES = 8 * 1024
    }
}

data class VN97GameEpisodeMemoryResult(
    val terminalRecordId: Long,
    val strategyRecordId: Long,
    val learnedStrategy: String,
) {
    init {
        require(terminalRecordId > 0L)
        require(strategyRecordId > 0L)
        require(learnedStrategy.isNotBlank())
    }
}

internal interface VN97GameMemoryPort {
    val vectorDim: Int

    fun append(
        kind: NativeMemoryKind,
        timestampNs: Long,
        importance: Float,
        source: String,
        content: String,
        vector: FloatArray,
        parentId: Long,
    ): Long

    fun retrieve(
        query: NativeMemoryQuery,
        topK: Int,
    ): List<NativeMemoryContextItem>
}

internal class NativeVN97GameMemoryPort(
    private val store: NativeMemoryStore,
) : VN97GameMemoryPort {
    override val vectorDim: Int
        get() = store.vectorDim

    override fun append(
        kind: NativeMemoryKind,
        timestampNs: Long,
        importance: Float,
        source: String,
        content: String,
        vector: FloatArray,
        parentId: Long,
    ): Long = store.append(
        kind = kind,
        timestampNs = timestampNs,
        importance = importance,
        source = source,
        content = content,
        vector = vector,
        parentId = parentId,
        durable = true,
    )

    override fun retrieve(
        query: NativeMemoryQuery,
        topK: Int,
    ): List<NativeMemoryContextItem> =
        store.retrieve(query, topK)
}

internal fun interface VN97GameMemoryEmbedder {
    fun embed(text: String, vectorDim: Int): FloatArray
}

internal fun interface VN97GameMemorySummarizer {
    fun summarize(prompt: String): String
}

class VN97GameEpisodeMemory internal constructor(
    private val memory: VN97GameMemoryPort,
    private val embedder: VN97GameMemoryEmbedder,
    private val summarizer: VN97GameMemorySummarizer,
    val packageName: String,
    val userGoal: String,
    val startedNs: Long,
) {
    private var context: VN97GameEpisodeMemoryContext? = null
    private var actionCount = 0
    private var finished = false
    private val recentOutcomes = ArrayDeque<String>()

    val recordedActions: Int
        @Synchronized get() = actionCount

    val episodeKey: String = sha256Hex(
        buildString {
            append("VN97GAMEEP1\n")
            append(packageName)
            append('\n')
            append(startedNs)
            append('\n')
            append(userGoal)
        }.toByteArray(StandardCharsets.UTF_8)
    )

    init {
        require(isCanonicalPackage(packageName)) {
            "game episode package is invalid"
        }
        require(userGoal.isNotBlank()) {
            "game episode goal must not be blank"
        }
        require(
            userGoal.toByteArray(StandardCharsets.UTF_8).size <=
                MAX_GOAL_BYTES
        ) {
            "game episode goal exceeds byte bound"
        }
        require(startedNs >= 0L) {
            "game episode startedNs must be non-negative"
        }
        require(memory.vectorDim > 0) {
            "game episode memory vector dimension must be positive"
        }
    }

    @Synchronized
    fun begin(): VN97GameEpisodeMemoryContext {
        check(context == null) {
            "game episode memory already began"
        }
        check(!finished) {
            "game episode memory is already finished"
        }

        val prior = recallPriorStrategies()
        val content = buildString {
            append("{\"episode_key\":")
            appendJson(episodeKey)
            append(",\"goal\":")
            appendJson(userGoal)
            append(",\"package\":")
            appendJson(packageName)
            append(",\"prior_strategy_count\":")
            append(if (prior.isBlank()) 0 else prior.count { it == '\n' } + 1)
            append(",\"started_ns\":")
            append(startedNs)
            append('}')
        }
        val rootId = appendMemory(
            kind = NativeMemoryKind.EPISODIC,
            importance = 0.85f,
            source = EPISODE_SOURCE,
            content = content,
            parentId = 0L,
            timestampNs = startedNs,
        )
        return VN97GameEpisodeMemoryContext(
            episodeKey = episodeKey,
            rootRecordId = rootId,
            priorStrategyContext = prior,
        ).also {
            context = it
        }
    }

    @Synchronized
    fun recordAction(
        actionIndex: Int,
        capabilityId: String,
        actionResult: String,
        beforeObservation: String,
        afterObservation: String,
        verification: String,
        timestampNs: Long,
    ): Long {
        val active = checkNotNull(context) {
            "game episode memory has not begun"
        }
        check(!finished) {
            "game episode memory is already finished"
        }
        require(actionIndex == actionCount) {
            "game action memory index is not contiguous"
        }
        require(capabilityId in GAME_CAPABILITIES) {
            "game action capability is not allowed"
        }
        require(timestampNs >= startedNs) {
            "game action timestamp precedes episode"
        }
        require(actionResult.isNotBlank())
        require(beforeObservation.isNotBlank())
        require(afterObservation.isNotBlank())
        require(verification.isNotBlank())

        val boundedBefore =
            boundedField(beforeObservation, MAX_OBSERVATION_CHARS)
        val boundedAfter =
            boundedField(afterObservation, MAX_OBSERVATION_CHARS)
        val boundedVerification =
            boundedField(verification, MAX_VERIFICATION_CHARS)
        val boundedResult =
            boundedField(actionResult, MAX_ACTION_RESULT_CHARS)

        val content = buildString {
            append("{\"action_index\":")
            append(actionIndex)
            append(",\"action_result\":")
            appendJson(boundedResult)
            append(",\"after\":")
            appendJson(boundedAfter)
            append(",\"before\":")
            appendJson(boundedBefore)
            append(",\"capability\":")
            appendJson(capabilityId)
            append(",\"episode_key\":")
            appendJson(episodeKey)
            append(",\"package\":")
            appendJson(packageName)
            append(",\"timestamp_ns\":")
            append(timestampNs)
            append(",\"verification\":")
            appendJson(boundedVerification)
            append('}')
        }
        val recordId = appendMemory(
            kind = NativeMemoryKind.EPISODIC,
            importance = 0.9f,
            source = ACTION_SOURCE,
            content = content,
            parentId = active.rootRecordId,
            timestampNs = timestampNs,
        )

        actionCount += 1
        recentOutcomes.addLast(
            buildString {
                append('#')
                append(actionIndex)
                append(' ')
                append(capabilityId)
                append(": ")
                append(boundedVerification)
            }
        )
        while (recentOutcomes.size > MAX_RECENT_OUTCOMES) {
            recentOutcomes.removeFirst()
        }
        return recordId
    }

    @Synchronized
    fun finish(
        terminal: VN97GameEpisodeTerminal,
        detail: String,
        timestampNs: Long,
    ): VN97GameEpisodeMemoryResult {
        val active = checkNotNull(context) {
            "game episode memory has not begun"
        }
        check(!finished) {
            "game episode memory is already finished"
        }
        require(timestampNs >= startedNs) {
            "game terminal timestamp precedes episode"
        }
        require(detail.isNotBlank()) {
            "game terminal detail must not be blank"
        }

        val terminalDetail =
            boundedField(detail, MAX_TERMINAL_DETAIL_CHARS)
        val terminalContent = buildString {
            append("{\"action_count\":")
            append(actionCount)
            append(",\"detail\":")
            appendJson(terminalDetail)
            append(",\"episode_key\":")
            appendJson(episodeKey)
            append(",\"package\":")
            appendJson(packageName)
            append(",\"terminal\":")
            appendJson(terminal.name)
            append(",\"timestamp_ns\":")
            append(timestampNs)
            append('}')
        }
        val terminalId = appendMemory(
            kind = NativeMemoryKind.EPISODIC,
            importance =
                if (terminal == VN97GameEpisodeTerminal.COMPLETED) {
                    0.95f
                } else {
                    0.8f
                },
            source = TERMINAL_SOURCE,
            content = terminalContent,
            parentId = active.rootRecordId,
            timestampNs = timestampNs,
        )

        val learned = summarizer
            .summarize(
                buildStrategyPrompt(
                    terminal = terminal,
                    terminalDetail = terminalDetail,
                    prior = active.priorStrategyContext,
                )
            )
            .trim()
            .ifEmpty {
                "Outcome ${terminal.name.lowercase()} after $actionCount governed actions."
            }
            .let { boundedField(it, MAX_STRATEGY_CHARS) }

        val strategyContent = buildString {
            append("{\"action_count\":")
            append(actionCount)
            append(",\"episode_key\":")
            appendJson(episodeKey)
            append(",\"goal\":")
            appendJson(boundedField(userGoal, MAX_GOAL_SUMMARY_CHARS))
            append(",\"package\":")
            appendJson(packageName)
            append(",\"strategy\":")
            appendJson(learned)
            append(",\"terminal\":")
            appendJson(terminal.name)
            append('}')
        }
        val strategyId = appendMemory(
            kind = NativeMemoryKind.SEMANTIC,
            importance =
                if (terminal == VN97GameEpisodeTerminal.COMPLETED) {
                    0.95f
                } else {
                    0.75f
                },
            source = STRATEGY_SOURCE,
            content = strategyContent,
            parentId = active.rootRecordId,
            timestampNs = timestampNs,
        )
        finished = true
        return VN97GameEpisodeMemoryResult(
            terminalRecordId = terminalId,
            strategyRecordId = strategyId,
            learnedStrategy = learned,
        )
    }

    private fun recallPriorStrategies(): String {
        val queryText = buildString {
            append("VN97 game strategy for package ")
            append(packageName)
            append(" goal ")
            append(userGoal)
        }
        val query = NativeMemoryQuery(
            vector = embedder.embed(
                queryText,
                memory.vectorDim,
            ),
            topK = MAX_RETRIEVAL_HITS,
            kinds = listOf(NativeMemoryKind.SEMANTIC),
            semanticWeight = 0.8,
            recencyWeight = 0.1,
            importanceWeight = 0.1,
            recencyHalfLifeNs = STRATEGY_HALF_LIFE_NS,
        )
        val packageMarker = "\"package\":\"$packageName\""
        return memory.retrieve(query, MAX_RETRIEVAL_HITS)
            .asSequence()
            .filter { it.source == STRATEGY_SOURCE }
            .filter { packageMarker in it.content }
            .take(MAX_PRIOR_STRATEGIES)
            .map {
                boundedField(
                    it.content,
                    MAX_PRIOR_ITEM_CHARS,
                )
            }
            .joinToString("\n")
            .let {
                boundedUtf8(
                    it,
                    VN97GameEpisodeMemoryContext
                        .MAX_PRIOR_CONTEXT_BYTES,
                )
            }
    }

    private fun appendMemory(
        kind: NativeMemoryKind,
        importance: Float,
        source: String,
        content: String,
        parentId: Long,
        timestampNs: Long,
    ): Long {
        val bounded = boundedUtf8(
            content,
            MAX_MEMORY_CONTENT_BYTES,
        )
        return memory.append(
            kind = kind,
            timestampNs = timestampNs,
            importance = importance,
            source = source,
            content = bounded,
            vector = embedder.embed(
                bounded,
                memory.vectorDim,
            ),
            parentId = parentId,
        )
    }

    private fun buildStrategyPrompt(
        terminal: VN97GameEpisodeTerminal,
        terminalDetail: String,
        prior: String,
    ): String = buildString {
        append("VN97GAMELEARN1\n")
        append("Summarize reusable game strategy learned from this episode. ")
        append("Do not invent unseen facts. Focus on observed action/outcome ")
        append("patterns, mistakes, successful tactics, and what should change ")
        append("next episode. Return concise plain UTF-8 text, not JSON.\n")
        append("package=")
        append(packageName)
        append("\ngoal=")
        append(boundedField(userGoal, MAX_GOAL_SUMMARY_CHARS))
        append("\nterminal=")
        append(terminal.name)
        append("\naction_count=")
        append(actionCount)
        append("\nterminal_detail=")
        append(terminalDetail)
        if (prior.isNotBlank()) {
            append("\nprior_strategy=")
            append(prior)
        }
        if (recentOutcomes.isNotEmpty()) {
            append("\nrecent_outcomes=")
            append(recentOutcomes.joinToString(" | "))
        }
        append("\nlearned_strategy=")
    }.let {
        boundedUtf8(it, MAX_SUMMARY_PROMPT_BYTES)
    }

    companion object {
        fun production(
            engine: NativeCognitionInferenceEngine,
            memory: NativeMemoryStore,
            packageName: String,
            userGoal: String,
            startedNs: Long,
        ): VN97GameEpisodeMemory =
            VN97GameEpisodeMemory(
                memory = NativeVN97GameMemoryPort(memory),
                embedder = VN97GameMemoryEmbedder {
                        text,
                        vectorDim,
                    ->
                    engine.embedText(text, vectorDim)
                },
                summarizer = VN97GameMemorySummarizer { prompt ->
                    engine.generateText(
                        prompt,
                        MAX_SUMMARY_TOKENS,
                    )
                },
                packageName = packageName,
                userGoal = userGoal,
                startedNs = startedNs,
            )

        const val EPISODE_SOURCE = "vn97.game.episode"
        const val ACTION_SOURCE = "vn97.game.action"
        const val TERMINAL_SOURCE = "vn97.game.terminal"
        const val STRATEGY_SOURCE = "vn97.game.strategy"

        private const val MAX_GOAL_BYTES = 32 * 1024
        private const val MAX_MEMORY_CONTENT_BYTES = 12 * 1024
        private const val MAX_SUMMARY_PROMPT_BYTES = 16 * 1024
        private const val MAX_SUMMARY_TOKENS = 384
        private const val MAX_RETRIEVAL_HITS = 16
        private const val MAX_PRIOR_STRATEGIES = 4
        private const val MAX_PRIOR_ITEM_CHARS = 1536
        private const val MAX_RECENT_OUTCOMES = 8
        private const val MAX_OBSERVATION_CHARS = 1536
        private const val MAX_VERIFICATION_CHARS = 2048
        private const val MAX_ACTION_RESULT_CHARS = 512
        private const val MAX_TERMINAL_DETAIL_CHARS = 2048
        private const val MAX_GOAL_SUMMARY_CHARS = 2048
        private const val MAX_STRATEGY_CHARS = 4096
        private const val STRATEGY_HALF_LIFE_NS =
            30L * 24L * 60L * 60L * 1_000_000_000L

        private val GAME_CAPABILITIES = setOf(
            "device.game.tap",
            "device.game.swipe",
            "device.game.multitouch",
            "device.game.back",
        )
    }
}

private val GAME_MEMORY_PACKAGE_RE =
    Regex("^[A-Za-z][A-Za-z0-9_]*(?:\\.[A-Za-z][A-Za-z0-9_]*)+$")

private fun isCanonicalPackage(value: String): Boolean =
    value.isNotEmpty() &&
        value.all { it.code <= 0x7f } &&
        GAME_MEMORY_PACKAGE_RE.matches(value)

private fun boundedField(
    value: String,
    maxChars: Int,
): String =
    value
        .replace('\u0000', ' ')
        .take(maxChars)

private fun boundedUtf8(
    value: String,
    maxBytes: Int,
): String {
    require(maxBytes > 0)
    if (
        value.toByteArray(StandardCharsets.UTF_8).size <=
        maxBytes
    ) {
        return value
    }
    val out = StringBuilder()
    var bytes = 0
    var index = 0
    while (index < value.length) {
        val ch = value[index]
        val piece =
            if (
                Character.isHighSurrogate(ch) &&
                index + 1 < value.length &&
                Character.isLowSurrogate(value[index + 1])
            ) {
                value.substring(index, index + 2)
            } else {
                ch.toString()
            }
        val encoded =
            piece.toByteArray(StandardCharsets.UTF_8)
        if (bytes + encoded.size > maxBytes) break
        out.append(piece)
        bytes += encoded.size
        index += piece.length
    }
    return out.toString()
}

private fun StringBuilder.appendJson(value: String) {
    append('"')
    var index = 0
    while (index < value.length) {
        val ch = value[index]
        when (ch) {
            '"' -> append("\\\"")
            '\\' -> append("\\\\")
            '\b' -> append("\\b")
            '\u000c' -> append("\\f")
            '\n' -> append("\\n")
            '\r' -> append("\\r")
            '\t' -> append("\\t")
            else -> when {
                ch.code < 0x20 ->
                    append("\\u%04x".format(ch.code))

                Character.isHighSurrogate(ch) -> {
                    require(
                        index + 1 < value.length &&
                            Character.isLowSurrogate(
                                value[index + 1]
                            )
                    ) {
                        "game memory text contains unpaired surrogate"
                    }
                    append(ch)
                    append(value[++index])
                }

                Character.isLowSurrogate(ch) ->
                    throw IllegalArgumentException(
                        "game memory text contains unpaired surrogate"
                    )

                else -> append(ch)
            }
        }
        index += 1
    }
    append('"')
}

private fun sha256Hex(bytes: ByteArray): String =
    MessageDigest.getInstance("SHA-256")
        .digest(bytes)
        .joinToString("") {
            "%02x".format(it.toInt() and 0xff)
        }
