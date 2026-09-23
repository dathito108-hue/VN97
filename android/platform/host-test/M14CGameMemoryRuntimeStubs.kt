package ai.vn97.runtime

enum class NativeMemoryKind {
    EPISODIC,
    SEMANTIC,
}

data class NativeMemoryQuery(
    val vector: FloatArray,
    val topK: Int,
    val kinds: List<NativeMemoryKind>?,
    val semanticWeight: Double,
    val recencyWeight: Double,
    val importanceWeight: Double,
    val recencyHalfLifeNs: Long,
)

data class NativeMemoryContextItem(
    val recordId: Long,
    val content: String,
    val source: String,
    val score: Double,
    val semanticScore: Double,
    val recencyScore: Double,
    val importanceScore: Double,
)

open class NativeMemoryStore {
    open val vectorDim: Int = 2

    open fun append(
        kind: NativeMemoryKind,
        timestampNs: Long,
        importance: Float,
        source: String,
        content: String,
        vector: FloatArray? = null,
        parentId: Long = 0L,
        durable: Boolean = true,
    ): Long = error("host stub")

    open fun retrieve(
        query: NativeMemoryQuery,
        topK: Int,
    ): List<NativeMemoryContextItem> = error("host stub")
}

open class NativeCognitionInferenceEngine {
    open fun embedText(
        text: String,
        vectorDim: Int,
    ): FloatArray = error("host stub")

    open fun generateText(
        prompt: String,
        maxNewTokens: Int,
    ): String = error("host stub")
}
