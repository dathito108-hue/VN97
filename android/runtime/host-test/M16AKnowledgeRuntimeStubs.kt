package ai.vn97.runtime

enum class NativeMemoryKind {
    EPISODIC,
    SEMANTIC,
}

data class NativeMemoryStoreStats(
    val lastRecordId: Long,
)

data class NativeMemoryRecord(
    val recordId: Long,
    val source: String,
    val content: String,
)

class NativeMemoryStore {
    val vectorDim: Int
        get() = error("host stub")

    fun stats(): NativeMemoryStoreStats =
        error("host stub")

    fun append(
        kind: NativeMemoryKind,
        timestampNs: Long,
        importance: Float,
        source: String,
        content: String,
        vector: FloatArray?,
        parentId: Long,
        durable: Boolean,
    ): Long = error("host stub")

    fun record(recordId: Long): NativeMemoryRecord =
        error("host stub")
}

interface NativeCognitionInference {
    fun embedText(
        text: String,
        vectorDim: Int,
    ): FloatArray
}
