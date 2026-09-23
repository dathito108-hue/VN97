package ai.vn97.runtime

import java.io.File
import java.nio.ByteBuffer
import java.nio.charset.CodingErrorAction
import java.nio.charset.StandardCharsets

enum class NativeMemoryStatus(val code: Int) {
    OK(0),
    NULL_ARGUMENT(1),
    BLOB_TOO_SHORT(2),
    BAD_MAGIC(3),
    UNSUPPORTED_VERSION(4),
    INVALID_HEADER(5),
    INVALID_FRAME(6),
    CHECKSUM_MISMATCH(7),
    INVALID_RECORD(8),
    TRUNCATED_TAIL(9),
    INVALID_PARAMETER(10),
    OUTPUT_TOO_SMALL(11),
    IO_ERROR(12),
    LOCKED(13);

    companion object {
        internal fun fromCode(code: Int): NativeMemoryStatus =
            entries.firstOrNull { it.code == code }
                ?: throw IllegalStateException("unknown native memory status: $code")
    }
}

class NativeMemoryException(
    val status: NativeMemoryStatus,
    operation: String,
) : IllegalStateException("$operation failed: ${status.name} (${status.code})")

data class NativeMemoryStoreStats(
    val vectorDim: Int,
    val recordCount: Long,
    val lastRecordId: Long,
    val validBytes: Long,
) {
    init {
        require(vectorDim >= 0) { "vectorDim must be non-negative" }
        require(recordCount >= 0L) { "recordCount must be non-negative" }
        require(lastRecordId >= 0L) { "lastRecordId must be non-negative" }
        require(validBytes >= 0L) { "validBytes must be non-negative" }
    }
}

data class NativeMemoryRecord(
    val recordId: Long,
    val timestampNs: Long,
    val parentId: Long,
    val kind: NativeMemoryKind,
    val importance: Float,
    val source: String,
    val content: String,
) {
    init {
        require(recordId > 0L) { "recordId must be positive" }
        require(timestampNs >= 0L) { "timestampNs must be non-negative" }
        require(parentId >= 0L) { "parentId must be non-negative" }
        require(importance.isFinite() && importance in 0.0f..1.0f) {
            "importance must be finite and in [0, 1]"
        }
    }
}

data class NativeMemoryRetentionPolicy(
    val maxRecords: Long = 0L,
    val maxAgeNs: Long = 0L,
    val minImportance: Float = 0.0f,
) {
    init {
        require(maxRecords >= 0L) { "maxRecords must be non-negative" }
        require(maxAgeNs >= 0L) { "maxAgeNs must be non-negative" }
        require(minImportance.isFinite() && minImportance in 0.0f..1.0f) {
            "minImportance must be finite and in [0, 1]"
        }
    }
}

data class NativeMemoryCompactionResult(
    val retained: Long,
    val removed: Long,
) {
    init {
        require(retained >= 0L && removed >= 0L) {
            "compaction counts must be non-negative"
        }
    }
}

internal object NativeMemoryBindings {
    init {
        System.loadLibrary("vn97_jni")
    }

    external fun nativeMemoryCreate(
        path: String,
        vectorDim: Int,
        overwrite: Boolean,
        handleOut: LongArray,
    ): Int

    external fun nativeMemoryOpen(
        path: String,
        recoverTornTail: Boolean,
        handleOut: LongArray,
    ): Int

    external fun nativeMemoryClose(handle: Long): Int

    external fun nativeMemoryStats(
        handle: Long,
        intsOut: IntArray,
        longsOut: LongArray,
    ): Int

    external fun nativeMemoryAppend(
        handle: Long,
        kind: Int,
        timestampNs: Long,
        importance: Float,
        source: ByteArray,
        content: ByteArray,
        vector: FloatArray?,
        parentId: Long,
        durable: Boolean,
        recordIdOut: LongArray,
    ): Int

    external fun nativeMemoryRetrieve(
        handle: Long,
        query: FloatArray,
        topK: Int,
        kindMask: Int,
        semanticWeight: Float,
        recencyWeight: Float,
        importanceWeight: Float,
        nowNs: Long,
        recencyHalfLifeNs: Long,
        recordIdsOut: LongArray,
        scoresOut: FloatArray,
        semanticScoresOut: FloatArray,
        recencyScoresOut: FloatArray,
        importanceScoresOut: FloatArray,
        countOut: IntArray,
    ): Int

    external fun nativeMemoryRecordInfo(
        handle: Long,
        recordId: Long,
        intsOut: IntArray,
        longsOut: LongArray,
        floatsOut: FloatArray,
    ): Int

    external fun nativeMemoryRecordRead(
        handle: Long,
        recordId: Long,
        sourceOut: ByteArray,
        contentOut: ByteArray,
    ): Int

    external fun nativeMemoryCompact(
        handle: Long,
        maxRecords: Long,
        maxAgeNs: Long,
        minImportance: Float,
        nowNs: Long,
        countsOut: LongArray,
    ): Int
}

class NativeMemoryStore private constructor(
    private var handle: Long,
    private val clockNs: () -> Long,
) : NativeMemoryRetriever, AutoCloseable {
    private val lock = Any()

    companion object {
        private const val MAX_TOP_K = 1024

        fun create(
            file: File,
            vectorDim: Int,
            overwrite: Boolean = false,
        ): NativeMemoryStore {
            require(vectorDim > 0) { "vectorDim must be positive" }
            val out = LongArray(1)
            checkStatus(
                NativeMemoryBindings.nativeMemoryCreate(
                    file.absolutePath,
                    vectorDim,
                    overwrite,
                    out,
                ),
                "memory create",
            )
            check(out[0] > 0L) { "native memory returned an invalid handle" }
            return NativeMemoryStore(out[0], ::wallClockNs).also { store ->
                check(store.vectorDim == vectorDim) {
                    "created memory vector dimension changed"
                }
            }
        }

        fun open(
            file: File,
            recoverTornTail: Boolean = true,
        ): NativeMemoryStore {
            val out = LongArray(1)
            checkStatus(
                NativeMemoryBindings.nativeMemoryOpen(
                    file.absolutePath,
                    recoverTornTail,
                    out,
                ),
                "memory open",
            )
            check(out[0] > 0L) { "native memory returned an invalid handle" }
            return NativeMemoryStore(out[0], ::wallClockNs)
        }

        internal fun openForTest(
            file: File,
            recoverTornTail: Boolean,
            clockNs: () -> Long,
        ): NativeMemoryStore {
            val out = LongArray(1)
            checkStatus(
                NativeMemoryBindings.nativeMemoryOpen(
                    file.absolutePath,
                    recoverTornTail,
                    out,
                ),
                "memory open",
            )
            check(out[0] > 0L) { "native memory returned an invalid handle" }
            return NativeMemoryStore(out[0], clockNs)
        }
    }

    override val vectorDim: Int
        get() = stats().vectorDim

    fun stats(): NativeMemoryStoreStats = withHandle { h ->
        val ints = IntArray(1)
        val longs = LongArray(3)
        checkStatus(
            NativeMemoryBindings.nativeMemoryStats(h, ints, longs),
            "memory stats",
        )
        NativeMemoryStoreStats(
            vectorDim = ints[0],
            recordCount = longs[0],
            lastRecordId = longs[1],
            validBytes = longs[2],
        )
    }

    fun append(
        kind: NativeMemoryKind,
        timestampNs: Long,
        importance: Float,
        source: String,
        content: String,
        vector: FloatArray? = null,
        parentId: Long = 0L,
        durable: Boolean = true,
    ): Long {
        require(timestampNs >= 0L) { "timestampNs must be non-negative" }
        require(parentId >= 0L) { "parentId must be non-negative" }
        require(importance.isFinite() && importance in 0.0f..1.0f) {
            "importance must be finite and in [0, 1]"
        }
        if (vector != null) {
            require(vector.size == vectorDim) {
                "memory vector dimension does not match store"
            }
            require(vector.all { it.isFinite() }) {
                "memory vector must contain only finite values"
            }
        }
        val out = LongArray(1)
        withHandle<Unit> { h ->
            checkStatus(
                NativeMemoryBindings.nativeMemoryAppend(
                    handle = h,
                    kind = kindCode(kind),
                    timestampNs = timestampNs,
                    importance = importance,
                    source = source.toByteArray(StandardCharsets.UTF_8),
                    content = content.toByteArray(StandardCharsets.UTF_8),
                    vector = vector,
                    parentId = parentId,
                    durable = durable,
                    recordIdOut = out,
                ),
                "memory append",
            )
        }
        check(out[0] > 0L) { "native memory returned an invalid record ID" }
        return out[0]
    }

    override fun retrieve(
        query: NativeMemoryQuery,
        topK: Int,
    ): List<NativeMemoryContextItem> {
        require(topK in 1..MAX_TOP_K) { "topK is outside bridge bound" }
        require(query.vector.size == vectorDim) {
            "memory query vector dimension does not match store"
        }
        require(query.vector.all { it.isFinite() }) {
            "memory query vector must contain only finite values"
        }
        val semantic = weightToFloat(query.semanticWeight, "semanticWeight")
        val recency = weightToFloat(query.recencyWeight, "recencyWeight")
        val importance = weightToFloat(query.importanceWeight, "importanceWeight")
        val nowNs = clockNs()
        require(nowNs >= 0L) { "memory clock returned a negative timestamp" }

        return withHandle { h ->
            val ids = LongArray(topK)
            val scores = FloatArray(topK)
            val semanticScores = FloatArray(topK)
            val recencyScores = FloatArray(topK)
            val importanceScores = FloatArray(topK)
            val count = IntArray(1)
            checkStatus(
                NativeMemoryBindings.nativeMemoryRetrieve(
                    handle = h,
                    query = query.vector,
                    topK = topK,
                    kindMask = kindMask(query.kinds),
                    semanticWeight = semantic,
                    recencyWeight = recency,
                    importanceWeight = importance,
                    nowNs = nowNs,
                    recencyHalfLifeNs = query.recencyHalfLifeNs,
                    recordIdsOut = ids,
                    scoresOut = scores,
                    semanticScoresOut = semanticScores,
                    recencyScoresOut = recencyScores,
                    importanceScoresOut = importanceScores,
                    countOut = count,
                ),
                "memory retrieve",
            )
            check(count[0] in 0..topK) { "native memory returned invalid hit count" }
            List(count[0]) { index ->
                val recordId = ids[index]
                check(recordId > 0L) { "native memory returned invalid record ID" }
                val record = readRecordForHandle(h, recordId)
                NativeMemoryContextItem(
                    recordId = recordId,
                    content = record.content,
                    source = record.source,
                    score = scores[index].toDouble(),
                    semanticScore = semanticScores[index].toDouble(),
                    recencyScore = recencyScores[index].toDouble(),
                    importanceScore = importanceScores[index].toDouble(),
                )
            }
        }
    }

    fun record(recordId: Long): NativeMemoryRecord {
        require(recordId > 0L) { "recordId must be positive" }
        return withHandle { h -> readRecordForHandle(h, recordId) }
    }

    fun compact(
        policy: NativeMemoryRetentionPolicy,
        nowNs: Long = clockNs(),
    ): NativeMemoryCompactionResult {
        require(nowNs >= 0L) { "nowNs must be non-negative" }
        val counts = LongArray(2)
        withHandle<Unit> { h ->
            checkStatus(
                NativeMemoryBindings.nativeMemoryCompact(
                    handle = h,
                    maxRecords = policy.maxRecords,
                    maxAgeNs = policy.maxAgeNs,
                    minImportance = policy.minImportance,
                    nowNs = nowNs,
                    countsOut = counts,
                ),
                "memory compact",
            )
        }
        return NativeMemoryCompactionResult(counts[0], counts[1])
    }

    override fun close() {
        val old = synchronized(lock) {
            val value = handle
            handle = 0L
            value
        }
        if (old != 0L) {
            val status = NativeMemoryStatus.fromCode(
                NativeMemoryBindings.nativeMemoryClose(old)
            )
            if (status != NativeMemoryStatus.OK &&
                status != NativeMemoryStatus.INVALID_PARAMETER
            ) {
                throw NativeMemoryException(status, "memory close")
            }
        }
    }

    private inline fun <T> withHandle(block: (Long) -> T): T = synchronized(lock) {
        check(handle > 0L) { "native memory store is closed" }
        block(handle)
    }

    private fun readRecordForHandle(
        handle: Long,
        recordId: Long,
    ): NativeMemoryRecord {
        val ints = IntArray(3)
        val longs = LongArray(2)
        val floats = FloatArray(1)
        checkStatus(
            NativeMemoryBindings.nativeMemoryRecordInfo(
                handle,
                recordId,
                ints,
                longs,
                floats,
            ),
            "memory record info",
        )
        check(ints[1] >= 0 && ints[2] >= 0) {
            "native memory returned invalid record byte lengths"
        }
        val source = ByteArray(ints[1])
        val content = ByteArray(ints[2])
        checkStatus(
            NativeMemoryBindings.nativeMemoryRecordRead(
                handle,
                recordId,
                source,
                content,
            ),
            "memory record read",
        )
        return NativeMemoryRecord(
            recordId = recordId,
            timestampNs = longs[0],
            parentId = longs[1],
            kind = memoryKindFromCode(ints[0]),
            importance = floats[0],
            source = strictUtf8(source),
            content = strictUtf8(content),
        )
    }
}

private fun memoryKindFromCode(code: Int): NativeMemoryKind = when (code) {
    1 -> NativeMemoryKind.EPISODIC
    2 -> NativeMemoryKind.SEMANTIC
    else -> throw IllegalStateException("native memory returned unknown kind: $code")
}

private fun kindCode(kind: NativeMemoryKind): Int = when (kind) {
    NativeMemoryKind.EPISODIC -> 1
    NativeMemoryKind.SEMANTIC -> 2
}

private fun kindMask(kinds: List<NativeMemoryKind>?): Int {
    if (kinds == null) return 0x3
    require(kinds.isNotEmpty()) { "memory query kinds must not be empty" }
    var mask = 0
    for (kind in kinds) {
        mask = mask or when (kind) {
            NativeMemoryKind.EPISODIC -> 0x1
            NativeMemoryKind.SEMANTIC -> 0x2
        }
    }
    return mask
}

private fun weightToFloat(value: Double, label: String): Float {
    require(value.isFinite() && value >= 0.0 && value <= Float.MAX_VALUE.toDouble()) {
        "$label is not representable by native float retrieval"
    }
    val converted = value.toFloat()
    require(value == 0.0 || converted > 0.0f) {
        "$label underflows native float retrieval"
    }
    return converted
}

private fun strictUtf8(bytes: ByteArray): String = try {
    StandardCharsets.UTF_8.newDecoder()
        .onMalformedInput(CodingErrorAction.REPORT)
        .onUnmappableCharacter(CodingErrorAction.REPORT)
        .decode(ByteBuffer.wrap(bytes))
        .toString()
} catch (exc: Exception) {
    throw IllegalStateException("native memory returned invalid UTF-8", exc)
}

private fun wallClockNs(): Long = Math.multiplyExact(
    System.currentTimeMillis(),
    1_000_000L,
)

private fun checkStatus(code: Int, operation: String) {
    val status = NativeMemoryStatus.fromCode(code)
    if (status != NativeMemoryStatus.OK) {
        throw NativeMemoryException(status, operation)
    }
}
