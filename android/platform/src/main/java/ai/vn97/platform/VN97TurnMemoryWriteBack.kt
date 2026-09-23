package ai.vn97.platform

import ai.vn97.runtime.NativeMemoryKind
import ai.vn97.runtime.NativeMemoryRecord
import ai.vn97.runtime.NativeMemoryStore
import ai.vn97.runtime.NativeMemoryStoreStats
import ai.vn97.runtime.NativePlanStatus
import ai.vn97.runtime.NativeStepKind
import ai.vn97.runtime.NativeStepStatus
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import java.nio.channels.FileLock
import java.nio.charset.StandardCharsets
import java.nio.file.Files
import java.nio.file.LinkOption
import java.nio.file.OpenOption
import java.nio.file.StandardOpenOption
import java.security.MessageDigest

private val TURN_JOURNAL_MAGIC = "VN97TWJ1".toByteArray(StandardCharsets.US_ASCII)
private const val TURN_JOURNAL_VERSION = 1
private const val TURN_JOURNAL_HEADER_BYTES = 16
private const val TURN_JOURNAL_ENTRY_BYTES = 192
private const val TURN_ENTRY_HASH_OFFSET = 160
private const val DEFAULT_MAX_JOURNAL_ENTRIES = 200_000
private const val ZERO_SHA256 =
    "0000000000000000000000000000000000000000000000000000000000000000"

class VN97TurnMemoryIntegrityException(
    message: String,
    cause: Throwable? = null,
) : IllegalStateException(message, cause)

class VN97TurnMemoryCapacityException(message: String) : IllegalStateException(message)

class VN97TurnMemoryPendingException(
    val pendingTurnKey: String,
) : IllegalStateException("turn memory journal has unresolved pending turn: $pendingTurnKey")

internal interface VN97TurnMemoryBackend {
    val vectorDim: Int
    fun stats(): NativeMemoryStoreStats
    fun append(
        kind: NativeMemoryKind,
        timestampNs: Long,
        importance: Float,
        source: String,
        content: String,
        vector: FloatArray,
    ): Long
    fun record(recordId: Long): NativeMemoryRecord
}

internal class NativeVN97TurnMemoryBackend(
    private val store: NativeMemoryStore,
) : VN97TurnMemoryBackend {
    override val vectorDim: Int get() = store.vectorDim
    override fun stats(): NativeMemoryStoreStats = store.stats()
    override fun append(
        kind: NativeMemoryKind,
        timestampNs: Long,
        importance: Float,
        source: String,
        content: String,
        vector: FloatArray,
    ): Long = store.append(
        kind = kind,
        timestampNs = timestampNs,
        importance = importance,
        source = source,
        content = content,
        vector = vector,
        durable = true,
    )
    override fun record(recordId: Long): NativeMemoryRecord = store.record(recordId)
}

internal fun interface VN97TurnMemoryEmbedder {
    fun embed(text: String, vectorDim: Int): FloatArray
}

internal enum class TurnJournalPhase(val code: Int) {
    PREPARED(1), COMMITTED(2);
    companion object {
        fun fromCode(code: Int): TurnJournalPhase = entries.firstOrNull { it.code == code }
            ?: throw VN97TurnMemoryIntegrityException("unknown turn journal phase: $code")
    }
}

internal data class TurnJournalEntry(
    val sequence: Long,
    val phase: TurnJournalPhase,
    val turnKey: String,
    val expectedRecordId: Long,
    val recordId: Long,
    val sourceSha256: String,
    val contentSha256: String,
    val previousEntryHash: String,
    val entryHash: String,
)

internal data class TurnJournalCommitted(
    val turnKey: String,
    val recordId: Long,
    val sourceSha256: String,
    val contentSha256: String,
)

internal class VN97TurnMemoryJournal(
    root: File,
    fileName: String = "turn-memory.vn97twj1",
    private val recoverTornTail: Boolean = true,
    private val maxEntries: Int = DEFAULT_MAX_JOURNAL_ENTRIES,
) : AutoCloseable {
    private val rootPath = root.toPath()
    private val target = root.resolve(fileName).toPath()
    private val channel: FileChannel
    private val lock: FileLock
    private val committed = LinkedHashMap<String, TurnJournalCommitted>()
    private var pending: TurnJournalEntry? = null
    private var entryCount = 0
    private var previousHash = ZERO_SHA256

    init {
        require(isSafeFileName(fileName)) { "journal file name must be one bounded path component" }
        require(maxEntries > 0) { "maxEntries must be positive" }
        Files.createDirectories(rootPath)
        if (!Files.isDirectory(rootPath, LinkOption.NOFOLLOW_LINKS) || Files.isSymbolicLink(rootPath)) {
            throw VN97TurnMemoryIntegrityException("turn memory journal root must be a real directory")
        }
        if (Files.exists(target, LinkOption.NOFOLLOW_LINKS) && Files.isSymbolicLink(target)) {
            throw VN97TurnMemoryIntegrityException("turn memory journal target must not be a symlink")
        }
        channel = try {
            FileChannel.open(
                target,
                StandardOpenOption.CREATE,
                StandardOpenOption.READ,
                StandardOpenOption.WRITE,
                LinkOption.NOFOLLOW_LINKS,
            )
        } catch (exc: Exception) {
            throw VN97TurnMemoryIntegrityException("turn memory journal could not be opened", exc)
        }
        try {
            lock = channel.tryLock() ?: throw VN97TurnMemoryIntegrityException(
                "turn memory journal is already locked"
            )
            initializeOrLoad()
        } catch (exc: Throwable) {
            try { channel.close() } catch (_: Throwable) { }
            throw exc
        }
    }

    val pendingTurnKey: String?
        @Synchronized get() = pending?.turnKey

    @Synchronized
    fun committed(turnKey: String): TurnJournalCommitted? = committed[turnKey]

    @Synchronized
    fun pending(turnKey: String): TurnJournalEntry? = pending?.takeIf { it.turnKey == turnKey }

    @Synchronized
    fun prepare(
        turnKey: String,
        expectedRecordId: Long,
        sourceSha256: String,
        contentSha256: String,
    ) {
        requireSha(turnKey, "turnKey")
        requireSha(sourceSha256, "sourceSha256")
        requireSha(contentSha256, "contentSha256")
        require(expectedRecordId > 0L) { "expectedRecordId must be positive" }
        pending?.let { unresolved ->
            if (
                unresolved.turnKey == turnKey &&
                unresolved.expectedRecordId == expectedRecordId &&
                unresolved.sourceSha256 == sourceSha256 &&
                unresolved.contentSha256 == contentSha256
            ) return
            throw VN97TurnMemoryPendingException(unresolved.turnKey)
        }
        if (committed.containsKey(turnKey)) {
            throw VN97TurnMemoryIntegrityException("cannot prepare an already committed turn")
        }
        val entry = nextEntry(
            TurnJournalPhase.PREPARED,
            turnKey,
            expectedRecordId,
            0L,
            sourceSha256,
            contentSha256,
        )
        append(entry)
        pending = entry
    }

    @Synchronized
    fun commit(turnKey: String, recordId: Long) {
        requireSha(turnKey, "turnKey")
        require(recordId > 0L) { "recordId must be positive" }
        val prepared = pending ?: throw VN97TurnMemoryIntegrityException(
            "turn journal commit lacks PREPARED entry"
        )
        if (prepared.turnKey != turnKey || prepared.expectedRecordId != recordId) {
            throw VN97TurnMemoryIntegrityException("turn journal commit does not match PREPARED entry")
        }
        val entry = nextEntry(
            TurnJournalPhase.COMMITTED,
            prepared.turnKey,
            prepared.expectedRecordId,
            recordId,
            prepared.sourceSha256,
            prepared.contentSha256,
        )
        append(entry)
        committed[turnKey] = TurnJournalCommitted(
            turnKey,
            recordId,
            prepared.sourceSha256,
            prepared.contentSha256,
        )
        pending = null
    }

    @Synchronized
    override fun close() {
        try {
            if (lock.isValid) lock.release()
        } finally {
            channel.close()
        }
    }

    private fun initializeOrLoad() {
        var size = channel.size()
        val maxBytes = TURN_JOURNAL_HEADER_BYTES.toLong() +
            maxEntries.toLong() * TURN_JOURNAL_ENTRY_BYTES.toLong()
        if (size > maxBytes) {
            throw VN97TurnMemoryCapacityException("turn memory journal exceeds byte limit")
        }
        if (size == 0L) {
            writeHeader()
            channel.force(true)
            forceDirectory()
            size = TURN_JOURNAL_HEADER_BYTES.toLong()
        }
        if (size < TURN_JOURNAL_HEADER_BYTES) {
            throw VN97TurnMemoryIntegrityException("turn memory journal is shorter than header")
        }
        val header = ByteArray(TURN_JOURNAL_HEADER_BYTES)
        readFully(0L, header)
        val h = ByteBuffer.wrap(header).order(ByteOrder.LITTLE_ENDIAN)
        val magic = ByteArray(8).also(h::get)
        if (!magic.contentEquals(TURN_JOURNAL_MAGIC)) {
            throw VN97TurnMemoryIntegrityException("bad VN97TWJ1 magic")
        }
        if (h.int != TURN_JOURNAL_VERSION || h.int != TURN_JOURNAL_ENTRY_BYTES) {
            throw VN97TurnMemoryIntegrityException("unsupported VN97TWJ1 header")
        }

        val payload = size - TURN_JOURNAL_HEADER_BYTES
        val remainder = payload % TURN_JOURNAL_ENTRY_BYTES
        if (remainder != 0L) {
            if (!recoverTornTail) {
                throw VN97TurnMemoryIntegrityException("turn memory journal has torn final entry")
            }
            val recovered = size - remainder
            channel.truncate(recovered)
            channel.force(true)
            forceDirectory()
            size = recovered
        }
        val count = ((size - TURN_JOURNAL_HEADER_BYTES) / TURN_JOURNAL_ENTRY_BYTES).toInt()
        if (count > maxEntries) {
            throw VN97TurnMemoryCapacityException("turn memory journal entry limit reached")
        }
        var expectedSequence = 1L
        var offset = TURN_JOURNAL_HEADER_BYTES.toLong()
        repeat(count) {
            val bytes = ByteArray(TURN_JOURNAL_ENTRY_BYTES)
            readFully(offset, bytes)
            val entry = decodeEntry(bytes)
            if (entry.sequence != expectedSequence) {
                throw VN97TurnMemoryIntegrityException("turn memory journal sequence is not contiguous")
            }
            if (entry.previousEntryHash != previousHash) {
                throw VN97TurnMemoryIntegrityException("turn memory journal hash chain is broken")
            }
            applyLoadedEntry(entry)
            previousHash = entry.entryHash
            entryCount += 1
            expectedSequence += 1L
            offset += TURN_JOURNAL_ENTRY_BYTES
        }
    }

    private fun applyLoadedEntry(entry: TurnJournalEntry) {
        when (entry.phase) {
            TurnJournalPhase.PREPARED -> {
                if (pending != null || committed.containsKey(entry.turnKey) || entry.recordId != 0L) {
                    throw VN97TurnMemoryIntegrityException("turn memory PREPARED entry is invalid")
                }
                pending = entry
            }
            TurnJournalPhase.COMMITTED -> {
                val prepared = pending ?: throw VN97TurnMemoryIntegrityException(
                    "turn memory COMMITTED entry lacks PREPARED entry"
                )
                if (
                    prepared.turnKey != entry.turnKey ||
                    prepared.expectedRecordId != entry.expectedRecordId ||
                    entry.recordId != entry.expectedRecordId ||
                    prepared.sourceSha256 != entry.sourceSha256 ||
                    prepared.contentSha256 != entry.contentSha256 ||
                    committed.containsKey(entry.turnKey)
                ) {
                    throw VN97TurnMemoryIntegrityException(
                        "turn memory COMMITTED entry does not match PREPARED entry"
                    )
                }
                committed[entry.turnKey] = TurnJournalCommitted(
                    entry.turnKey,
                    entry.recordId,
                    entry.sourceSha256,
                    entry.contentSha256,
                )
                pending = null
            }
        }
    }

    private fun nextEntry(
        phase: TurnJournalPhase,
        turnKey: String,
        expectedRecordId: Long,
        recordId: Long,
        sourceSha256: String,
        contentSha256: String,
    ): TurnJournalEntry {
        if (entryCount >= maxEntries) {
            throw VN97TurnMemoryCapacityException("turn memory journal entry limit reached")
        }
        val base = TurnJournalEntry(
            sequence = entryCount.toLong() + 1L,
            phase = phase,
            turnKey = turnKey,
            expectedRecordId = expectedRecordId,
            recordId = recordId,
            sourceSha256 = sourceSha256,
            contentSha256 = contentSha256,
            previousEntryHash = previousHash,
            entryHash = ZERO_SHA256,
        )
        val encoded = encodeEntry(base)
        val digest = sha256Hex(encoded.copyOfRange(0, TURN_ENTRY_HASH_OFFSET))
        return base.copy(entryHash = digest)
    }

    private fun append(entry: TurnJournalEntry) {
        val bytes = encodeEntry(entry)
        val expectedPosition = TURN_JOURNAL_HEADER_BYTES.toLong() +
            entryCount.toLong() * TURN_JOURNAL_ENTRY_BYTES.toLong()
        if (channel.size() != expectedPosition) {
            throw VN97TurnMemoryIntegrityException(
                "turn memory journal size changed outside this instance"
            )
        }
        channel.position(expectedPosition)
        val buffer = ByteBuffer.wrap(bytes)
        while (buffer.hasRemaining()) {
            if (channel.write(buffer) <= 0) {
                throw VN97TurnMemoryIntegrityException(
                    "turn memory journal append made no progress"
                )
            }
        }
        channel.force(true)
        entryCount += 1
        previousHash = entry.entryHash
    }

    private fun encodeEntry(entry: TurnJournalEntry): ByteArray {
        val bytes = ByteArray(TURN_JOURNAL_ENTRY_BYTES)
        val b = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        b.putLong(entry.sequence)
        b.putInt(entry.phase.code)
        b.putInt(0)
        b.put(hexToBytes(entry.turnKey))
        b.putLong(entry.expectedRecordId)
        b.putLong(entry.recordId)
        b.put(hexToBytes(entry.sourceSha256))
        b.put(hexToBytes(entry.contentSha256))
        b.put(hexToBytes(entry.previousEntryHash))
        if (entry.entryHash != ZERO_SHA256) b.put(hexToBytes(entry.entryHash))
        return bytes
    }

    private fun decodeEntry(bytes: ByteArray): TurnJournalEntry {
        val storedHash = bytes.copyOfRange(TURN_ENTRY_HASH_OFFSET, TURN_JOURNAL_ENTRY_BYTES)
        val expectedHash = MessageDigest.getInstance("SHA-256").digest(
            bytes.copyOfRange(0, TURN_ENTRY_HASH_OFFSET)
        )
        if (!storedHash.contentEquals(expectedHash)) {
            throw VN97TurnMemoryIntegrityException(
                "turn memory journal entry SHA-256 mismatch"
            )
        }
        val b = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        val sequence = b.long
        if (sequence <= 0L) {
            throw VN97TurnMemoryIntegrityException("turn memory sequence must be positive")
        }
        val phase = TurnJournalPhase.fromCode(b.int)
        if (b.int != 0) {
            throw VN97TurnMemoryIntegrityException("turn memory reserved field must be zero")
        }
        val turnKey = ByteArray(32).also(b::get).toLowerHex()
        val expectedRecordId = b.long
        val recordId = b.long
        val sourceSha = ByteArray(32).also(b::get).toLowerHex()
        val contentSha = ByteArray(32).also(b::get).toLowerHex()
        val previous = ByteArray(32).also(b::get).toLowerHex()
        val entryHash = ByteArray(32).also(b::get).toLowerHex()
        if (expectedRecordId <= 0L || recordId < 0L) {
            throw VN97TurnMemoryIntegrityException("turn memory record IDs are invalid")
        }
        return TurnJournalEntry(
            sequence,
            phase,
            turnKey,
            expectedRecordId,
            recordId,
            sourceSha,
            contentSha,
            previous,
            entryHash,
        )
    }

    private fun writeHeader() {
        val bytes = ByteBuffer.allocate(TURN_JOURNAL_HEADER_BYTES)
            .order(ByteOrder.LITTLE_ENDIAN)
            .put(TURN_JOURNAL_MAGIC)
            .putInt(TURN_JOURNAL_VERSION)
            .putInt(TURN_JOURNAL_ENTRY_BYTES)
            .array()
        channel.position(0L)
        val buffer = ByteBuffer.wrap(bytes)
        while (buffer.hasRemaining()) {
            if (channel.write(buffer) <= 0) {
                throw VN97TurnMemoryIntegrityException(
                    "turn memory journal header write made no progress"
                )
            }
        }
    }

    private fun readFully(position: Long, out: ByteArray) {
        channel.position(position)
        val buffer = ByteBuffer.wrap(out)
        while (buffer.hasRemaining()) {
            val read = channel.read(buffer)
            if (read < 0) {
                throw VN97TurnMemoryIntegrityException(
                    "turn memory journal ended unexpectedly"
                )
            }
            if (read == 0) {
                throw VN97TurnMemoryIntegrityException(
                    "turn memory journal read made no progress"
                )
            }
        }
    }

    private fun forceDirectory() {
        try {
            FileChannel.open(rootPath, StandardOpenOption.READ).use { it.force(true) }
        } catch (exc: Exception) {
            throw VN97TurnMemoryIntegrityException(
                "turn memory journal directory fsync failed",
                exc,
            )
        }
    }
}

class VN97TurnMemoryWriter internal constructor(
    private val backend: VN97TurnMemoryBackend,
    private val embedder: VN97TurnMemoryEmbedder,
    journalRoot: File,
    journalFileName: String = "turn-memory.vn97twj1",
    recoverTornTail: Boolean = true,
    maxJournalEntries: Int = DEFAULT_MAX_JOURNAL_ENTRIES,
    private val importance: Float = 0.75f,
) : AutoCloseable {
    private val journal = VN97TurnMemoryJournal(
        journalRoot,
        journalFileName,
        recoverTornTail,
        maxJournalEntries,
    )

    init {
        require(backend.vectorDim > 0) { "memory vector dimension must be positive" }
        require(importance.isFinite() && importance in 0.0f..1.0f) {
            "importance must be finite and in [0,1]"
        }
    }

    val pendingTurnKey: String?
        @Synchronized get() = journal.pendingTurnKey

    @Synchronized
    fun commitCompletedTurn(
        turn: VN97AssistantTurn,
        finalResponse: String,
        timestampNs: Long,
    ): Long {
        require(timestampNs >= 0L) { "timestampNs must be non-negative" }
        require(finalResponse.isNotEmpty()) { "finalResponse must not be empty" }
        val plan = turn.controller.plan
        check(plan.status == NativePlanStatus.COMPLETED) {
            "turn planner must be COMPLETED before memory write-back"
        }
        val canonicalResponse = plan.steps.asReversed().firstOrNull {
            it.spec.kind == NativeStepKind.RESPOND &&
                it.status == NativeStepStatus.SUCCEEDED
        }?.result.orEmpty()
        check(canonicalResponse == finalResponse) {
            "finalResponse does not match the completed canonical RESPOND step"
        }

        val turnKey = computeTurnKey(turn)
        val source = "vn97.assistant.turn.v1/$turnKey"
        val content = canonicalTurnContent(turn, finalResponse, turnKey)
        val sourceSha = sha256Hex(source.toByteArray(StandardCharsets.UTF_8))
        val contentSha = sha256Hex(content.toByteArray(StandardCharsets.UTF_8))

        journal.pendingTurnKey?.let {
            if (it != turnKey) throw VN97TurnMemoryPendingException(it)
        }

        journal.committed(turnKey)?.let { committed ->
            requireDigests(
                committed.sourceSha256,
                committed.contentSha256,
                sourceSha,
                contentSha,
            )
            verifyRecord(committed.recordId, source, content)
            return committed.recordId
        }

        val prepared = journal.pending(turnKey)
        val expected = if (prepared != null) {
            requireDigests(
                prepared.sourceSha256,
                prepared.contentSha256,
                sourceSha,
                contentSha,
            )
            prepared.expectedRecordId
        } else {
            val next = try {
                Math.addExact(backend.stats().lastRecordId, 1L)
            } catch (exc: ArithmeticException) {
                throw VN97TurnMemoryCapacityException("VN97MEM1 record ID exhausted")
            }
            journal.prepare(turnKey, next, sourceSha, contentSha)
            next
        }

        val recordId = if (backend.stats().lastRecordId < expected) {
            appendExpected(expected, timestampNs, source, content)
        } else {
            verifyRecord(expected, source, content)
            expected
        }
        journal.commit(turnKey, recordId)
        return recordId
    }

    @Synchronized
    override fun close() {
        journal.close()
    }

    private fun appendExpected(
        expectedRecordId: Long,
        timestampNs: Long,
        source: String,
        content: String,
    ): Long {
        val vector = embedder.embed(content, backend.vectorDim)
        if (vector.size != backend.vectorDim || vector.any { !it.isFinite() }) {
            throw VN97TurnMemoryIntegrityException("VN97 turn embedding is invalid")
        }
        val recordId = backend.append(
            NativeMemoryKind.EPISODIC,
            timestampNs,
            importance,
            source,
            content,
            vector,
        )
        if (recordId != expectedRecordId) {
            throw VN97TurnMemoryIntegrityException(
                "VN97MEM1 record ID changed between PREPARED and append"
            )
        }
        verifyRecord(recordId, source, content)
        return recordId
    }

    private fun verifyRecord(recordId: Long, source: String, content: String) {
        val record = try {
            backend.record(recordId)
        } catch (exc: Exception) {
            throw VN97TurnMemoryIntegrityException(
                "committed/prepared VN97MEM1 record is missing",
                exc,
            )
        }
        if (
            record.kind != NativeMemoryKind.EPISODIC ||
            record.source != source ||
            record.content != content
        ) {
            throw VN97TurnMemoryIntegrityException(
                "VN97MEM1 record does not match turn write-back transaction"
            )
        }
    }

    private fun requireDigests(
        storedSource: String,
        storedContent: String,
        source: String,
        content: String,
    ) {
        if (storedSource != source || storedContent != content) {
            throw VN97TurnMemoryIntegrityException(
                "turn write-back key was reused with different content"
            )
        }
    }
}

private fun computeTurnKey(turn: VN97AssistantTurn): String {
    val plan = turn.controller.plan
    return sha256Hex(
        buildString {
            append("VN97TURN1\n")
            append("plan_id=").append(turn.planId).append('\n')
            append("created_ns=").append(plan.createdNs).append('\n')
            append("principal=").append(turn.principal).append('\n')
        }.toByteArray(StandardCharsets.UTF_8)
    )
}

private fun canonicalTurnContent(
    turn: VN97AssistantTurn,
    finalResponse: String,
    turnKey: String,
): String = buildString {
    append("{\"assistant\":")
    appendJson(finalResponse)
    append(",\"created_ns\":").append(turn.controller.plan.createdNs)
    append(",\"plan_id\":"); appendJson(turn.planId)
    append(",\"principal\":"); appendJson(turn.principal)
    append(",\"turn_key\":"); appendJson(turnKey)
    append(",\"user\":"); appendJson(turn.goal)
    append('}')
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
            '\u000C' -> append("\\f")
            '\n' -> append("\\n")
            '\r' -> append("\\r")
            '\t' -> append("\\t")
            else -> when {
                ch.code < 0x20 -> append("\\u%04x".format(ch.code))
                Character.isHighSurrogate(ch) -> {
                    if (
                        index + 1 >= value.length ||
                        !Character.isLowSurrogate(value[index + 1])
                    ) {
                        throw VN97TurnMemoryIntegrityException(
                            "turn content contains unpaired surrogate"
                        )
                    }
                    append(ch); append(value[index + 1]); index += 1
                }
                Character.isLowSurrogate(ch) ->
                    throw VN97TurnMemoryIntegrityException(
                        "turn content contains unpaired surrogate"
                    )
                else -> append(ch)
            }
        }
        index += 1
    }
    append('"')
}

private fun isSafeFileName(value: String): Boolean =
    value.isNotEmpty() && value.length <= 128 && value != "." && value != ".." &&
        value.all { ch ->
            ch in 'a'..'z' || ch in 'A'..'Z' || ch in '0'..'9' ||
                ch == '.' || ch == '_' || ch == '-'
        }

private fun requireSha(value: String, label: String) {
    require(value.length == 64 && value.all { it in "0123456789abcdef" }) {
        "$label must be lowercase SHA-256 hex"
    }
}

private fun hexToBytes(value: String): ByteArray {
    requireSha(value, "SHA-256")
    return ByteArray(32) { index ->
        value.substring(index * 2, index * 2 + 2).toInt(16).toByte()
    }
}

private fun ByteArray.toLowerHex(): String =
    joinToString("") { "%02x".format(it.toInt() and 0xff) }

private fun sha256Hex(bytes: ByteArray): String =
    MessageDigest.getInstance("SHA-256").digest(bytes).toLowerHex()
