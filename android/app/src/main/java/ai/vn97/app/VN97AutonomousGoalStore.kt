package ai.vn97.app

import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.charset.CodingErrorAction
import java.nio.charset.StandardCharsets
import java.nio.file.AtomicMoveNotSupportedException
import java.nio.file.Files
import java.nio.file.LinkOption
import java.nio.file.StandardCopyOption
import java.security.MessageDigest

private const val VN97_GOA_VERSION = 3
private const val VN97_GOA_HEADER_BYTES = 48
private val VN97_GOA_MAGIC =
    "VN97GOA1".toByteArray(StandardCharsets.US_ASCII)

enum class VN97AutonomousGoalState(val code: Int) {
    SCHEDULED(1),
    RUNNING(2),
    WAITING_APPROVAL(3),
    PAUSED(4),
    COMPLETED(5),
    FAILED(6),
    CANCELLED(7),
    BUDGET_EXHAUSTED(8),
    WAITING_DEPENDENCY(9);

    companion object {
        fun fromCode(code: Int): VN97AutonomousGoalState =
            entries.firstOrNull { it.code == code }
                ?: throw VN97AutonomousGoalStoreException(
                    "unknown VN97GOA1 state code: $code"
                )
    }
}

enum class VN97AutonomousPowerPolicy(val code: Int) {
    ADAPTIVE(0),
    BATTERY_NOT_LOW(1),
    CHARGING_ONLY(2);

    companion object {
        fun fromCode(code: Int): VN97AutonomousPowerPolicy =
            entries.firstOrNull { it.code == code }
                ?: throw VN97AutonomousGoalStoreException(
                    "unknown VN97GOA1 power policy code: $code"
                )
    }
}

data class VN97AutonomousGoalRecord(
    val jobId: Int,
    val planId: String,
    val modelIdHex: String,
    val principal: String,
    val goal: String,
    val state: VN97AutonomousGoalState,
    val rootJobId: Int = jobId,
    val generation: Int = 0,
    val previousJobId: Int = 0,
    val createdNs: Long,
    val updatedNs: Long,
    val wakeCount: Int = 0,
    val finalResponse: String = "",
    val terminalReason: String = "",
    val notBeforeWallTimeMillis: Long = 0L,
    val deadlineWallTimeMillis: Long = 0L,
    val dependencyKey: String = "",
    val dependencySatisfied: Boolean = dependencyKey.isEmpty(),
    val powerPolicy: VN97AutonomousPowerPolicy =
        VN97AutonomousPowerPolicy.ADAPTIVE,
    val scheduleAttemptCount: Int = 0,
    val lastScheduledWallTimeMillis: Long = 0L,
    val terminalNotificationSent: Boolean = false,
) {
    init {
        require(jobId > 0) { "autonomous jobId must be positive" }
        require(rootJobId > 0) {
            "autonomous rootJobId must be positive"
        }
        require(generation >= 0) {
            "autonomous generation must be non-negative"
        }
        require(previousJobId >= 0) {
            "autonomous previousJobId must be non-negative"
        }
        if (generation == 0) {
            require(rootJobId == jobId && previousJobId == 0) {
                "generation zero must be the root autonomous job"
            }
        } else {
            require(previousJobId > 0 && previousJobId != jobId) {
                "replan generation requires a distinct predecessor job"
            }
        }
        requireHex64(planId, "planId")
        requireHex64(modelIdHex, "modelIdHex")
        requirePrincipal(principal)
        requireUtf8Bound(goal, MAX_GOAL_BYTES, "goal", nonEmpty = true)
        require(createdNs >= 0L && updatedNs >= createdNs) {
            "autonomous goal timestamps are invalid"
        }
        require(wakeCount >= 0) {
            "autonomous wakeCount must be non-negative"
        }
        require(notBeforeWallTimeMillis >= 0L) {
            "autonomous not-before time must be non-negative"
        }
        require(deadlineWallTimeMillis >= 0L) {
            "autonomous deadline must be non-negative"
        }
        if (
            deadlineWallTimeMillis > 0L &&
            notBeforeWallTimeMillis > 0L
        ) {
            require(deadlineWallTimeMillis >= notBeforeWallTimeMillis) {
                "autonomous deadline precedes not-before time"
            }
        }
        requireUtf8Bound(
            dependencyKey,
            MAX_DEPENDENCY_KEY_BYTES,
            "dependencyKey",
            nonEmpty = false,
        )
        require(dependencyKey.isEmpty() || dependencyKey.isNotBlank()) {
            "dependencyKey must not contain only whitespace"
        }
        if (dependencyKey.isEmpty()) {
            require(dependencySatisfied) {
                "empty dependency must be satisfied"
            }
        }
        require(scheduleAttemptCount >= 0) {
            "scheduleAttemptCount must be non-negative"
        }
        require(lastScheduledWallTimeMillis >= 0L) {
            "lastScheduledWallTimeMillis must be non-negative"
        }
        requireUtf8Bound(
            finalResponse,
            MAX_RESPONSE_BYTES,
            "finalResponse",
            nonEmpty = false,
        )
        requireUtf8Bound(
            terminalReason,
            MAX_REASON_BYTES,
            "terminalReason",
            nonEmpty = false,
        )
        when (state) {
            VN97AutonomousGoalState.COMPLETED -> {
                require(finalResponse.isNotBlank()) {
                    "completed autonomous goal requires final response"
                }
            }

            VN97AutonomousGoalState.FAILED,
            VN97AutonomousGoalState.CANCELLED,
            VN97AutonomousGoalState.BUDGET_EXHAUSTED,
            -> require(terminalReason.isNotBlank()) {
                "terminal autonomous goal requires terminal reason"
            }

            else -> {
                require(finalResponse.isEmpty()) {
                    "non-completed autonomous goal cannot carry final response"
                }
            }
        }
    }

    val terminal: Boolean
        get() = state in setOf(
            VN97AutonomousGoalState.COMPLETED,
            VN97AutonomousGoalState.FAILED,
            VN97AutonomousGoalState.CANCELLED,
            VN97AutonomousGoalState.BUDGET_EXHAUSTED,
        )

    internal fun sameIdentity(other: VN97AutonomousGoalRecord): Boolean =
        jobId == other.jobId &&
            planId == other.planId &&
            modelIdHex == other.modelIdHex &&
            principal == other.principal &&
            goal == other.goal &&
            rootJobId == other.rootJobId &&
            generation == other.generation &&
            previousJobId == other.previousJobId &&
            createdNs == other.createdNs &&
            notBeforeWallTimeMillis ==
                other.notBeforeWallTimeMillis &&
            deadlineWallTimeMillis ==
                other.deadlineWallTimeMillis &&
            dependencyKey == other.dependencyKey &&
            powerPolicy == other.powerPolicy

    companion object {
        internal const val MAX_GOAL_BYTES = 32 * 1024
        internal const val MAX_RESPONSE_BYTES = 128 * 1024
        internal const val MAX_REASON_BYTES = 16 * 1024
        internal const val MAX_DEPENDENCY_KEY_BYTES = 512
    }
}

class VN97AutonomousGoalStoreException(
    message: String,
    cause: Throwable? = null,
) : IllegalStateException(message, cause)

class VN97AutonomousGoalStore(
    private val root: File,
) {
    init {
        Files.createDirectories(root.toPath())
        require(
            Files.isDirectory(root.toPath(), LinkOption.NOFOLLOW_LINKS) &&
                !Files.isSymbolicLink(root.toPath())
        ) {
            "autonomous goal root must be a real directory"
        }
    }

    @Synchronized
    fun save(record: VN97AutonomousGoalRecord) {
        val directory = goalDirectory(record.jobId)
        Files.createDirectories(directory.toPath())
        require(
            Files.isDirectory(
                directory.toPath(),
                LinkOption.NOFOLLOW_LINKS,
            ) && !Files.isSymbolicLink(directory.toPath())
        ) {
            "autonomous goal directory must be real"
        }
        val target = directory.resolve(FILE_NAME)
        if (
            Files.exists(
                target.toPath(),
                LinkOption.NOFOLLOW_LINKS,
            ) && Files.isSymbolicLink(target.toPath())
        ) {
            fail("autonomous goal file must not be a symlink")
        }

        val existing = loadOrNull(record.jobId)
        if (existing != null && !existing.sameIdentity(record)) {
            fail("autonomous goal identity is immutable")
        }
        if (
            existing != null &&
            record.updatedNs < existing.updatedNs
        ) {
            fail("autonomous goal update timestamp moved backwards")
        }
        if (
            existing != null &&
            record.wakeCount < existing.wakeCount
        ) {
            fail("autonomous goal wake count moved backwards")
        }
        if (
            existing != null &&
            existing.dependencySatisfied &&
            !record.dependencySatisfied
        ) {
            fail("autonomous dependency satisfaction moved backwards")
        }
        if (
            existing != null &&
            record.scheduleAttemptCount <
                existing.scheduleAttemptCount
        ) {
            fail("autonomous schedule attempts moved backwards")
        }
        if (
            existing != null &&
            record.lastScheduledWallTimeMillis <
                existing.lastScheduledWallTimeMillis
        ) {
            fail("autonomous last scheduled time moved backwards")
        }
        if (
            existing != null &&
            existing.terminalNotificationSent &&
            !record.terminalNotificationSent
        ) {
            fail("autonomous terminal notification state moved backwards")
        }

        val bytes = encode(record)
        val temp = Files.createTempFile(
            directory.toPath(),
            ".vn97-goal-",
            ".tmp",
        ).toFile()
        try {
            FileOutputStream(temp).use { output ->
                output.write(bytes)
                output.flush()
                output.fd.sync()
            }
            try {
                Files.move(
                    temp.toPath(),
                    target.toPath(),
                    StandardCopyOption.ATOMIC_MOVE,
                    StandardCopyOption.REPLACE_EXISTING,
                )
            } catch (exc: AtomicMoveNotSupportedException) {
                throw VN97AutonomousGoalStoreException(
                    "VN97GOA1 requires atomic replace",
                    exc,
                )
            }
            forceDirectory(directory)
        } catch (exc: VN97AutonomousGoalStoreException) {
            throw exc
        } catch (exc: Exception) {
            throw VN97AutonomousGoalStoreException(
                "autonomous goal persistence failed",
                exc,
            )
        } finally {
            Files.deleteIfExists(temp.toPath())
        }
        val verified = loadOrNull(record.jobId)
            ?: fail("autonomous goal disappeared after write")
        if (verified != record) {
            fail("autonomous goal post-write verification failed")
        }
    }

    @Synchronized
    fun loadOrNull(jobId: Int): VN97AutonomousGoalRecord? {
        require(jobId > 0)
        val target = goalDirectory(jobId).resolve(FILE_NAME)
        if (
            !Files.exists(
                target.toPath(),
                LinkOption.NOFOLLOW_LINKS,
            )
        ) {
            return null
        }
        if (Files.isSymbolicLink(target.toPath())) {
            fail("autonomous goal file must not be a symlink")
        }
        val size = try {
            Files.size(target.toPath())
        } catch (exc: Exception) {
            throw VN97AutonomousGoalStoreException(
                "autonomous goal size read failed",
                exc,
            )
        }
        if (size !in VN97_GOA_HEADER_BYTES.toLong()..MAX_FILE_BYTES.toLong()) {
            fail("autonomous goal file size is outside bounds")
        }
        val bytes = try {
            Files.readAllBytes(target.toPath())
        } catch (exc: Exception) {
            throw VN97AutonomousGoalStoreException(
                "autonomous goal read failed",
                exc,
            )
        }
        if (bytes.size.toLong() != size) {
            fail("autonomous goal changed while reading")
        }
        val record = decode(bytes)
        if (record.jobId != jobId) {
            fail("autonomous goal jobId does not match directory")
        }
        return record
    }

    @Synchronized
    fun list(): List<VN97AutonomousGoalRecord> {
        val entries = root.listFiles()
            ?.filter { file ->
                file.isDirectory &&
                    !Files.isSymbolicLink(file.toPath()) &&
                    file.name.toIntOrNull()?.let { it > 0 } == true
            }
            ?.sortedBy { it.name.toInt() }
            .orEmpty()
        require(entries.size <= MAX_GOALS) {
            "autonomous goal directory count exceeds bound"
        }
        return entries.mapNotNull { directory ->
            loadOrNull(directory.name.toInt())
        }
    }

    @Synchronized
    fun delete(jobId: Int) {
        require(jobId > 0)
        val directory = goalDirectory(jobId)
        val target = directory.resolve(FILE_NAME)
        try {
            Files.deleteIfExists(target.toPath())
            if (
                directory.exists() &&
                directory.list().orEmpty().isEmpty()
            ) {
                Files.deleteIfExists(directory.toPath())
            }
            forceDirectory(root)
        } catch (exc: Exception) {
            throw VN97AutonomousGoalStoreException(
                "autonomous goal delete failed",
                exc,
            )
        }
    }

    fun contains(jobId: Int): Boolean =
        loadOrNull(jobId) != null

    private fun goalDirectory(jobId: Int): File =
        root.resolve(jobId.toString())

    private fun forceDirectory(directory: File) {
        try {
            java.nio.channels.FileChannel.open(
                directory.toPath(),
                java.nio.file.StandardOpenOption.READ,
            ).use { it.force(true) }
        } catch (exc: IOException) {
            throw VN97AutonomousGoalStoreException(
                "autonomous goal directory fsync failed",
                exc,
            )
        }
    }

    companion object {
        private const val FILE_NAME = "goal.vn97goa1"
        private const val MAX_GOALS = 128
        private const val MAX_FILE_BYTES =
            VN97_GOA_HEADER_BYTES +
                4 + 8 + 8 + 4 + 4 + 4 + 4 + 4 +
                4 + 64 +
                4 + 64 +
                4 + 256 +
                4 + VN97AutonomousGoalRecord.MAX_GOAL_BYTES +
                4 + VN97AutonomousGoalRecord.MAX_RESPONSE_BYTES +
                4 + VN97AutonomousGoalRecord.MAX_REASON_BYTES +
                8 + 8 + 4 + 4 + 4 + 8 + 4 +
                4 + VN97AutonomousGoalRecord.MAX_DEPENDENCY_KEY_BYTES
    }
}

private fun encode(record: VN97AutonomousGoalRecord): ByteArray {
    val plan = record.planId.toByteArray(StandardCharsets.US_ASCII)
    val model = record.modelIdHex.toByteArray(StandardCharsets.US_ASCII)
    val principal = record.principal.toByteArray(StandardCharsets.UTF_8)
    val goal = record.goal.toByteArray(StandardCharsets.UTF_8)
    val response =
        record.finalResponse.toByteArray(StandardCharsets.UTF_8)
    val reason =
        record.terminalReason.toByteArray(StandardCharsets.UTF_8)
    val dependency =
        record.dependencyKey.toByteArray(StandardCharsets.UTF_8)

    val payloadSize = 4 + 8 + 8 + 4 + 4 + 4 + 4 + 4 +
        stringBytes(plan) +
        stringBytes(model) +
        stringBytes(principal) +
        stringBytes(goal) +
        stringBytes(response) +
        stringBytes(reason) +
        8 + 8 + 4 + 4 + 4 + 8 + 4 +
        stringBytes(dependency)
    val payload = ByteBuffer.allocate(payloadSize)
        .order(ByteOrder.BIG_ENDIAN)
        .apply {
            putInt(record.jobId)
            putLong(record.createdNs)
            putLong(record.updatedNs)
            putInt(record.wakeCount)
            putInt(record.state.code)
            putInt(record.rootJobId)
            putInt(record.generation)
            putInt(record.previousJobId)
            putBytes(plan)
            putBytes(model)
            putBytes(principal)
            putBytes(goal)
            putBytes(response)
            putBytes(reason)
            putLong(record.notBeforeWallTimeMillis)
            putLong(record.deadlineWallTimeMillis)
            putInt(if (record.dependencySatisfied) 1 else 0)
            putInt(record.powerPolicy.code)
            putInt(record.scheduleAttemptCount)
            putLong(record.lastScheduledWallTimeMillis)
            putInt(if (record.terminalNotificationSent) 1 else 0)
            putBytes(dependency)
        }
        .array()
    val digest =
        MessageDigest.getInstance("SHA-256").digest(payload)
    return ByteBuffer.allocate(VN97_GOA_HEADER_BYTES + payload.size)
        .order(ByteOrder.BIG_ENDIAN)
        .apply {
            put(VN97_GOA_MAGIC)
            putInt(VN97_GOA_VERSION)
            putInt(payload.size)
            put(digest)
            put(payload)
        }
        .array()
}

private fun decode(bytes: ByteArray): VN97AutonomousGoalRecord {
    if (bytes.size < 48) fail("VN97GOA1 is shorter than header")
    val header = ByteBuffer.wrap(bytes, 0, 48)
        .order(ByteOrder.BIG_ENDIAN)
    val magic = ByteArray(8).also(header::get)
    if (!magic.contentEquals(
            "VN97GOA1".toByteArray(StandardCharsets.US_ASCII)
        )
    ) {
        fail("bad VN97GOA1 magic")
    }
    val version = header.int
    if (version !in 1..VN97_GOA_VERSION) {
        fail("unsupported VN97GOA1 version")
    }
    val payloadSize = header.int
    if (
        payloadSize <= 0 ||
        bytes.size != 48 + payloadSize
    ) {
        fail("VN97GOA1 payload length is invalid")
    }
    val expected = ByteArray(32).also(header::get)
    val payload = bytes.copyOfRange(48, bytes.size)
    val actual =
        MessageDigest.getInstance("SHA-256").digest(payload)
    if (!actual.contentEquals(expected)) {
        fail("VN97GOA1 SHA-256 mismatch")
    }

    val source = ByteBuffer.wrap(payload)
        .order(ByteOrder.BIG_ENDIAN)
    val jobId = source.int
    val createdNs = source.long
    val updatedNs = source.long
    val wakeCount = source.int
    val state = VN97AutonomousGoalState.fromCode(source.int)
    val rootJobId: Int
    val generation: Int
    val previousJobId: Int
    if (version >= 2) {
        if (source.remaining() < 12) {
            fail("VN97GOA1 lineage fields are truncated")
        }
        rootJobId = source.int
        generation = source.int
        previousJobId = source.int
    } else {
        rootJobId = jobId
        generation = 0
        previousJobId = 0
    }
    val planId = source.readText(64, "planId")
    val modelId = source.readText(64, "modelIdHex")
    val principal = source.readText(256, "principal")
    val goal = source.readText(
        VN97AutonomousGoalRecord.MAX_GOAL_BYTES,
        "goal",
    )
    val response = source.readText(
        VN97AutonomousGoalRecord.MAX_RESPONSE_BYTES,
        "finalResponse",
    )
    val reason = source.readText(
        VN97AutonomousGoalRecord.MAX_REASON_BYTES,
        "terminalReason",
    )
    var notBeforeWallTimeMillis = 0L
    var deadlineWallTimeMillis = 0L
    var dependencyKey = ""
    var dependencySatisfied = true
    var powerPolicy = VN97AutonomousPowerPolicy.ADAPTIVE
    var scheduleAttemptCount = 0
    var lastScheduledWallTimeMillis = 0L
    var terminalNotificationSent = false
    if (version >= 3) {
        if (source.remaining() < 40) {
            fail("VN97GOA1 scheduling fields are truncated")
        }
        notBeforeWallTimeMillis = source.long
        deadlineWallTimeMillis = source.long
        dependencySatisfied =
            decodeBoolean(source.int, "dependencySatisfied")
        powerPolicy =
            VN97AutonomousPowerPolicy.fromCode(source.int)
        scheduleAttemptCount = source.int
        lastScheduledWallTimeMillis = source.long
        terminalNotificationSent =
            decodeBoolean(source.int, "terminalNotificationSent")
        dependencyKey = source.readText(
            VN97AutonomousGoalRecord.MAX_DEPENDENCY_KEY_BYTES,
            "dependencyKey",
        )
    }
    if (source.hasRemaining()) {
        fail("VN97GOA1 payload has trailing bytes")
    }
    return VN97AutonomousGoalRecord(
        jobId = jobId,
        planId = planId,
        modelIdHex = modelId,
        principal = principal,
        goal = goal,
        state = state,
        rootJobId = rootJobId,
        generation = generation,
        previousJobId = previousJobId,
        createdNs = createdNs,
        updatedNs = updatedNs,
        wakeCount = wakeCount,
        finalResponse = response,
        terminalReason = reason,
        notBeforeWallTimeMillis = notBeforeWallTimeMillis,
        deadlineWallTimeMillis = deadlineWallTimeMillis,
        dependencyKey = dependencyKey,
        dependencySatisfied = dependencySatisfied,
        powerPolicy = powerPolicy,
        scheduleAttemptCount = scheduleAttemptCount,
        lastScheduledWallTimeMillis = lastScheduledWallTimeMillis,
        terminalNotificationSent = terminalNotificationSent,
    )
}

private fun decodeBoolean(value: Int, label: String): Boolean =
    when (value) {
        0 -> false
        1 -> true
        else -> fail("$label must be encoded as 0 or 1")
    }

private fun stringBytes(bytes: ByteArray): Int =
    Math.addExact(4, bytes.size)

private fun ByteBuffer.putBytes(bytes: ByteArray) {
    putInt(bytes.size)
    put(bytes)
}

private fun ByteBuffer.readText(
    maxBytes: Int,
    label: String,
): String {
    if (remaining() < 4) fail("$label length is truncated")
    val size = int
    if (size < 0 || size > maxBytes || remaining() < size) {
        fail("$label length is invalid")
    }
    val bytes = ByteArray(size).also(::get)
    return try {
        StandardCharsets.UTF_8.newDecoder()
            .onMalformedInput(CodingErrorAction.REPORT)
            .onUnmappableCharacter(CodingErrorAction.REPORT)
            .decode(ByteBuffer.wrap(bytes))
            .toString()
    } catch (exc: Exception) {
        throw VN97AutonomousGoalStoreException(
            "$label is not strict UTF-8",
            exc,
        )
    }
}

private fun requireHex64(value: String, label: String) {
    require(
        value.length == 64 &&
            value.all { it in "0123456789abcdef" }
    ) {
        "$label must be lowercase SHA-256 hex"
    }
}

private fun requirePrincipal(value: String) {
    val bytes = value.toByteArray(StandardCharsets.UTF_8)
    val allowed =
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._:-/"
            .toSet()
    require(
        value.isNotEmpty() &&
            bytes.size <= 256 &&
            value.all { it in allowed }
    ) {
        "principal is invalid"
    }
}

private fun requireUtf8Bound(
    value: String,
    maxBytes: Int,
    label: String,
    nonEmpty: Boolean,
) {
    if (nonEmpty) require(value.isNotBlank()) {
        "$label must not be blank"
    }
    require(
        value.toByteArray(StandardCharsets.UTF_8).size <= maxBytes
    ) {
        "$label exceeds UTF-8 byte bound"
    }
}

private fun fail(message: String): Nothing =
    throw VN97AutonomousGoalStoreException(message)
