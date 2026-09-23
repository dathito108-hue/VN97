package ai.vn97.app

import java.io.File
import java.io.FileOutputStream
import java.net.URI
import java.nio.ByteBuffer
import java.nio.charset.CodingErrorAction
import java.nio.charset.StandardCharsets
import java.nio.file.AtomicMoveNotSupportedException
import java.nio.file.Files
import java.nio.file.LinkOption
import java.nio.file.StandardCopyOption
import java.security.MessageDigest
import java.util.Base64

enum class VN97PaperTradingSessionState {
    SCHEDULED,
    RUNNING,
    PAUSED,
    STOPPED,
    COMPLETED,
    FAILED,
}

data class VN97PaperTradingSessionRecord(
    val jobId: Int,
    val sessionId: String,
    val modelIdHex: String,
    val endpoint: String,
    val sourceId: String,
    val symbols: List<String>,
    val userGoal: String,
    val state: VN97PaperTradingSessionState,
    val createdWallTimeMillis: Long,
    val updatedWallTimeMillis: Long,
    val intervalMillis: Long,
    val nextRunWallTimeMillis: Long,
    val deadlineWallTimeMillis: Long,
    val maxEpisodes: Int,
    val wakeCount: Int = 0,
    val episodesAttempted: Int = 0,
    val lastSnapshotId: String = "",
    val lastObservedNs: Long = -1L,
    val lastPlanId: String = "",
    val lastDecision: String = "",
    val lastOutcome: String = "",
    val terminalReason: String = "",
    val requiresBatteryNotLow: Boolean = true,
    val requiresCharging: Boolean = false,
    val scheduleAttemptCount: Int = 0,
    val lastScheduledWallTimeMillis: Long = 0L,
) {
    init {
        require(jobId > 0) { "paper session jobId must be positive" }
        requireHex64(sessionId, "sessionId")
        requireHex64(modelIdHex, "modelIdHex")
        requireHttpsEndpoint(endpoint)
        require(SOURCE_RE.matches(sourceId)) {
            "paper session sourceId is invalid"
        }
        require(symbols.size in 1..64) {
            "paper session symbol count is outside bounds"
        }
        require(symbols == symbols.distinct().sorted()) {
            "paper session symbols must be unique canonical order"
        }
        require(symbols.all(SYMBOL_RE::matches)) {
            "paper session contains invalid symbol"
        }
        requireUtf8Bound(userGoal, MAX_GOAL_BYTES, "userGoal", true)
        require(createdWallTimeMillis >= 0L)
        require(updatedWallTimeMillis >= createdWallTimeMillis)
        require(intervalMillis in MIN_INTERVAL_MILLIS..MAX_INTERVAL_MILLIS) {
            "paper session interval is outside bounds"
        }
        require(nextRunWallTimeMillis >= 0L)
        require(deadlineWallTimeMillis >= 0L)
        if (deadlineWallTimeMillis > 0L) {
            require(deadlineWallTimeMillis >= createdWallTimeMillis) {
                "paper session deadline predates creation"
            }
        }
        require(maxEpisodes in 1..MAX_EPISODES) {
            "paper session episode bound is invalid"
        }
        require(wakeCount in 0..MAX_WAKE_COUNT) {
            "paper session wake count is outside bounds"
        }
        require(episodesAttempted in 0..maxEpisodes) {
            "paper session episode count is outside bounds"
        }
        require(lastObservedNs >= -1L)
        if (lastSnapshotId.isNotEmpty()) {
            requireHex64(lastSnapshotId, "lastSnapshotId")
            require(lastObservedNs >= 0L) {
                "paper session snapshot requires observation timestamp"
            }
        } else {
            require(lastObservedNs == -1L) {
                "paper session observation requires snapshot identity"
            }
        }
        if (lastPlanId.isNotEmpty()) {
            requireHex64(lastPlanId, "lastPlanId")
        }
        requireUtf8Bound(lastDecision, MAX_DECISION_BYTES, "lastDecision", false)
        requireUtf8Bound(lastOutcome, MAX_OUTCOME_BYTES, "lastOutcome", false)
        requireUtf8Bound(
            terminalReason,
            MAX_TERMINAL_REASON_BYTES,
            "terminalReason",
            false,
        )
        require(scheduleAttemptCount in 0..MAX_SCHEDULE_ATTEMPTS) {
            "paper session schedule-attempt count is outside bounds"
        }
        require(lastScheduledWallTimeMillis >= 0L)

        if (terminal) {
            require(terminalReason.isNotBlank()) {
                "terminal paper session requires terminal reason"
            }
        } else {
            require(terminalReason.isEmpty()) {
                "non-terminal paper session cannot carry terminal reason"
            }
        }
    }

    val terminal: Boolean
        get() = state == VN97PaperTradingSessionState.STOPPED ||
            state == VN97PaperTradingSessionState.COMPLETED ||
            state == VN97PaperTradingSessionState.FAILED

    internal fun sameIdentity(other: VN97PaperTradingSessionRecord): Boolean =
        jobId == other.jobId &&
            sessionId == other.sessionId &&
            modelIdHex == other.modelIdHex &&
            endpoint == other.endpoint &&
            sourceId == other.sourceId &&
            symbols == other.symbols &&
            userGoal == other.userGoal &&
            createdWallTimeMillis == other.createdWallTimeMillis &&
            intervalMillis == other.intervalMillis &&
            deadlineWallTimeMillis == other.deadlineWallTimeMillis &&
            maxEpisodes == other.maxEpisodes &&
            requiresBatteryNotLow == other.requiresBatteryNotLow &&
            requiresCharging == other.requiresCharging

    companion object {
        const val MIN_INTERVAL_MILLIS = 15_000L
        const val MAX_INTERVAL_MILLIS = 6L * 60L * 60L * 1000L
        const val MAX_EPISODES = 2_048
        const val MAX_WAKE_COUNT = 8_192
        const val MAX_SCHEDULE_ATTEMPTS = 8_192
        const val MAX_GOAL_BYTES = 8 * 1024
        const val MAX_DECISION_BYTES = 4 * 1024
        const val MAX_OUTCOME_BYTES = 8 * 1024
        const val MAX_TERMINAL_REASON_BYTES = 4 * 1024
    }
}

class VN97PaperTradingSessionStore(
    private val root: File,
) {
    init {
        Files.createDirectories(root.toPath())
        require(
            Files.isDirectory(root.toPath(), LinkOption.NOFOLLOW_LINKS) &&
                !Files.isSymbolicLink(root.toPath())
        ) {
            "paper session root must be a real directory"
        }
    }

    @Synchronized
    fun save(record: VN97PaperTradingSessionRecord) {
        val directory = sessionDirectory(record.jobId)
        Files.createDirectories(directory.toPath())
        require(
            Files.isDirectory(directory.toPath(), LinkOption.NOFOLLOW_LINKS) &&
                !Files.isSymbolicLink(directory.toPath())
        ) {
            "paper session directory must be real"
        }
        val target = directory.resolve(FILE_NAME)
        if (
            Files.exists(target.toPath(), LinkOption.NOFOLLOW_LINKS) &&
            Files.isSymbolicLink(target.toPath())
        ) {
            error("paper session state must not be a symlink")
        }

        val previous = loadOrNull(record.jobId)
        if (previous != null) {
            check(previous.sameIdentity(record)) {
                "paper session identity is immutable"
            }
            check(record.updatedWallTimeMillis >= previous.updatedWallTimeMillis) {
                "paper session update time moved backwards"
            }
            check(record.wakeCount >= previous.wakeCount) {
                "paper session wake count moved backwards"
            }
            check(record.episodesAttempted >= previous.episodesAttempted) {
                "paper session episode count moved backwards"
            }
            check(record.scheduleAttemptCount >= previous.scheduleAttemptCount) {
                "paper session schedule attempts moved backwards"
            }
            check(
                record.lastScheduledWallTimeMillis >=
                    previous.lastScheduledWallTimeMillis
            ) {
                "paper session last schedule time moved backwards"
            }
            if (previous.lastObservedNs >= 0L) {
                check(record.lastObservedNs >= previous.lastObservedNs) {
                    "paper session observation time moved backwards"
                }
            }
            check(!previous.terminal || record == previous) {
                "terminal paper session is immutable"
            }
        }

        val bytes = encode(record)
        require(bytes.size <= MAX_FILE_BYTES) {
            "paper session encoded state exceeds bound"
        }
        val temp = Files.createTempFile(
            directory.toPath(),
            ".vn97-paper-session-",
            ".tmp",
        ).toFile()
        try {
            FileOutputStream(temp).use { out ->
                out.write(bytes)
                out.flush()
                out.fd.sync()
            }
            try {
                Files.move(
                    temp.toPath(),
                    target.toPath(),
                    StandardCopyOption.ATOMIC_MOVE,
                    StandardCopyOption.REPLACE_EXISTING,
                )
            } catch (exc: AtomicMoveNotSupportedException) {
                throw IllegalStateException(
                    "paper session persistence requires atomic replace",
                    exc,
                )
            }
            forceDirectory(directory)
        } finally {
            Files.deleteIfExists(temp.toPath())
        }
        check(loadOrNull(record.jobId) == record) {
            "paper session post-write verification failed"
        }
    }

    @Synchronized
    fun loadOrNull(jobId: Int): VN97PaperTradingSessionRecord? {
        require(jobId > 0)
        val target = sessionDirectory(jobId).resolve(FILE_NAME)
        if (!Files.exists(target.toPath(), LinkOption.NOFOLLOW_LINKS)) {
            return null
        }
        check(!Files.isSymbolicLink(target.toPath())) {
            "paper session state must not be a symlink"
        }
        val size = Files.size(target.toPath())
        require(size in HEADER_MIN_BYTES.toLong()..MAX_FILE_BYTES.toLong()) {
            "paper session state size is outside bounds"
        }
        val bytes = Files.readAllBytes(target.toPath())
        check(bytes.size.toLong() == size) {
            "paper session state changed while reading"
        }
        val decoded = decode(bytes)
        check(decoded.jobId == jobId) {
            "paper session jobId does not match directory"
        }
        return decoded
    }

    @Synchronized
    fun list(): List<VN97PaperTradingSessionRecord> {
        val directories = root.listFiles()
            ?.filter {
                it.isDirectory &&
                    !Files.isSymbolicLink(it.toPath()) &&
                    it.name.toIntOrNull()?.let { id -> id > 0 } == true
            }
            ?.sortedBy { it.name.toInt() }
            .orEmpty()
        require(directories.size <= MAX_SESSIONS) {
            "paper session directory count exceeds bound"
        }
        return directories.mapNotNull {
            loadOrNull(it.name.toInt())
        }
    }

    fun contains(jobId: Int): Boolean = loadOrNull(jobId) != null

    private fun sessionDirectory(jobId: Int): File =
        root.resolve(jobId.toString())

    private fun forceDirectory(directory: File) {
        java.nio.channels.FileChannel.open(
            directory.toPath(),
            java.nio.file.StandardOpenOption.READ,
        ).use { it.force(true) }
    }

    companion object {
        private const val FILE_NAME = "session.vn97pts1"
        private const val MAX_SESSIONS = 64
        private const val HEADER_MIN_BYTES = 80
        private const val MAX_FILE_BYTES = 64 * 1024
    }
}

private fun encode(record: VN97PaperTradingSessionRecord): ByteArray {
    val payload = buildString {
        appendField("job_id", record.jobId.toString())
        appendField("session_id", record.sessionId)
        appendField("model_id", record.modelIdHex)
        appendField("endpoint_b64", encodeText(record.endpoint))
        appendField("source_id", record.sourceId)
        appendField("symbols", record.symbols.joinToString(","))
        appendField("goal_b64", encodeText(record.userGoal))
        appendField("state", record.state.name)
        appendField("created_ms", record.createdWallTimeMillis.toString())
        appendField("updated_ms", record.updatedWallTimeMillis.toString())
        appendField("interval_ms", record.intervalMillis.toString())
        appendField("next_run_ms", record.nextRunWallTimeMillis.toString())
        appendField("deadline_ms", record.deadlineWallTimeMillis.toString())
        appendField("max_episodes", record.maxEpisodes.toString())
        appendField("wake_count", record.wakeCount.toString())
        appendField("episodes_attempted", record.episodesAttempted.toString())
        appendField("last_snapshot", record.lastSnapshotId)
        appendField("last_observed_ns", record.lastObservedNs.toString())
        appendField("last_plan", record.lastPlanId)
        appendField("last_decision_b64", encodeText(record.lastDecision))
        appendField("last_outcome_b64", encodeText(record.lastOutcome))
        appendField("terminal_reason_b64", encodeText(record.terminalReason))
        appendField(
            "requires_battery_not_low",
            if (record.requiresBatteryNotLow) "1" else "0",
        )
        appendField(
            "requires_charging",
            if (record.requiresCharging) "1" else "0",
        )
        appendField(
            "schedule_attempt_count",
            record.scheduleAttemptCount.toString(),
        )
        appendField(
            "last_scheduled_ms",
            record.lastScheduledWallTimeMillis.toString(),
        )
    }.toByteArray(StandardCharsets.UTF_8)
    val digest = MessageDigest.getInstance("SHA-256")
        .digest(payload)
        .toHex()
    return buildString {
        append("VN97PTS1")
        append(10.toChar())
        append("sha256=")
        append(digest)
        append(10.toChar())
        append(payload.toString(StandardCharsets.UTF_8))
    }.toByteArray(StandardCharsets.UTF_8)
}

private fun decode(bytes: ByteArray): VN97PaperTradingSessionRecord {
    val text = strictUtf8(bytes)
    val lines = text.split(10.toChar())
    require(lines.size >= 4 && lines[0] == "VN97PTS1") {
        "paper session magic is invalid"
    }
    require(lines[1].startsWith("sha256=")) {
        "paper session digest header is missing"
    }
    val expectedDigest = lines[1].removePrefix("sha256=")
    requireHex64(expectedDigest, "paper session digest")
    val payloadText = lines.drop(2).joinToString(10.toChar().toString())
    val payload = payloadText.toByteArray(StandardCharsets.UTF_8)
    val actualDigest = MessageDigest.getInstance("SHA-256")
        .digest(payload)
        .toHex()
    check(actualDigest == expectedDigest) {
        "paper session SHA-256 mismatch"
    }

    val values = linkedMapOf<String, String>()
    payloadText.split(10.toChar())
        .filter { it.isNotEmpty() }
        .forEach { line ->
            val index = line.indexOf('=')
            require(index > 0) { "paper session field is malformed" }
            val key = line.substring(0, index)
            val value = line.substring(index + 1)
            require(key in EXPECTED_KEYS) {
                "paper session contains unknown field: $key"
            }
            require(values.put(key, value) == null) {
                "paper session contains duplicate field: $key"
            }
        }
    require(values.keys == EXPECTED_KEYS) {
        "paper session fields are incomplete"
    }

    fun field(name: String): String =
        checkNotNull(values[name]) { "missing paper session field: $name" }
    fun intField(name: String): Int =
        field(name).toIntOrNull()
            ?: error("invalid integer paper session field: $name")
    fun longField(name: String): Long =
        field(name).toLongOrNull()
            ?: error("invalid long paper session field: $name")
    fun boolField(name: String): Boolean =
        when (field(name)) {
            "0" -> false
            "1" -> true
            else -> error("invalid boolean paper session field: $name")
        }

    return VN97PaperTradingSessionRecord(
        jobId = intField("job_id"),
        sessionId = field("session_id"),
        modelIdHex = field("model_id"),
        endpoint = decodeText(field("endpoint_b64")),
        sourceId = field("source_id"),
        symbols = field("symbols").split(',').filter { it.isNotEmpty() },
        userGoal = decodeText(field("goal_b64")),
        state = VN97PaperTradingSessionState.valueOf(field("state")),
        createdWallTimeMillis = longField("created_ms"),
        updatedWallTimeMillis = longField("updated_ms"),
        intervalMillis = longField("interval_ms"),
        nextRunWallTimeMillis = longField("next_run_ms"),
        deadlineWallTimeMillis = longField("deadline_ms"),
        maxEpisodes = intField("max_episodes"),
        wakeCount = intField("wake_count"),
        episodesAttempted = intField("episodes_attempted"),
        lastSnapshotId = field("last_snapshot"),
        lastObservedNs = longField("last_observed_ns"),
        lastPlanId = field("last_plan"),
        lastDecision = decodeText(field("last_decision_b64")),
        lastOutcome = decodeText(field("last_outcome_b64")),
        terminalReason = decodeText(field("terminal_reason_b64")),
        requiresBatteryNotLow = boolField("requires_battery_not_low"),
        requiresCharging = boolField("requires_charging"),
        scheduleAttemptCount = intField("schedule_attempt_count"),
        lastScheduledWallTimeMillis = longField("last_scheduled_ms"),
    )
}

private fun StringBuilder.appendField(key: String, value: String) {
    append(key)
    append('=')
    append(value)
    append(10.toChar())
}

private fun encodeText(value: String): String =
    Base64.getUrlEncoder().withoutPadding()
        .encodeToString(value.toByteArray(StandardCharsets.UTF_8))

private fun decodeText(value: String): String {
    val bytes = try {
        Base64.getUrlDecoder().decode(value)
    } catch (exc: IllegalArgumentException) {
        throw IllegalArgumentException(
            "paper session base64 field is invalid",
            exc,
        )
    }
    return strictUtf8(bytes)
}

private fun strictUtf8(bytes: ByteArray): String =
    StandardCharsets.UTF_8.newDecoder()
        .onMalformedInput(CodingErrorAction.REPORT)
        .onUnmappableCharacter(CodingErrorAction.REPORT)
        .decode(ByteBuffer.wrap(bytes))
        .toString()

private fun requireHttpsEndpoint(value: String) {
    require(value.toByteArray(StandardCharsets.UTF_8).size <= 2048) {
        "paper session endpoint exceeds byte bound"
    }
    val uri = URI(value)
    require(uri.scheme?.equals("https", ignoreCase = true) == true) {
        "paper session endpoint must use HTTPS"
    }
    require(!uri.host.isNullOrBlank()) {
        "paper session endpoint host is missing"
    }
    require(uri.rawUserInfo == null && uri.rawFragment == null) {
        "paper session endpoint contains forbidden URL components"
    }
}

private fun requireHex64(value: String, label: String) {
    require(value.length == 64 && value.all { it in "0123456789abcdef" }) {
        "$label must be lowercase SHA-256 hex"
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
    require(value.toByteArray(StandardCharsets.UTF_8).size <= maxBytes) {
        "$label exceeds UTF-8 byte bound"
    }
}

private fun ByteArray.toHex(): String =
    joinToString("") { "%02x".format(it.toInt() and 0xff) }

private val SOURCE_RE =
    Regex("^[A-Za-z0-9][A-Za-z0-9._:-]{0,63}$")
private val SYMBOL_RE =
    Regex("^[A-Z][A-Z0-9._-]{0,23}$")

private val EXPECTED_KEYS = linkedSetOf(
    "job_id",
    "session_id",
    "model_id",
    "endpoint_b64",
    "source_id",
    "symbols",
    "goal_b64",
    "state",
    "created_ms",
    "updated_ms",
    "interval_ms",
    "next_run_ms",
    "deadline_ms",
    "max_episodes",
    "wake_count",
    "episodes_attempted",
    "last_snapshot",
    "last_observed_ns",
    "last_plan",
    "last_decision_b64",
    "last_outcome_b64",
    "terminal_reason_b64",
    "requires_battery_not_low",
    "requires_charging",
    "schedule_attempt_count",
    "last_scheduled_ms",
)
