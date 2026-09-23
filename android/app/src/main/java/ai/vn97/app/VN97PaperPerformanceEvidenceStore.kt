package ai.vn97.app

import ai.vn97.platform.VN97PaperPerformance
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.charset.CodingErrorAction
import java.nio.charset.StandardCharsets
import java.nio.file.AtomicMoveNotSupportedException
import java.nio.file.Files
import java.nio.file.LinkOption
import java.nio.file.StandardCopyOption
import java.security.MessageDigest
import java.util.Base64

data class VN97PaperPerformanceEvidence(
    val sessionId: String,
    val jobId: Int,
    val episode: Int,
    val snapshotId: String,
    val observedNs: Long,
    val recordedWallTimeMillis: Long,
    val accountSequence: Long,
    val startingCashMicros: Long,
    val cashMicros: Long,
    val grossBookCostMicros: Long,
    val markedPositionValueMicros: Long,
    val markedEquityMicros: Long,
    val unrealizedPnlMicros: Long,
    val totalPnlMicros: Long,
    val totalReturnBasisPoints: Long,
    val positionCount: Int,
    val decision: String,
    val outcome: String,
) {
    init {
        requireHex64(sessionId, "performance sessionId")
        require(jobId > 0)
        require(episode > 0)
        requireHex64(snapshotId, "performance snapshotId")
        require(observedNs >= 0L)
        require(recordedWallTimeMillis >= 0L)
        require(accountSequence >= 0L)
        require(startingCashMicros > 0L)
        require(cashMicros >= 0L)
        require(grossBookCostMicros >= 0L)
        require(markedPositionValueMicros >= 0L)
        require(markedEquityMicros >= 0L)
        require(positionCount in 0..128)
        requireUtf8Bound(decision, MAX_DECISION_BYTES, "decision")
        requireUtf8Bound(outcome, MAX_OUTCOME_BYTES, "outcome")
    }

    companion object {
        const val MAX_DECISION_BYTES = 4 * 1024
        const val MAX_OUTCOME_BYTES = 8 * 1024

        fun from(
            sessionId: String,
            jobId: Int,
            episode: Int,
            recordedWallTimeMillis: Long,
            performance: VN97PaperPerformance,
            decision: String,
            outcome: String,
        ): VN97PaperPerformanceEvidence =
            VN97PaperPerformanceEvidence(
                sessionId = sessionId,
                jobId = jobId,
                episode = episode,
                snapshotId = performance.snapshotId,
                observedNs = performance.observedNs,
                recordedWallTimeMillis = recordedWallTimeMillis,
                accountSequence = performance.accountSequence,
                startingCashMicros = performance.startingCashMicros,
                cashMicros = performance.cashMicros,
                grossBookCostMicros =
                    performance.grossBookCostMicros,
                markedPositionValueMicros =
                    performance.markedPositionValueMicros,
                markedEquityMicros =
                    performance.markedEquityMicros,
                unrealizedPnlMicros =
                    performance.unrealizedPnlMicros,
                totalPnlMicros = performance.totalPnlMicros,
                totalReturnBasisPoints =
                    performance.totalReturnBasisPoints,
                positionCount = performance.positions.size,
                decision = decision.take(MAX_DECISION_BYTES),
                outcome = outcome.take(MAX_OUTCOME_BYTES),
            )
    }
}

data class VN97PaperSessionPerformanceSummary(
    val sessionId: String,
    val jobId: Int,
    val evidenceCount: Int,
    val firstEpisode: Int,
    val latestEpisode: Int,
    val latestEquityMicros: Long,
    val latestTotalPnlMicros: Long,
    val latestReturnBasisPoints: Long,
    val peakEquityMicros: Long,
    val maxDrawdownMicros: Long,
    val maxDrawdownBasisPoints: Long,
    val holdCount: Int,
    val orderCount: Int,
)

data class VN97PaperPerformanceAggregate(
    val evidenceCount: Int,
    val sessionCount: Int,
    val sessions: List<VN97PaperSessionPerformanceSummary>,
)

class VN97PaperPerformanceEvidenceStore(
    private val root: File,
) {
    init {
        Files.createDirectories(root.toPath())
        require(
            Files.isDirectory(root.toPath(), LinkOption.NOFOLLOW_LINKS) &&
                !Files.isSymbolicLink(root.toPath())
        ) {
            "paper performance root must be a real directory"
        }
    }

    @Synchronized
    fun append(record: VN97PaperPerformanceEvidence) {
        val directory = sessionDirectory(record.sessionId)
        Files.createDirectories(directory.toPath())
        require(
            Files.isDirectory(directory.toPath(), LinkOption.NOFOLLOW_LINKS) &&
                !Files.isSymbolicLink(directory.toPath())
        ) {
            "paper performance session directory must be real"
        }
        val target = File(
            directory,
            "%08d.vn97ppe1".format(record.episode),
        )
        if (target.exists()) {
            val existing = decode(target.readBytes())
            check(existing == record) {
                "paper performance evidence identity is immutable"
            }
            return
        }
        val bytes = encode(record)
        require(bytes.size <= MAX_RECORD_BYTES) {
            "paper performance evidence exceeds byte bound"
        }
        val temp = Files.createTempFile(
            directory.toPath(),
            ".vn97-ppe-",
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
                )
            } catch (exc: AtomicMoveNotSupportedException) {
                throw IllegalStateException(
                    "paper performance evidence requires atomic create",
                    exc,
                )
            }
            forceDirectory(directory)
        } finally {
            Files.deleteIfExists(temp.toPath())
        }
        check(loadSession(record.sessionId).lastOrNull() == record) {
            "paper performance post-write verification failed"
        }
    }

    @Synchronized
    fun loadSession(
        sessionId: String,
    ): List<VN97PaperPerformanceEvidence> {
        requireHex64(sessionId, "performance sessionId")
        val directory = sessionDirectory(sessionId)
        if (!directory.exists()) return emptyList()
        require(
            directory.isDirectory &&
                !Files.isSymbolicLink(directory.toPath())
        )
        val files = directory.listFiles()
            ?.filter {
                it.isFile &&
                    !Files.isSymbolicLink(it.toPath()) &&
                    it.name.endsWith(".vn97ppe1")
            }
            ?.sortedBy { it.name }
            .orEmpty()
        require(files.size <= MAX_EVIDENCE_PER_SESSION) {
            "paper performance evidence count exceeds bound"
        }
        val records = files.map { file ->
            val size = Files.size(file.toPath())
            require(size in 1L..MAX_RECORD_BYTES.toLong()) {
                "paper performance evidence size is outside bound"
            }
            decode(Files.readAllBytes(file.toPath()))
        }
        require(records.zipWithNext().all { (a, b) ->
            b.episode == a.episode + 1 &&
                b.observedNs > a.observedNs
        }) {
            "paper performance evidence sequence is not contiguous"
        }
        return records
    }

    @Synchronized
    fun aggregate(): VN97PaperPerformanceAggregate {
        val directories = root.listFiles()
            ?.filter {
                it.isDirectory &&
                    !Files.isSymbolicLink(it.toPath()) &&
                    SESSION_RE.matches(it.name)
            }
            ?.sortedBy { it.name }
            .orEmpty()
        require(directories.size <= MAX_SESSIONS) {
            "paper performance session count exceeds bound"
        }
        val summaries = directories.mapNotNull { directory ->
            val records = loadSession(directory.name)
            if (records.isEmpty()) null else summarize(records)
        }
        return VN97PaperPerformanceAggregate(
            evidenceCount = summaries.sumOf { it.evidenceCount },
            sessionCount = summaries.size,
            sessions = summaries.sortedByDescending { it.jobId },
        )
    }

    private fun summarize(
        records: List<VN97PaperPerformanceEvidence>,
    ): VN97PaperSessionPerformanceSummary {
        val first = records.first()
        val latest = records.last()
        var peak = first.markedEquityMicros
        var maxDrawdown = 0L
        var maxDrawdownBps = 0L
        var holds = 0
        var orders = 0
        records.forEach { record ->
            peak = maxOf(peak, record.markedEquityMicros)
            val drawdown =
                Math.subtractExact(peak, record.markedEquityMicros)
            if (drawdown > maxDrawdown) {
                maxDrawdown = drawdown
            }
            val drawdownBps =
                if (peak == 0L) 0L
                else scaledRatio(drawdown, 10_000L, peak)
            if (drawdownBps > maxDrawdownBps) {
                maxDrawdownBps = drawdownBps
            }
            if (record.decision == "HOLD") holds += 1 else orders += 1
        }
        return VN97PaperSessionPerformanceSummary(
            sessionId = first.sessionId,
            jobId = first.jobId,
            evidenceCount = records.size,
            firstEpisode = first.episode,
            latestEpisode = latest.episode,
            latestEquityMicros = latest.markedEquityMicros,
            latestTotalPnlMicros = latest.totalPnlMicros,
            latestReturnBasisPoints = latest.totalReturnBasisPoints,
            peakEquityMicros = peak,
            maxDrawdownMicros = maxDrawdown,
            maxDrawdownBasisPoints = maxDrawdownBps,
            holdCount = holds,
            orderCount = orders,
        )
    }

    private fun sessionDirectory(sessionId: String): File =
        root.resolve(sessionId)

    private fun forceDirectory(directory: File) {
        java.nio.channels.FileChannel.open(
            directory.toPath(),
            java.nio.file.StandardOpenOption.READ,
        ).use { it.force(true) }
    }

    companion object {
        private const val MAX_RECORD_BYTES = 32 * 1024
        private const val MAX_EVIDENCE_PER_SESSION = 2_048
        private const val MAX_SESSIONS = 64
        private val SESSION_RE = Regex("^[0-9a-f]{64}$")
    }
}

private fun encode(record: VN97PaperPerformanceEvidence): ByteArray {
    val payload = buildString {
        appendField("session_id", record.sessionId)
        appendField("job_id", record.jobId.toString())
        appendField("episode", record.episode.toString())
        appendField("snapshot_id", record.snapshotId)
        appendField("observed_ns", record.observedNs.toString())
        appendField(
            "recorded_ms",
            record.recordedWallTimeMillis.toString(),
        )
        appendField("account_sequence", record.accountSequence.toString())
        appendField("starting_cash", record.startingCashMicros.toString())
        appendField("cash", record.cashMicros.toString())
        appendField("gross_book_cost", record.grossBookCostMicros.toString())
        appendField(
            "marked_position_value",
            record.markedPositionValueMicros.toString(),
        )
        appendField("marked_equity", record.markedEquityMicros.toString())
        appendField(
            "unrealized_pnl",
            record.unrealizedPnlMicros.toString(),
        )
        appendField("total_pnl", record.totalPnlMicros.toString())
        appendField(
            "return_bps",
            record.totalReturnBasisPoints.toString(),
        )
        appendField("position_count", record.positionCount.toString())
        appendField("decision_b64", encodeText(record.decision))
        appendField("outcome_b64", encodeText(record.outcome))
    }.toByteArray(StandardCharsets.UTF_8)
    val digest = MessageDigest.getInstance("SHA-256")
        .digest(payload)
        .toHex()
    return buildString {
        append("VN97PPE1")
        append(10.toChar())
        append("sha256=")
        append(digest)
        append(10.toChar())
        append(payload.toString(StandardCharsets.UTF_8))
    }.toByteArray(StandardCharsets.UTF_8)
}

private fun decode(bytes: ByteArray): VN97PaperPerformanceEvidence {
    val text = strictUtf8(bytes)
    val lines = text.split(10.toChar())
    require(lines.size >= 4 && lines[0] == "VN97PPE1") {
        "paper performance magic is invalid"
    }
    require(lines[1].startsWith("sha256=")) {
        "paper performance digest header is missing"
    }
    val expected = lines[1].removePrefix("sha256=")
    requireHex64(expected, "paper performance digest")
    val payloadText =
        lines.drop(2).joinToString(10.toChar().toString())
    val payload = payloadText.toByteArray(StandardCharsets.UTF_8)
    val actual = MessageDigest.getInstance("SHA-256")
        .digest(payload)
        .toHex()
    check(actual == expected) {
        "paper performance SHA-256 mismatch"
    }
    val values = linkedMapOf<String, String>()
    payloadText.split(10.toChar())
        .filter { it.isNotEmpty() }
        .forEach { line ->
            val index = line.indexOf('=')
            require(index > 0)
            val key = line.substring(0, index)
            val value = line.substring(index + 1)
            require(key in PPE_KEYS) {
                "paper performance contains unknown field"
            }
            require(values.put(key, value) == null) {
                "paper performance contains duplicate field"
            }
        }
    require(values.keys == PPE_KEYS) {
        "paper performance fields are incomplete"
    }

    fun field(name: String) =
        checkNotNull(values[name])
    fun longField(name: String) =
        field(name).toLongOrNull()
            ?: error("invalid paper performance long: $name")
    fun intField(name: String) =
        field(name).toIntOrNull()
            ?: error("invalid paper performance int: $name")

    return VN97PaperPerformanceEvidence(
        sessionId = field("session_id"),
        jobId = intField("job_id"),
        episode = intField("episode"),
        snapshotId = field("snapshot_id"),
        observedNs = longField("observed_ns"),
        recordedWallTimeMillis = longField("recorded_ms"),
        accountSequence = longField("account_sequence"),
        startingCashMicros = longField("starting_cash"),
        cashMicros = longField("cash"),
        grossBookCostMicros = longField("gross_book_cost"),
        markedPositionValueMicros =
            longField("marked_position_value"),
        markedEquityMicros = longField("marked_equity"),
        unrealizedPnlMicros = longField("unrealized_pnl"),
        totalPnlMicros = longField("total_pnl"),
        totalReturnBasisPoints = longField("return_bps"),
        positionCount = intField("position_count"),
        decision = decodeText(field("decision_b64")),
        outcome = decodeText(field("outcome_b64")),
    )
}

private fun StringBuilder.appendField(
    key: String,
    value: String,
) {
    append(key)
    append('=')
    append(value)
    append(10.toChar())
}

private fun encodeText(value: String): String =
    Base64.getUrlEncoder().withoutPadding()
        .encodeToString(value.toByteArray(StandardCharsets.UTF_8))

private fun decodeText(value: String): String =
    strictUtf8(
        try {
            Base64.getUrlDecoder().decode(value)
        } catch (exc: IllegalArgumentException) {
            throw IllegalArgumentException(
                "paper performance base64 field is invalid",
                exc,
            )
        }
    )

private fun strictUtf8(bytes: ByteArray): String =
    StandardCharsets.UTF_8.newDecoder()
        .onMalformedInput(CodingErrorAction.REPORT)
        .onUnmappableCharacter(CodingErrorAction.REPORT)
        .decode(ByteBuffer.wrap(bytes))
        .toString()

private fun requireHex64(value: String, label: String) {
    require(
        value.length == 64 &&
            value.all { it in "0123456789abcdef" }
    ) {
        "$label must be lowercase SHA-256 hex"
    }
}

private fun requireUtf8Bound(
    value: String,
    maxBytes: Int,
    label: String,
) {
    require(
        value.toByteArray(StandardCharsets.UTF_8).size <= maxBytes
    ) {
        "$label exceeds UTF-8 byte bound"
    }
}

private fun scaledRatio(
    numerator: Long,
    scale: Long,
    denominator: Long,
): Long =
    java.math.BigInteger.valueOf(numerator)
        .multiply(java.math.BigInteger.valueOf(scale))
        .divide(java.math.BigInteger.valueOf(denominator))
        .longValueExact()

private fun ByteArray.toHex(): String =
    joinToString("") {
        "%02x".format(it.toInt() and 0xff)
    }

private val PPE_KEYS = linkedSetOf(
    "session_id",
    "job_id",
    "episode",
    "snapshot_id",
    "observed_ns",
    "recorded_ms",
    "account_sequence",
    "starting_cash",
    "cash",
    "gross_book_cost",
    "marked_position_value",
    "marked_equity",
    "unrealized_pnl",
    "total_pnl",
    "return_bps",
    "position_count",
    "decision_b64",
    "outcome_b64",
)
