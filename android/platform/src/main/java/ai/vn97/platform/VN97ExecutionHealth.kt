package ai.vn97.platform

import java.io.File
import java.io.FileOutputStream
import java.nio.charset.StandardCharsets
import java.nio.file.AtomicMoveNotSupportedException
import java.nio.file.Files
import java.nio.file.LinkOption
import java.nio.file.StandardCopyOption
import java.security.MessageDigest
import java.util.Base64

enum class VN97ExecutionHealthDomain {
    MOBILE_RECOVERY,
    CONTINUATION_JOB,
    PAPER_TRADING_JOB,
    GAME_AGENT,
}

enum class VN97ExecutionHealthState {
    RUNNING,
    SUCCEEDED,
    FAILED,
    TIMED_OUT,
    ABANDONED,
    CANCELLED,
}

data class VN97ExecutionHealthRecord(
    val domain: VN97ExecutionHealthDomain,
    val key: String,
    val generation: Long,
    val state: VN97ExecutionHealthState,
    val consecutiveFailures: Int,
    val startedWallTimeMillis: Long,
    val updatedWallTimeMillis: Long,
    val deadlineWallTimeMillis: Long,
    val suppressUntilWallTimeMillis: Long,
    val detail: String = "",
) {
    init {
        require(key.isNotBlank())
        require(
            key.toByteArray(
                StandardCharsets.UTF_8
            ).size <= MAX_KEY_BYTES
        )
        require(generation > 0L)
        require(consecutiveFailures >= 0)
        require(startedWallTimeMillis >= 0L)
        require(
            updatedWallTimeMillis >=
                startedWallTimeMillis
        )
        require(
            deadlineWallTimeMillis >
                startedWallTimeMillis
        )
        require(
            suppressUntilWallTimeMillis >=
                0L
        )
        require(
            detail.toByteArray(
                StandardCharsets.UTF_8
            ).size <= MAX_DETAIL_BYTES
        )
        if (
            state ==
                VN97ExecutionHealthState
                    .SUCCEEDED
        ) {
            require(consecutiveFailures == 0)
            require(
                suppressUntilWallTimeMillis ==
                    0L
            )
        }
    }

    val running: Boolean
        get() =
            state ==
                VN97ExecutionHealthState
                    .RUNNING

    companion object {
        const val MAX_KEY_BYTES = 512
        const val MAX_DETAIL_BYTES = 2048
    }
}

data class VN97ExecutionHealthLease(
    val domain: VN97ExecutionHealthDomain,
    val key: String,
    val generation: Long,
    val deadlineWallTimeMillis: Long,
)

data class VN97ExecutionBeginResult(
    val lease: VN97ExecutionHealthLease?,
    val record: VN97ExecutionHealthRecord,
    val suppressed: Boolean,
) {
    init {
        require(suppressed == (lease == null))
        if (!suppressed) {
            require(record.running)
            require(
                lease?.generation ==
                    record.generation
            )
        }
    }
}

object VN97ExecutionHealthPolicy {
    const val MAX_CONSECUTIVE_FAILURES = 3
    const val FAILURE_COOLDOWN_MILLIS =
        15 * 60 * 1000L
    const val STALE_GRACE_MILLIS =
        30 * 1000L

    fun abandoned(
        record: VN97ExecutionHealthRecord,
        nowWallTimeMillis: Long,
    ): VN97ExecutionHealthRecord {
        require(record.running)
        require(
            nowWallTimeMillis >
                Math.addExact(
                    record.deadlineWallTimeMillis,
                    STALE_GRACE_MILLIS,
                )
        )
        val failures =
            Math.addExact(
                record.consecutiveFailures,
                1,
            )
        return record.copy(
            state =
                VN97ExecutionHealthState
                    .ABANDONED,
            consecutiveFailures = failures,
            updatedWallTimeMillis =
                nowWallTimeMillis,
            suppressUntilWallTimeMillis =
                suppressionDeadline(
                    failures,
                    nowWallTimeMillis,
                ),
            detail =
                "previous execution lease was abandoned",
        )
    }

    fun shouldSuppress(
        record: VN97ExecutionHealthRecord,
        nowWallTimeMillis: Long,
    ): Boolean =
        !record.running &&
            record.consecutiveFailures >=
                MAX_CONSECUTIVE_FAILURES &&
            record.suppressUntilWallTimeMillis >
                nowWallTimeMillis

    fun terminal(
        running: VN97ExecutionHealthRecord,
        state: VN97ExecutionHealthState,
        nowWallTimeMillis: Long,
        detail: String,
    ): VN97ExecutionHealthRecord {
        require(running.running)
        require(
            state !=
                VN97ExecutionHealthState
                    .RUNNING
        )
        require(
            nowWallTimeMillis >=
                running.startedWallTimeMillis
        )
        val boundedDetail =
            boundUtf8(
                detail,
                VN97ExecutionHealthRecord
                    .MAX_DETAIL_BYTES,
            )
        if (
            state ==
                VN97ExecutionHealthState
                    .SUCCEEDED
        ) {
            return running.copy(
                state = state,
                consecutiveFailures = 0,
                updatedWallTimeMillis =
                    nowWallTimeMillis,
                suppressUntilWallTimeMillis =
                    0L,
                detail = boundedDetail,
            )
        }
        if (
            state ==
                VN97ExecutionHealthState
                    .CANCELLED
        ) {
            return running.copy(
                state = state,
                updatedWallTimeMillis =
                    nowWallTimeMillis,
                detail = boundedDetail,
            )
        }
        val failures =
            Math.addExact(
                running.consecutiveFailures,
                1,
            )
        return running.copy(
            state = state,
            consecutiveFailures = failures,
            updatedWallTimeMillis =
                nowWallTimeMillis,
            suppressUntilWallTimeMillis =
                suppressionDeadline(
                    failures,
                    nowWallTimeMillis,
                ),
            detail = boundedDetail,
        )
    }

    private fun suppressionDeadline(
        failures: Int,
        nowWallTimeMillis: Long,
    ): Long =
        if (
            failures >=
                MAX_CONSECUTIVE_FAILURES
        ) {
            Math.addExact(
                nowWallTimeMillis,
                FAILURE_COOLDOWN_MILLIS,
            )
        } else {
            0L
        }
}

class VN97ExecutionHealthStore(
    private val root: File,
) {
    init {
        Files.createDirectories(root.toPath())
        require(
            Files.isDirectory(
                root.toPath(),
                LinkOption.NOFOLLOW_LINKS,
            ) &&
                !Files.isSymbolicLink(
                    root.toPath()
                )
        ) {
            "execution health root must be a real directory"
        }
    }

    @Synchronized
    fun begin(
        domain: VN97ExecutionHealthDomain,
        key: String,
        nowWallTimeMillis: Long,
        maxRunMillis: Long,
    ): VN97ExecutionBeginResult {
        require(nowWallTimeMillis >= 0L)
        require(maxRunMillis > 0L)
        require(key.isNotBlank())

        var previous =
            loadOrNull(domain, key)
        if (
            previous?.running == true &&
            nowWallTimeMillis >
                Math.addExact(
                    previous.deadlineWallTimeMillis,
                    VN97ExecutionHealthPolicy
                        .STALE_GRACE_MILLIS,
                )
        ) {
            previous =
                VN97ExecutionHealthPolicy
                    .abandoned(
                        previous,
                        nowWallTimeMillis,
                    )
            save(previous)
        }

        if (
            previous?.running == true ||
            (
                previous != null &&
                    VN97ExecutionHealthPolicy
                        .shouldSuppress(
                            previous,
                            nowWallTimeMillis,
                        )
                )
        ) {
            return VN97ExecutionBeginResult(
                lease = null,
                record = previous,
                suppressed = true,
            )
        }

        val generation =
            Math.addExact(
                previous?.generation ?: 0L,
                1L,
            )
        val record =
            VN97ExecutionHealthRecord(
                domain = domain,
                key = key,
                generation = generation,
                state =
                    VN97ExecutionHealthState
                        .RUNNING,
                consecutiveFailures =
                    previous
                        ?.consecutiveFailures
                        ?: 0,
                startedWallTimeMillis =
                    nowWallTimeMillis,
                updatedWallTimeMillis =
                    nowWallTimeMillis,
                deadlineWallTimeMillis =
                    Math.addExact(
                        nowWallTimeMillis,
                        maxRunMillis,
                    ),
                suppressUntilWallTimeMillis =
                    0L,
            )
        save(record)
        return VN97ExecutionBeginResult(
            lease =
                VN97ExecutionHealthLease(
                    domain = domain,
                    key = key,
                    generation = generation,
                    deadlineWallTimeMillis =
                        record.deadlineWallTimeMillis,
                ),
            record = record,
            suppressed = false,
        )
    }

    @Synchronized
    fun finish(
        lease: VN97ExecutionHealthLease,
        state: VN97ExecutionHealthState,
        nowWallTimeMillis: Long,
        detail: String = "",
    ): VN97ExecutionHealthRecord {
        require(
            state !=
                VN97ExecutionHealthState
                    .RUNNING
        )
        val current =
            checkNotNull(
                loadOrNull(
                    lease.domain,
                    lease.key,
                )
            ) {
                "execution health lease record is missing"
            }
        if (
            current.generation !=
                lease.generation ||
            !current.running
        ) {
            return current
        }
        val terminal =
            VN97ExecutionHealthPolicy
                .terminal(
                    running = current,
                    state = state,
                    nowWallTimeMillis =
                        maxOf(
                            nowWallTimeMillis,
                            current
                                .startedWallTimeMillis,
                        ),
                    detail = detail,
                )
        save(terminal)
        return terminal
    }

    @Synchronized
    fun loadOrNull(
        domain: VN97ExecutionHealthDomain,
        key: String,
    ): VN97ExecutionHealthRecord? {
        val target = target(domain, key)
        if (
            !Files.exists(
                target.toPath(),
                LinkOption.NOFOLLOW_LINKS,
            )
        ) {
            return null
        }
        require(
            !Files.isSymbolicLink(
                target.toPath()
            )
        ) {
            "execution health record must not be a symlink"
        }
        val bytes =
            Files.readAllBytes(target.toPath())
        require(
            bytes.size in 1..MAX_FILE_BYTES
        ) {
            "execution health record size is outside bound"
        }
        return decode(bytes).also {
            require(
                it.domain == domain &&
                    it.key == key
            ) {
                "execution health identity mismatch"
            }
        }
    }

    private fun save(
        record: VN97ExecutionHealthRecord,
    ) {
        val target =
            target(
                record.domain,
                record.key,
            )
        val bytes = encode(record)
        val temp =
            Files.createTempFile(
                root.toPath(),
                ".vn97-health-",
                ".tmp",
            ).toFile()
        try {
            FileOutputStream(temp).use {
                output ->
                output.write(bytes)
                output.flush()
                output.fd.sync()
            }
            try {
                Files.move(
                    temp.toPath(),
                    target.toPath(),
                    StandardCopyOption
                        .ATOMIC_MOVE,
                    StandardCopyOption
                        .REPLACE_EXISTING,
                )
            } catch (
                exc:
                    AtomicMoveNotSupportedException
            ) {
                throw IllegalStateException(
                    "execution health requires atomic replace",
                    exc,
                )
            }
            java.nio.channels.FileChannel
                .open(
                    root.toPath(),
                    java.nio.file
                        .StandardOpenOption.READ,
                )
                .use { it.force(true) }
        } finally {
            Files.deleteIfExists(
                temp.toPath()
            )
        }
        check(
            loadOrNull(
                record.domain,
                record.key,
            ) == record
        ) {
            "execution health post-write verification failed"
        }
    }

    private fun target(
        domain: VN97ExecutionHealthDomain,
        key: String,
    ): File {
        require(key.isNotBlank())
        require(
            key.toByteArray(
                StandardCharsets.UTF_8
            ).size <=
                VN97ExecutionHealthRecord
                    .MAX_KEY_BYTES
        )
        val digest =
            MessageDigest
                .getInstance("SHA-256")
                .digest(
                    (
                        domain.name +
                            "\n" +
                            key
                        ).toByteArray(
                            StandardCharsets.UTF_8
                        )
                )
                .joinToString("") {
                    "%02x".format(
                        it.toInt() and 0xff
                    )
                }
        return File(
            root,
            "$digest.vn97health1",
        )
    }

    companion object {
        private const val MAX_FILE_BYTES =
            16 * 1024
    }
}

private fun encode(
    record: VN97ExecutionHealthRecord,
): ByteArray {
    val detail =
        Base64.getEncoder()
            .encodeToString(
                record.detail.toByteArray(
                    StandardCharsets.UTF_8
                )
            )
    val payload =
        buildString {
            append("VN97HEALTH1\n")
            append(record.domain.name)
            append('\n')
            append(
                Base64.getEncoder()
                    .encodeToString(
                        record.key.toByteArray(
                            StandardCharsets.UTF_8
                        )
                    )
            )
            append('\n')
            append(record.generation)
            append('\n')
            append(record.state.name)
            append('\n')
            append(record.consecutiveFailures)
            append('\n')
            append(record.startedWallTimeMillis)
            append('\n')
            append(record.updatedWallTimeMillis)
            append('\n')
            append(record.deadlineWallTimeMillis)
            append('\n')
            append(record.suppressUntilWallTimeMillis)
            append('\n')
            append(detail)
            append('\n')
        }
    val digest =
        MessageDigest
            .getInstance("SHA-256")
            .digest(
                payload.toByteArray(
                    StandardCharsets.UTF_8
                )
            )
            .joinToString("") {
                "%02x".format(
                    it.toInt() and 0xff
                )
            }
    return (
        payload +
            "sha256=" +
            digest +
            "\n"
        ).toByteArray(
            StandardCharsets.UTF_8
        )
}

private fun decode(
    bytes: ByteArray,
): VN97ExecutionHealthRecord {
    val text =
        bytes.toString(
            StandardCharsets.UTF_8
        )
    val lines = text.split('\n')
    require(lines.size == 13) {
        "execution health record line count is invalid"
    }
    require(lines[0] == "VN97HEALTH1")
    require(
        lines[11].startsWith(
            "sha256="
        )
    )
    val payload =
        lines.take(11)
            .joinToString(
                separator = "\n",
                postfix = "\n",
            )
    val expected =
        MessageDigest
            .getInstance("SHA-256")
            .digest(
                payload.toByteArray(
                    StandardCharsets.UTF_8
                )
            )
            .joinToString("") {
                "%02x".format(
                    it.toInt() and 0xff
                )
            }
    require(
        lines[11] ==
            "sha256=$expected"
    ) {
        "execution health digest mismatch"
    }
    return VN97ExecutionHealthRecord(
        domain =
            VN97ExecutionHealthDomain
                .valueOf(lines[1]),
        key =
            Base64.getDecoder()
                .decode(lines[2])
                .toString(
                    StandardCharsets.UTF_8
                ),
        generation = lines[3].toLong(),
        state =
            VN97ExecutionHealthState
                .valueOf(lines[4]),
        consecutiveFailures =
            lines[5].toInt(),
        startedWallTimeMillis =
            lines[6].toLong(),
        updatedWallTimeMillis =
            lines[7].toLong(),
        deadlineWallTimeMillis =
            lines[8].toLong(),
        suppressUntilWallTimeMillis =
            lines[9].toLong(),
        detail =
            Base64.getDecoder()
                .decode(lines[10])
                .toString(
                    StandardCharsets.UTF_8
                ),
    )
}

private fun boundUtf8(
    value: String,
    maxBytes: Int,
): String {
    val normalized =
        value.replace('\n', ' ')
            .replace('\r', ' ')
            .trim()
    if (
        normalized.toByteArray(
            StandardCharsets.UTF_8
        ).size <= maxBytes
    ) {
        return normalized
    }
    val out = StringBuilder()
    var index = 0
    var used = 0
    while (
        index < normalized.length
    ) {
        val cp =
            Character.codePointAt(
                normalized,
                index,
            )
        val chars =
            String(
                Character.toChars(cp)
            )
        val size =
            chars.toByteArray(
                StandardCharsets.UTF_8
            ).size
        if (used + size > maxBytes) break
        out.append(chars)
        used += size
        index += Character.charCount(cp)
    }
    return out.toString()
}
