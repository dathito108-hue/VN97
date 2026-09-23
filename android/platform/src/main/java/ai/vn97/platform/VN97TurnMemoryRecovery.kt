package ai.vn97.platform

import ai.vn97.runtime.NativePlanStatus
import ai.vn97.runtime.NativePlannerCheckpoint
import ai.vn97.runtime.NativeStepKind
import ai.vn97.runtime.NativeStepStatus
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

private val TURN_RECOVERY_MAGIC = "VN97TMR1".toByteArray(StandardCharsets.US_ASCII)
private const val TURN_RECOVERY_VERSION = 1
private const val TURN_RECOVERY_HEADER_BYTES = 48
private const val TURN_RECOVERY_FIXED_PAYLOAD_BYTES = 48
private const val TURN_RECOVERY_MAX_PRINCIPAL_BYTES = 256
private const val TURN_RECOVERY_MAX_PLANNER_BYTES = (16 shl 20) + 48
private const val TURN_RECOVERY_MAX_PAYLOAD_BYTES =
    TURN_RECOVERY_FIXED_PAYLOAD_BYTES +
        TURN_RECOVERY_MAX_PRINCIPAL_BYTES +
        TURN_RECOVERY_MAX_PLANNER_BYTES

data class VN97RecoveredTurnMemoryCommit(
    val turnKey: String,
    val planId: String,
    val recordId: Long,
) {
    init {
        require(turnKey.length == 64 && turnKey.all { it in "0123456789abcdef" })
        require(planId.length == 64 && planId.all { it in "0123456789abcdef" })
        require(recordId > 0L)
    }
}

private data class TurnRecoverySnapshot(
    val turnKey: String,
    val timestampNs: Long,
    val principal: String,
    val plannerCheckpoint: ByteArray,
)

/**
 * Durable cold-process recovery for one completed-turn M7W transaction.
 *
 * The envelope is continuity metadata only: the canonical turn content remains the completed
 * VN97PLN1 planner checkpoint and the memory record remains exclusively in VN97MEM1.
 */
class VN97TurnMemoryRecovery internal constructor(
    private val writer: VN97TurnMemoryWriter,
    root: File,
    fileName: String = "turn-memory-recovery.vn97tmr1",
) {
    private val rootPath = root.toPath()
    private val target = root.resolve(fileName).toPath()

    init {
        require(isSafeRecoveryFileName(fileName)) {
            "recovery file name must be one bounded path component"
        }
        Files.createDirectories(rootPath)
        if (
            !Files.isDirectory(rootPath, LinkOption.NOFOLLOW_LINKS) ||
            Files.isSymbolicLink(rootPath)
        ) {
            throw VN97TurnMemoryIntegrityException(
                "turn memory recovery root must be a real directory"
            )
        }
        if (
            Files.exists(target, LinkOption.NOFOLLOW_LINKS) &&
            Files.isSymbolicLink(target)
        ) {
            throw VN97TurnMemoryIntegrityException(
                "turn memory recovery target must not be a symlink"
            )
        }
    }

    @Synchronized
    fun commitCompletedTurn(
        turn: VN97AssistantTurn,
        finalResponse: String,
        timestampNs: Long,
    ): Long {
        require(timestampNs >= 0L) { "timestampNs must be non-negative" }
        requireCompletedResponse(turn, finalResponse)
        val candidate = TurnRecoverySnapshot(
            turnKey = computeTurnKey(turn),
            timestampNs = timestampNs,
            principal = turn.principal,
            plannerCheckpoint = NativePlannerCheckpoint.encode(turn.controller.plan),
        )
        val durable = ensureSnapshot(candidate)
        val recordId = writer.commitCompletedTurn(
            turn = turn,
            finalResponse = finalResponse,
            timestampNs = durable.timestampNs,
        )
        deleteSnapshot()
        return recordId
    }

    /**
     * Reconcile a durable completed-turn envelope with VN97TWJ1/VN97MEM1 after process death.
     * Returns null when there is no recovery work. Any inconsistent partial state fails closed.
     */
    @Synchronized
    fun recoverPendingOrNull(): VN97RecoveredTurnMemoryCommit? {
        val snapshot = loadSnapshotOrNull()
        val pendingTurnKey = writer.pendingTurnKey
        if (snapshot == null) {
            if (pendingTurnKey != null) {
                throw VN97TurnMemoryIntegrityException(
                    "VN97TWJ1 has a pending turn but VN97TMR1 recovery metadata is missing"
                )
            }
            return null
        }

        val controller = try {
            NativePlannerCheckpoint.decode(snapshot.plannerCheckpoint)
        } catch (exc: Exception) {
            throw VN97TurnMemoryIntegrityException(
                "VN97TMR1 planner checkpoint is invalid",
                exc,
            )
        }
        if (controller.plan.status != NativePlanStatus.COMPLETED) {
            throw VN97TurnMemoryIntegrityException(
                "VN97TMR1 planner checkpoint is not COMPLETED"
            )
        }
        val finalResponse = completedResponse(controller.plan.steps)
        val turn = VN97AssistantTurn(
            turnId = 0L,
            controller = controller,
            principal = snapshot.principal,
        )
        val computedTurnKey = computeTurnKey(turn)
        if (computedTurnKey != snapshot.turnKey) {
            throw VN97TurnMemoryIntegrityException(
                "VN97TMR1 turn key does not match recovered planner/principal"
            )
        }
        if (pendingTurnKey != null && pendingTurnKey != computedTurnKey) {
            throw VN97TurnMemoryIntegrityException(
                "VN97TWJ1 pending turn does not match VN97TMR1 recovery metadata"
            )
        }

        val recordId = writer.commitCompletedTurn(
            turn = turn,
            finalResponse = finalResponse,
            timestampNs = snapshot.timestampNs,
        )
        deleteSnapshot()
        return VN97RecoveredTurnMemoryCommit(
            turnKey = computedTurnKey,
            planId = turn.planId,
            recordId = recordId,
        )
    }

    private fun ensureSnapshot(candidate: TurnRecoverySnapshot): TurnRecoverySnapshot {
        validateSnapshot(candidate)
        val existing = loadSnapshotOrNull()
        if (existing != null) {
            if (
                existing.turnKey != candidate.turnKey ||
                existing.principal != candidate.principal ||
                !existing.plannerCheckpoint.contentEquals(candidate.plannerCheckpoint)
            ) {
                throw VN97TurnMemoryIntegrityException(
                    "another completed-turn recovery transaction is already durable"
                )
            }
            return existing
        }
        saveSnapshot(candidate)
        return candidate
    }

    private fun saveSnapshot(snapshot: TurnRecoverySnapshot) {
        val bytes = encodeSnapshot(snapshot)
        val temp = Files.createTempFile(
            rootPath,
            ".vn97-turn-memory-recovery-",
            ".tmp",
        ).toFile()
        try {
            FileOutputStream(temp).use { stream ->
                stream.write(bytes)
                stream.flush()
                stream.fd.sync()
            }
            try {
                Files.move(
                    temp.toPath(),
                    target,
                    StandardCopyOption.ATOMIC_MOVE,
                    StandardCopyOption.REPLACE_EXISTING,
                )
            } catch (exc: AtomicMoveNotSupportedException) {
                throw VN97TurnMemoryIntegrityException(
                    "turn memory recovery filesystem does not support atomic replace",
                    exc,
                )
            }
            forceDirectory()
        } catch (exc: VN97TurnMemoryIntegrityException) {
            throw exc
        } catch (exc: Exception) {
            throw VN97TurnMemoryIntegrityException(
                "turn memory recovery metadata could not be persisted",
                exc,
            )
        } finally {
            Files.deleteIfExists(temp.toPath())
        }
    }

    private fun loadSnapshotOrNull(): TurnRecoverySnapshot? {
        if (!Files.exists(target, LinkOption.NOFOLLOW_LINKS)) return null
        if (Files.isSymbolicLink(target)) {
            throw VN97TurnMemoryIntegrityException(
                "turn memory recovery target must not be a symlink"
            )
        }
        val size = try {
            Files.size(target)
        } catch (exc: Exception) {
            throw VN97TurnMemoryIntegrityException(
                "turn memory recovery metadata size could not be read",
                exc,
            )
        }
        val maxBytes = TURN_RECOVERY_HEADER_BYTES.toLong() +
            TURN_RECOVERY_MAX_PAYLOAD_BYTES.toLong()
        if (size !in TURN_RECOVERY_HEADER_BYTES.toLong()..maxBytes) {
            throw VN97TurnMemoryIntegrityException(
                "turn memory recovery metadata size is outside bounds"
            )
        }
        val bytes = try {
            Files.readAllBytes(target)
        } catch (exc: Exception) {
            throw VN97TurnMemoryIntegrityException(
                "turn memory recovery metadata could not be read",
                exc,
            )
        }
        if (bytes.size.toLong() != size) {
            throw VN97TurnMemoryIntegrityException(
                "turn memory recovery metadata changed while being read"
            )
        }
        return decodeSnapshot(bytes)
    }

    private fun deleteSnapshot() {
        val deleted = try {
            Files.deleteIfExists(target)
        } catch (exc: Exception) {
            throw VN97TurnMemoryIntegrityException(
                "committed turn recovery metadata could not be deleted",
                exc,
            )
        }
        if (deleted) forceDirectory()
    }

    private fun encodeSnapshot(snapshot: TurnRecoverySnapshot): ByteArray {
        validateSnapshot(snapshot)
        val principal = snapshot.principal.toByteArray(StandardCharsets.UTF_8)
        val payloadSize = TURN_RECOVERY_FIXED_PAYLOAD_BYTES +
            principal.size + snapshot.plannerCheckpoint.size
        if (payloadSize > TURN_RECOVERY_MAX_PAYLOAD_BYTES) {
            throw VN97TurnMemoryIntegrityException(
                "turn memory recovery payload exceeds format limit"
            )
        }
        val payload = ByteBuffer.allocate(payloadSize)
            .order(ByteOrder.LITTLE_ENDIAN)
            .apply {
                put(hexToRecoveryBytes(snapshot.turnKey))
                putLong(snapshot.timestampNs)
                putInt(principal.size)
                putInt(snapshot.plannerCheckpoint.size)
                put(principal)
                put(snapshot.plannerCheckpoint)
            }
            .array()
        val digest = MessageDigest.getInstance("SHA-256").digest(payload)
        return ByteBuffer.allocate(TURN_RECOVERY_HEADER_BYTES + payload.size)
            .order(ByteOrder.LITTLE_ENDIAN)
            .apply {
                put(TURN_RECOVERY_MAGIC)
                putInt(TURN_RECOVERY_VERSION)
                putInt(payload.size)
                put(digest)
                put(payload)
            }
            .array()
    }

    private fun decodeSnapshot(bytes: ByteArray): TurnRecoverySnapshot {
        if (bytes.size < TURN_RECOVERY_HEADER_BYTES) {
            throw VN97TurnMemoryIntegrityException(
                "turn memory recovery metadata is shorter than header"
            )
        }
        val header = ByteBuffer.wrap(bytes, 0, TURN_RECOVERY_HEADER_BYTES)
            .order(ByteOrder.LITTLE_ENDIAN)
        val magic = ByteArray(8).also(header::get)
        if (!magic.contentEquals(TURN_RECOVERY_MAGIC)) {
            throw VN97TurnMemoryIntegrityException("bad VN97TMR1 magic")
        }
        if (header.int != TURN_RECOVERY_VERSION) {
            throw VN97TurnMemoryIntegrityException("unsupported VN97TMR1 version")
        }
        val payloadSize = header.int
        if (
            payloadSize < TURN_RECOVERY_FIXED_PAYLOAD_BYTES ||
            payloadSize > TURN_RECOVERY_MAX_PAYLOAD_BYTES ||
            bytes.size != TURN_RECOVERY_HEADER_BYTES + payloadSize
        ) {
            throw VN97TurnMemoryIntegrityException(
                "VN97TMR1 payload length is invalid"
            )
        }
        val expected = ByteArray(32).also(header::get)
        val payload = bytes.copyOfRange(TURN_RECOVERY_HEADER_BYTES, bytes.size)
        val actual = MessageDigest.getInstance("SHA-256").digest(payload)
        if (!actual.contentEquals(expected)) {
            throw VN97TurnMemoryIntegrityException("VN97TMR1 SHA-256 mismatch")
        }

        val input = ByteBuffer.wrap(payload).order(ByteOrder.LITTLE_ENDIAN)
        val turnKey = ByteArray(32).also(input::get).toRecoveryHex()
        val timestampNs = input.long
        val principalSize = input.int
        val plannerSize = input.int
        if (
            timestampNs < 0L ||
            principalSize !in 1..TURN_RECOVERY_MAX_PRINCIPAL_BYTES ||
            plannerSize !in 48..TURN_RECOVERY_MAX_PLANNER_BYTES ||
            input.remaining() != principalSize + plannerSize
        ) {
            throw VN97TurnMemoryIntegrityException("VN97TMR1 fields are invalid")
        }
        val principalBytes = ByteArray(principalSize).also(input::get)
        val planner = ByteArray(plannerSize).also(input::get)
        val snapshot = TurnRecoverySnapshot(
            turnKey = turnKey,
            timestampNs = timestampNs,
            principal = strictRecoveryUtf8(principalBytes),
            plannerCheckpoint = planner,
        )
        validateSnapshot(snapshot)
        return snapshot
    }

    private fun validateSnapshot(snapshot: TurnRecoverySnapshot) {
        if (
            snapshot.turnKey.length != 64 ||
            snapshot.turnKey.any { it !in "0123456789abcdef" }
        ) {
            throw VN97TurnMemoryIntegrityException("VN97TMR1 turn key is invalid")
        }
        if (snapshot.timestampNs < 0L) {
            throw VN97TurnMemoryIntegrityException("VN97TMR1 timestamp is invalid")
        }
        validateRecoveryPrincipal(snapshot.principal)
        if (
            snapshot.plannerCheckpoint.size !in
            48..TURN_RECOVERY_MAX_PLANNER_BYTES
        ) {
            throw VN97TurnMemoryIntegrityException(
                "VN97TMR1 planner checkpoint size is invalid"
            )
        }
    }

    private fun forceDirectory() {
        try {
            java.nio.channels.FileChannel.open(
                rootPath,
                java.nio.file.StandardOpenOption.READ,
            ).use { it.force(true) }
        } catch (exc: IOException) {
            throw VN97TurnMemoryIntegrityException(
                "turn memory recovery directory fsync failed",
                exc,
            )
        }
    }
}

private fun requireCompletedResponse(
    turn: VN97AssistantTurn,
    finalResponse: String,
) {
    if (turn.controller.plan.status != NativePlanStatus.COMPLETED) {
        throw VN97TurnMemoryIntegrityException(
            "turn planner must be COMPLETED before recovery persistence"
        )
    }
    val canonical = completedResponse(turn.controller.plan.steps)
    if (canonical != finalResponse) {
        throw VN97TurnMemoryIntegrityException(
            "finalResponse does not match completed canonical RESPOND step"
        )
    }
}

private fun completedResponse(
    steps: List<ai.vn97.runtime.NativePlannerStep>,
): String {
    val response = steps.asReversed().firstOrNull {
        it.spec.kind == NativeStepKind.RESPOND &&
            it.status == NativeStepStatus.SUCCEEDED
    }?.result.orEmpty()
    if (response.isEmpty()) {
        throw VN97TurnMemoryIntegrityException(
            "completed planner has no canonical RESPOND result"
        )
    }
    return response
}

private fun validateRecoveryPrincipal(value: String) {
    val bytes = value.toByteArray(StandardCharsets.UTF_8)
    val allowed =
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._:-/".toSet()
    if (
        value.isEmpty() ||
        bytes.size > TURN_RECOVERY_MAX_PRINCIPAL_BYTES ||
        value.any { it !in allowed }
    ) {
        throw VN97TurnMemoryIntegrityException("VN97TMR1 principal is invalid")
    }
}

private fun strictRecoveryUtf8(bytes: ByteArray): String = try {
    StandardCharsets.UTF_8.newDecoder()
        .onMalformedInput(CodingErrorAction.REPORT)
        .onUnmappableCharacter(CodingErrorAction.REPORT)
        .decode(ByteBuffer.wrap(bytes))
        .toString()
} catch (exc: Exception) {
    throw VN97TurnMemoryIntegrityException(
        "VN97TMR1 principal is not UTF-8",
        exc,
    )
}

private fun isSafeRecoveryFileName(value: String): Boolean =
    value.isNotEmpty() && value.length <= 128 && value != "." && value != ".." &&
        value.all { ch ->
            ch in 'a'..'z' || ch in 'A'..'Z' || ch in '0'..'9' ||
                ch == '.' || ch == '_' || ch == '-'
        }

private fun hexToRecoveryBytes(value: String): ByteArray =
    ByteArray(32) { index ->
        value.substring(index * 2, index * 2 + 2).toInt(16).toByte()
    }

private fun ByteArray.toRecoveryHex(): String =
    joinToString("") { "%02x".format(it.toInt() and 0xff) }
