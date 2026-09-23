package ai.vn97.runtime

import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.charset.CodingErrorAction
import java.nio.charset.StandardCharsets
import java.nio.channels.FileChannel
import java.nio.file.AtomicMoveNotSupportedException
import java.nio.file.Files
import java.nio.file.LinkOption
import java.nio.file.StandardCopyOption
import java.nio.file.StandardOpenOption
import java.security.MessageDigest

internal enum class VN97KnowledgeAcquisitionState {
    IMPORTING,
    COMPLETED,
}

internal data class VN97KnowledgeAcquisitionLedgerRecord(
    val packageSha256: String,
    val capabilityId: String,
    val capabilityVersion: Long,
    val publisherKeyId: String,
    val signatureSha256: String,
    val payloadSha256: String,
    val totalRecords: Int,
    val state: VN97KnowledgeAcquisitionState,
    val nextIndex: Int,
    val recordIds: List<Long>,
    val pendingIndex: Int = -1,
    val pendingPreAppendLastRecordId: Long = 0L,
    val pendingContentSha256: String = "",
    val createdWallTimeMillis: Long,
    val updatedWallTimeMillis: Long,
) {
    init {
        requireKnowledgeSha(packageSha256, "packageSha256")
        requireKnowledgeId(capabilityId, "capabilityId")
        require(capabilityId == "knowledge" || capabilityId.startsWith("knowledge.")) {
            "knowledge acquisition capability namespace is invalid"
        }
        require(capabilityVersion in 1L..0xffff_ffffL)
        requireKnowledgeId(publisherKeyId, "publisherKeyId")
        requireKnowledgeSha(signatureSha256, "signatureSha256")
        requireKnowledgeSha(payloadSha256, "payloadSha256")
        require(totalRecords in 1..VN97KnowledgePayload.MAX_RECORDS)
        require(nextIndex in 0..totalRecords)
        require(recordIds.size == nextIndex)
        require(recordIds.all { it > 0L })
        require(recordIds.zipWithNext().all { (a, b) -> b > a }) {
            "knowledge acquisition memory IDs must increase"
        }
        require(createdWallTimeMillis >= 0L)
        require(updatedWallTimeMillis >= createdWallTimeMillis)
        if (pendingIndex >= 0) {
            require(state == VN97KnowledgeAcquisitionState.IMPORTING)
            require(pendingIndex == nextIndex)
            require(pendingPreAppendLastRecordId >= 0L)
            requireKnowledgeSha(
                pendingContentSha256,
                "pendingContentSha256",
            )
        } else {
            require(pendingPreAppendLastRecordId == 0L)
            require(pendingContentSha256.isEmpty())
        }
        if (state == VN97KnowledgeAcquisitionState.COMPLETED) {
            require(nextIndex == totalRecords)
            require(pendingIndex < 0)
        }
    }

    fun sameIdentity(
        other: VN97KnowledgeAcquisitionLedgerRecord,
    ): Boolean =
        packageSha256 == other.packageSha256 &&
            capabilityId == other.capabilityId &&
            capabilityVersion == other.capabilityVersion &&
            publisherKeyId == other.publisherKeyId &&
            signatureSha256 == other.signatureSha256 &&
            payloadSha256 == other.payloadSha256 &&
            totalRecords == other.totalRecords &&
            createdWallTimeMillis == other.createdWallTimeMillis
}

internal class VN97KnowledgeAcquisitionLedger(
    private val root: File,
) {
    private val rootPath = root.toPath()

    init {
        Files.createDirectories(rootPath)
        require(
            Files.isDirectory(rootPath, LinkOption.NOFOLLOW_LINKS) &&
                !Files.isSymbolicLink(rootPath)
        ) {
            "knowledge acquisition ledger root must be a real directory"
        }
    }

    @Synchronized
    fun loadOrNull(
        packageSha256: String,
    ): VN97KnowledgeAcquisitionLedgerRecord? {
        requireKnowledgeSha(packageSha256, "packageSha256")
        val target = target(packageSha256)
        if (!Files.exists(target, LinkOption.NOFOLLOW_LINKS)) {
            return null
        }
        require(
            Files.isRegularFile(target, LinkOption.NOFOLLOW_LINKS) &&
                !Files.isSymbolicLink(target)
        ) {
            "knowledge acquisition ledger target must be a regular file"
        }
        val size = Files.size(target)
        require(size in 1L..MAX_LEDGER_BYTES.toLong()) {
            "knowledge acquisition ledger size is outside bounds"
        }
        val bytes = Files.readAllBytes(target)
        require(bytes.size.toLong() == size) {
            "knowledge acquisition ledger changed while reading"
        }
        return decode(bytes)
    }

    @Synchronized
    fun save(record: VN97KnowledgeAcquisitionLedgerRecord) {
        val target = target(record.packageSha256)
        if (
            Files.exists(target, LinkOption.NOFOLLOW_LINKS) &&
            Files.isSymbolicLink(target)
        ) {
            throw IllegalStateException(
                "knowledge acquisition ledger target must not be a symlink"
            )
        }
        val existing = loadOrNull(record.packageSha256)
        if (existing != null) {
            check(existing.sameIdentity(record)) {
                "knowledge acquisition ledger identity is immutable"
            }
            check(record.updatedWallTimeMillis >= existing.updatedWallTimeMillis) {
                "knowledge acquisition ledger timestamp moved backwards"
            }
            check(record.nextIndex >= existing.nextIndex) {
                "knowledge acquisition progress moved backwards"
            }
            check(
                record.recordIds.take(existing.recordIds.size) ==
                    existing.recordIds
            ) {
                "knowledge acquisition record IDs changed"
            }
            if (
                existing.state ==
                    VN97KnowledgeAcquisitionState.COMPLETED
            ) {
                check(record == existing) {
                    "completed knowledge acquisition is immutable"
                }
                return
            }
        }

        val bytes = encode(record)
        require(bytes.size <= MAX_LEDGER_BYTES) {
            "knowledge acquisition ledger exceeds byte bound"
        }
        val temp = Files.createTempFile(
            rootPath,
            ".vn97kal-",
            ".tmp",
        )
        try {
            FileOutputStream(temp.toFile()).use { out ->
                out.write(bytes)
                out.flush()
                out.fd.sync()
            }
            try {
                Files.move(
                    temp,
                    target,
                    StandardCopyOption.ATOMIC_MOVE,
                    StandardCopyOption.REPLACE_EXISTING,
                )
            } catch (exc: AtomicMoveNotSupportedException) {
                throw IllegalStateException(
                    "knowledge acquisition ledger requires atomic replace",
                    exc,
                )
            }
            FileChannel.open(
                rootPath,
                StandardOpenOption.READ,
            ).use { it.force(true) }
        } finally {
            Files.deleteIfExists(temp)
        }
        check(loadOrNull(record.packageSha256) == record) {
            "knowledge acquisition ledger post-write verification failed"
        }
    }

    private fun target(packageSha256: String) =
        rootPath.resolve("$packageSha256.vn97kal1")

    companion object {
        private const val MAX_LEDGER_BYTES = 128 * 1024
    }
}

private fun encode(
    record: VN97KnowledgeAcquisitionLedgerRecord,
): ByteArray {
    val payload = VnStrictJson.canonical(
        VnStrictJson.objectOf(
            "capability_id" to
                VnStrictJson.string(record.capabilityId),
            "capability_version" to
                VnStrictJson.long(record.capabilityVersion),
            "created_ms" to
                VnStrictJson.long(record.createdWallTimeMillis),
            "next_index" to
                VnStrictJson.int(record.nextIndex),
            "package_sha256" to
                VnStrictJson.string(record.packageSha256),
            "payload_sha256" to
                VnStrictJson.string(record.payloadSha256),
            "pending_content_sha256" to
                VnStrictJson.string(record.pendingContentSha256),
            "pending_index" to
                VnStrictJson.int(record.pendingIndex),
            "pending_pre_append_last_record_id" to
                VnStrictJson.long(
                    record.pendingPreAppendLastRecordId
                ),
            "publisher_key_id" to
                VnStrictJson.string(record.publisherKeyId),
            "record_ids" to
                VnStrictJson.array(
                    record.recordIds.map(VnStrictJson::long)
                ),
            "schema" to VnStrictJson.string("VN97KAL1"),
            "signature_sha256" to
                VnStrictJson.string(record.signatureSha256),
            "state" to
                VnStrictJson.string(record.state.name),
            "total_records" to
                VnStrictJson.int(record.totalRecords),
            "updated_ms" to
                VnStrictJson.long(record.updatedWallTimeMillis),
        )
    )
    val payloadBytes =
        payload.toByteArray(StandardCharsets.UTF_8)
    val digest = MessageDigest.getInstance("SHA-256")
        .digest(payloadBytes)
        .toKnowledgeHex()
    return buildString {
        append("VN97KAL1")
        append(10.toChar())
        append("sha256=")
        append(digest)
        append(10.toChar())
        append(payload)
    }.toByteArray(StandardCharsets.UTF_8)
}

private fun decode(
    bytes: ByteArray,
): VN97KnowledgeAcquisitionLedgerRecord {
    val text = strictKnowledgeUtf8(bytes)
    val firstBreak = text.indexOf(10.toChar())
    val secondBreak =
        if (firstBreak >= 0) {
            text.indexOf(10.toChar(), firstBreak + 1)
        } else {
            -1
        }
    require(
        firstBreak > 0 &&
            secondBreak > firstBreak &&
            text.substring(0, firstBreak) == "VN97KAL1"
    ) {
        "knowledge acquisition ledger header is invalid"
    }
    val digestLine =
        text.substring(firstBreak + 1, secondBreak)
    require(digestLine.startsWith("sha256=")) {
        "knowledge acquisition ledger digest is missing"
    }
    val expected = digestLine.removePrefix("sha256=")
    requireKnowledgeSha(expected, "ledger digest")
    val payload = text.substring(secondBreak + 1)
    val actual = MessageDigest.getInstance("SHA-256")
        .digest(payload.toByteArray(StandardCharsets.UTF_8))
        .toKnowledgeHex()
    check(actual == expected) {
        "knowledge acquisition ledger SHA-256 mismatch"
    }

    val root = VnStrictJson.parseObject(
        payload,
        VnJsonLimits(
            maxInputUtf8Bytes = 128 * 1024,
            maxDepth = 8,
            maxNodes = 4096,
            maxStringUtf8Bytes = 16 * 1024,
        ),
    )
    check(VnStrictJson.canonical(root) == payload) {
        "knowledge acquisition ledger is not canonical JSON"
    }
    val expectedKeys = setOf(
        "capability_id",
        "capability_version",
        "created_ms",
        "next_index",
        "package_sha256",
        "payload_sha256",
        "pending_content_sha256",
        "pending_index",
        "pending_pre_append_last_record_id",
        "publisher_key_id",
        "record_ids",
        "schema",
        "signature_sha256",
        "state",
        "total_records",
        "updated_ms",
    )
    require(root.values.keys == expectedKeys) {
        "knowledge acquisition ledger fields are invalid"
    }
    require(root.string("schema") == "VN97KAL1") {
        "knowledge acquisition ledger schema mismatch"
    }
    val state = try {
        VN97KnowledgeAcquisitionState.valueOf(
            root.string("state")
        )
    } catch (exc: IllegalArgumentException) {
        throw IllegalArgumentException(
            "knowledge acquisition ledger state is invalid",
            exc,
        )
    }
    val recordIds = root.array("record_ids").values.map {
        val number = it as? VnJsonNumber
            ?: throw IllegalArgumentException(
                "knowledge acquisition record ID must be integer"
            )
        number.canonical.toLongOrNull()
            ?: throw IllegalArgumentException(
                "knowledge acquisition record ID is invalid"
            )
    }

    return VN97KnowledgeAcquisitionLedgerRecord(
        packageSha256 = root.string("package_sha256"),
        capabilityId = root.string("capability_id"),
        capabilityVersion =
            root.long("capability_version"),
        publisherKeyId =
            root.string("publisher_key_id"),
        signatureSha256 =
            root.string("signature_sha256"),
        payloadSha256 =
            root.string("payload_sha256"),
        totalRecords =
            root.int("total_records"),
        state = state,
        nextIndex = root.int("next_index"),
        recordIds = recordIds,
        pendingIndex = root.int("pending_index"),
        pendingPreAppendLastRecordId =
            root.long(
                "pending_pre_append_last_record_id"
            ),
        pendingContentSha256 =
            root.string("pending_content_sha256"),
        createdWallTimeMillis =
            root.long("created_ms"),
        updatedWallTimeMillis =
            root.long("updated_ms"),
    )
}

private fun VnJsonObject.string(key: String): String =
    (values[key] as? VnJsonString)?.value
        ?: throw IllegalArgumentException(
            "knowledge acquisition $key must be string"
        )

private fun VnJsonObject.array(key: String): VnJsonArray =
    values[key] as? VnJsonArray
        ?: throw IllegalArgumentException(
            "knowledge acquisition $key must be array"
        )

private fun VnJsonObject.long(key: String): Long {
    val raw = (values[key] as? VnJsonNumber)?.canonical
        ?: throw IllegalArgumentException(
            "knowledge acquisition $key must be integer"
        )
    require(
        raw.none { it == '.' || it == 'e' || it == 'E' }
    ) {
        "knowledge acquisition $key must be integer"
    }
    return raw.toLongOrNull()
        ?: throw IllegalArgumentException(
            "knowledge acquisition $key is outside Long range"
        )
}

private fun VnJsonObject.int(key: String): Int {
    val value = long(key)
    require(value in Int.MIN_VALUE.toLong()..Int.MAX_VALUE.toLong()) {
        "knowledge acquisition $key is outside Int range"
    }
    return value.toInt()
}

internal fun requireKnowledgeId(
    value: String,
    label: String,
) {
    require(
        value.isNotEmpty() &&
            value.length <= 128 &&
            value[0] in 'a'..'z' &&
            value.all {
                it in 'a'..'z' ||
                    it in '0'..'9' ||
                    it == '.' ||
                    it == '_' ||
                    it == '-'
            }
    ) {
        "$label is invalid"
    }
}

internal fun requireKnowledgeSha(
    value: String,
    label: String,
) {
    require(
        value.length == 64 &&
            value.all { it in "0123456789abcdef" }
    ) {
        "$label must be lowercase SHA-256 hex"
    }
}

internal fun ByteArray.toKnowledgeHex(): String =
    joinToString("") {
        "%02x".format(it.toInt() and 0xff)
    }

internal fun strictKnowledgeUtf8(
    bytes: ByteArray,
): String =
    try {
        StandardCharsets.UTF_8.newDecoder()
            .onMalformedInput(CodingErrorAction.REPORT)
            .onUnmappableCharacter(
                CodingErrorAction.REPORT
            )
            .decode(ByteBuffer.wrap(bytes))
            .toString()
    } catch (exc: Exception) {
        throw IllegalArgumentException(
            "knowledge acquisition data is not strict UTF-8",
            exc,
        )
    }
