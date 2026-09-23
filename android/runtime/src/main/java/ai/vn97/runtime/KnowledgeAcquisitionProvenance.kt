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

data class VN97AcquisitionProvenanceRecord(
    val packageSha256: String,
    val capabilityId: String,
    val capabilityVersion: Long,
    val publisherKeyId: String,
    val publisherKeySha256: String,
    val signatureSha256: String,
    val payloadSha256: String,
    val sourceOrigin: String,
    val sourceLicense: String,
    val recordIds: List<Long>,
    val alreadyAcquired: Boolean,
    val proposalId: String = "",
    val proposalCapabilityId: String = "",
    val proposalEvidenceRecordIds: List<Long> = emptyList(),
    val fetchReceiptId: String = "",
    val fetchCanonicalUrl: String = "",
    val createdWallTimeMillis: Long,
) {
    init {
        requireKnowledgeSha(packageSha256, "packageSha256")
        requireKnowledgeId(capabilityId, "capabilityId")
        require(capabilityId == "knowledge" || capabilityId.startsWith("knowledge."))
        require(capabilityVersion in 1L..0xffff_ffffL)
        requireKnowledgeId(publisherKeyId, "publisherKeyId")
        requireKnowledgeSha(publisherKeySha256, "publisherKeySha256")
        requireKnowledgeSha(signatureSha256, "signatureSha256")
        requireKnowledgeSha(payloadSha256, "payloadSha256")
        requireBoundedProvenanceText(sourceOrigin, MAX_ORIGIN_BYTES, "sourceOrigin", true)
        requireBoundedProvenanceText(sourceLicense, MAX_LICENSE_BYTES, "sourceLicense", true)
        require(recordIds.isNotEmpty() && recordIds.all { it > 0L })
        require(recordIds.zipWithNext().all { (a, b) -> b > a })
        require(createdWallTimeMillis >= 0L)

        if (proposalId.isNotEmpty()) {
            requireKnowledgeSha(proposalId, "proposalId")
            requireKnowledgeId(proposalCapabilityId, "proposalCapabilityId")
            require(proposalCapabilityId.startsWith("knowledge."))
            require(proposalCapabilityId == capabilityId) {
                "acquisition proposal capability does not match reviewed capability"
            }
            require(
                proposalEvidenceRecordIds.size <=
                    VN97KnowledgeAcquisitionProposalEngine.MAX_EVIDENCE &&
                    proposalEvidenceRecordIds.all { it > 0L } &&
                    proposalEvidenceRecordIds.distinct().size ==
                    proposalEvidenceRecordIds.size
            ) {
                "acquisition proposal evidence IDs are invalid"
            }
        } else {
            require(proposalCapabilityId.isEmpty())
            require(proposalEvidenceRecordIds.isEmpty())
        }

        if (fetchReceiptId.isNotEmpty()) {
            requireKnowledgeSha(fetchReceiptId, "fetchReceiptId")
            requireBoundedProvenanceText(
                fetchCanonicalUrl,
                MAX_FETCH_URL_BYTES,
                "fetchCanonicalUrl",
                true,
            )
            require(fetchCanonicalUrl.startsWith("https://")) {
                "provenance fetch URL must use HTTPS"
            }
        } else {
            require(fetchCanonicalUrl.isEmpty())
        }
    }

    val provenanceId: String by lazy(LazyThreadSafetyMode.PUBLICATION) {
        provenanceSha(
            VnStrictJson.canonical(toJson())
                .toByteArray(StandardCharsets.UTF_8)
        )
    }

    internal fun toJson(): VnJsonObject =
        VnStrictJson.objectOf(
            "already_acquired" to
                VnStrictJson.bool(alreadyAcquired),
            "capability_id" to
                VnStrictJson.string(capabilityId),
            "capability_version" to
                VnStrictJson.long(capabilityVersion),
            "created_ms" to
                VnStrictJson.long(createdWallTimeMillis),
            "fetch_receipt_id" to
                VnStrictJson.string(fetchReceiptId),
            "fetch_url" to
                VnStrictJson.string(fetchCanonicalUrl),
            "package_sha256" to
                VnStrictJson.string(packageSha256),
            "payload_sha256" to
                VnStrictJson.string(payloadSha256),
            "proposal_capability_id" to
                VnStrictJson.string(proposalCapabilityId),
            "proposal_evidence_record_ids" to
                VnStrictJson.array(
                    proposalEvidenceRecordIds.map(VnStrictJson::long)
                ),
            "proposal_id" to
                VnStrictJson.string(proposalId),
            "publisher_key_id" to
                VnStrictJson.string(publisherKeyId),
            "publisher_key_sha256" to
                VnStrictJson.string(publisherKeySha256),
            "record_ids" to
                VnStrictJson.array(recordIds.map(VnStrictJson::long)),
            "schema" to
                VnStrictJson.string(SCHEMA),
            "signature_sha256" to
                VnStrictJson.string(signatureSha256),
            "source_license" to
                VnStrictJson.string(sourceLicense),
            "source_origin" to
                VnStrictJson.string(sourceOrigin),
        )

    companion object {
        const val SCHEMA = "VN97KPROV1"
        const val MAX_ORIGIN_BYTES = 4 * 1024
        const val MAX_LICENSE_BYTES = 512
        const val MAX_FETCH_URL_BYTES = 2_048
    }
}

class VN97AcquisitionProvenanceLedger(
    root: File,
) {
    private val rootPath =
        root.toPath().toAbsolutePath().normalize()

    init {
        Files.createDirectories(rootPath)
        require(
            Files.isDirectory(rootPath, LinkOption.NOFOLLOW_LINKS) &&
                !Files.isSymbolicLink(rootPath)
        ) {
            "acquisition provenance root must be a real directory"
        }
    }

    @Synchronized
    fun loadOrNull(
        packageSha256: String,
    ): VN97AcquisitionProvenanceRecord? {
        requireKnowledgeSha(packageSha256, "packageSha256")
        val target = target(packageSha256)
        if (!Files.exists(target, LinkOption.NOFOLLOW_LINKS)) {
            return null
        }
        require(
            Files.isRegularFile(target, LinkOption.NOFOLLOW_LINKS) &&
                !Files.isSymbolicLink(target)
        ) {
            "acquisition provenance target must be a regular file"
        }
        val size = Files.size(target)
        require(size in 1L..MAX_FILE_BYTES.toLong()) {
            "acquisition provenance file size is outside bounds"
        }
        val bytes = Files.readAllBytes(target)
        require(bytes.size.toLong() == size) {
            "acquisition provenance file changed while reading"
        }
        return decodeProvenance(bytes)
    }

    @Synchronized
    fun saveCompleted(
        record: VN97AcquisitionProvenanceRecord,
    ): VN97AcquisitionProvenanceRecord {
        val existing = loadOrNull(record.packageSha256)
        if (existing != null) {
            check(
                existing.packageSha256 == record.packageSha256 &&
                    existing.capabilityId == record.capabilityId &&
                    existing.capabilityVersion == record.capabilityVersion &&
                    existing.publisherKeyId == record.publisherKeyId &&
                    existing.publisherKeySha256 == record.publisherKeySha256 &&
                    existing.signatureSha256 == record.signatureSha256 &&
                    existing.payloadSha256 == record.payloadSha256 &&
                    existing.recordIds == record.recordIds
            ) {
                "completed acquisition provenance identity is immutable"
            }
            return existing
        }

        val bytes = encodeProvenance(record)
        require(bytes.size <= MAX_FILE_BYTES) {
            "acquisition provenance exceeds byte bound"
        }
        val target = target(record.packageSha256)
        require(
            !Files.exists(target, LinkOption.NOFOLLOW_LINKS) ||
                !Files.isSymbolicLink(target)
        ) {
            "acquisition provenance target must not be a symlink"
        }
        val temp = Files.createTempFile(
            rootPath,
            ".vn97kprov-",
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
                )
            } catch (exc: AtomicMoveNotSupportedException) {
                throw IllegalStateException(
                    "acquisition provenance requires atomic create",
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
        return checkNotNull(loadOrNull(record.packageSha256)) {
            "acquisition provenance post-write verification failed"
        }.also {
            check(it == record) {
                "acquisition provenance post-write identity changed"
            }
        }
    }

    private fun target(packageSha256: String) =
        rootPath.resolve("$packageSha256.vn97kprov1")

    companion object {
        private const val MAX_FILE_BYTES = 256 * 1024
    }
}

private fun encodeProvenance(
    record: VN97AcquisitionProvenanceRecord,
): ByteArray {
    val payload = VnStrictJson.canonical(record.toJson())
    val digest = provenanceSha(
        payload.toByteArray(StandardCharsets.UTF_8)
    )
    return buildString {
        append(VN97AcquisitionProvenanceRecord.SCHEMA)
        append('\n')
        append("sha256=")
        append(digest)
        append('\n')
        append(payload)
    }.toByteArray(StandardCharsets.UTF_8)
}

private fun decodeProvenance(
    bytes: ByteArray,
): VN97AcquisitionProvenanceRecord {
    val text = strictProvenanceUtf8(bytes)
    val first = text.indexOf('\n')
    val second =
        if (first >= 0) text.indexOf('\n', first + 1)
        else -1
    require(
        first > 0 &&
            second > first &&
            text.substring(0, first) ==
                VN97AcquisitionProvenanceRecord.SCHEMA
    ) {
        "acquisition provenance header is invalid"
    }
    val digestLine = text.substring(first + 1, second)
    require(digestLine.startsWith("sha256=")) {
        "acquisition provenance digest is missing"
    }
    val expected = digestLine.removePrefix("sha256=")
    requireKnowledgeSha(expected, "provenance digest")
    val payload = text.substring(second + 1)
    check(
        provenanceSha(
            payload.toByteArray(StandardCharsets.UTF_8)
        ) == expected
    ) {
        "acquisition provenance SHA-256 mismatch"
    }

    val root = VnStrictJson.parseObject(
        payload,
        VnJsonLimits(
            maxInputUtf8Bytes = 256 * 1024,
            maxDepth = 8,
            maxNodes = 4096,
            maxStringUtf8Bytes = 16 * 1024,
        ),
    )
    check(VnStrictJson.canonical(root) == payload) {
        "acquisition provenance is not canonical JSON"
    }
    val keys = setOf(
        "already_acquired",
        "capability_id",
        "capability_version",
        "created_ms",
        "fetch_receipt_id",
        "fetch_url",
        "package_sha256",
        "payload_sha256",
        "proposal_capability_id",
        "proposal_evidence_record_ids",
        "proposal_id",
        "publisher_key_id",
        "publisher_key_sha256",
        "record_ids",
        "schema",
        "signature_sha256",
        "source_license",
        "source_origin",
    )
    require(root.values.keys == keys) {
        "acquisition provenance fields are invalid"
    }
    require(
        root.provenanceString("schema") ==
            VN97AcquisitionProvenanceRecord.SCHEMA
    ) {
        "acquisition provenance schema mismatch"
    }
    return VN97AcquisitionProvenanceRecord(
        packageSha256 =
            root.provenanceString("package_sha256"),
        capabilityId =
            root.provenanceString("capability_id"),
        capabilityVersion =
            root.provenanceLong("capability_version"),
        publisherKeyId =
            root.provenanceString("publisher_key_id"),
        publisherKeySha256 =
            root.provenanceString("publisher_key_sha256"),
        signatureSha256 =
            root.provenanceString("signature_sha256"),
        payloadSha256 =
            root.provenanceString("payload_sha256"),
        sourceOrigin =
            root.provenanceString("source_origin"),
        sourceLicense =
            root.provenanceString("source_license"),
        recordIds =
            root.provenanceLongArray("record_ids"),
        alreadyAcquired =
            root.provenanceBool("already_acquired"),
        proposalId =
            root.provenanceString("proposal_id"),
        proposalCapabilityId =
            root.provenanceString("proposal_capability_id"),
        proposalEvidenceRecordIds =
            root.provenanceLongArray(
                "proposal_evidence_record_ids"
            ),
        fetchReceiptId =
            root.provenanceString("fetch_receipt_id"),
        fetchCanonicalUrl =
            root.provenanceString("fetch_url"),
        createdWallTimeMillis =
            root.provenanceLong("created_ms"),
    )
}

private fun VnJsonObject.provenanceString(
    key: String,
): String =
    (values[key] as? VnJsonString)?.value
        ?: throw IllegalArgumentException(
            "acquisition provenance $key must be string"
        )

private fun VnJsonObject.provenanceBool(
    key: String,
): Boolean =
    (values[key] as? VnJsonBoolean)?.value
        ?: throw IllegalArgumentException(
            "acquisition provenance $key must be bool"
        )

private fun VnJsonObject.provenanceLong(
    key: String,
): Long {
    val raw = (values[key] as? VnJsonNumber)?.canonical
        ?: throw IllegalArgumentException(
            "acquisition provenance $key must be integer"
        )
    require(raw.none { it == '.' || it == 'e' || it == 'E' })
    return raw.toLongOrNull()
        ?: throw IllegalArgumentException(
            "acquisition provenance $key is outside Long range"
        )
}

private fun VnJsonObject.provenanceLongArray(
    key: String,
): List<Long> {
    val array = values[key] as? VnJsonArray
        ?: throw IllegalArgumentException(
            "acquisition provenance $key must be array"
        )
    return array.values.mapIndexed { index, value ->
        val raw = (value as? VnJsonNumber)?.canonical
            ?: throw IllegalArgumentException(
                "acquisition provenance $key[$index] must be integer"
            )
        require(raw.none { it == '.' || it == 'e' || it == 'E' })
        raw.toLongOrNull()
            ?: throw IllegalArgumentException(
                "acquisition provenance $key[$index] is outside Long range"
            )
    }
}

private fun requireBoundedProvenanceText(
    value: String,
    maxBytes: Int,
    label: String,
    nonBlank: Boolean,
) {
    if (nonBlank) require(value.isNotBlank()) {
        "$label must not be blank"
    }
    require('\u0000' !in value) {
        "$label contains NUL"
    }
    require(
        value.toByteArray(StandardCharsets.UTF_8).size <=
            maxBytes
    ) {
        "$label exceeds UTF-8 byte bound"
    }
}

private fun strictProvenanceUtf8(
    bytes: ByteArray,
): String =
    try {
        StandardCharsets.UTF_8.newDecoder()
            .onMalformedInput(CodingErrorAction.REPORT)
            .onUnmappableCharacter(CodingErrorAction.REPORT)
            .decode(ByteBuffer.wrap(bytes))
            .toString()
    } catch (exc: Exception) {
        throw IllegalArgumentException(
            "acquisition provenance is not strict UTF-8",
            exc,
        )
    }

private fun provenanceSha(
    bytes: ByteArray,
): String =
    MessageDigest.getInstance("SHA-256")
        .digest(bytes)
        .toKnowledgeHex()
