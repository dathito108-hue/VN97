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

enum class VN97ImprovementCandidateState {
    REVIEWED,
    REJECTED,
}

data class VN97ImprovementCandidateSpec(
    val baselineActivationId: String,
    val baselineArtifactSha256: String,
    val baselinePackageSha256: String,
    val baselineCapabilityVersion: Long,
    val candidatePackageSha256: String,
    val candidateCapabilityVersion: Long,
    val candidatePublisherKeyId: String,
    val candidatePublisherKeySha256: String,
    val candidatePlanSha256: String,
    val sourceOrigin: String,
    val sourceLicense: String,
    val objective: String,
) {
    init {
        requireImprovementSha(
            baselineActivationId,
            "baselineActivationId",
        )
        requireImprovementSha(
            baselineArtifactSha256,
            "baselineArtifactSha256",
        )
        requireImprovementSha(
            baselinePackageSha256,
            "baselinePackageSha256",
        )
        require(
            baselineCapabilityVersion in
                1L..0xffff_ffffL
        ) {
            "baseline capability version is invalid"
        }
        requireImprovementSha(
            candidatePackageSha256,
            "candidatePackageSha256",
        )
        require(
            candidateCapabilityVersion in
                1L..0xffff_ffffL
        ) {
            "candidate capability version is invalid"
        }
        require(
            candidateCapabilityVersion >
                baselineCapabilityVersion
        ) {
            "self-improvement candidate must be a strict version upgrade"
        }
        require(
            candidatePackageSha256 !=
                baselinePackageSha256
        ) {
            "self-improvement candidate must differ from baseline package"
        }
        requireImprovementId(
            candidatePublisherKeyId,
            "candidatePublisherKeyId",
        )
        requireImprovementSha(
            candidatePublisherKeySha256,
            "candidatePublisherKeySha256",
        )
        requireImprovementSha(
            candidatePlanSha256,
            "candidatePlanSha256",
        )
        requireImprovementText(
            sourceOrigin,
            MAX_ORIGIN_BYTES,
            "sourceOrigin",
        )
        requireImprovementText(
            sourceLicense,
            MAX_LICENSE_BYTES,
            "sourceLicense",
        )
        requireImprovementText(
            objective,
            MAX_OBJECTIVE_BYTES,
            "objective",
        )
    }

    val candidateId: String by lazy(
        LazyThreadSafetyMode.PUBLICATION
    ) {
        improvementSha(
            VnStrictJson.canonical(identityJson())
                .toByteArray(StandardCharsets.UTF_8)
        )
    }

    internal fun identityJson(): VnJsonObject =
        VnStrictJson.objectOf(
            "baseline_activation_id" to
                VnStrictJson.string(
                    baselineActivationId
                ),
            "baseline_artifact_sha256" to
                VnStrictJson.string(
                    baselineArtifactSha256
                ),
            "baseline_capability_version" to
                VnStrictJson.long(
                    baselineCapabilityVersion
                ),
            "baseline_package_sha256" to
                VnStrictJson.string(
                    baselinePackageSha256
                ),
            "candidate_capability_version" to
                VnStrictJson.long(
                    candidateCapabilityVersion
                ),
            "candidate_package_sha256" to
                VnStrictJson.string(
                    candidatePackageSha256
                ),
            "candidate_plan_sha256" to
                VnStrictJson.string(
                    candidatePlanSha256
                ),
            "candidate_publisher_key_id" to
                VnStrictJson.string(
                    candidatePublisherKeyId
                ),
            "candidate_publisher_key_sha256" to
                VnStrictJson.string(
                    candidatePublisherKeySha256
                ),
            "objective" to
                VnStrictJson.string(objective),
            "schema" to
                VnStrictJson.string(
                    "VN97IMPIDENT1"
                ),
            "source_license" to
                VnStrictJson.string(
                    sourceLicense
                ),
            "source_origin" to
                VnStrictJson.string(
                    sourceOrigin
                ),
        )

    companion object {
        const val MAX_OBJECTIVE_BYTES =
            4 * 1024
        const val MAX_ORIGIN_BYTES =
            4 * 1024
        const val MAX_LICENSE_BYTES = 512
    }
}

data class VN97ImprovementCandidateRecord(
    val candidateId: String,
    val spec: VN97ImprovementCandidateSpec,
    val state: VN97ImprovementCandidateState,
    val rejectionReason: String,
    val createdWallTimeMillis: Long,
    val updatedWallTimeMillis: Long,
) {
    init {
        require(candidateId == spec.candidateId) {
            "self-improvement candidate identity mismatch"
        }
        require(createdWallTimeMillis >= 0L)
        require(
            updatedWallTimeMillis >=
                createdWallTimeMillis
        )
        when (state) {
            VN97ImprovementCandidateState.REVIEWED ->
                require(rejectionReason.isEmpty()) {
                    "reviewed candidate must not have rejection reason"
                }
            VN97ImprovementCandidateState.REJECTED ->
                requireImprovementText(
                    rejectionReason,
                    MAX_REJECTION_BYTES,
                    "rejectionReason",
                )
        }
    }

    companion object {
        const val MAX_REJECTION_BYTES =
            2 * 1024
    }
}

class VN97ImprovementCandidateLedger(
    root: File,
) {
    private val rootPath =
        root.toPath().toAbsolutePath().normalize()

    init {
        Files.createDirectories(rootPath)
        require(
            Files.isDirectory(
                rootPath,
                LinkOption.NOFOLLOW_LINKS,
            ) &&
                !Files.isSymbolicLink(rootPath)
        ) {
            "self-improvement ledger root must be a real directory"
        }
    }

    @Synchronized
    fun registerReviewed(
        spec: VN97ImprovementCandidateSpec,
        nowWallTimeMillis: Long =
            System.currentTimeMillis(),
    ): VN97ImprovementCandidateRecord {
        require(nowWallTimeMillis >= 0L)
        val existing =
            loadOrNull(spec.candidateId)
        if (existing != null) {
            check(existing.spec == spec) {
                "self-improvement candidate identity collision"
            }
            return existing
        }
        val record =
            VN97ImprovementCandidateRecord(
                candidateId =
                    spec.candidateId,
                spec = spec,
                state =
                    VN97ImprovementCandidateState
                        .REVIEWED,
                rejectionReason = "",
                createdWallTimeMillis =
                    nowWallTimeMillis,
                updatedWallTimeMillis =
                    nowWallTimeMillis,
            )
        saveNew(record)
        return record
    }

    @Synchronized
    fun reject(
        candidateId: String,
        reason: String,
        nowWallTimeMillis: Long =
            System.currentTimeMillis(),
    ): VN97ImprovementCandidateRecord {
        requireImprovementSha(
            candidateId,
            "candidateId",
        )
        requireImprovementText(
            reason,
            VN97ImprovementCandidateRecord
                .MAX_REJECTION_BYTES,
            "rejectionReason",
        )
        require(nowWallTimeMillis >= 0L)
        val current = checkNotNull(
            loadOrNull(candidateId)
        ) {
            "self-improvement candidate does not exist"
        }
        if (
            current.state ==
                VN97ImprovementCandidateState
                    .REJECTED
        ) {
            check(
                current.rejectionReason == reason
            ) {
                "rejected candidate is immutable"
            }
            return current
        }
        val next = current.copy(
            state =
                VN97ImprovementCandidateState
                    .REJECTED,
            rejectionReason = reason,
            updatedWallTimeMillis =
                maxOf(
                    current.updatedWallTimeMillis,
                    nowWallTimeMillis,
                ),
        )
        replace(current, next)
        return next
    }

    @Synchronized
    fun loadOrNull(
        candidateId: String,
    ): VN97ImprovementCandidateRecord? {
        requireImprovementSha(
            candidateId,
            "candidateId",
        )
        val target = target(candidateId)
        if (
            !Files.exists(
                target,
                LinkOption.NOFOLLOW_LINKS,
            )
        ) {
            return null
        }
        require(
            Files.isRegularFile(
                target,
                LinkOption.NOFOLLOW_LINKS,
            ) &&
                !Files.isSymbolicLink(target)
        ) {
            "self-improvement ledger target must be a regular file"
        }
        val size = Files.size(target)
        require(
            size in 1L..MAX_FILE_BYTES.toLong()
        ) {
            "self-improvement ledger size is outside bounds"
        }
        val bytes = Files.readAllBytes(target)
        require(bytes.size.toLong() == size) {
            "self-improvement ledger changed while reading"
        }
        return decodeImprovement(bytes).also {
            check(it.candidateId == candidateId) {
                "self-improvement ledger filename identity mismatch"
            }
        }
    }

    private fun saveNew(
        record: VN97ImprovementCandidateRecord,
    ) {
        val target =
            target(record.candidateId)
        check(
            !Files.exists(
                target,
                LinkOption.NOFOLLOW_LINKS,
            )
        ) {
            "self-improvement candidate already exists"
        }
        writeAtomic(
            target = target,
            bytes =
                encodeImprovement(record),
            replace = false,
        )
        check(loadOrNull(record.candidateId) == record) {
            "self-improvement candidate post-write verification failed"
        }
    }

    private fun replace(
        previous: VN97ImprovementCandidateRecord,
        next: VN97ImprovementCandidateRecord,
    ) {
        check(
            previous.candidateId ==
                next.candidateId &&
                previous.spec == next.spec &&
                previous.createdWallTimeMillis ==
                next.createdWallTimeMillis
        ) {
            "self-improvement immutable identity changed"
        }
        check(
            previous.state ==
                VN97ImprovementCandidateState
                    .REVIEWED &&
                next.state ==
                    VN97ImprovementCandidateState
                        .REJECTED
        ) {
            "self-improvement state transition is not allowed"
        }
        writeAtomic(
            target =
                target(next.candidateId),
            bytes =
                encodeImprovement(next),
            replace = true,
        )
        check(loadOrNull(next.candidateId) == next) {
            "self-improvement transition post-write verification failed"
        }
    }

    private fun writeAtomic(
        target: java.nio.file.Path,
        bytes: ByteArray,
        replace: Boolean,
    ) {
        require(bytes.size <= MAX_FILE_BYTES)
        if (
            Files.exists(
                target,
                LinkOption.NOFOLLOW_LINKS,
            ) &&
                Files.isSymbolicLink(target)
        ) {
            throw IllegalStateException(
                "self-improvement ledger target must not be a symlink"
            )
        }
        val temp = Files.createTempFile(
            rootPath,
            ".vn97imp-",
            ".tmp",
        )
        try {
            FileOutputStream(
                temp.toFile()
            ).use { out ->
                out.write(bytes)
                out.flush()
                out.fd.sync()
            }
            try {
                if (replace) {
                    Files.move(
                        temp,
                        target,
                        StandardCopyOption
                            .ATOMIC_MOVE,
                        StandardCopyOption
                            .REPLACE_EXISTING,
                    )
                } else {
                    Files.move(
                        temp,
                        target,
                        StandardCopyOption
                            .ATOMIC_MOVE,
                    )
                }
            } catch (
                exc: AtomicMoveNotSupportedException
            ) {
                throw IllegalStateException(
                    "self-improvement ledger requires atomic move",
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
    }

    private fun target(
        candidateId: String,
    ) =
        rootPath.resolve(
            "$candidateId.vn97imp1"
        )

    companion object {
        private const val MAX_FILE_BYTES =
            64 * 1024
    }
}

private fun encodeImprovement(
    record: VN97ImprovementCandidateRecord,
): ByteArray {
    val spec = record.spec
    val payload =
        VnStrictJson.canonical(
            VnStrictJson.objectOf(
                "baseline_activation_id" to
                    VnStrictJson.string(
                        spec.baselineActivationId
                    ),
                "baseline_artifact_sha256" to
                    VnStrictJson.string(
                        spec.baselineArtifactSha256
                    ),
                "baseline_capability_version" to
                    VnStrictJson.long(
                        spec.baselineCapabilityVersion
                    ),
                "baseline_package_sha256" to
                    VnStrictJson.string(
                        spec.baselinePackageSha256
                    ),
                "candidate_capability_version" to
                    VnStrictJson.long(
                        spec.candidateCapabilityVersion
                    ),
                "candidate_id" to
                    VnStrictJson.string(
                        record.candidateId
                    ),
                "candidate_package_sha256" to
                    VnStrictJson.string(
                        spec.candidatePackageSha256
                    ),
                "candidate_plan_sha256" to
                    VnStrictJson.string(
                        spec.candidatePlanSha256
                    ),
                "candidate_publisher_key_id" to
                    VnStrictJson.string(
                        spec.candidatePublisherKeyId
                    ),
                "candidate_publisher_key_sha256" to
                    VnStrictJson.string(
                        spec.candidatePublisherKeySha256
                    ),
                "created_ms" to
                    VnStrictJson.long(
                        record.createdWallTimeMillis
                    ),
                "objective" to
                    VnStrictJson.string(
                        spec.objective
                    ),
                "rejection_reason" to
                    VnStrictJson.string(
                        record.rejectionReason
                    ),
                "schema" to
                    VnStrictJson.string(
                        "VN97IMP1"
                    ),
                "source_license" to
                    VnStrictJson.string(
                        spec.sourceLicense
                    ),
                "source_origin" to
                    VnStrictJson.string(
                        spec.sourceOrigin
                    ),
                "state" to
                    VnStrictJson.string(
                        record.state.name
                    ),
                "updated_ms" to
                    VnStrictJson.long(
                        record.updatedWallTimeMillis
                    ),
            )
        )
    val digest = improvementSha(
        payload.toByteArray(
            StandardCharsets.UTF_8
        )
    )
    return buildString {
        append("VN97IMP1")
        append('\n')
        append("sha256=")
        append(digest)
        append('\n')
        append(payload)
    }.toByteArray(StandardCharsets.UTF_8)
}

private fun decodeImprovement(
    bytes: ByteArray,
): VN97ImprovementCandidateRecord {
    val text = strictImprovementUtf8(bytes)
    val first = text.indexOf('\n')
    val second =
        if (first >= 0) {
            text.indexOf('\n', first + 1)
        } else {
            -1
        }
    require(
        first > 0 &&
            second > first &&
            text.substring(0, first) ==
                "VN97IMP1"
    ) {
        "self-improvement ledger header is invalid"
    }
    val digestLine =
        text.substring(first + 1, second)
    require(
        digestLine.startsWith("sha256=")
    ) {
        "self-improvement ledger digest is missing"
    }
    val expected =
        digestLine.removePrefix("sha256=")
    requireImprovementSha(
        expected,
        "ledger digest",
    )
    val payload =
        text.substring(second + 1)
    check(
        improvementSha(
            payload.toByteArray(
                StandardCharsets.UTF_8
            )
        ) == expected
    ) {
        "self-improvement ledger SHA-256 mismatch"
    }
    val root = VnStrictJson.parseObject(
        payload,
        VnJsonLimits(
            maxInputUtf8Bytes = 64 * 1024,
            maxDepth = 8,
            maxNodes = 1024,
            maxStringUtf8Bytes = 8 * 1024,
        ),
    )
    check(
        VnStrictJson.canonical(root) ==
            payload
    ) {
        "self-improvement ledger is not canonical JSON"
    }
    val expectedKeys = setOf(
        "baseline_activation_id",
        "baseline_artifact_sha256",
        "baseline_capability_version",
        "baseline_package_sha256",
        "candidate_capability_version",
        "candidate_id",
        "candidate_package_sha256",
        "candidate_plan_sha256",
        "candidate_publisher_key_id",
        "candidate_publisher_key_sha256",
        "created_ms",
        "objective",
        "rejection_reason",
        "schema",
        "source_license",
        "source_origin",
        "state",
        "updated_ms",
    )
    require(root.values.keys == expectedKeys) {
        "self-improvement ledger fields are invalid"
    }
    require(
        root.impString("schema") ==
            "VN97IMP1"
    )
    val spec = VN97ImprovementCandidateSpec(
        baselineActivationId =
            root.impString(
                "baseline_activation_id"
            ),
        baselineArtifactSha256 =
            root.impString(
                "baseline_artifact_sha256"
            ),
        baselinePackageSha256 =
            root.impString(
                "baseline_package_sha256"
            ),
        baselineCapabilityVersion =
            root.impLong(
                "baseline_capability_version"
            ),
        candidatePackageSha256 =
            root.impString(
                "candidate_package_sha256"
            ),
        candidateCapabilityVersion =
            root.impLong(
                "candidate_capability_version"
            ),
        candidatePublisherKeyId =
            root.impString(
                "candidate_publisher_key_id"
            ),
        candidatePublisherKeySha256 =
            root.impString(
                "candidate_publisher_key_sha256"
            ),
        candidatePlanSha256 =
            root.impString(
                "candidate_plan_sha256"
            ),
        sourceOrigin =
            root.impString("source_origin"),
        sourceLicense =
            root.impString("source_license"),
        objective =
            root.impString("objective"),
    )
    val state = try {
        VN97ImprovementCandidateState
            .valueOf(root.impString("state"))
    } catch (exc: IllegalArgumentException) {
        throw IllegalArgumentException(
            "self-improvement candidate state is invalid",
            exc,
        )
    }
    return VN97ImprovementCandidateRecord(
        candidateId =
            root.impString("candidate_id"),
        spec = spec,
        state = state,
        rejectionReason =
            root.impString(
                "rejection_reason"
            ),
        createdWallTimeMillis =
            root.impLong("created_ms"),
        updatedWallTimeMillis =
            root.impLong("updated_ms"),
    )
}

private fun VnJsonObject.impString(
    key: String,
): String =
    (values[key] as? VnJsonString)?.value
        ?: throw IllegalArgumentException(
            "self-improvement $key must be string"
        )

private fun VnJsonObject.impLong(
    key: String,
): Long {
    val raw =
        (values[key] as? VnJsonNumber)
            ?.canonical
            ?: throw IllegalArgumentException(
                "self-improvement $key must be integer"
            )
    require(
        raw.none {
            it == '.' ||
                it == 'e' ||
                it == 'E'
        }
    )
    return raw.toLongOrNull()
        ?: throw IllegalArgumentException(
            "self-improvement $key is outside Long range"
        )
}

private fun requireImprovementId(
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

private fun requireImprovementSha(
    value: String,
    label: String,
) {
    require(
        value.length == 64 &&
            value.all {
                it in "0123456789abcdef"
            }
    ) {
        "$label must be lowercase SHA-256 hex"
    }
}

private fun requireImprovementText(
    value: String,
    maxBytes: Int,
    label: String,
) {
    require(value.isNotBlank()) {
        "$label must not be blank"
    }
    require('\u0000' !in value) {
        "$label contains NUL"
    }
    require(
        value.toByteArray(
            StandardCharsets.UTF_8
        ).size <= maxBytes
    ) {
        "$label exceeds UTF-8 byte bound"
    }
}

private fun strictImprovementUtf8(
    bytes: ByteArray,
): String =
    try {
        StandardCharsets.UTF_8.newDecoder()
            .onMalformedInput(
                CodingErrorAction.REPORT
            )
            .onUnmappableCharacter(
                CodingErrorAction.REPORT
            )
            .decode(ByteBuffer.wrap(bytes))
            .toString()
    } catch (exc: Exception) {
        throw IllegalArgumentException(
            "self-improvement ledger is not strict UTF-8",
            exc,
        )
    }

private fun improvementSha(
    bytes: ByteArray,
): String =
    MessageDigest.getInstance("SHA-256")
        .digest(bytes)
        .joinToString("") {
            "%02x".format(
                it.toInt() and 0xff
            )
        }
