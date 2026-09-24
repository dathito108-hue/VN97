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

enum class VN97ImprovementPromotionState {
    PENDING,
    PROMOTED,
    ROLLED_BACK,
    ABORTED,
}

data class VN97ImprovementPromotionSpec(
    val candidateId: String,
    val evaluationId: String,
    val baselineActivationId: String,
    val baselineArtifactSha256: String,
    val baselinePackageSha256: String,
    val baselineCapabilityVersion: Long,
    val candidatePackageSha256: String,
    val candidateArtifactSha256: String,
    val candidateCapabilityVersion: Long,
    val candidatePlanSha256: String,
    val candidatePublisherKeyId: String,
    val requestedWallTimeMillis: Long,
) {
    init {
        listOf(
            candidateId,
            evaluationId,
            baselineActivationId,
            baselineArtifactSha256,
            baselinePackageSha256,
            candidatePackageSha256,
            candidateArtifactSha256,
            candidatePlanSha256,
        ).forEach {
            requirePromotionSha(it, "promotion identity")
        }
        requirePromotionId(
            candidatePublisherKeyId,
            "candidatePublisherKeyId",
        )
        require(
            baselineCapabilityVersion in
                1L..0xffff_ffffL &&
                candidateCapabilityVersion in
                    1L..0xffff_ffffL &&
                candidateCapabilityVersion >
                    baselineCapabilityVersion
        )
        require(requestedWallTimeMillis >= 0L)
    }

    val promotionId: String by lazy(
        LazyThreadSafetyMode.PUBLICATION
    ) {
        promotionSha(
            VnStrictJson.canonical(
                VnStrictJson.objectOf(
                    "baseline_activation_id" to
                        VnStrictJson.string(
                            baselineActivationId
                        ),
                    "candidate_id" to
                        VnStrictJson.string(
                            candidateId
                        ),
                    "candidate_package_sha256" to
                        VnStrictJson.string(
                            candidatePackageSha256
                        ),
                    "evaluation_id" to
                        VnStrictJson.string(
                            evaluationId
                        ),
                    "requested_ms" to
                        VnStrictJson.long(
                            requestedWallTimeMillis
                        ),
                    "schema" to
                        VnStrictJson.string(
                            "VN97IMPPROMIDENT1"
                        ),
                )
            ).toByteArray(StandardCharsets.UTF_8)
        )
    }
}

data class VN97ImprovementPromotionRecord(
    val promotionId: String,
    val spec: VN97ImprovementPromotionSpec,
    val state: VN97ImprovementPromotionState,
    val resultingActivationId: String,
    val resultingArtifactSha256: String,
    val rollbackRestoredActivationId: String,
    val detail: String,
    val completedWallTimeMillis: Long,
) {
    init {
        require(promotionId == spec.promotionId)
        when (state) {
            VN97ImprovementPromotionState.PENDING -> {
                require(resultingActivationId.isEmpty())
                require(resultingArtifactSha256.isEmpty())
                require(rollbackRestoredActivationId.isEmpty())
                require(detail.isEmpty())
                require(completedWallTimeMillis == 0L)
            }
            VN97ImprovementPromotionState.PROMOTED -> {
                requirePromotionSha(
                    resultingActivationId,
                    "resultingActivationId",
                )
                require(
                    resultingArtifactSha256 ==
                        spec.candidateArtifactSha256
                )
                require(rollbackRestoredActivationId.isEmpty())
                require(detail.isEmpty())
                require(
                    completedWallTimeMillis >=
                        spec.requestedWallTimeMillis
                )
            }
            VN97ImprovementPromotionState.ROLLED_BACK -> {
                if (resultingActivationId.isNotEmpty()) {
                    requirePromotionSha(
                        resultingActivationId,
                        "resultingActivationId",
                    )
                }
                if (resultingArtifactSha256.isNotEmpty()) {
                    require(
                        resultingArtifactSha256 ==
                            spec.candidateArtifactSha256
                    )
                }
                require(
                    rollbackRestoredActivationId ==
                        spec.baselineActivationId
                )
                requirePromotionText(
                    detail,
                    MAX_DETAIL_BYTES,
                    "detail",
                )
                require(
                    completedWallTimeMillis >=
                        spec.requestedWallTimeMillis
                )
            }
            VN97ImprovementPromotionState.ABORTED -> {
                require(resultingActivationId.isEmpty())
                require(resultingArtifactSha256.isEmpty())
                require(
                    rollbackRestoredActivationId.isEmpty() ||
                        rollbackRestoredActivationId ==
                            spec.baselineActivationId
                )
                requirePromotionText(
                    detail,
                    MAX_DETAIL_BYTES,
                    "detail",
                )
                require(
                    completedWallTimeMillis >=
                        spec.requestedWallTimeMillis
                )
            }
        }
    }

    companion object {
        const val MAX_DETAIL_BYTES = 2 * 1024
    }
}

class VN97ImprovementPromotionLedger(
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
            "promotion ledger root must be a real directory"
        }
    }

    @Synchronized
    fun begin(
        spec: VN97ImprovementPromotionSpec,
    ): VN97ImprovementPromotionRecord {
        pendingForCandidate(spec.candidateId)
            ?.let { current ->
                check(current.spec == spec) {
                    "candidate already has a different pending promotion"
                }
                return current
            }
        val record =
            VN97ImprovementPromotionRecord(
                promotionId = spec.promotionId,
                spec = spec,
                state =
                    VN97ImprovementPromotionState.PENDING,
                resultingActivationId = "",
                resultingArtifactSha256 = "",
                rollbackRestoredActivationId = "",
                detail = "",
                completedWallTimeMillis = 0L,
            )
        writeNew(record)
        return record
    }

    @Synchronized
    fun completePromoted(
        promotionId: String,
        activationId: String,
        artifactSha256: String,
        nowWallTimeMillis: Long =
            System.currentTimeMillis(),
    ): VN97ImprovementPromotionRecord =
        complete(
            promotionId = promotionId,
            state =
                VN97ImprovementPromotionState.PROMOTED,
            activationId = activationId,
            artifactSha256 = artifactSha256,
            restoredActivationId = "",
            detail = "",
            nowWallTimeMillis = nowWallTimeMillis,
        )

    @Synchronized
    fun completeRolledBack(
        promotionId: String,
        activationId: String,
        artifactSha256: String,
        detail: String,
        nowWallTimeMillis: Long =
            System.currentTimeMillis(),
    ): VN97ImprovementPromotionRecord {
        val pending = checkNotNull(
            loadOrNull(promotionId)
        )
        return complete(
            promotionId = promotionId,
            state =
                VN97ImprovementPromotionState.ROLLED_BACK,
            activationId = activationId,
            artifactSha256 = artifactSha256,
            restoredActivationId =
                pending.spec.baselineActivationId,
            detail = detail,
            nowWallTimeMillis = nowWallTimeMillis,
        )
    }

    @Synchronized
    fun completeAborted(
        promotionId: String,
        detail: String,
        baselineStillActive: Boolean,
        nowWallTimeMillis: Long =
            System.currentTimeMillis(),
    ): VN97ImprovementPromotionRecord {
        val pending = checkNotNull(
            loadOrNull(promotionId)
        )
        return complete(
            promotionId = promotionId,
            state =
                VN97ImprovementPromotionState.ABORTED,
            activationId = "",
            artifactSha256 = "",
            restoredActivationId =
                if (baselineStillActive) {
                    pending.spec.baselineActivationId
                } else {
                    ""
                },
            detail = detail,
            nowWallTimeMillis = nowWallTimeMillis,
        )
    }

    @Synchronized
    fun loadOrNull(
        promotionId: String,
    ): VN97ImprovementPromotionRecord? {
        requirePromotionSha(
            promotionId,
            "promotionId",
        )
        val target = target(promotionId)
        if (!Files.exists(target, LinkOption.NOFOLLOW_LINKS)) {
            return null
        }
        require(
            Files.isRegularFile(
                target,
                LinkOption.NOFOLLOW_LINKS,
            ) &&
                !Files.isSymbolicLink(target)
        ) {
            "promotion ledger target must be a regular file"
        }
        val size = Files.size(target)
        require(size in 1L..MAX_FILE_BYTES.toLong())
        return decodePromotion(
            Files.readAllBytes(target)
        ).also {
            check(it.promotionId == promotionId)
        }
    }

    @Synchronized
    fun pendingForCandidate(
        candidateId: String,
    ): VN97ImprovementPromotionRecord? {
        requirePromotionSha(
            candidateId,
            "candidateId",
        )
        var found:
            VN97ImprovementPromotionRecord? = null
        var count = 0
        Files.newDirectoryStream(
            rootPath,
            "*.vn97impprom1",
        ).use { entries ->
            for (path in entries) {
                count += 1
                require(count <= MAX_FILES) {
                    "promotion ledger file count exceeds bound"
                }
                require(
                    Files.isRegularFile(
                        path,
                        LinkOption.NOFOLLOW_LINKS,
                    ) &&
                        !Files.isSymbolicLink(path)
                )
                val name =
                    path.fileName.toString()
                val id =
                    name.removeSuffix(
                        ".vn97impprom1"
                    )
                require(
                    name.endsWith(
                        ".vn97impprom1"
                    ) &&
                        id.length == 64
                )
                val record =
                    loadOrNull(id)
                        ?: throw IllegalStateException(
                            "promotion ledger entry disappeared"
                        )
                if (
                    record.spec.candidateId ==
                        candidateId &&
                        record.state ==
                            VN97ImprovementPromotionState
                                .PENDING
                ) {
                    check(found == null) {
                        "multiple pending promotions exist for candidate"
                    }
                    found = record
                }
            }
        }
        return found
    }

    private fun complete(
        promotionId: String,
        state: VN97ImprovementPromotionState,
        activationId: String,
        artifactSha256: String,
        restoredActivationId: String,
        detail: String,
        nowWallTimeMillis: Long,
    ): VN97ImprovementPromotionRecord {
        require(
            state !=
                VN97ImprovementPromotionState.PENDING
        )
        val current = checkNotNull(
            loadOrNull(promotionId)
        ) {
            "promotion attempt does not exist"
        }
        if (
            current.state !=
                VN97ImprovementPromotionState.PENDING
        ) {
            return current
        }
        val next = current.copy(
            state = state,
            resultingActivationId =
                activationId,
            resultingArtifactSha256 =
                artifactSha256,
            rollbackRestoredActivationId =
                restoredActivationId,
            detail = detail,
            completedWallTimeMillis =
                nowWallTimeMillis,
        )
        replace(current, next)
        return next
    }

    private fun writeNew(
        record: VN97ImprovementPromotionRecord,
    ) {
        val target = target(record.promotionId)
        check(
            !Files.exists(
                target,
                LinkOption.NOFOLLOW_LINKS,
            )
        ) {
            "promotion attempt already exists"
        }
        writeAtomic(
            target,
            encodePromotion(record),
            replace = false,
        )
        check(
            loadOrNull(record.promotionId) ==
                record
        )
    }

    private fun replace(
        previous: VN97ImprovementPromotionRecord,
        next: VN97ImprovementPromotionRecord,
    ) {
        check(
            previous.promotionId ==
                next.promotionId &&
                previous.spec == next.spec &&
                previous.state ==
                    VN97ImprovementPromotionState
                        .PENDING &&
                next.state !=
                    VN97ImprovementPromotionState
                        .PENDING
        ) {
            "promotion ledger transition is invalid"
        }
        writeAtomic(
            target(next.promotionId),
            encodePromotion(next),
            replace = true,
        )
        check(loadOrNull(next.promotionId) == next)
    }

    private fun writeAtomic(
        target: java.nio.file.Path,
        bytes: ByteArray,
        replace: Boolean,
    ) {
        require(bytes.size <= MAX_FILE_BYTES)
        require(
            !Files.exists(
                target,
                LinkOption.NOFOLLOW_LINKS,
            ) ||
                !Files.isSymbolicLink(target)
        )
        val temp =
            Files.createTempFile(
                rootPath,
                ".vn97prom-",
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
                val options =
                    if (replace) {
                        arrayOf(
                            StandardCopyOption
                                .ATOMIC_MOVE,
                            StandardCopyOption
                                .REPLACE_EXISTING,
                        )
                    } else {
                        arrayOf(
                            StandardCopyOption
                                .ATOMIC_MOVE
                        )
                    }
                Files.move(
                    temp,
                    target,
                    *options,
                )
            } catch (
                exc: AtomicMoveNotSupportedException
            ) {
                throw IllegalStateException(
                    "promotion ledger requires atomic move",
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
        promotionId: String,
    ) =
        rootPath.resolve(
            "$promotionId.vn97impprom1"
        )

    companion object {
        private const val MAX_FILE_BYTES =
            128 * 1024
        private const val MAX_FILES = 256
    }
}

private fun encodePromotion(
    record: VN97ImprovementPromotionRecord,
): ByteArray {
    val payload =
        VnStrictJson.canonical(
            promotionJson(record)
        )
    val digest =
        promotionSha(
            payload.toByteArray(
                StandardCharsets.UTF_8
            )
        )
    return buildString {
        append("VN97IMPPROM1")
        append('\n')
        append("sha256=")
        append(digest)
        append('\n')
        append(payload)
    }.toByteArray(StandardCharsets.UTF_8)
}

private fun promotionJson(
    record: VN97ImprovementPromotionRecord,
): VnJsonObject {
    val s = record.spec
    return VnStrictJson.objectOf(
        "baseline_activation_id" to
            VnStrictJson.string(
                s.baselineActivationId
            ),
        "baseline_artifact_sha256" to
            VnStrictJson.string(
                s.baselineArtifactSha256
            ),
        "baseline_capability_version" to
            VnStrictJson.long(
                s.baselineCapabilityVersion
            ),
        "baseline_package_sha256" to
            VnStrictJson.string(
                s.baselinePackageSha256
            ),
        "candidate_artifact_sha256" to
            VnStrictJson.string(
                s.candidateArtifactSha256
            ),
        "candidate_capability_version" to
            VnStrictJson.long(
                s.candidateCapabilityVersion
            ),
        "candidate_id" to
            VnStrictJson.string(
                s.candidateId
            ),
        "candidate_package_sha256" to
            VnStrictJson.string(
                s.candidatePackageSha256
            ),
        "candidate_plan_sha256" to
            VnStrictJson.string(
                s.candidatePlanSha256
            ),
        "candidate_publisher_key_id" to
            VnStrictJson.string(
                s.candidatePublisherKeyId
            ),
        "completed_ms" to
            VnStrictJson.long(
                record.completedWallTimeMillis
            ),
        "detail" to
            VnStrictJson.string(
                record.detail
            ),
        "evaluation_id" to
            VnStrictJson.string(
                s.evaluationId
            ),
        "promotion_id" to
            VnStrictJson.string(
                record.promotionId
            ),
        "requested_ms" to
            VnStrictJson.long(
                s.requestedWallTimeMillis
            ),
        "resulting_activation_id" to
            VnStrictJson.string(
                record.resultingActivationId
            ),
        "resulting_artifact_sha256" to
            VnStrictJson.string(
                record.resultingArtifactSha256
            ),
        "rollback_restored_activation_id" to
            VnStrictJson.string(
                record.rollbackRestoredActivationId
            ),
        "schema" to
            VnStrictJson.string(
                "VN97IMPPROM1"
            ),
        "state" to
            VnStrictJson.string(
                record.state.name
            ),
    )
}

private fun decodePromotion(
    bytes: ByteArray,
): VN97ImprovementPromotionRecord {
    val text = strictPromotionUtf8(bytes)
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
                "VN97IMPPROM1"
    )
    val digestLine =
        text.substring(first + 1, second)
    require(digestLine.startsWith("sha256="))
    val expected =
        digestLine.removePrefix("sha256=")
    requirePromotionSha(
        expected,
        "promotion digest",
    )
    val payload =
        text.substring(second + 1)
    check(
        promotionSha(
            payload.toByteArray(
                StandardCharsets.UTF_8
            )
        ) == expected
    ) {
        "promotion ledger SHA-256 mismatch"
    }

    val root =
        VnStrictJson.parseObject(
            payload,
            VnJsonLimits(
                maxInputUtf8Bytes =
                    128 * 1024,
                maxDepth = 8,
                maxNodes = 2048,
                maxStringUtf8Bytes =
                    8 * 1024,
            ),
        )
    check(
        VnStrictJson.canonical(root) ==
            payload
    )
    val keys = setOf(
        "baseline_activation_id",
        "baseline_artifact_sha256",
        "baseline_capability_version",
        "baseline_package_sha256",
        "candidate_artifact_sha256",
        "candidate_capability_version",
        "candidate_id",
        "candidate_package_sha256",
        "candidate_plan_sha256",
        "candidate_publisher_key_id",
        "completed_ms",
        "detail",
        "evaluation_id",
        "promotion_id",
        "requested_ms",
        "resulting_activation_id",
        "resulting_artifact_sha256",
        "rollback_restored_activation_id",
        "schema",
        "state",
    )
    require(root.values.keys == keys)
    require(
        root.promString("schema") ==
            "VN97IMPPROM1"
    )
    val spec =
        VN97ImprovementPromotionSpec(
            candidateId =
                root.promString(
                    "candidate_id"
                ),
            evaluationId =
                root.promString(
                    "evaluation_id"
                ),
            baselineActivationId =
                root.promString(
                    "baseline_activation_id"
                ),
            baselineArtifactSha256 =
                root.promString(
                    "baseline_artifact_sha256"
                ),
            baselinePackageSha256 =
                root.promString(
                    "baseline_package_sha256"
                ),
            baselineCapabilityVersion =
                root.promLong(
                    "baseline_capability_version"
                ),
            candidatePackageSha256 =
                root.promString(
                    "candidate_package_sha256"
                ),
            candidateArtifactSha256 =
                root.promString(
                    "candidate_artifact_sha256"
                ),
            candidateCapabilityVersion =
                root.promLong(
                    "candidate_capability_version"
                ),
            candidatePlanSha256 =
                root.promString(
                    "candidate_plan_sha256"
                ),
            candidatePublisherKeyId =
                root.promString(
                    "candidate_publisher_key_id"
                ),
            requestedWallTimeMillis =
                root.promLong(
                    "requested_ms"
                ),
        )
    val state = try {
        VN97ImprovementPromotionState
            .valueOf(
                root.promString("state")
            )
    } catch (exc: Exception) {
        throw IllegalArgumentException(
            "promotion state is invalid",
            exc,
        )
    }
    return VN97ImprovementPromotionRecord(
        promotionId =
            root.promString(
                "promotion_id"
            ),
        spec = spec,
        state = state,
        resultingActivationId =
            root.promString(
                "resulting_activation_id"
            ),
        resultingArtifactSha256 =
            root.promString(
                "resulting_artifact_sha256"
            ),
        rollbackRestoredActivationId =
            root.promString(
                "rollback_restored_activation_id"
            ),
        detail =
            root.promString("detail"),
        completedWallTimeMillis =
            root.promLong("completed_ms"),
    )
}

private fun VnJsonObject.promString(
    key: String,
): String =
    (values[key] as? VnJsonString)?.value
        ?: throw IllegalArgumentException(
            "promotion $key must be string"
        )

private fun VnJsonObject.promLong(
    key: String,
): Long {
    val raw =
        (values[key] as? VnJsonNumber)
            ?.canonical
            ?: throw IllegalArgumentException(
                "promotion $key must be integer"
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
            "promotion $key is outside Long range"
        )
}

private fun requirePromotionSha(
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

private fun requirePromotionId(
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

private fun requirePromotionText(
    value: String,
    maxBytes: Int,
    label: String,
) {
    require(value.isNotBlank())
    require('\u0000' !in value)
    require(
        value.toByteArray(
            StandardCharsets.UTF_8
        ).size <= maxBytes
    ) {
        "$label exceeds UTF-8 byte bound"
    }
}

private fun strictPromotionUtf8(
    bytes: ByteArray,
): String =
    try {
        StandardCharsets.UTF_8
            .newDecoder()
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
            "promotion ledger is not strict UTF-8",
            exc,
        )
    }

private fun promotionSha(
    bytes: ByteArray,
): String =
    MessageDigest.getInstance("SHA-256")
        .digest(bytes)
        .joinToString("") {
            "%02x".format(
                it.toInt() and 0xff
            )
        }
