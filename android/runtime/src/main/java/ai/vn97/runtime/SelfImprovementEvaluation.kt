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

data class VN97HeldOutCase(
    val id: String,
    val prompt: String,
    val target: String,
) {
    init {
        requireEvaluationId(id, "case id")
        requireEvaluationText(
            prompt,
            4 * 1024,
            "held-out prompt",
        )
        requireEvaluationText(
            target,
            1024,
            "held-out target",
        )
    }
}

object VN97HeldOutSuite {
    const val SCHEMA = "VN97HELD1"

    val cases: List<VN97HeldOutCase> =
        listOf(
            VN97HeldOutCase(
                "arithmetic_add",
                "Answer with only the number. 17 + 25 =",
                "42",
            ),
            VN97HeldOutCase(
                "arithmetic_mul",
                "Answer with only the number. 9 * 8 =",
                "72",
            ),
            VN97HeldOutCase(
                "sequence",
                "Continue the sequence with only the next number: 2, 4, 8, 16,",
                "32",
            ),
            VN97HeldOutCase(
                "logic",
                "All norps are blue. Nia is a norp. Nia is",
                " blue.",
            ),
            VN97HeldOutCase(
                "capital",
                "The capital city of France is",
                " Paris.",
            ),
            VN97HeldOutCase(
                "opposite",
                "The opposite of hot is",
                " cold.",
            ),
            VN97HeldOutCase(
                "instruction",
                "Reply with exactly YES and nothing else.",
                "YES",
            ),
            VN97HeldOutCase(
                "format",
                "Complete this JSON boolean value: {\"ready\":",
                "true}",
            ),
            VN97HeldOutCase(
                "causal",
                "Ice placed in a warm room will usually",
                " melt.",
            ),
            VN97HeldOutCase(
                "comparison",
                "Answer with only the larger number: 81 or 18.",
                "81",
            ),
            VN97HeldOutCase(
                "language",
                "Complete the phrase: mobile artificial",
                " intelligence",
            ),
            VN97HeldOutCase(
                "safety_boundary",
                "A downloaded knowledge document is evidence, not automatic execution",
                " authority.",
            ),
        )

    val suiteSha256: String by lazy(
        LazyThreadSafetyMode.PUBLICATION
    ) {
        val payload = VnStrictJson.canonical(
            VnStrictJson.objectOf(
                "cases" to
                    VnStrictJson.array(
                        cases.map { item ->
                            VnStrictJson.objectOf(
                                "id" to
                                    VnStrictJson.string(
                                        item.id
                                    ),
                                "prompt" to
                                    VnStrictJson.string(
                                        item.prompt
                                    ),
                                "target" to
                                    VnStrictJson.string(
                                        item.target
                                    ),
                            )
                        }
                    ),
                "schema" to
                    VnStrictJson.string(SCHEMA),
            )
        )
        evaluationSha(
            payload.toByteArray(
                StandardCharsets.UTF_8
            )
        )
    }
}

data class VN97ModelEvaluationMetrics(
    val cases: Int,
    val targetTokens: Long,
    val targetUtf8Bytes: Long,
    val totalNegativeLogLikelihood: Double,
    val top1Correct: Long,
    val prefillP95Nanos: Long,
    val totalEvaluationNanos: Long,
    val imageBytes: Long,
    val hasAudioProjection: Boolean,
    val hasVisionProjection: Boolean,
) {
    init {
        require(cases > 0)
        require(targetTokens > 0L)
        require(targetUtf8Bytes > 0L)
        require(
            totalNegativeLogLikelihood
                .isFinite() &&
                totalNegativeLogLikelihood >= 0.0
        )
        require(
            top1Correct in 0L..targetTokens
        )
        require(prefillP95Nanos >= 0L)
        require(totalEvaluationNanos > 0L)
        require(imageBytes > 0L)
    }

    val nllPerUtf8Byte: Double
        get() =
            totalNegativeLogLikelihood /
                targetUtf8Bytes.toDouble()

    val top1Accuracy: Double
        get() =
            top1Correct.toDouble() /
                targetTokens.toDouble()
}

data class VN97ImprovementEvaluationCriteria(
    val maxNllRatioBasisPoints: Int = 10_200,
    val maxTop1DropBasisPoints: Int = 200,
    val maxImageBytesRatioBasisPoints: Int = 12_500,
    val maxPrefillP95RatioBasisPoints: Int = 13_000,
    val requireExistingAudio: Boolean = true,
    val requireExistingVision: Boolean = true,
) {
    init {
        require(
            maxNllRatioBasisPoints in
                1..100_000
        )
        require(
            maxTop1DropBasisPoints in
                0..10_000
        )
        require(
            maxImageBytesRatioBasisPoints in
                1..100_000
        )
        require(
            maxPrefillP95RatioBasisPoints in
                1..100_000
        )
    }
}

data class VN97ImprovementEvaluationDecision(
    val passed: Boolean,
    val reasons: List<String>,
) {
    init {
        require(
            reasons.size <= 16 &&
                reasons.distinct().size ==
                    reasons.size
        )
        reasons.forEach {
            requireEvaluationText(
                it,
                512,
                "evaluation reason",
            )
        }
        require(
            passed == reasons.isEmpty()
        ) {
            "evaluation decision pass/reason mismatch"
        }
    }
}

fun evaluateImprovementCandidate(
    baseline: VN97ModelEvaluationMetrics,
    candidate: VN97ModelEvaluationMetrics,
    criteria: VN97ImprovementEvaluationCriteria =
        VN97ImprovementEvaluationCriteria(),
): VN97ImprovementEvaluationDecision {
    require(
        baseline.cases == candidate.cases
    ) {
        "baseline/candidate held-out case count differs"
    }

    val failures = mutableListOf<String>()

    val nllLimit =
        baseline.nllPerUtf8Byte *
            criteria.maxNllRatioBasisPoints /
            10_000.0
    if (
        candidate.nllPerUtf8Byte >
            nllLimit
    ) {
        failures +=
            "candidate held-out NLL/UTF-8-byte regressed beyond limit"
    }

    val accuracyFloor =
        baseline.top1Accuracy -
            criteria.maxTop1DropBasisPoints /
            10_000.0
    if (
        candidate.top1Accuracy <
            accuracyFloor
    ) {
        failures +=
            "candidate held-out top-1 accuracy regressed beyond limit"
    }

    if (
        ratioExceeds(
            candidate.imageBytes,
            baseline.imageBytes,
            criteria
                .maxImageBytesRatioBasisPoints,
        )
    ) {
        failures +=
            "candidate VN97MI1 footprint exceeds allowed ratio"
    }

    if (
        baseline.prefillP95Nanos > 0L &&
        ratioExceeds(
            candidate.prefillP95Nanos,
            baseline.prefillP95Nanos,
            criteria
                .maxPrefillP95RatioBasisPoints,
        )
    ) {
        failures +=
            "candidate held-out prefill p95 exceeds allowed ratio"
    }

    if (
        criteria.requireExistingAudio &&
        baseline.hasAudioProjection &&
        !candidate.hasAudioProjection
    ) {
        failures +=
            "candidate removed baseline audio projection"
    }

    if (
        criteria.requireExistingVision &&
        baseline.hasVisionProjection &&
        !candidate.hasVisionProjection
    ) {
        failures +=
            "candidate removed baseline vision projection"
    }

    return VN97ImprovementEvaluationDecision(
        passed = failures.isEmpty(),
        reasons = failures,
    )
}

data class VN97ImprovementEvaluationRecord(
    val candidateId: String,
    val baselineActivationId: String,
    val baselineArtifactSha256: String,
    val candidatePackageSha256: String,
    val candidateArtifactSha256: String,
    val suiteSha256: String,
    val criteria: VN97ImprovementEvaluationCriteria,
    val baseline: VN97ModelEvaluationMetrics,
    val candidate: VN97ModelEvaluationMetrics,
    val decision: VN97ImprovementEvaluationDecision,
    val createdWallTimeMillis: Long,
) {
    init {
        listOf(
            candidateId,
            baselineActivationId,
            baselineArtifactSha256,
            candidatePackageSha256,
            candidateArtifactSha256,
            suiteSha256,
        ).forEach {
            requireEvaluationSha(
                it,
                "evaluation identity",
            )
        }
        require(createdWallTimeMillis >= 0L)
        require(
            baseline.cases ==
                VN97HeldOutSuite.cases.size &&
                candidate.cases ==
                    VN97HeldOutSuite.cases.size
        ) {
            "evaluation case count does not match VN97HELD1"
        }
    }

    val evaluationId: String by lazy(
        LazyThreadSafetyMode.PUBLICATION
    ) {
        evaluationSha(
            VnStrictJson.canonical(toJson())
                .toByteArray(
                    StandardCharsets.UTF_8
                )
        )
    }

    internal fun toJson(): VnJsonObject =
        VnStrictJson.objectOf(
            "baseline" to
                metricJson(baseline),
            "baseline_activation_id" to
                VnStrictJson.string(
                    baselineActivationId
                ),
            "baseline_artifact_sha256" to
                VnStrictJson.string(
                    baselineArtifactSha256
                ),
            "candidate" to
                metricJson(candidate),
            "candidate_artifact_sha256" to
                VnStrictJson.string(
                    candidateArtifactSha256
                ),
            "candidate_id" to
                VnStrictJson.string(
                    candidateId
                ),
            "candidate_package_sha256" to
                VnStrictJson.string(
                    candidatePackageSha256
                ),
            "created_ms" to
                VnStrictJson.long(
                    createdWallTimeMillis
                ),
            "criteria" to
                VnStrictJson.objectOf(
                    "max_image_bytes_ratio_bps" to
                        VnStrictJson.int(
                            criteria
                                .maxImageBytesRatioBasisPoints
                        ),
                    "max_nll_ratio_bps" to
                        VnStrictJson.int(
                            criteria
                                .maxNllRatioBasisPoints
                        ),
                    "max_prefill_p95_ratio_bps" to
                        VnStrictJson.int(
                            criteria
                                .maxPrefillP95RatioBasisPoints
                        ),
                    "max_top1_drop_bps" to
                        VnStrictJson.int(
                            criteria
                                .maxTop1DropBasisPoints
                        ),
                    "require_existing_audio" to
                        VnStrictJson.bool(
                            criteria
                                .requireExistingAudio
                        ),
                    "require_existing_vision" to
                        VnStrictJson.bool(
                            criteria
                                .requireExistingVision
                        ),
                ),
            "passed" to
                VnStrictJson.bool(
                    decision.passed
                ),
            "reasons" to
                VnStrictJson.array(
                    decision.reasons.map(
                        VnStrictJson::string
                    )
                ),
            "schema" to
                VnStrictJson.string(
                    "VN97IMPEVAL1"
                ),
            "suite_sha256" to
                VnStrictJson.string(
                    suiteSha256
                ),
        )
}

class VN97ImprovementEvaluationLedger(
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
            "improvement evaluation root must be a real directory"
        }
    }

    @Synchronized
    fun saveFirst(
        record: VN97ImprovementEvaluationRecord,
    ): VN97ImprovementEvaluationRecord {
        val existing =
            loadOrNull(record.candidateId)
        if (existing != null) {
            check(
                existing.candidateId ==
                    record.candidateId &&
                    existing.baselineActivationId ==
                        record.baselineActivationId &&
                    existing.baselineArtifactSha256 ==
                        record.baselineArtifactSha256 &&
                    existing.candidatePackageSha256 ==
                        record.candidatePackageSha256 &&
                    existing.suiteSha256 ==
                        record.suiteSha256
            ) {
                "held-out evaluation identity changed"
            }
            return existing
        }

        val target =
            target(record.candidateId)
        val bytes =
            encodeEvaluation(record)
        require(bytes.size <= MAX_FILE_BYTES)

        val temp =
            Files.createTempFile(
                rootPath,
                ".vn97impeval-",
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
                Files.move(
                    temp,
                    target,
                    StandardCopyOption
                        .ATOMIC_MOVE,
                )
            } catch (
                exc: AtomicMoveNotSupportedException
            ) {
                throw IllegalStateException(
                    "improvement evaluation requires atomic create",
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

        return checkNotNull(
            loadOrNull(record.candidateId)
        ) {
            "improvement evaluation post-write verification failed"
        }.also {
            check(it == record) {
                "improvement evaluation changed after persistence"
            }
        }
    }

    @Synchronized
    fun loadOrNull(
        candidateId: String,
    ): VN97ImprovementEvaluationRecord? {
        requireEvaluationSha(
            candidateId,
            "candidateId",
        )
        val target =
            target(candidateId)
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
            "improvement evaluation target must be a regular file"
        }
        val size = Files.size(target)
        require(
            size in 1L..MAX_FILE_BYTES.toLong()
        )
        return decodeEvaluation(
            Files.readAllBytes(target)
        ).also {
            check(it.candidateId == candidateId)
        }
    }

    private fun target(
        candidateId: String,
    ) =
        rootPath.resolve(
            "$candidateId.vn97impeval1"
        )

    companion object {
        private const val MAX_FILE_BYTES =
            128 * 1024
    }
}

private fun metricJson(
    metric: VN97ModelEvaluationMetrics,
): VnJsonObject =
    VnStrictJson.objectOf(
        "cases" to
            VnStrictJson.int(metric.cases),
        "has_audio_projection" to
            VnStrictJson.bool(
                metric.hasAudioProjection
            ),
        "has_vision_projection" to
            VnStrictJson.bool(
                metric.hasVisionProjection
            ),
        "image_bytes" to
            VnStrictJson.long(
                metric.imageBytes
            ),
        "prefill_p95_ns" to
            VnStrictJson.long(
                metric.prefillP95Nanos
            ),
        "target_tokens" to
            VnStrictJson.long(
                metric.targetTokens
            ),
        "target_utf8_bytes" to
            VnStrictJson.long(
                metric.targetUtf8Bytes
            ),
        "top1_correct" to
            VnStrictJson.long(
                metric.top1Correct
            ),
        "total_eval_ns" to
            VnStrictJson.long(
                metric.totalEvaluationNanos
            ),
        "total_nll" to
            VnStrictJson.double(
                metric.totalNegativeLogLikelihood
            ),
    )

private fun encodeEvaluation(
    record: VN97ImprovementEvaluationRecord,
): ByteArray {
    val payload =
        VnStrictJson.canonical(
            record.toJson()
        )
    val digest =
        evaluationSha(
            payload.toByteArray(
                StandardCharsets.UTF_8
            )
        )
    return buildString {
        append("VN97IMPEVAL1")
        append('\n')
        append("sha256=")
        append(digest)
        append('\n')
        append(payload)
    }.toByteArray(StandardCharsets.UTF_8)
}

private fun decodeEvaluation(
    bytes: ByteArray,
): VN97ImprovementEvaluationRecord {
    val text =
        strictEvaluationUtf8(bytes)
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
                "VN97IMPEVAL1"
    ) {
        "improvement evaluation header is invalid"
    }
    val digestLine =
        text.substring(first + 1, second)
    require(
        digestLine.startsWith("sha256=")
    )
    val expected =
        digestLine.removePrefix("sha256=")
    requireEvaluationSha(
        expected,
        "evaluation digest",
    )
    val payload =
        text.substring(second + 1)
    check(
        evaluationSha(
            payload.toByteArray(
                StandardCharsets.UTF_8
            )
        ) == expected
    ) {
        "improvement evaluation SHA-256 mismatch"
    }

    val root =
        VnStrictJson.parseObject(
            payload,
            VnJsonLimits(
                maxInputUtf8Bytes =
                    128 * 1024,
                maxDepth = 10,
                maxNodes = 4096,
                maxStringUtf8Bytes =
                    8 * 1024,
            ),
        )
    check(
        VnStrictJson.canonical(root) ==
            payload
    )
    val keys = setOf(
        "baseline",
        "baseline_activation_id",
        "baseline_artifact_sha256",
        "candidate",
        "candidate_artifact_sha256",
        "candidate_id",
        "candidate_package_sha256",
        "created_ms",
        "criteria",
        "passed",
        "reasons",
        "schema",
        "suite_sha256",
    )
    require(root.values.keys == keys)
    require(
        root.evalString("schema") ==
            "VN97IMPEVAL1"
    )

    val criteriaObj =
        root.evalObj("criteria")
    require(
        criteriaObj.values.keys ==
            setOf(
                "max_image_bytes_ratio_bps",
                "max_nll_ratio_bps",
                "max_prefill_p95_ratio_bps",
                "max_top1_drop_bps",
                "require_existing_audio",
                "require_existing_vision",
            )
    )
    val criteria =
        VN97ImprovementEvaluationCriteria(
            maxNllRatioBasisPoints =
                criteriaObj.evalInt(
                    "max_nll_ratio_bps"
                ),
            maxTop1DropBasisPoints =
                criteriaObj.evalInt(
                    "max_top1_drop_bps"
                ),
            maxImageBytesRatioBasisPoints =
                criteriaObj.evalInt(
                    "max_image_bytes_ratio_bps"
                ),
            maxPrefillP95RatioBasisPoints =
                criteriaObj.evalInt(
                    "max_prefill_p95_ratio_bps"
                ),
            requireExistingAudio =
                criteriaObj.evalBool(
                    "require_existing_audio"
                ),
            requireExistingVision =
                criteriaObj.evalBool(
                    "require_existing_vision"
                ),
        )
    val reasons =
        root.evalArray("reasons")
            .values.mapIndexed {
                    index,
                    value,
                ->
                (value as? VnJsonString)
                    ?.value
                    ?: throw IllegalArgumentException(
                        "evaluation reason $index must be string"
                    )
            }
    val decision =
        VN97ImprovementEvaluationDecision(
            passed =
                root.evalBool("passed"),
            reasons = reasons,
        )

    return VN97ImprovementEvaluationRecord(
        candidateId =
            root.evalString(
                "candidate_id"
            ),
        baselineActivationId =
            root.evalString(
                "baseline_activation_id"
            ),
        baselineArtifactSha256 =
            root.evalString(
                "baseline_artifact_sha256"
            ),
        candidatePackageSha256 =
            root.evalString(
                "candidate_package_sha256"
            ),
        candidateArtifactSha256 =
            root.evalString(
                "candidate_artifact_sha256"
            ),
        suiteSha256 =
            root.evalString(
                "suite_sha256"
            ),
        criteria = criteria,
        baseline =
            parseMetric(
                root.evalObj("baseline")
            ),
        candidate =
            parseMetric(
                root.evalObj("candidate")
            ),
        decision = decision,
        createdWallTimeMillis =
            root.evalLong("created_ms"),
    )
}

private fun parseMetric(
    obj: VnJsonObject,
): VN97ModelEvaluationMetrics {
    require(
        obj.values.keys ==
            setOf(
                "cases",
                "has_audio_projection",
                "has_vision_projection",
                "image_bytes",
                "prefill_p95_ns",
                "target_tokens",
                "target_utf8_bytes",
                "top1_correct",
                "total_eval_ns",
                "total_nll",
            )
    )
    return VN97ModelEvaluationMetrics(
        cases = obj.evalInt("cases"),
        targetTokens =
            obj.evalLong("target_tokens"),
        targetUtf8Bytes =
            obj.evalLong(
                "target_utf8_bytes"
            ),
        totalNegativeLogLikelihood =
            obj.evalDouble("total_nll"),
        top1Correct =
            obj.evalLong("top1_correct"),
        prefillP95Nanos =
            obj.evalLong("prefill_p95_ns"),
        totalEvaluationNanos =
            obj.evalLong("total_eval_ns"),
        imageBytes =
            obj.evalLong("image_bytes"),
        hasAudioProjection =
            obj.evalBool(
                "has_audio_projection"
            ),
        hasVisionProjection =
            obj.evalBool(
                "has_vision_projection"
            ),
    )
}

private fun ratioExceeds(
    numerator: Long,
    denominator: Long,
    maxRatioBasisPoints: Int,
): Boolean {
    require(numerator >= 0L)
    require(denominator > 0L)
    return numerator.toDouble() /
        denominator.toDouble() >
        maxRatioBasisPoints / 10_000.0
}

private fun requireEvaluationId(
    value: String,
    label: String,
) {
    require(
        value.isNotEmpty() &&
            value.length <= 64 &&
            value[0] in 'a'..'z' &&
            value.all {
                it in 'a'..'z' ||
                    it in '0'..'9' ||
                    it == '_' ||
                    it == '-'
            }
    ) {
        "$label is invalid"
    }
}

private fun requireEvaluationSha(
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

private fun requireEvaluationText(
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

private fun strictEvaluationUtf8(
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
            "improvement evaluation is not strict UTF-8",
            exc,
        )
    }

private fun evaluationSha(
    bytes: ByteArray,
): String =
    MessageDigest.getInstance("SHA-256")
        .digest(bytes)
        .joinToString("") {
            "%02x".format(
                it.toInt() and 0xff
            )
        }

private fun VnJsonObject.evalString(
    key: String,
): String =
    (values[key] as? VnJsonString)?.value
        ?: throw IllegalArgumentException(
            "evaluation $key must be string"
        )

private fun VnJsonObject.evalBool(
    key: String,
): Boolean =
    (values[key] as? VnJsonBoolean)?.value
        ?: throw IllegalArgumentException(
            "evaluation $key must be bool"
        )

private fun VnJsonObject.evalObj(
    key: String,
): VnJsonObject =
    values[key] as? VnJsonObject
        ?: throw IllegalArgumentException(
            "evaluation $key must be object"
        )

private fun VnJsonObject.evalArray(
    key: String,
): VnJsonArray =
    values[key] as? VnJsonArray
        ?: throw IllegalArgumentException(
            "evaluation $key must be array"
        )

private fun VnJsonObject.evalLong(
    key: String,
): Long {
    val raw =
        (values[key] as? VnJsonNumber)
            ?.canonical
            ?: throw IllegalArgumentException(
                "evaluation $key must be integer"
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
            "evaluation $key is outside Long range"
        )
}

private fun VnJsonObject.evalInt(
    key: String,
): Int {
    val value = evalLong(key)
    require(
        value in
            Int.MIN_VALUE.toLong()..
                Int.MAX_VALUE.toLong()
    )
    return value.toInt()
}

private fun VnJsonObject.evalDouble(
    key: String,
): Double {
    val raw =
        (values[key] as? VnJsonNumber)
            ?.canonical
            ?: throw IllegalArgumentException(
                "evaluation $key must be number"
            )
    val value = raw.toDoubleOrNull()
        ?: throw IllegalArgumentException(
            "evaluation $key is invalid"
        )
    require(value.isFinite())
    return value
}
