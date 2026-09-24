package ai.vn97.runtime

import java.io.File
import kotlin.io.path.createTempDirectory

private inline fun expectFailure(
    label: String,
    block: () -> Unit,
) {
    check(runCatching(block).isFailure) {
        "expected M17B failure: $label"
    }
}

private fun metrics(
    nll: Double,
    correct: Long,
    imageBytes: Long = 1000L,
    p95: Long = 100L,
    audio: Boolean = true,
    vision: Boolean = true,
): VN97ModelEvaluationMetrics =
    VN97ModelEvaluationMetrics(
        cases = VN97HeldOutSuite.cases.size,
        targetTokens = 100L,
        targetUtf8Bytes = 100L,
        totalNegativeLogLikelihood = nll,
        top1Correct = correct,
        prefillP95Nanos = p95,
        totalEvaluationNanos = 1000L,
        imageBytes = imageBytes,
        hasAudioProjection = audio,
        hasVisionProjection = vision,
    )

fun main() {
    check(VN97HeldOutSuite.cases.size == 12)
    check(VN97HeldOutSuite.suiteSha256.length == 64)
    check(
        VN97HeldOutSuite.suiteSha256 ==
            VN97HeldOutSuite.suiteSha256
    )

    val baseline =
        metrics(
            nll = 100.0,
            correct = 50L,
        )
    val improved =
        metrics(
            nll = 90.0,
            correct = 55L,
            imageBytes = 1100L,
            p95 = 110L,
        )
    val pass =
        evaluateImprovementCandidate(
            baseline,
            improved,
        )
    check(pass.passed)
    check(pass.reasons.isEmpty())

    val regressed =
        metrics(
            nll = 130.0,
            correct = 40L,
            imageBytes = 1400L,
            p95 = 150L,
            audio = false,
            vision = false,
        )
    val fail =
        evaluateImprovementCandidate(
            baseline,
            regressed,
        )
    check(!fail.passed)
    check(
        fail.reasons.any {
            "NLL" in it
        }
    )
    check(
        fail.reasons.any {
            "top-1" in it
        }
    )
    check(
        fail.reasons.any {
            "footprint" in it
        }
    )
    check(
        fail.reasons.any {
            "prefill" in it
        }
    )
    check(
        fail.reasons.any {
            "audio" in it
        }
    )
    check(
        fail.reasons.any {
            "vision" in it
        }
    )

    val record =
        VN97ImprovementEvaluationRecord(
            candidateId =
                "11".repeat(32),
            baselineActivationId =
                "22".repeat(32),
            baselineArtifactSha256 =
                "33".repeat(32),
            candidatePackageSha256 =
                "44".repeat(32),
            candidateArtifactSha256 =
                "55".repeat(32),
            suiteSha256 =
                VN97HeldOutSuite
                    .suiteSha256,
            criteria =
                VN97ImprovementEvaluationCriteria(),
            baseline = baseline,
            candidate = improved,
            decision = pass,
            createdWallTimeMillis = 1234L,
        )
    check(record.evaluationId.length == 64)

    val root =
        createTempDirectory(
            "m17b-eval-"
        ).toFile()
    try {
        val ledger =
            VN97ImprovementEvaluationLedger(
                root
            )
        val saved =
            ledger.saveFirst(record)
        check(saved == record)
        check(
            ledger.loadOrNull(
                record.candidateId
            ) == record
        )

        val second =
            ledger.saveFirst(
                record.copy(
                    createdWallTimeMillis =
                        9999L,
                )
            )
        check(second == record)

        expectFailure(
            "identity mutation"
        ) {
            ledger.saveFirst(
                record.copy(
                    candidateArtifactSha256 =
                        "66".repeat(32)
                )
            )
        }

        val target = File(
            root,
            record.candidateId +
                ".vn97impeval1",
        )
        val original =
            target.readBytes()
        target.writeBytes(
            original.copyOf().also {
                it[it.lastIndex] =
                    (
                        it.last().toInt() xor 1
                    ).toByte()
            }
        )
        expectFailure("tamper") {
            ledger.loadOrNull(
                record.candidateId
            )
        }

        println(
            "M17B held-out evaluation contracts: PASS"
        )
    } finally {
        root.deleteRecursively()
    }
}
