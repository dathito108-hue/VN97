package ai.vn97.runtime

import java.io.File
import kotlin.io.path.createTempDirectory

private inline fun expectM17CFailure(
    label: String,
    block: () -> Unit,
) {
    check(runCatching(block).isFailure) {
        "expected M17C failure: $label"
    }
}

private fun sha(pair: String): String {
    require(
        pair.length == 2 &&
            pair.all {
                it in "0123456789abcdef"
            }
    )
    return pair.repeat(32)
}

private fun spec(
    candidate: String,
    evaluation: String,
    baselineActivation: String,
    baselineArtifact: String,
    baselinePackage: String,
    candidatePackage: String,
    candidateArtifact: String,
    plan: String,
    requestedMs: Long,
    baselineVersion: Long = 7L,
    candidateVersion: Long = 8L,
): VN97ImprovementPromotionSpec =
    VN97ImprovementPromotionSpec(
        candidateId = sha(candidate),
        evaluationId = sha(evaluation),
        baselineActivationId =
            sha(baselineActivation),
        baselineArtifactSha256 =
            sha(baselineArtifact),
        baselinePackageSha256 =
            sha(baselinePackage),
        baselineCapabilityVersion =
            baselineVersion,
        candidatePackageSha256 =
            sha(candidatePackage),
        candidateArtifactSha256 =
            sha(candidateArtifact),
        candidateCapabilityVersion =
            candidateVersion,
        candidatePlanSha256 =
            sha(plan),
        candidatePublisherKeyId =
            "publisher-key",
        requestedWallTimeMillis =
            requestedMs,
    )

fun main() {
    val root =
        createTempDirectory(
            "m17c-promotion-"
        ).toFile()
    try {
        val ledger =
            VN97ImprovementPromotionLedger(
                root
            )
        val firstSpec =
            spec(
                candidate = "11",
                evaluation = "22",
                baselineActivation = "33",
                baselineArtifact = "44",
                baselinePackage = "55",
                candidatePackage = "66",
                candidateArtifact = "77",
                plan = "88",
                requestedMs = 100L,
            )

        val pending =
            ledger.begin(firstSpec)
        check(
            pending.state ==
                VN97ImprovementPromotionState
                    .PENDING
        )
        check(
            pending.promotionId ==
                firstSpec.promotionId
        )
        check(
            ledger.pendingAttempts() ==
                listOf(pending)
        )
        check(
            ledger.pendingForCandidate(
                firstSpec.candidateId
            ) == pending
        )
        check(
            ledger.begin(firstSpec) ==
                pending
        )

        expectM17CFailure(
            "same candidate different pending intent"
        ) {
            ledger.begin(
                firstSpec.copy(
                    requestedWallTimeMillis =
                        101L
                )
            )
        }

        val promoted =
            ledger.completePromoted(
                promotionId =
                    pending.promotionId,
                activationId =
                    sha("99"),
                artifactSha256 =
                    firstSpec
                        .candidateArtifactSha256,
                nowWallTimeMillis = 200L,
            )
        check(
            promoted.state ==
                VN97ImprovementPromotionState
                    .PROMOTED
        )
        check(
            promoted.resultingArtifactSha256 ==
                firstSpec
                    .candidateArtifactSha256
        )
        check(ledger.pendingAttempts().isEmpty())
        check(
            ledger.loadOrNull(
                pending.promotionId
            ) == promoted
        )

        val terminalReplay =
            ledger.completeAborted(
                promotionId =
                    promoted.promotionId,
                detail =
                    "must not replace terminal state",
                baselineStillActive = true,
                nowWallTimeMillis = 300L,
            )
        check(terminalReplay == promoted)

        val rollbackSpec =
            spec(
                candidate = "aa",
                evaluation = "ab",
                baselineActivation = "ac",
                baselineArtifact = "ad",
                baselinePackage = "ae",
                candidatePackage = "af",
                candidateArtifact = "ba",
                plan = "bb",
                requestedMs = 400L,
                baselineVersion = 9L,
                candidateVersion = 10L,
            )
        val rollbackPending =
            ledger.begin(
                rollbackSpec
            )
        val rolledBack =
            ledger.completeRolledBack(
                promotionId =
                    rollbackPending.promotionId,
                activationId =
                    sha("bc"),
                artifactSha256 =
                    rollbackSpec
                        .candidateArtifactSha256,
                detail =
                    "post-activation verification failed; exact baseline restored",
                nowWallTimeMillis = 500L,
            )
        check(
            rolledBack.state ==
                VN97ImprovementPromotionState
                    .ROLLED_BACK
        )
        check(
            rolledBack
                .rollbackRestoredActivationId ==
                rollbackSpec
                    .baselineActivationId
        )
        check(rolledBack.detail.isNotBlank())

        val abortedSpec =
            spec(
                candidate = "c1",
                evaluation = "c2",
                baselineActivation = "c3",
                baselineArtifact = "c4",
                baselinePackage = "c5",
                candidatePackage = "c6",
                candidateArtifact = "c7",
                plan = "c8",
                requestedMs = 600L,
                baselineVersion = 11L,
                candidateVersion = 12L,
            )
        val abortedPending =
            ledger.begin(
                abortedSpec
            )
        val aborted =
            ledger.completeAborted(
                promotionId =
                    abortedPending.promotionId,
                detail =
                    "recovered pending promotion with exact baseline still active",
                baselineStillActive = true,
                nowWallTimeMillis = 700L,
            )
        check(
            aborted.state ==
                VN97ImprovementPromotionState
                    .ABORTED
        )
        check(
            aborted
                .rollbackRestoredActivationId ==
                abortedSpec
                    .baselineActivationId
        )

        expectM17CFailure(
            "candidate version must advance"
        ) {
            spec(
                candidate = "d1",
                evaluation = "d2",
                baselineActivation = "d3",
                baselineArtifact = "d4",
                baselinePackage = "d5",
                candidatePackage = "d6",
                candidateArtifact = "d7",
                plan = "d8",
                requestedMs = 800L,
                baselineVersion = 20L,
                candidateVersion = 20L,
            )
        }

        val tamperSpec =
            spec(
                candidate = "e1",
                evaluation = "e2",
                baselineActivation = "e3",
                baselineArtifact = "e4",
                baselinePackage = "e5",
                candidatePackage = "e6",
                candidateArtifact = "e7",
                plan = "e8",
                requestedMs = 900L,
                baselineVersion = 30L,
                candidateVersion = 31L,
            )
        val tamperPending =
            ledger.begin(
                tamperSpec
            )
        val target = File(
            root,
            tamperPending.promotionId +
                ".vn97impprom1",
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
        expectM17CFailure("tamper") {
            ledger.loadOrNull(
                tamperPending.promotionId
            )
        }

        println(
            "M17C controlled promotion ledger contracts: PASS"
        )
    } finally {
        root.deleteRecursively()
    }
}
