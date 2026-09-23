package ai.vn97.runtime

import java.io.File
import kotlin.io.path.createTempDirectory

private inline fun expectFailure(
    label: String,
    block: () -> Unit,
) {
    check(runCatching(block).isFailure) {
        "expected M17A failure: $label"
    }
}

private fun spec(
    candidateVersion: Long = 8L,
    objective: String =
        "Improve held-out assistant quality without regressing mobile limits.",
    candidatePackage: String =
        "22".repeat(32),
): VN97ImprovementCandidateSpec =
    VN97ImprovementCandidateSpec(
        baselineActivationId =
            "01".repeat(32),
        baselineArtifactSha256 =
            "02".repeat(32),
        baselinePackageSha256 =
            "03".repeat(32),
        baselineCapabilityVersion = 7L,
        candidatePackageSha256 =
            candidatePackage,
        candidateCapabilityVersion =
            candidateVersion,
        candidatePublisherKeyId =
            "publisher.selftest",
        candidatePublisherKeySha256 =
            "04".repeat(32),
        candidatePlanSha256 =
            "05".repeat(32),
        sourceOrigin =
            "local-training-campaign",
        sourceLicense = "test",
        objective = objective,
    )

fun main() {
    val firstSpec = spec()
    check(firstSpec.candidateId.length == 64)
    check(
        firstSpec.candidateId ==
            spec().candidateId
    )
    check(
        firstSpec.candidateId !=
            spec(
                objective =
                    "Improve reasoning on another held-out objective."
            ).candidateId
    )

    expectFailure("same version") {
        spec(candidateVersion = 7L)
    }
    expectFailure("downgrade") {
        spec(candidateVersion = 6L)
    }
    expectFailure("same package as baseline") {
        VN97ImprovementCandidateSpec(
            baselineActivationId =
                "01".repeat(32),
            baselineArtifactSha256 =
                "02".repeat(32),
            baselinePackageSha256 =
                "03".repeat(32),
            baselineCapabilityVersion = 7L,
            candidatePackageSha256 =
                "03".repeat(32),
            candidateCapabilityVersion = 8L,
            candidatePublisherKeyId =
                "publisher.selftest",
            candidatePublisherKeySha256 =
                "04".repeat(32),
            candidatePlanSha256 =
                "05".repeat(32),
            sourceOrigin = "local",
            sourceLicense = "test",
            objective = "upgrade",
        )
    }

    val root =
        createTempDirectory(
            "m17a-candidate-"
        ).toFile()
    try {
        val ledger =
            VN97ImprovementCandidateLedger(root)
        val reviewed =
            ledger.registerReviewed(
                firstSpec,
                nowWallTimeMillis = 100L,
            )
        check(
            reviewed.state ==
                VN97ImprovementCandidateState
                    .REVIEWED
        )
        check(
            ledger.loadOrNull(
                reviewed.candidateId
            ) == reviewed
        )
        check(
            ledger.reviewedForPackageOrNull(
                firstSpec.candidatePackageSha256
            ) == reviewed
        )
        check(
            ledger.registerReviewed(
                firstSpec,
                nowWallTimeMillis = 999L,
            ) == reviewed
        )

        expectFailure(
            "different reviewed objective for same package"
        ) {
            ledger.registerReviewed(
                spec(
                    objective =
                        "Different objective for same signed package."
                ),
                nowWallTimeMillis = 101L,
            )
        }

        val rejected = ledger.reject(
            candidateId =
                reviewed.candidateId,
            reason =
                "candidate has not passed held-out evaluation",
            nowWallTimeMillis = 200L,
        )
        check(
            rejected.state ==
                VN97ImprovementCandidateState
                    .REJECTED
        )
        check(
            ledger.reviewedForPackageOrNull(
                firstSpec.candidatePackageSha256
            ) == null
        )
        check(
            ledger.reject(
                candidateId =
                    rejected.candidateId,
                reason =
                    "candidate has not passed held-out evaluation",
                nowWallTimeMillis = 300L,
            ) == rejected
        )
        expectFailure(
            "rejected reason immutable"
        ) {
            ledger.reject(
                candidateId =
                    rejected.candidateId,
                reason = "different reason",
                nowWallTimeMillis = 300L,
            )
        }

        val target = File(
            root,
            reviewed.candidateId +
                ".vn97imp1",
        )
        val good = target.readBytes()
        target.writeBytes(
            good.copyOf().also {
                it[it.lastIndex] =
                    (it.last().toInt() xor 1)
                        .toByte()
            }
        )
        expectFailure("tamper") {
            ledger.loadOrNull(
                reviewed.candidateId
            )
        }

        println(
            "M17A controlled improvement candidate contracts: PASS"
        )
    } finally {
        root.deleteRecursively()
    }
}
