package ai.vn97.platform

import ai.vn97.runtime.NativeActivatedInventoryModelLoader
import ai.vn97.runtime.NativeSelfImprovementEvaluator
import ai.vn97.runtime.VN97CapabilityPackageParser
import ai.vn97.runtime.VN97HeldOutSuite
import ai.vn97.runtime.VN97ImprovementCandidateRecord
import ai.vn97.runtime.VN97ImprovementCandidateState
import ai.vn97.runtime.VN97ImprovementEvaluationCriteria
import ai.vn97.runtime.VN97ImprovementEvaluationLedger
import ai.vn97.runtime.VN97ImprovementEvaluationRecord
import ai.vn97.runtime.VN97ModelImageActivationBackend
import ai.vn97.runtime.evaluateImprovementCandidate
import android.content.Context
import android.os.ParcelFileDescriptor
import java.io.File
import java.io.RandomAccessFile
import java.nio.channels.FileChannel
import java.nio.file.Files
import java.nio.file.LinkOption

class VN97ControlledImprovementEvaluator(
    context: Context,
) {
    private val appContext =
        context.applicationContext
    private val provisioner =
        AndroidVN97CapabilityProvisioner(
            appContext
        )
    private val evaluationLedger =
        VN97ImprovementEvaluationLedger(
            File(
                provisioner.capabilityRoot,
                "self-improvement-evaluations",
            )
        )

    fun evaluate(
        candidate:
            VN97ImprovementCandidateRecord,
        criteria:
            VN97ImprovementEvaluationCriteria =
            VN97ImprovementEvaluationCriteria(),
    ): VN97ImprovementEvaluationRecord {
        check(
            candidate.state ==
                VN97ImprovementCandidateState
                    .REVIEWED
        ) {
            "only REVIEWED self-improvement candidates can be evaluated"
        }

        val baselineBefore =
            requireExactBaseline(candidate)

        evaluationLedger
            .loadOrNull(
                candidate.candidateId
            )
            ?.let { existing ->
                requireEvaluationBinding(
                    existing,
                    candidate,
                    baselineBefore,
                )
                return existing
            }

        val baselineMetrics =
            checkNotNull(
                NativeActivatedInventoryModelLoader
                    .openOrNull(
                        provisioner
                            .capabilityRoot
                    )
            ) {
                "active baseline model is unavailable"
            }.use { model ->
                check(
                    model.info.modelId
                        .toLowerHex() ==
                        candidate.spec
                            .baselineArtifactSha256
                ) {
                    "loaded baseline artifact does not match candidate baseline"
                }
                NativeSelfImprovementEvaluator
                    .evaluateActivatedModel(
                        model
                    )
            }

        val candidatePackage =
            stagedCandidatePackage(
                candidate
            )
        val parsed =
            RandomAccessFile(
                candidatePackage,
                "r",
            ).use { file ->
                val size = file.length()
                check(size > 0L) {
                    "candidate package is empty"
                }
                val mapped =
                    file.channel.map(
                        FileChannel
                            .MapMode.READ_ONLY,
                        0L,
                        size,
                    )
                VN97CapabilityPackageParser
                    .parse(mapped)
            }

        check(
            parsed.packageSha256 ==
                candidate.spec
                    .candidatePackageSha256
        ) {
            "staged candidate package identity changed"
        }
        val manifest = parsed.manifest
        check(
            manifest.capabilityId ==
                VN97ModelImageActivationBackend
                    .CAPABILITY_ID &&
                manifest.kind == "weights" &&
                manifest.capabilityVersion ==
                    candidate.spec
                        .candidateCapabilityVersion &&
                manifest.sections.size == 1
        ) {
            "staged candidate no longer matches canonical model.language identity"
        }

        val section =
            manifest.sections.single()
        check(
            section.role ==
                VN97ModelImageActivationBackend
                    .MODEL_IMAGE_ROLE &&
                section.format ==
                    VN97ModelImageActivationBackend
                        .MODEL_IMAGE_FORMAT
        ) {
            "staged candidate is not direct VN97MI1 model_image"
        }

        val candidateMetrics =
            ParcelFileDescriptor.open(
                candidatePackage,
                ParcelFileDescriptor
                    .MODE_READ_ONLY,
            ).use { descriptor ->
                val packageSize =
                    descriptor.statSize
                check(
                    packageSize ==
                        candidatePackage.length() &&
                        section.packageOffset >= 0L &&
                        section.size > 0L &&
                        section.packageOffset <=
                            packageSize &&
                        section.size <=
                            packageSize -
                                section.packageOffset
                ) {
                    "candidate VN97MI1 descriptor range is invalid"
                }
                NativeSelfImprovementEvaluator
                    .evaluateCandidateDescriptor(
                        fd = descriptor.fd,
                        offset =
                            section.packageOffset,
                        length =
                            section.size,
                        artifactSha256 =
                            section.sha256
                                .hexToBytes(),
                    )
            }

        val baselineAfter =
            requireExactBaseline(candidate)
        check(
            baselineAfter ==
                baselineBefore
        ) {
            "active baseline changed during held-out evaluation"
        }

        val decision =
            evaluateImprovementCandidate(
                baseline =
                    baselineMetrics,
                candidate =
                    candidateMetrics,
                criteria = criteria,
            )

        val record =
            VN97ImprovementEvaluationRecord(
                candidateId =
                    candidate.candidateId,
                baselineActivationId =
                    candidate.spec
                        .baselineActivationId,
                baselineArtifactSha256 =
                    candidate.spec
                        .baselineArtifactSha256,
                candidatePackageSha256 =
                    candidate.spec
                        .candidatePackageSha256,
                candidateArtifactSha256 =
                    section.sha256,
                suiteSha256 =
                    VN97HeldOutSuite
                        .suiteSha256,
                criteria = criteria,
                baseline =
                    baselineMetrics,
                candidate =
                    candidateMetrics,
                decision = decision,
                createdWallTimeMillis =
                    System.currentTimeMillis(),
            )

        return evaluationLedger
            .saveFirst(record)
    }

    fun loadOrNull(
        candidateId: String,
    ): VN97ImprovementEvaluationRecord? =
        evaluationLedger
            .loadOrNull(candidateId)

    private fun requireExactBaseline(
        candidate:
            VN97ImprovementCandidateRecord,
    ) =
        checkNotNull(
            provisioner
                .currentModelActivation()
        ) {
            "controlled evaluation requires an active canonical baseline"
        }.also { active ->
            check(
                active.capabilityId ==
                    VN97ModelImageActivationBackend
                        .CAPABILITY_ID &&
                    active.activationId ==
                        candidate.spec
                            .baselineActivationId &&
                    active.artifactSha256 ==
                        candidate.spec
                            .baselineArtifactSha256 &&
                    active.packageSha256 ==
                        candidate.spec
                            .baselinePackageSha256 &&
                    active.capabilityVersion ==
                        candidate.spec
                            .baselineCapabilityVersion
            ) {
                "active baseline no longer matches the candidate binding"
            }
        }

    private fun stagedCandidatePackage(
        candidate:
            VN97ImprovementCandidateRecord,
    ): File {
        val target =
            File(
                provisioner.stageRoot,
                candidate.spec
                    .candidatePackageSha256 +
                    ".vn97cap1",
            )
        val path = target.toPath()
        check(
            Files.isRegularFile(
                path,
                LinkOption.NOFOLLOW_LINKS,
            ) &&
                !Files.isSymbolicLink(path)
        ) {
            "reviewed candidate package is missing or unsafe"
        }
        return target
    }

    private fun requireEvaluationBinding(
        record:
            VN97ImprovementEvaluationRecord,
        candidate:
            VN97ImprovementCandidateRecord,
        baseline:
            ai.vn97.runtime.VN97CapabilityInventoryItem,
    ) {
        check(
            record.candidateId ==
                candidate.candidateId &&
                record.candidatePackageSha256 ==
                    candidate.spec
                        .candidatePackageSha256 &&
                record.baselineActivationId ==
                    baseline.activationId &&
                record.baselineArtifactSha256 ==
                    baseline.artifactSha256 &&
                record.suiteSha256 ==
                    VN97HeldOutSuite
                        .suiteSha256
        ) {
            "persisted held-out evaluation does not match current candidate binding"
        }
    }

    private fun String.hexToBytes():
        ByteArray {
        check(
            length == 64 &&
                all {
                    it in
                        "0123456789abcdef"
                }
        ) {
            "candidate artifact SHA-256 is invalid"
        }
        return ByteArray(32) {
                index,
            ->
            substring(
                index * 2,
                index * 2 + 2,
            ).toInt(16).toByte()
        }
    }

    private fun ByteArray.toLowerHex():
        String =
        joinToString("") {
            "%02x".format(
                it.toInt() and 0xff
            )
        }
}
