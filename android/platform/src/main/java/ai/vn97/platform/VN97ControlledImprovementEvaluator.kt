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
import java.nio.ByteBuffer
import java.nio.channels.FileChannel
import java.nio.file.Files
import java.nio.file.LinkOption
import java.nio.file.StandardOpenOption
import java.security.MessageDigest

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

        val evaluationImage =
            extractEvaluationImage(
                packageFile =
                    candidatePackage,
                offset =
                    section.packageOffset,
                length =
                    section.size,
                expectedSha256 =
                    section.sha256,
            )
        val candidateMetrics =
            try {
                ParcelFileDescriptor.open(
                    evaluationImage,
                    ParcelFileDescriptor
                        .MODE_READ_ONLY,
                ).use { descriptor ->
                    check(
                        descriptor.statSize ==
                            section.size &&
                            evaluationImage.length() ==
                                section.size
                    ) {
                        "evaluation VN97MI1 descriptor size changed"
                    }
                    NativeSelfImprovementEvaluator
                        .evaluateCandidateDescriptor(
                            fd = descriptor.fd,
                            offset = 0L,
                            length =
                                section.size,
                            artifactSha256 =
                                section.sha256
                                    .hexToBytes(),
                        )
                }
            } finally {
                check(
                    evaluationImage.delete() ||
                        !evaluationImage.exists()
                ) {
                    "failed to delete temporary evaluation VN97MI1"
                }
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

    private fun extractEvaluationImage(
        packageFile: File,
        offset: Long,
        length: Long,
        expectedSha256: String,
    ): File {
        check(
            offset >= 0L &&
                length > 0L &&
                offset <=
                    packageFile.length() &&
                length <=
                    packageFile.length() -
                        offset
        ) {
            "candidate VN97MI1 range is invalid"
        }

        val temp =
            File.createTempFile(
                "vn97-impeval-",
                ".vn97mi1",
                appContext.cacheDir,
            )
        try {
            FileChannel.open(
                packageFile.toPath(),
                StandardOpenOption.READ,
                LinkOption.NOFOLLOW_LINKS,
            ).use { input ->
                FileChannel.open(
                    temp.toPath(),
                    StandardOpenOption.WRITE,
                    LinkOption.NOFOLLOW_LINKS,
                ).use { output ->
                    input.position(offset)
                    var remaining = length
                    val buffer =
                        ByteBuffer.allocate(
                            64 * 1024
                        )
                    while (remaining > 0L) {
                        buffer.clear()
                        buffer.limit(
                            minOf(
                                buffer.capacity()
                                    .toLong(),
                                remaining,
                            ).toInt()
                        )
                        val read =
                            input.read(buffer)
                        check(read > 0) {
                            "candidate VN97MI1 extraction was truncated"
                        }
                        remaining -=
                            read.toLong()
                        buffer.flip()
                        while (
                            buffer.hasRemaining()
                        ) {
                            check(
                                output.write(
                                    buffer
                                ) > 0
                            ) {
                                "candidate VN97MI1 extraction made no write progress"
                            }
                        }
                    }
                    output.force(true)
                }
            }

            check(
                temp.length() == length
            ) {
                "temporary evaluation VN97MI1 length mismatch"
            }
            check(
                hashFile(temp) ==
                    expectedSha256
            ) {
                "temporary evaluation VN97MI1 SHA-256 mismatch"
            }
            return temp
        } catch (exc: Throwable) {
            temp.delete()
            throw exc
        }
    }

    private fun hashFile(
        file: File,
    ): String {
        val digest =
            MessageDigest.getInstance(
                "SHA-256"
            )
        FileChannel.open(
            file.toPath(),
            StandardOpenOption.READ,
            LinkOption.NOFOLLOW_LINKS,
        ).use { channel ->
            val buffer =
                ByteBuffer.allocate(
                    64 * 1024
                )
            while (true) {
                buffer.clear()
                val read =
                    channel.read(buffer)
                if (read < 0) break
                check(read > 0) {
                    "evaluation VN97MI1 hash read made no progress"
                }
                buffer.flip()
                digest.update(buffer)
            }
        }
        return digest.digest()
            .toLowerHex()
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
