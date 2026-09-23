package ai.vn97.runtime

import java.io.File
import kotlin.io.path.createTempDirectory

private inline fun expectFailure(
    label: String,
    block: () -> Unit,
) {
    check(runCatching(block).isFailure) {
        "expected M16E failure: $label"
    }
}

private fun record(
    packageSha: String = "11".repeat(32),
    proposalId: String = "22".repeat(32),
    fetchReceipt: String = "33".repeat(32),
): VN97AcquisitionProvenanceRecord =
    VN97AcquisitionProvenanceRecord(
        packageSha256 = packageSha,
        capabilityId = "knowledge.android.power",
        capabilityVersion = 3L,
        publisherKeyId = "publisher.test",
        publisherKeySha256 = "44".repeat(32),
        signatureSha256 = "55".repeat(32),
        payloadSha256 = "66".repeat(32),
        sourceOrigin = "https://example.com/source",
        sourceLicense = "test-license",
        recordIds = listOf(101L, 102L),
        alreadyAcquired = false,
        proposalId = proposalId,
        proposalCapabilityId =
            if (proposalId.isEmpty()) "" else
                "knowledge.android.power",
        proposalEvidenceRecordIds =
            if (proposalId.isEmpty()) emptyList()
            else listOf(7L, 9L),
        fetchReceiptId = fetchReceipt,
        fetchCanonicalUrl =
            if (fetchReceipt.isEmpty()) ""
            else "https://example.com/cap.vn97cap1",
        createdWallTimeMillis = 1234L,
    )

fun main() {
    val root = createTempDirectory(
        "m16e-provenance-"
    ).toFile()
    try {
        val ledger =
            VN97AcquisitionProvenanceLedger(root)
        val first = record()
        val saved = ledger.saveCompleted(first)
        check(saved == first)
        check(saved.provenanceId.length == 64)
        check(ledger.loadOrNull(first.packageSha256) == first)

        val again = ledger.saveCompleted(
            first.copy(
                alreadyAcquired = true,
                createdWallTimeMillis = 9999L,
            )
        )
        check(again == first)

        expectFailure("core identity mutation") {
            ledger.saveCompleted(
                first.copy(
                    payloadSha256 = "77".repeat(32)
                )
            )
        }

        val target = File(
            root,
            first.packageSha256 +
                ".vn97kprov1",
        )
        val original = target.readBytes()
        target.writeBytes(
            original.copyOf().also {
                it[it.lastIndex] =
                    (it.last().toInt() xor 1)
                        .toByte()
            }
        )
        expectFailure("tampered digest") {
            ledger.loadOrNull(
                first.packageSha256
            )
        }
        target.writeBytes(original)
        check(
            ledger.loadOrNull(
                first.packageSha256
            ) == first
        )

        expectFailure("proposal mismatch") {
            record().copy(
                proposalCapabilityId =
                    "knowledge.other"
            )
        }

        expectFailure("actionable non-https provenance") {
            record().copy(
                fetchCanonicalUrl =
                    "http://example.com/cap"
            )
        }

        val manual = record(
            packageSha = "88".repeat(32),
            proposalId = "",
            fetchReceipt = "",
        )
        check(manual.proposalId.isEmpty())
        check(manual.fetchReceiptId.isEmpty())
        check(
            ledger.saveCompleted(manual) ==
                manual
        )

        println(
            "M16E acquisition provenance contracts: PASS"
        )
    } finally {
        root.deleteRecursively()
    }
}
