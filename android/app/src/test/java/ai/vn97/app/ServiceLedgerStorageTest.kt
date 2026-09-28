package ai.vn97.app

import java.io.File
import java.io.IOException
import java.nio.file.Files
import java.util.concurrent.Callable
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import org.junit.Assert.*
import org.junit.Test

class ServiceLedgerStorageTest {
    private fun order(id: String) = VN97ServiceOrder(
        orderId = id, title = id, service = VN97DigitalService.values().first(),
        quotedPriceVnd = 100L, estimatedCostVnd = 10L,
        outputSha256 = "a".repeat(64), createdAtEpochMs = 1L,
    )

    private fun rejected(block: () -> Unit) {
        try { block(); fail("operation should fail") } catch (_: Exception) { }
    }

    @Test fun missingStoreCreatesAndRoundTripsOrder() {
        var stored: String? = null
        val repository = VN97ServiceLedgerRepository({ stored }, { stored = it })
        assertEquals(0, repository.load().summary().orderCount)
        repository.update { it.recordOrder(order("first")) }
        assertEquals(listOf(order("first")), repository.load().orders())
    }

    @Test fun paymentEvidenceSurvivesCodecRoundTrip() {
        val ledger = VN97ServiceRevenueLedger(listOf(order("paid")))
        ledger.recordPayment(VN97PaymentRecord(
            orderId = "paid", provider = "test", providerAccountFingerprint = "",
            externalEventId = "manual", grossVnd = 100L, providerFeeVnd = 2L,
            refundVnd = 0L, realizedCostVnd = 10L,
            verification = VN97PaymentVerification.MANUAL_UNVERIFIED,
            evidenceSha256 = null, settledAtEpochMs = 2L,
        ))
        val restored = VN97ServiceLedgerCodec.decode(VN97ServiceLedgerCodec.encode(ledger))
        assertEquals(ledger.payments(), restored.payments())
        assertEquals(0L, restored.summary().verifiedNetVnd)
        assertEquals(88L, restored.summary().unverifiedClaimNetVnd)
    }

    @Test fun malformedAndUnsupportedFilesNeverBecomeEmptyOrGetWritten() {
        val broken = listOf("", "{", "[]", "{\"schema\":99}",
            "{\"schema\":1,\"orders\":[],\"payments\":[{}]}",
            "x".repeat(VN97ServiceLedgerCodec.MAX_BYTES + 1))
        for (original in broken) {
            var stored = original
            var changed = false
            var writes = 0
            val repository = VN97ServiceLedgerRepository({ stored }, { writes++; stored = it })
            rejected { repository.load() }
            rejected { repository.update { changed = true; it.recordOrder(order("new")) } }
            assertFalse(changed)
            assertEquals(0, writes)
            assertEquals(original, stored)
        }
    }

    @Test fun ioFailureBlocksMutationAndPreservesExistingBytes() {
        val original = VN97ServiceLedgerCodec.encode(VN97ServiceRevenueLedger(listOf(order("old"))))
        var writes = 0
        val repository = VN97ServiceLedgerRepository({ throw IOException("read denied") }, { writes++ })
        rejected { repository.update { it.recordOrder(order("new")) } }
        assertEquals(0, writes)
        assertEquals("old", VN97ServiceLedgerCodec.decode(original).orders().single().orderId)
    }

    @Test fun failedCommitAndFailedMutationDoNotChangeDurableSnapshot() {
        var stored = VN97ServiceLedgerCodec.encode(VN97ServiceRevenueLedger(listOf(order("old"))))
        val original = stored
        var failWrite = true
        val repository = VN97ServiceLedgerRepository({ stored }, {
            if (failWrite) throw IOException("disk full") else stored = it
        })
        rejected { repository.update { it.recordOrder(order("new")) } }
        assertEquals(original, stored)
        failWrite = false
        rejected { repository.update { it.recordOrder(order("new")); error("cancel transaction") } }
        assertEquals(original, stored)
        repository.update { it.recordOrder(order("new")) }
        assertEquals(2, repository.load().summary().orderCount)
    }

    @Test fun separateRepositoriesCannotLoseConcurrentOrders() {
        var stored: String? = null
        val first = VN97ServiceLedgerRepository({ stored }, { stored = it })
        val second = VN97ServiceLedgerRepository({ stored }, { stored = it })
        val pool = Executors.newFixedThreadPool(4)
        try {
            val futures = (0 until 40).map { index ->
                pool.submit(Callable {
                    (if (index % 2 == 0) first else second).update { it.recordOrder(order("order-$index")) }
                })
            }
            futures.forEach { it.get(10, TimeUnit.SECONDS) }
            assertEquals(40, first.load().summary().orderCount)
        } finally { pool.shutdownNow() }
    }

    @Test fun backupOrPendingFileIsNotConfirmedAbsence() {
        val directory = Files.createTempDirectory("vn97-ledger").toFile()
        try {
            val base = File(directory, "ledger.json")
            assertTrue(VN97ServiceLedgerRepository.definitelyMissing(base))
            for (path in listOf(base, File(base.path + ".bak"), File(base.path + ".new"))) {
                path.writeText("stored")
                assertFalse(VN97ServiceLedgerRepository.definitelyMissing(base))
                assertTrue(path.delete())
            }
            Files.createSymbolicLink(base.toPath(), File(directory, "absent").toPath())
            assertFalse(VN97ServiceLedgerRepository.definitelyMissing(base))
        } finally { directory.deleteRecursively() }
    }
}
