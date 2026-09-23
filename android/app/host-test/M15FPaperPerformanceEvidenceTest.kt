package ai.vn97.app

import ai.vn97.platform.VN97MarketQuote
import ai.vn97.platform.VN97MarketSnapshot
import ai.vn97.platform.VN97PaperPerformanceEvaluator
import ai.vn97.platform.VN97PaperPosition
import ai.vn97.platform.VN97PaperTradingSnapshot
import java.nio.file.Files

private fun expectM15FFailure(
    label: String,
    block: () -> Unit,
) {
    check(runCatching(block).isFailure) {
        "expected M15F failure: $label"
    }
}

private fun market(
    observedNs: Long,
    bid: Long,
): VN97MarketSnapshot =
    VN97MarketSnapshot.create(
        sourceId = "fixture.market",
        observedNs = observedNs,
        quotes = listOf(
            VN97MarketQuote(
                symbol = "ABC",
                bidPriceMicros = bid,
                askPriceMicros = bid + 1_000_000L,
                timestampNs = observedNs - 1L,
            )
        ),
        maxQuoteAgeNs = 10L,
    )

private fun account(): VN97PaperTradingSnapshot =
    VN97PaperTradingSnapshot(
        startingCashMicros = 100_000_000L,
        cashMicros = 50_000_000L,
        positions = mapOf(
            "ABC" to VN97PaperPosition(
                symbol = "ABC",
                quantityMicrounits = 500_000L,
                bookCostMicros = 50_000_000L,
            )
        ),
        executedOrderIds = emptySet(),
        sequence = 7L,
    )

fun main() {
    val firstMarket = market(
        observedNs = 1_000L,
        bid = 110_000_000L,
    )
    val first =
        VN97PaperPerformanceEvaluator.evaluate(
            account = account(),
            market = firstMarket,
        )
    check(first.markedPositionValueMicros == 55_000_000L)
    check(first.markedEquityMicros == 105_000_000L)
    check(first.grossBookCostMicros == 50_000_000L)
    check(first.unrealizedPnlMicros == 5_000_000L)
    check(first.totalPnlMicros == 5_000_000L)
    check(first.totalReturnBasisPoints == 500L)
    check(first.positions.single().bidPriceMicros == 110_000_000L)

    val noHeldSymbol = VN97MarketSnapshot.create(
        sourceId = "fixture.market",
        observedNs = 1_100L,
        quotes = listOf(
            VN97MarketQuote(
                symbol = "XYZ",
                bidPriceMicros = 10_000_000L,
                askPriceMicros = 11_000_000L,
                timestampNs = 1_099L,
            )
        ),
        maxQuoteAgeNs = 10L,
    )
    expectM15FFailure("missing held symbol") {
        VN97PaperPerformanceEvaluator.evaluate(
            account = account(),
            market = noHeldSymbol,
        )
    }

    val second =
        VN97PaperPerformanceEvaluator.evaluate(
            account = account(),
            market = market(
                observedNs = 2_000L,
                bid = 90_000_000L,
            ),
        )
    check(second.markedEquityMicros == 95_000_000L)
    check(second.totalPnlMicros == -5_000_000L)
    check(second.totalReturnBasisPoints == -500L)

    val root = Files.createTempDirectory("vn97-m15f").toFile()
    try {
        val store = VN97PaperPerformanceEvidenceStore(root)
        val sessionId = "a".repeat(64)
        val firstEvidence = VN97PaperPerformanceEvidence.from(
            sessionId = sessionId,
            jobId = 601,
            episode = 1,
            recordedWallTimeMillis = 10_000L,
            performance = first,
            decision = "HOLD",
            outcome = "HOLD snapshot=" + first.snapshotId,
        )
        store.append(firstEvidence)
        store.append(firstEvidence)
        check(store.loadSession(sessionId).size == 1)

        val thirdEvidence = VN97PaperPerformanceEvidence.from(
            sessionId = sessionId,
            jobId = 601,
            episode = 3,
            recordedWallTimeMillis = 20_000L,
            performance = second,
            decision = "ORDER|SELL|ABC|1|1999",
            outcome = "PAPER_FILL sequence=8",
        )
        store.append(thirdEvidence)
        val loaded = store.loadSession(sessionId)
        check(loaded.map { it.episode } == listOf(1, 3))

        expectM15FFailure("immutable episode evidence") {
            store.append(
                firstEvidence.copy(
                    outcome = "different outcome",
                )
            )
        }

        val aggregate = store.aggregate()
        check(aggregate.evidenceCount == 2)
        check(aggregate.sessionCount == 1)
        val summary = aggregate.sessions.single()
        check(summary.firstEpisode == 1)
        check(summary.latestEpisode == 3)
        check(summary.latestEquityMicros == 95_000_000L)
        check(summary.latestTotalPnlMicros == -5_000_000L)
        check(summary.latestReturnBasisPoints == -500L)
        check(summary.peakEquityMicros == 105_000_000L)
        check(summary.maxDrawdownMicros == 10_000_000L)
        check(summary.maxDrawdownBasisPoints == 952L)
        check(summary.holdCount == 1)
        check(summary.orderCount == 1)
    } finally {
        root.deleteRecursively()
    }

    println("M15F paper performance evidence contracts: PASS")
}
