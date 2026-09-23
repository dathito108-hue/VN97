package ai.vn97.platform

import java.nio.file.Files

private fun expectM15BFailure(label: String, block: () -> Unit) {
    check(runCatching(block).isFailure) {
        "expected M15B failure: $label"
    }
}

private fun quote(
    symbol: String,
    bid: Long,
    ask: Long,
    timestampNs: Long,
): VN97MarketQuote = VN97MarketQuote(
    symbol = symbol,
    bidPriceMicros = bid,
    askPriceMicros = ask,
    timestampNs = timestampNs,
)

fun main() {
    val abc = quote("ABC", 99_000_000L, 100_000_000L, 990L)
    val xyz = quote("XYZ", 49_000_000L, 50_000_000L, 995L)

    val first = VN97MarketSnapshot.create(
        sourceId = "fixture.market",
        observedNs = 1_000L,
        quotes = listOf(xyz, abc),
        maxQuoteAgeNs = 20L,
    )
    val second = VN97MarketSnapshot.create(
        sourceId = "fixture.market",
        observedNs = 1_000L,
        quotes = listOf(abc, xyz),
        maxQuoteAgeNs = 20L,
    )
    check(first.snapshotId == second.snapshotId)
    check(first.quotes.map { it.symbol } == listOf("ABC", "XYZ"))
    check(first.quote("ABC") == abc)
    check(first.quote("MISSING") == null)
    check(
        first.canonicalObservation()
            .contains("snapshot_id=${first.snapshotId}")
    )

    expectM15BFailure("duplicate symbol") {
        VN97MarketSnapshot.create(
            sourceId = "fixture.market",
            observedNs = 1_000L,
            quotes = listOf(abc, abc),
            maxQuoteAgeNs = 20L,
        )
    }

    expectM15BFailure("stale quote") {
        VN97MarketSnapshot.create(
            sourceId = "fixture.market",
            observedNs = 2_000L,
            quotes = listOf(abc),
            maxQuoteAgeNs = 20L,
        )
    }

    expectM15BFailure("future quote") {
        VN97MarketSnapshot.create(
            sourceId = "fixture.market",
            observedNs = 900L,
            quotes = listOf(abc),
            maxQuoteAgeNs = 200L,
        )
    }

    expectM15BFailure("invalid source") {
        VN97MarketSnapshot.create(
            sourceId = "bad source",
            observedNs = 1_000L,
            quotes = listOf(abc),
            maxQuoteAgeNs = 20L,
        )
    }

    val tooMany = (0..VN97MarketSnapshot.MAX_QUOTES).map { index ->
        quote(
            symbol = "S${index.toString().padStart(2, '0')}",
            bid = 1_000_000L,
            ask = 1_000_001L,
            timestampNs = 1_000L,
        )
    }
    expectM15BFailure("too many quotes") {
        VN97MarketSnapshot.create(
            sourceId = "fixture.market",
            observedNs = 1_000L,
            quotes = tooMany,
            maxQuoteAgeNs = 20L,
        )
    }

    val hold = VN97PaperTradingDecision.parseCanonical("HOLD")
    check(hold.kind == VN97PaperTradingDecisionKind.HOLD)

    val buy = VN97PaperTradingDecision.parseCanonical(
        "ORDER|BUY|ABC|2000000|990"
    )
    check(buy.kind == VN97PaperTradingDecisionKind.ORDER)
    check(buy.side == VN97TradeSide.BUY)
    check(buy.symbol == "ABC")
    check(buy.quantityMicrounits == 2_000_000L)
    check(buy.quoteTimestampNs == 990L)

    expectM15BFailure("decision prose") {
        VN97PaperTradingDecision.parseCanonical(
            "I think ORDER|BUY|ABC|2000000|990"
        )
    }

    expectM15BFailure("zero quantity") {
        VN97PaperTradingDecision.parseCanonical(
            "ORDER|BUY|ABC|0|990"
        )
    }

    expectM15BFailure("lowercase symbol") {
        VN97PaperTradingDecision.parseCanonical(
            "ORDER|BUY|abc|1000000|990"
        )
    }

    val root = Files.createTempDirectory("vn97-m15b").toFile()
    val journal = root.resolve("paper.vn97trd1")
    val account = VN97PaperTradingAccount.openOrCreate(
        file = journal,
        startingCashMicros = 1_000_000_000L,
        riskPolicy = VN97PaperTradingRiskPolicy(
            maxOrderNotionalMicros = 500_000_000L,
            maxGrossBookCostMicros = 800_000_000L,
            minCashReserveMicros = 100_000_000L,
            feeBasisPoints = 0,
            maxSymbols = 4,
        ),
    )

    val holdResult = VN97PaperTradingDecisionExecutor.execute(
        account = account,
        snapshot = first,
        decision = hold,
        executedNs = 1_001L,
    )
    check(holdResult.fill == null)
    check(account.snapshot().sequence == 0L)

    expectM15BFailure("quote timestamp binding") {
        VN97PaperTradingDecisionExecutor.execute(
            account = account,
            snapshot = first,
            decision = VN97PaperTradingDecision.parseCanonical(
                "ORDER|BUY|ABC|1000000|995"
            ),
            executedNs = 1_001L,
        )
    }
    check(account.snapshot().sequence == 0L)

    val executed = VN97PaperTradingDecisionExecutor.execute(
        account = account,
        snapshot = first,
        decision = buy,
        executedNs = 1_001L,
    )
    check(executed.fill != null)
    check(executed.fill?.fillPriceMicros == 100_000_000L)
    check(executed.fill?.quantityMicrounits == 2_000_000L)
    check(account.snapshot().sequence == 1L)

    expectM15BFailure("one successful order per snapshot") {
        VN97PaperTradingDecisionExecutor.execute(
            account = account,
            snapshot = first,
            decision = VN97PaperTradingDecision.parseCanonical(
                "ORDER|BUY|XYZ|1000000|995"
            ),
            executedNs = 1_002L,
        )
    }
    check(account.snapshot().sequence == 1L)

    expectM15BFailure("execution before observation") {
        VN97PaperTradingDecisionExecutor.execute(
            account = account,
            snapshot = second,
            decision = hold,
            executedNs = 999L,
        )
    }

    println("M15B market observation and paper decision contracts: PASS")
}
