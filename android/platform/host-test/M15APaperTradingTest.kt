package ai.vn97.platform

import java.nio.file.Files

private fun expectPaperFailure(label: String, block: () -> Unit) {
    check(runCatching(block).isFailure) {
        "expected paper-trading failure: $label"
    }
}

fun main() {
    val root = Files.createTempDirectory("vn97-m15a").toFile()
    val file = root.resolve("paper.vn97trd1")
    val policy = VN97PaperTradingRiskPolicy(
        maxOrderNotionalMicros = 500_000_000L,
        maxGrossBookCostMicros = 800_000_000L,
        minCashReserveMicros = 100_000_000L,
        feeBasisPoints = 10,
        maxSymbols = 2,
    )
    val account = VN97PaperTradingAccount.openOrCreate(
        file = file,
        startingCashMicros = 1_000_000_000L,
        riskPolicy = policy,
    )

    val quote1 = VN97MarketQuote(
        symbol = "ABC",
        bidPriceMicros = 99_000_000L,
        askPriceMicros = 100_000_000L,
        timestampNs = 10L,
    )
    val buy = VN97PaperOrder(
        orderId = "ord-1",
        symbol = "ABC",
        side = VN97TradeSide.BUY,
        quantityMicrounits = 2_000_000L,
        quoteTimestampNs = 10L,
        createdNs = 10L,
    )
    val buyFill = account.execute(
        order = buy,
        quote = quote1,
        executedNs = 11L,
    )
    check(buyFill.fillPriceMicros == 100_000_000L)
    check(buyFill.notionalMicros == 200_000_000L)
    check(buyFill.feeMicros == 200_000L)
    check(buyFill.cashAfterMicros == 799_800_000L)
    check(buyFill.positionQuantityAfterMicrounits == 2_000_000L)
    check(buyFill.positionBookCostAfterMicros == 200_200_000L)

    expectPaperFailure("duplicate order id") {
        account.execute(buy, quote1, 12L)
    }

    expectPaperFailure("stale quote mismatch") {
        account.execute(
            VN97PaperOrder(
                orderId = "ord-stale",
                symbol = "ABC",
                side = VN97TradeSide.BUY,
                quantityMicrounits = 1_000_000L,
                quoteTimestampNs = 12L,
                createdNs = 12L,
            ),
            quote1,
            13L,
        )
    }

    expectPaperFailure("max order notional") {
        account.execute(
            VN97PaperOrder(
                orderId = "ord-too-large",
                symbol = "ABC",
                side = VN97TradeSide.BUY,
                quantityMicrounits = 6_000_000L,
                quoteTimestampNs = 10L,
                createdNs = 10L,
            ),
            quote1,
            14L,
        )
    }

    expectPaperFailure("oversell") {
        account.execute(
            VN97PaperOrder(
                orderId = "ord-short",
                symbol = "ABC",
                side = VN97TradeSide.SELL,
                quantityMicrounits = 3_000_000L,
                quoteTimestampNs = 10L,
                createdNs = 10L,
            ),
            quote1,
            15L,
        )
    }

    val quote2 = VN97MarketQuote(
        symbol = "ABC",
        bidPriceMicros = 110_000_000L,
        askPriceMicros = 111_000_000L,
        timestampNs = 20L,
    )
    val sell = VN97PaperOrder(
        orderId = "ord-2",
        symbol = "ABC",
        side = VN97TradeSide.SELL,
        quantityMicrounits = 1_000_000L,
        quoteTimestampNs = 20L,
        createdNs = 20L,
    )
    val sellFill = account.execute(
        order = sell,
        quote = quote2,
        executedNs = 21L,
    )
    check(sellFill.fillPriceMicros == 110_000_000L)
    check(sellFill.notionalMicros == 110_000_000L)
    check(sellFill.feeMicros == 110_000L)
    check(sellFill.cashAfterMicros == 909_690_000L)
    check(sellFill.positionQuantityAfterMicrounits == 1_000_000L)
    check(sellFill.positionBookCostAfterMicros == 100_100_000L)

    val beforeRestart = account.snapshot()
    check(beforeRestart.sequence == 2L)
    check(beforeRestart.positions["ABC"]?.quantityMicrounits == 1_000_000L)
    check(beforeRestart.grossBookCostMicros == 100_100_000L)
    check(beforeRestart.executedOrderIds == setOf("ord-1", "ord-2"))

    val reopened = VN97PaperTradingAccount.openOrCreate(
        file = file,
        startingCashMicros = 1_000_000_000L,
        riskPolicy = policy,
    )
    val afterRestart = reopened.snapshot()
    check(afterRestart == beforeRestart)

    expectPaperFailure("risk policy mismatch on replay") {
        VN97PaperTradingAccount.openOrCreate(
            file = file,
            startingCashMicros = 1_000_000_000L,
            riskPolicy = policy.copy(feeBasisPoints = 0),
        )
    }

    expectPaperFailure("invalid lowercase symbol") {
        VN97MarketQuote(
            symbol = "abc",
            bidPriceMicros = 1L,
            askPriceMicros = 1L,
            timestampNs = 1L,
        )
    }

    val lines = file.readLines()
    check(lines.size == 3) {
        "failed paper orders must not be appended to journal"
    }
    check(lines.first().startsWith("VN97TRD1|G|"))
    check(lines[1].startsWith("VN97TRD1|O|ord-1|ABC|BUY|"))
    check(lines[2].startsWith("VN97TRD1|O|ord-2|ABC|SELL|"))

    println("M15A paper trading foundation: PASS")
}
