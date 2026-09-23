package ai.vn97.platform

import java.math.BigInteger

data class VN97PaperPositionMark(
    val symbol: String,
    val quantityMicrounits: Long,
    val bookCostMicros: Long,
    val bidPriceMicros: Long,
    val marketValueMicros: Long,
    val unrealizedPnlMicros: Long,
) {
    init {
        require(symbol.isNotBlank())
        require(quantityMicrounits > 0L)
        require(bookCostMicros >= 0L)
        require(bidPriceMicros > 0L)
        require(marketValueMicros >= 0L)
    }
}

data class VN97PaperPerformance(
    val snapshotId: String,
    val observedNs: Long,
    val accountSequence: Long,
    val startingCashMicros: Long,
    val cashMicros: Long,
    val grossBookCostMicros: Long,
    val markedPositionValueMicros: Long,
    val markedEquityMicros: Long,
    val unrealizedPnlMicros: Long,
    val totalPnlMicros: Long,
    val totalReturnBasisPoints: Long,
    val positions: List<VN97PaperPositionMark>,
) {
    init {
        require(snapshotId.length == 64)
        require(observedNs >= 0L)
        require(accountSequence >= 0L)
        require(startingCashMicros > 0L)
        require(cashMicros >= 0L)
        require(grossBookCostMicros >= 0L)
        require(markedPositionValueMicros >= 0L)
        require(markedEquityMicros >= 0L)
        require(positions == positions.sortedBy { it.symbol })
        require(
            positions.sumOfExact { it.bookCostMicros } ==
                grossBookCostMicros
        )
        require(
            positions.sumOfExact { it.marketValueMicros } ==
                markedPositionValueMicros
        )
        require(
            Math.addExact(cashMicros, markedPositionValueMicros) ==
                markedEquityMicros
        )
        require(
            Math.subtractExact(
                markedPositionValueMicros,
                grossBookCostMicros,
            ) == unrealizedPnlMicros
        )
        require(
            Math.subtractExact(
                markedEquityMicros,
                startingCashMicros,
            ) == totalPnlMicros
        )
    }
}

object VN97PaperPerformanceEvaluator {
    /**
     * Conservative paper mark: every open long position is valued at the
     * current bid, i.e. the price available to liquidate it in the supplied
     * fresh market snapshot. No model output participates in this arithmetic.
     */
    fun evaluate(
        account: VN97PaperTradingSnapshot,
        market: VN97MarketSnapshot,
    ): VN97PaperPerformance {
        val marks = account.positions.values
            .sortedBy { it.symbol }
            .map { position ->
                val quote = requireNotNull(market.quote(position.symbol)) {
                    "paper performance snapshot is missing held symbol " +
                        position.symbol
                }
                val value = scaledMultiplyExact(
                    quote.bidPriceMicros,
                    position.quantityMicrounits,
                    VN97PaperTradingAccount.QUANTITY_SCALE,
                )
                VN97PaperPositionMark(
                    symbol = position.symbol,
                    quantityMicrounits = position.quantityMicrounits,
                    bookCostMicros = position.bookCostMicros,
                    bidPriceMicros = quote.bidPriceMicros,
                    marketValueMicros = value,
                    unrealizedPnlMicros =
                        Math.subtractExact(
                            value,
                            position.bookCostMicros,
                        ),
                )
            }
        val grossBookCost = marks.sumOfExact { it.bookCostMicros }
        val markedPositionValue =
            marks.sumOfExact { it.marketValueMicros }
        val equity =
            Math.addExact(account.cashMicros, markedPositionValue)
        val unrealized =
            Math.subtractExact(
                markedPositionValue,
                grossBookCost,
            )
        val totalPnl =
            Math.subtractExact(equity, account.startingCashMicros)
        val returnBps = scaledSignedRatioExact(
            numerator = totalPnl,
            scale = 10_000L,
            denominator = account.startingCashMicros,
        )
        return VN97PaperPerformance(
            snapshotId = market.snapshotId,
            observedNs = market.observedNs,
            accountSequence = account.sequence,
            startingCashMicros = account.startingCashMicros,
            cashMicros = account.cashMicros,
            grossBookCostMicros = grossBookCost,
            markedPositionValueMicros = markedPositionValue,
            markedEquityMicros = equity,
            unrealizedPnlMicros = unrealized,
            totalPnlMicros = totalPnl,
            totalReturnBasisPoints = returnBps,
            positions = marks,
        )
    }

    private fun scaledMultiplyExact(
        left: Long,
        right: Long,
        divisor: Long,
    ): Long {
        require(left >= 0L && right >= 0L && divisor > 0L)
        return BigInteger.valueOf(left)
            .multiply(BigInteger.valueOf(right))
            .divide(BigInteger.valueOf(divisor))
            .longValueExact()
    }

    private fun scaledSignedRatioExact(
        numerator: Long,
        scale: Long,
        denominator: Long,
    ): Long {
        require(scale > 0L && denominator > 0L)
        return BigInteger.valueOf(numerator)
            .multiply(BigInteger.valueOf(scale))
            .divide(BigInteger.valueOf(denominator))
            .longValueExact()
    }
}

private inline fun <T> Iterable<T>.sumOfExact(
    value: (T) -> Long,
): Long =
    fold(0L) { total, item ->
        Math.addExact(total, value(item))
    }
