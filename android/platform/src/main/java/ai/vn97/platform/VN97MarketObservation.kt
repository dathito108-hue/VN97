package ai.vn97.platform

import java.nio.charset.StandardCharsets
import java.security.MessageDigest

data class VN97MarketSnapshot private constructor(
    val snapshotId: String,
    val sourceId: String,
    val observedNs: Long,
    val maxQuoteAgeNs: Long,
    val quotes: List<VN97MarketQuote>,
) {
    init {
        require(SNAPSHOT_ID_RE.matches(snapshotId)) {
            "market snapshot id is invalid"
        }
        require(SOURCE_ID_RE.matches(sourceId)) {
            "market source id is invalid"
        }
        require(observedNs >= 0L) {
            "market observation time must be non-negative"
        }
        require(maxQuoteAgeNs in 1L..MAX_QUOTE_AGE_NS) {
            "market max quote age is outside bounds"
        }
        require(quotes.size in 1..MAX_QUOTES) {
            "market snapshot quote count is outside bounds"
        }
        require(quotes.map { it.symbol }.distinct().size == quotes.size) {
            "market snapshot symbols must be unique"
        }
        require(quotes == quotes.sortedBy { it.symbol }) {
            "market snapshot quotes must be canonically sorted"
        }
        quotes.forEach { quote ->
            require(quote.timestampNs <= observedNs) {
                "market quote cannot be newer than observation"
            }
            require(observedNs - quote.timestampNs <= maxQuoteAgeNs) {
                "market quote is stale"
            }
        }
    }

    fun quote(symbol: String): VN97MarketQuote? =
        quotes.binarySearchBy(symbol) { it.symbol }
            .takeIf { it >= 0 }
            ?.let(quotes::get)

    fun canonicalObservation(): String = buildString {
        append("snapshot_id=")
        append(snapshotId)
        append("\nsource=")
        append(sourceId)
        append("\nobserved_ns=")
        append(observedNs)
        append("\nquotes=")
        quotes.forEachIndexed { index, quote ->
            if (index > 0) append(';')
            append(quote.symbol)
            append(',')
            append(quote.bidPriceMicros)
            append(',')
            append(quote.askPriceMicros)
            append(',')
            append(quote.timestampNs)
        }
    }

    companion object {
        const val DEFAULT_MAX_QUOTE_AGE_NS = 30_000_000_000L
        const val MAX_QUOTE_AGE_NS = 300_000_000_000L
        const val MAX_QUOTES = 64

        fun create(
            sourceId: String,
            observedNs: Long,
            quotes: List<VN97MarketQuote>,
            maxQuoteAgeNs: Long = DEFAULT_MAX_QUOTE_AGE_NS,
        ): VN97MarketSnapshot {
            require(SOURCE_ID_RE.matches(sourceId)) {
                "market source id is invalid"
            }
            require(observedNs >= 0L) {
                "market observation time must be non-negative"
            }
            require(maxQuoteAgeNs in 1L..MAX_QUOTE_AGE_NS) {
                "market max quote age is outside bounds"
            }
            require(quotes.size in 1..MAX_QUOTES) {
                "market snapshot quote count is outside bounds"
            }
            val sorted = quotes.sortedBy { it.symbol }
            require(sorted.map { it.symbol }.distinct().size == sorted.size) {
                "market snapshot symbols must be unique"
            }
            sorted.forEach { quote ->
                require(quote.timestampNs <= observedNs) {
                    "market quote cannot be newer than observation"
                }
                require(observedNs - quote.timestampNs <= maxQuoteAgeNs) {
                    "market quote is stale"
                }
            }
            val snapshotId = computeId(
                sourceId = sourceId,
                observedNs = observedNs,
                maxQuoteAgeNs = maxQuoteAgeNs,
                quotes = sorted,
            )
            return VN97MarketSnapshot(
                snapshotId = snapshotId,
                sourceId = sourceId,
                observedNs = observedNs,
                maxQuoteAgeNs = maxQuoteAgeNs,
                quotes = sorted,
            )
        }

        private fun computeId(
            sourceId: String,
            observedNs: Long,
            maxQuoteAgeNs: Long,
            quotes: List<VN97MarketQuote>,
        ): String {
            val payload = buildString {
                append("VN97MKT1\n")
                append(sourceId)
                append('\n')
                append(observedNs)
                append('\n')
                append(maxQuoteAgeNs)
                quotes.forEach { quote ->
                    append('\n')
                    append(quote.symbol)
                    append('|')
                    append(quote.bidPriceMicros)
                    append('|')
                    append(quote.askPriceMicros)
                    append('|')
                    append(quote.timestampNs)
                }
            }
            return MessageDigest.getInstance("SHA-256")
                .digest(payload.toByteArray(StandardCharsets.UTF_8))
                .joinToString("") { "%02x".format(it.toInt() and 0xff) }
        }

        private val SOURCE_ID_RE =
            Regex("^[A-Za-z0-9][A-Za-z0-9._:-]{0,63}$")
        private val SNAPSHOT_ID_RE =
            Regex("^[0-9a-f]{64}$")
    }
}

enum class VN97PaperTradingDecisionKind {
    HOLD,
    ORDER,
}

data class VN97PaperTradingDecision(
    val kind: VN97PaperTradingDecisionKind,
    val side: VN97TradeSide? = null,
    val symbol: String = "",
    val quantityMicrounits: Long = 0L,
    val quoteTimestampNs: Long = 0L,
) {
    init {
        when (kind) {
            VN97PaperTradingDecisionKind.HOLD -> {
                require(side == null)
                require(symbol.isEmpty())
                require(quantityMicrounits == 0L)
                require(quoteTimestampNs == 0L)
            }
            VN97PaperTradingDecisionKind.ORDER -> {
                require(side != null)
                require(symbol.isNotEmpty())
                require(quantityMicrounits > 0L)
                require(quoteTimestampNs >= 0L)
            }
        }
    }

    companion object {
        fun parseCanonical(value: String): VN97PaperTradingDecision {
            if (value == "HOLD") {
                return VN97PaperTradingDecision(
                    kind = VN97PaperTradingDecisionKind.HOLD,
                )
            }
            val match = ORDER_RE.matchEntire(value)
                ?: throw IllegalArgumentException(
                    "trading decision must be HOLD or canonical ORDER"
                )
            val side = VN97TradeSide.valueOf(match.groupValues[1])
            val symbol = match.groupValues[2]
            val quantity = match.groupValues[3].toLongOrNull()
                ?: throw IllegalArgumentException(
                    "trading decision quantity overflows"
                )
            val quoteTimestamp = match.groupValues[4].toLongOrNull()
                ?: throw IllegalArgumentException(
                    "trading decision quote timestamp overflows"
                )
            return VN97PaperTradingDecision(
                kind = VN97PaperTradingDecisionKind.ORDER,
                side = side,
                symbol = symbol,
                quantityMicrounits = quantity,
                quoteTimestampNs = quoteTimestamp,
            )
        }

        private val ORDER_RE =
            Regex(
                "^ORDER\\|(BUY|SELL)\\|" +
                    "([A-Z][A-Z0-9._-]{0,23})\\|" +
                    "([1-9][0-9]*)\\|([0-9]+)$"
            )
    }
}

data class VN97PaperTradingDecisionResult(
    val snapshotId: String,
    val decision: VN97PaperTradingDecision,
    val fill: VN97PaperFill?,
) {
    init {
        if (decision.kind == VN97PaperTradingDecisionKind.HOLD) {
            require(fill == null) {
                "HOLD decision cannot contain a fill"
            }
        } else {
            require(fill != null) {
                "ORDER decision requires a fill"
            }
        }
    }
}

object VN97PaperTradingDecisionExecutor {
    fun execute(
        account: VN97PaperTradingAccount,
        snapshot: VN97MarketSnapshot,
        decision: VN97PaperTradingDecision,
        executedNs: Long,
    ): VN97PaperTradingDecisionResult {
        require(executedNs >= snapshot.observedNs) {
            "paper decision execution cannot predate market observation"
        }
        if (decision.kind == VN97PaperTradingDecisionKind.HOLD) {
            return VN97PaperTradingDecisionResult(
                snapshotId = snapshot.snapshotId,
                decision = decision,
                fill = null,
            )
        }

        val quote = checkNotNull(snapshot.quote(decision.symbol)) {
            "paper decision symbol is absent from market snapshot"
        }
        require(decision.quoteTimestampNs == quote.timestampNs) {
            "paper decision is not bound to the snapshot quote timestamp"
        }
        val order = VN97PaperOrder(
            orderId = orderIdForSnapshot(snapshot.snapshotId),
            symbol = decision.symbol,
            side = checkNotNull(decision.side),
            quantityMicrounits = decision.quantityMicrounits,
            quoteTimestampNs = quote.timestampNs,
            createdNs = snapshot.observedNs,
        )
        return VN97PaperTradingDecisionResult(
            snapshotId = snapshot.snapshotId,
            decision = decision,
            fill = account.execute(
                order = order,
                quote = quote,
                executedNs = executedNs,
            ),
        )
    }

    private fun orderIdForSnapshot(snapshotId: String): String =
        "m15b-" + snapshotId.take(40)
}
