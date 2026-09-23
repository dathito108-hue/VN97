package ai.vn97.platform

import java.io.File
import java.io.FileOutputStream
import java.math.BigInteger
import java.nio.charset.StandardCharsets

enum class VN97TradeSide {
    BUY,
    SELL,
}

data class VN97MarketQuote(
    val symbol: String,
    val bidPriceMicros: Long,
    val askPriceMicros: Long,
    val timestampNs: Long,
) {
    init {
        require(isTradingSymbol(symbol)) { "trading symbol is invalid" }
        require(bidPriceMicros > 0L) { "bid price must be positive" }
        require(askPriceMicros >= bidPriceMicros) {
            "ask price must be greater than or equal to bid"
        }
        require(timestampNs >= 0L) { "quote timestamp must be non-negative" }
    }
}

data class VN97PaperOrder(
    val orderId: String,
    val symbol: String,
    val side: VN97TradeSide,
    val quantityMicrounits: Long,
    val quoteTimestampNs: Long,
    val createdNs: Long,
) {
    init {
        require(isPaperOrderId(orderId)) { "paper order id is invalid" }
        require(isTradingSymbol(symbol)) { "paper order symbol is invalid" }
        require(quantityMicrounits > 0L) {
            "paper order quantity must be positive"
        }
        require(quoteTimestampNs >= 0L) {
            "paper order quote timestamp must be non-negative"
        }
        require(createdNs >= quoteTimestampNs) {
            "paper order cannot predate its observed quote"
        }
    }
}

data class VN97PaperTradingRiskPolicy(
    val maxOrderNotionalMicros: Long = 5_000_000_000L,
    val maxGrossBookCostMicros: Long = 20_000_000_000L,
    val minCashReserveMicros: Long = 0L,
    val feeBasisPoints: Int = 0,
    val maxSymbols: Int = 32,
) {
    init {
        require(maxOrderNotionalMicros > 0L) {
            "max paper order notional must be positive"
        }
        require(maxGrossBookCostMicros > 0L) {
            "max paper gross exposure must be positive"
        }
        require(minCashReserveMicros >= 0L) {
            "minimum cash reserve must be non-negative"
        }
        require(feeBasisPoints in 0..100) {
            "paper fee basis points are outside bounds"
        }
        require(maxSymbols in 1..128) {
            "paper max symbols is outside bounds"
        }
    }
}

data class VN97PaperPosition(
    val symbol: String,
    val quantityMicrounits: Long,
    val bookCostMicros: Long,
) {
    init {
        require(isTradingSymbol(symbol)) { "position symbol is invalid" }
        require(quantityMicrounits > 0L) {
            "position quantity must be positive"
        }
        require(bookCostMicros >= 0L) {
            "position book cost must be non-negative"
        }
    }
}

data class VN97PaperTradingSnapshot(
    val startingCashMicros: Long,
    val cashMicros: Long,
    val positions: Map<String, VN97PaperPosition>,
    val executedOrderIds: Set<String>,
    val sequence: Long,
) {
    init {
        require(startingCashMicros > 0L) {
            "starting paper cash must be positive"
        }
        require(cashMicros >= 0L) { "paper cash must be non-negative" }
        require(sequence >= 0L) { "paper sequence must be non-negative" }
        require(
            positions.keys == positions.values.map { it.symbol }.toSet()
        ) {
            "paper position keys do not match symbols"
        }
    }

    val grossBookCostMicros: Long
        get() = positions.values.fold(0L) { acc, position ->
            Math.addExact(acc, position.bookCostMicros)
        }
}

data class VN97PaperFill(
    val sequence: Long,
    val orderId: String,
    val symbol: String,
    val side: VN97TradeSide,
    val quantityMicrounits: Long,
    val fillPriceMicros: Long,
    val notionalMicros: Long,
    val feeMicros: Long,
    val quoteTimestampNs: Long,
    val executedNs: Long,
    val cashAfterMicros: Long,
    val positionQuantityAfterMicrounits: Long,
    val positionBookCostAfterMicros: Long,
)

class VN97PaperTradingAccount private constructor(
    private val journalFile: File,
    val riskPolicy: VN97PaperTradingRiskPolicy,
    startingCashMicros: Long,
    initialSnapshot: VN97PaperTradingSnapshot,
) {
    private var current = initialSnapshot

    init {
        require(startingCashMicros == initialSnapshot.startingCashMicros) {
            "paper account starting cash mismatch"
        }
    }

    @Synchronized
    fun snapshot(): VN97PaperTradingSnapshot =
        current.copy(
            positions = current.positions.toMap(),
            executedOrderIds = current.executedOrderIds.toSet(),
        )

    @Synchronized
    fun execute(
        order: VN97PaperOrder,
        quote: VN97MarketQuote,
        executedNs: Long,
    ): VN97PaperFill {
        val transition =
            transition(
                snapshot = current,
                policy = riskPolicy,
                order = order,
                quote = quote,
                executedNs = executedNs,
            )
        appendDurable(
            journalFile,
            encodeOrderRecord(order, quote, executedNs),
        )
        current = transition.second
        return transition.first
    }

    companion object {
        const val PRICE_SCALE = 1_000_000L
        const val QUANTITY_SCALE = 1_000_000L

        fun openOrCreate(
            file: File,
            startingCashMicros: Long,
            riskPolicy: VN97PaperTradingRiskPolicy =
                VN97PaperTradingRiskPolicy(),
        ): VN97PaperTradingAccount {
            require(startingCashMicros > 0L) {
                "starting paper cash must be positive"
            }
            require(
                startingCashMicros >= riskPolicy.minCashReserveMicros
            ) {
                "starting paper cash is below minimum reserve"
            }
            requireSafeJournalFile(file)

            if (!file.exists()) {
                file.parentFile?.let { parent ->
                    if (parent.exists()) {
                        require(parent.isDirectory) {
                            "paper trading parent is not a directory"
                        }
                    } else {
                        check(parent.mkdirs()) {
                            "failed to create paper trading directory"
                        }
                    }
                }
                appendDurable(
                    file,
                    encodeGenesis(startingCashMicros, riskPolicy),
                )
            }

            val lines =
                file.readLines(StandardCharsets.UTF_8)
                    .filter { it.isNotEmpty() }
            require(lines.isNotEmpty()) {
                "paper trading journal is empty"
            }
            val genesis = decodeGenesis(lines.first())
            require(genesis.first == startingCashMicros) {
                "paper trading starting cash does not match journal"
            }
            require(genesis.second == riskPolicy) {
                "paper trading risk policy does not match journal"
            }

            var snapshot = VN97PaperTradingSnapshot(
                startingCashMicros = startingCashMicros,
                cashMicros = startingCashMicros,
                positions = emptyMap(),
                executedOrderIds = emptySet(),
                sequence = 0L,
            )
            lines.drop(1).forEach { line ->
                val record = decodeOrderRecord(line)
                snapshot =
                    transition(
                        snapshot = snapshot,
                        policy = riskPolicy,
                        order = record.order,
                        quote = record.quote,
                        executedNs = record.executedNs,
                    ).second
            }

            return VN97PaperTradingAccount(
                journalFile = file,
                riskPolicy = riskPolicy,
                startingCashMicros = startingCashMicros,
                initialSnapshot = snapshot,
            )
        }

        internal fun transition(
            snapshot: VN97PaperTradingSnapshot,
            policy: VN97PaperTradingRiskPolicy,
            order: VN97PaperOrder,
            quote: VN97MarketQuote,
            executedNs: Long,
        ): Pair<VN97PaperFill, VN97PaperTradingSnapshot> {
            require(executedNs >= order.createdNs) {
                "paper execution cannot predate order creation"
            }
            require(order.orderId !in snapshot.executedOrderIds) {
                "paper order id was already executed"
            }
            require(order.symbol == quote.symbol) {
                "paper order symbol does not match quote"
            }
            require(order.quoteTimestampNs == quote.timestampNs) {
                "paper order quote timestamp does not match supplied quote"
            }

            val fillPrice =
                if (order.side == VN97TradeSide.BUY) {
                    quote.askPriceMicros
                } else {
                    quote.bidPriceMicros
                }
            val notional =
                scaledMultiply(
                    fillPrice,
                    order.quantityMicrounits,
                    QUANTITY_SCALE,
                )
            require(notional > 0L) {
                "paper order notional rounded to zero"
            }
            require(notional <= policy.maxOrderNotionalMicros) {
                "paper order exceeds max notional"
            }
            val fee =
                scaledMultiply(
                    notional,
                    policy.feeBasisPoints.toLong(),
                    10_000L,
                )

            val positions = snapshot.positions.toMutableMap()
            val before = positions[order.symbol]
            val nextCash: Long
            val nextQuantity: Long
            val nextBookCost: Long

            if (order.side == VN97TradeSide.BUY) {
                val cashOut = Math.addExact(notional, fee)
                nextCash = Math.subtractExact(snapshot.cashMicros, cashOut)
                require(nextCash >= policy.minCashReserveMicros) {
                    "paper buy violates minimum cash reserve"
                }

                val createsNewSymbol = before == null
                if (createsNewSymbol) {
                    require(positions.size < policy.maxSymbols) {
                        "paper account symbol limit reached"
                    }
                }
                nextQuantity = Math.addExact(
                    before?.quantityMicrounits ?: 0L,
                    order.quantityMicrounits,
                )
                nextBookCost = Math.addExact(
                    before?.bookCostMicros ?: 0L,
                    cashOut,
                )
                val projectedGross =
                    Math.addExact(
                        snapshot.grossBookCostMicros,
                        cashOut,
                    )
                require(
                    projectedGross <= policy.maxGrossBookCostMicros
                ) {
                    "paper buy exceeds gross book-cost limit"
                }
                positions[order.symbol] = VN97PaperPosition(
                    symbol = order.symbol,
                    quantityMicrounits = nextQuantity,
                    bookCostMicros = nextBookCost,
                )
            } else {
                val existing = checkNotNull(before) {
                    "paper sell requires an existing long position"
                }
                require(
                    order.quantityMicrounits <=
                        existing.quantityMicrounits
                ) {
                    "paper sell exceeds held quantity"
                }
                val cashIn = Math.subtractExact(notional, fee)
                require(cashIn >= 0L) {
                    "paper fee exceeds sell proceeds"
                }
                nextCash = Math.addExact(snapshot.cashMicros, cashIn)
                nextQuantity = Math.subtractExact(
                    existing.quantityMicrounits,
                    order.quantityMicrounits,
                )
                val removedBookCost =
                    if (nextQuantity == 0L) {
                        existing.bookCostMicros
                    } else {
                        scaledMultiply(
                            existing.bookCostMicros,
                            order.quantityMicrounits,
                            existing.quantityMicrounits,
                        )
                    }
                nextBookCost = Math.subtractExact(
                    existing.bookCostMicros,
                    removedBookCost,
                )
                if (nextQuantity == 0L) {
                    positions.remove(order.symbol)
                } else {
                    positions[order.symbol] = VN97PaperPosition(
                        symbol = order.symbol,
                        quantityMicrounits = nextQuantity,
                        bookCostMicros = nextBookCost,
                    )
                }
            }

            val sequence = Math.addExact(snapshot.sequence, 1L)
            val next = VN97PaperTradingSnapshot(
                startingCashMicros = snapshot.startingCashMicros,
                cashMicros = nextCash,
                positions = positions.toSortedMap(),
                executedOrderIds =
                    snapshot.executedOrderIds + order.orderId,
                sequence = sequence,
            )
            val fill = VN97PaperFill(
                sequence = sequence,
                orderId = order.orderId,
                symbol = order.symbol,
                side = order.side,
                quantityMicrounits = order.quantityMicrounits,
                fillPriceMicros = fillPrice,
                notionalMicros = notional,
                feeMicros = fee,
                quoteTimestampNs = quote.timestampNs,
                executedNs = executedNs,
                cashAfterMicros = nextCash,
                positionQuantityAfterMicrounits = nextQuantity,
                positionBookCostAfterMicros = nextBookCost,
            )
            return fill to next
        }

        private fun scaledMultiply(
            left: Long,
            right: Long,
            divisor: Long,
        ): Long {
            require(left >= 0L && right >= 0L && divisor > 0L)
            val value =
                BigInteger.valueOf(left)
                    .multiply(BigInteger.valueOf(right))
                    .divide(BigInteger.valueOf(divisor))
            return value.longValueExact()
        }

        private fun requireSafeJournalFile(file: File) {
            require(file.name == file.nameWithoutSeparators()) {
                "paper journal file name must be one component"
            }
            if (file.exists()) {
                require(file.isFile) {
                    "paper trading journal target is not a file"
                }
            }
        }

        private fun encodeGenesis(
            startingCashMicros: Long,
            policy: VN97PaperTradingRiskPolicy,
        ): String =
            listOf(
                MAGIC,
                "G",
                startingCashMicros.toString(),
                policy.maxOrderNotionalMicros.toString(),
                policy.maxGrossBookCostMicros.toString(),
                policy.minCashReserveMicros.toString(),
                policy.feeBasisPoints.toString(),
                policy.maxSymbols.toString(),
            ).joinToString("|")

        private fun decodeGenesis(
            line: String,
        ): Pair<Long, VN97PaperTradingRiskPolicy> {
            val fields = line.split('|')
            require(fields.size == 8) {
                "paper trading genesis field count is invalid"
            }
            require(fields[0] == MAGIC && fields[1] == "G") {
                "paper trading genesis magic is invalid"
            }
            val startingCash = fields[2].toLongStrict()
            val policy = VN97PaperTradingRiskPolicy(
                maxOrderNotionalMicros = fields[3].toLongStrict(),
                maxGrossBookCostMicros = fields[4].toLongStrict(),
                minCashReserveMicros = fields[5].toLongStrict(),
                feeBasisPoints = fields[6].toIntStrict(),
                maxSymbols = fields[7].toIntStrict(),
            )
            return startingCash to policy
        }

        private fun encodeOrderRecord(
            order: VN97PaperOrder,
            quote: VN97MarketQuote,
            executedNs: Long,
        ): String =
            listOf(
                MAGIC,
                "O",
                order.orderId,
                order.symbol,
                order.side.name,
                order.quantityMicrounits.toString(),
                quote.bidPriceMicros.toString(),
                quote.askPriceMicros.toString(),
                quote.timestampNs.toString(),
                order.createdNs.toString(),
                executedNs.toString(),
            ).joinToString("|")

        private data class JournalOrderRecord(
            val order: VN97PaperOrder,
            val quote: VN97MarketQuote,
            val executedNs: Long,
        )

        private fun decodeOrderRecord(line: String): JournalOrderRecord {
            val fields = line.split('|')
            require(fields.size == 11) {
                "paper trading order record field count is invalid"
            }
            require(fields[0] == MAGIC && fields[1] == "O") {
                "paper trading order record magic is invalid"
            }
            val orderId = fields[2]
            val symbol = fields[3]
            val side = runCatching {
                VN97TradeSide.valueOf(fields[4])
            }.getOrElse {
                throw IllegalArgumentException(
                    "paper trading side is invalid"
                )
            }
            val quantity = fields[5].toLongStrict()
            val bid = fields[6].toLongStrict()
            val ask = fields[7].toLongStrict()
            val quoteTimestamp = fields[8].toLongStrict()
            val createdNs = fields[9].toLongStrict()
            val executedNs = fields[10].toLongStrict()
            return JournalOrderRecord(
                order = VN97PaperOrder(
                    orderId = orderId,
                    symbol = symbol,
                    side = side,
                    quantityMicrounits = quantity,
                    quoteTimestampNs = quoteTimestamp,
                    createdNs = createdNs,
                ),
                quote = VN97MarketQuote(
                    symbol = symbol,
                    bidPriceMicros = bid,
                    askPriceMicros = ask,
                    timestampNs = quoteTimestamp,
                ),
                executedNs = executedNs,
            )
        }

        private fun appendDurable(file: File, line: String) {
            val bytes =
                (line + "\n").toByteArray(StandardCharsets.UTF_8)
            FileOutputStream(file, true).use { stream ->
                stream.write(bytes)
                stream.flush()
                stream.fd.sync()
            }
        }

        private const val MAGIC = "VN97TRD1"
    }
}

private val TRADING_SYMBOL_RE =
    Regex("^[A-Z][A-Z0-9._-]{0,23}$")

private val PAPER_ORDER_ID_RE =
    Regex("^[A-Za-z0-9][A-Za-z0-9._:-]{0,63}$")

private fun isTradingSymbol(value: String): Boolean =
    TRADING_SYMBOL_RE.matches(value)

private fun isPaperOrderId(value: String): Boolean =
    PAPER_ORDER_ID_RE.matches(value)

private fun File.nameWithoutSeparators(): String =
    name.replace("/", "").replace("\\", "")

private fun String.toLongStrict(): Long {
    require(matches(Regex("^-?[0-9]+$"))) {
        "paper trading integer field is invalid"
    }
    return toLong()
}

private fun String.toIntStrict(): Int {
    require(matches(Regex("^-?[0-9]+$"))) {
        "paper trading integer field is invalid"
    }
    return toInt()
}
