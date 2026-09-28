package ai.vn97.app

private val SHA256_HEX = Regex("^[0-9a-f]{64}$")
private val SAFE_ID = Regex("^[A-Za-z0-9._:-]{1,160}$")

enum class VN97PaymentVerification {
    MANUAL_UNVERIFIED,
    PROVIDER_API_VERIFIED,
}

data class VN97ServiceOrder(
    val orderId: String,
    val title: String,
    val service: VN97DigitalService,
    val quotedPriceVnd: Long,
    val estimatedCostVnd: Long,
    val outputSha256: String,
    val createdAtEpochMs: Long,
    val exportedAtEpochMs: Long? = null,
) {
    init {
        require(SAFE_ID.matches(orderId)) { "order ID is invalid" }
        require(title.isNotBlank() && title.length <= 120) { "order title is invalid" }
        VN97ServiceQuote(quotedPriceVnd, estimatedCostVnd)
        require(SHA256_HEX.matches(outputSha256)) { "output digest is invalid" }
        require(createdAtEpochMs > 0) { "created time is invalid" }
        require(exportedAtEpochMs == null || exportedAtEpochMs >= createdAtEpochMs) {
            "export time predates the order"
        }
    }
}

/**
 * A provider adapter may emit PROVIDER_API_VERIFIED only after an authenticated, read-only
 * provider response says the money is settled. UI-entered claims must remain MANUAL_UNVERIFIED.
 * This type classifies evidence; it is not an authority boundary or a substitute for M6.
 */
data class VN97PaymentRecord(
    val orderId: String,
    val provider: String,
    val providerAccountFingerprint: String,
    val externalEventId: String,
    val grossVnd: Long,
    val providerFeeVnd: Long,
    val refundVnd: Long,
    val realizedCostVnd: Long,
    val verification: VN97PaymentVerification,
    val evidenceSha256: String?,
    val settledAtEpochMs: Long,
) {
    init {
        require(SAFE_ID.matches(orderId)) { "payment order ID is invalid" }
        require(SAFE_ID.matches(provider)) { "payment provider is invalid" }
        require(SAFE_ID.matches(externalEventId)) { "external event ID is invalid" }
        require(providerAccountFingerprint.isEmpty() || SHA256_HEX.matches(providerAccountFingerprint)) {
            "provider account fingerprint is invalid"
        }
        require(grossVnd in 0..MAX_MONEY_VND)
        require(providerFeeVnd in 0..MAX_MONEY_VND)
        require(refundVnd in 0..MAX_MONEY_VND)
        require(realizedCostVnd in 0..MAX_MONEY_VND)
        require(settledAtEpochMs > 0)
        if (verification == VN97PaymentVerification.PROVIDER_API_VERIFIED) {
            require(providerAccountFingerprint.isNotEmpty()) {
                "verified provider evidence needs an account fingerprint"
            }
            require(evidenceSha256 != null && SHA256_HEX.matches(evidenceSha256)) {
                "verified provider evidence needs a SHA-256 identity"
            }
        } else {
            require(evidenceSha256 == null) {
                "manual claims cannot carry a provider-verification digest"
            }
        }
    }

    val netVnd: Long
        get() = Math.subtractExact(
            Math.subtractExact(
                Math.subtractExact(grossVnd, providerFeeVnd),
                refundVnd,
            ),
            realizedCostVnd,
        )

    val idempotencyKey: String
        get() = "$provider:$providerAccountFingerprint:$externalEventId"

    companion object {
        const val MAX_MONEY_VND = 1_000_000_000_000L
    }
}

data class VN97RevenueSummary(
    val orderCount: Int,
    val exportedOrderCount: Int,
    val verifiedSettlementCount: Int,
    val verifiedGrossVnd: Long,
    val verifiedFeesVnd: Long,
    val verifiedRefundsVnd: Long,
    val verifiedCostsVnd: Long,
    val verifiedNetVnd: Long,
    val unverifiedClaimCount: Int,
    val unverifiedClaimNetVnd: Long,
)

class VN97ServiceRevenueLedger(
    orders: List<VN97ServiceOrder> = emptyList(),
    payments: List<VN97PaymentRecord> = emptyList(),
) {
    private val ordersById = LinkedHashMap<String, VN97ServiceOrder>()
    private val paymentsByKey = LinkedHashMap<String, VN97PaymentRecord>()

    init {
        require(orders.size <= MAX_ORDERS) { "too many service orders" }
        require(payments.size <= MAX_PAYMENTS) { "too many payment records" }
        orders.forEach { require(ordersById.put(it.orderId, it) == null) { "duplicate order ID" } }
        payments.forEach { recordPayment(it) }
    }

    fun recordOrder(order: VN97ServiceOrder): Boolean {
        val current = ordersById[order.orderId]
        if (current != null) {
            require(current == order) { "order ID collision" }
            return false
        }
        require(ordersById.size < MAX_ORDERS) { "service order limit reached" }
        ordersById[order.orderId] = order
        return true
    }

    fun markExported(orderId: String, exportedAtEpochMs: Long): Boolean {
        val order = requireNotNull(ordersById[orderId]) { "unknown service order" }
        require(exportedAtEpochMs >= order.createdAtEpochMs) { "export time predates the order" }
        if (order.exportedAtEpochMs != null) return false
        ordersById[orderId] = order.copy(exportedAtEpochMs = exportedAtEpochMs)
        return true
    }

    /** Exact retries are no-ops; a reused provider event ID with different money/evidence fails. */
    fun recordPayment(payment: VN97PaymentRecord): Boolean {
        require(ordersById.containsKey(payment.orderId)) { "payment references an unknown order" }
        val current = paymentsByKey[payment.idempotencyKey]
        if (current != null) {
            require(current == payment) { "conflicting duplicate provider event" }
            return false
        }
        require(paymentsByKey.size < MAX_PAYMENTS) { "payment record limit reached" }
        paymentsByKey[payment.idempotencyKey] = payment
        return true
    }

    fun orders(): List<VN97ServiceOrder> = ordersById.values.toList()

    fun payments(): List<VN97PaymentRecord> = paymentsByKey.values.toList()

    fun summary(): VN97RevenueSummary {
        val verified = paymentsByKey.values.filter {
            it.verification == VN97PaymentVerification.PROVIDER_API_VERIFIED
        }
        val manual = paymentsByKey.values.filter {
            it.verification == VN97PaymentVerification.MANUAL_UNVERIFIED
        }
        fun sum(values: Iterable<Long>): Long =
            values.fold(0L) { total, value -> Math.addExact(total, value) }
        val gross = sum(verified.map { it.grossVnd })
        val fees = sum(verified.map { it.providerFeeVnd })
        val refunds = sum(verified.map { it.refundVnd })
        val costs = sum(verified.map { it.realizedCostVnd })
        val net = Math.subtractExact(Math.subtractExact(Math.subtractExact(gross, fees), refunds), costs)
        return VN97RevenueSummary(
            orderCount = ordersById.size,
            exportedOrderCount = ordersById.values.count { it.exportedAtEpochMs != null },
            verifiedSettlementCount = verified.size,
            verifiedGrossVnd = gross,
            verifiedFeesVnd = fees,
            verifiedRefundsVnd = refunds,
            verifiedCostsVnd = costs,
            verifiedNetVnd = net,
            unverifiedClaimCount = manual.size,
            unverifiedClaimNetVnd = sum(manual.map { it.netVnd }),
        )
    }

    companion object {
        const val MAX_ORDERS = 500
        const val MAX_PAYMENTS = 2_000
    }
}
