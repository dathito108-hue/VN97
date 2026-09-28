package ai.vn97.app

fun main() {
    val hashA = "a".repeat(64)
    val hashB = "b".repeat(64)
    val account = "c".repeat(64)
    val order = VN97ServiceOrder(
        orderId = "svc-1",
        title = "Dọn bảng khách hàng",
        service = VN97DigitalService.TABLE_CLEANUP,
        quotedPriceVnd = 500_000,
        estimatedCostVnd = 40_000,
        outputSha256 = hashA,
        createdAtEpochMs = 1_000,
    )
    val ledger = VN97ServiceRevenueLedger()
    check(ledger.recordOrder(order))
    check(!ledger.recordOrder(order))
    check(ledger.markExported("svc-1", 2_000))
    check(!ledger.markExported("svc-1", 3_000))

    val manual = VN97PaymentRecord(
        orderId = "svc-1",
        provider = "bank-manual",
        providerAccountFingerprint = "",
        externalEventId = "claim-1",
        grossVnd = 500_000,
        providerFeeVnd = 5_000,
        refundVnd = 0,
        realizedCostVnd = 40_000,
        verification = VN97PaymentVerification.MANUAL_UNVERIFIED,
        evidenceSha256 = null,
        settledAtEpochMs = 3_000,
    )
    check(ledger.recordPayment(manual))
    check(!ledger.recordPayment(manual))
    check(ledger.summary().verifiedNetVnd == 0L)
    check(ledger.summary().unverifiedClaimNetVnd == 455_000L)

    val verified = manual.copy(
        provider = "provider-api",
        providerAccountFingerprint = account,
        externalEventId = "evt-42",
        providerFeeVnd = 15_000,
        refundVnd = 10_000,
        verification = VN97PaymentVerification.PROVIDER_API_VERIFIED,
        evidenceSha256 = hashB,
    )
    check(ledger.recordPayment(verified))
    check(!ledger.recordPayment(verified))
    val summary = ledger.summary()
    check(summary.orderCount == 1 && summary.exportedOrderCount == 1)
    check(summary.verifiedSettlementCount == 1)
    check(summary.verifiedGrossVnd == 500_000L)
    check(summary.verifiedFeesVnd == 15_000L)
    check(summary.verifiedRefundsVnd == 10_000L)
    check(summary.verifiedCostsVnd == 40_000L)
    check(summary.verifiedNetVnd == 435_000L)

    check(runCatching { ledger.recordPayment(verified.copy(grossVnd = 600_000)) }.isFailure)
    check(runCatching { ledger.recordPayment(verified.copy(orderId = "missing")) }.isFailure)
    check(runCatching {
        VN97PaymentRecord(
            orderId = "svc-1", provider = "provider-api", providerAccountFingerprint = "",
            externalEventId = "evt-43", grossVnd = 1, providerFeeVnd = 0, refundVnd = 0,
            realizedCostVnd = 0, verification = VN97PaymentVerification.PROVIDER_API_VERIFIED,
            evidenceSha256 = hashB, settledAtEpochMs = 4_000,
        )
    }.isFailure)
    check(runCatching { VN97ServiceRevenueLedger(emptyList(), listOf(manual)) }.isFailure)
    println("Service revenue ledger: durable orders/idempotency/verified-net separation PASS")
}
