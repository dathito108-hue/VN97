package ai.vn97.platform

fun main() {
    check(
        canonicalPositiveDecimal(
            "0.01",
            "volume",
        ) == "0.01"
    )
    check(
        canonicalPositiveDecimal(
            "1",
            "volume",
        ) == "1"
    )
    check(
        runCatching {
            canonicalPositiveDecimal(
                "1.00",
                "volume",
            )
        }.getOrThrow() == "1"
    )
    check(
        runCatching {
            requireCanonicalPositiveDecimal(
                "1.00",
                "volume",
            )
        }.isFailure
    )
    check(
        runCatching {
            canonicalPositiveDecimal(
                "-1",
                "volume",
            )
        }.isFailure
    )
    check(
        runCatching {
            canonicalPositiveDecimal(
                "1e-2",
                "volume",
            )
        }.isFailure
    )

    val request =
        VN97RevenueOpenPositionRequest(
            accountId = "123456",
            campaignId = "a".repeat(64),
            instrument = "XAUUSD",
            side = VN97RevenueTradeSide.BUY,
            volume = "0.01",
            idempotencyKey =
                "vn97-open-" +
                    "b".repeat(48),
        )
    check(request.volume == "0.01")
    check(
        runCatching {
            request.copy(
                idempotencyKey = "bad key"
            )
        }.isFailure
    )

    println("M20G revenue trade contract: PASS")
}
