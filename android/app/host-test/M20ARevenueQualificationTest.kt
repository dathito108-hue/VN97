package ai.vn97.app

private fun summary(
    jobId: Int,
    evidence: Int,
    returnBps: Long,
    drawdownBps: Long,
    orders: Int,
): VN97PaperSessionPerformanceSummary =
    VN97PaperSessionPerformanceSummary(
        sessionId = jobId.toString().padStart(64, '0'),
        jobId = jobId,
        evidenceCount = evidence,
        firstEpisode = 1,
        latestEpisode = evidence,
        latestEquityMicros = 100_000_000L + returnBps * 10_000L,
        latestTotalPnlMicros = returnBps * 10_000L,
        latestReturnBasisPoints = returnBps,
        peakEquityMicros = 105_000_000L,
        maxDrawdownMicros = drawdownBps * 10_000L,
        maxDrawdownBasisPoints = drawdownBps,
        holdCount = maxOf(0, evidence - orders),
        orderCount = orders,
    )

private fun aggregate(
    vararg sessions: VN97PaperSessionPerformanceSummary,
): VN97PaperPerformanceAggregate =
    VN97PaperPerformanceAggregate(
        evidenceCount = sessions.sumOf { it.evidenceCount },
        sessionCount = sessions.size,
        sessions = sessions.toList(),
    )

fun main() {
    val insufficient =
        VN97RevenueQualificationGate.evaluate(
            aggregate(
                summary(1, 31, 100L, 100L, 4),
                summary(2, 31, 90L, 100L, 4),
                summary(3, 31, 80L, 100L, 4),
            )
        )
    check(
        insufficient.stage ==
            VN97RevenueQualificationStage.BUILD_EVIDENCE
    )
    check(!insufficient.paperQualified)
    check(!insufficient.productionMoneyMovementAuthorized)

    val risky =
        VN97RevenueQualificationGate.evaluate(
            aggregate(
                summary(1, 48, 180L, 200L, 4),
                summary(2, 48, 90L, 650L, 4),
                summary(3, 48, 70L, 220L, 4),
            )
        )
    check(risky.stage == VN97RevenueQualificationStage.REPAIR_RISK)
    check(risky.worstDrawdownBasisPoints == 650L)

    val weakReturn =
        VN97RevenueQualificationGate.evaluate(
            aggregate(
                summary(1, 48, 20L, 200L, 4),
                summary(2, 48, 10L, 180L, 4),
                summary(3, 48, -15L, 150L, 4),
            )
        )
    check(
        weakReturn.stage ==
            VN97RevenueQualificationStage.REPAIR_RETURN
    )
    check(weakReturn.medianReturnBasisPoints == 10L)

    val qualified =
        VN97RevenueQualificationGate.evaluate(
            aggregate(
                summary(1, 48, 160L, 210L, 4),
                summary(2, 48, 90L, 180L, 4),
                summary(3, 48, -80L, 230L, 4),
            )
        )
    check(
        qualified.stage ==
            VN97RevenueQualificationStage.PAPER_QUALIFIED
    )
    check(qualified.paperQualified)
    check(qualified.sessionsConsidered == 3)
    check(qualified.evidenceConsidered == 144)
    check(qualified.profitableSessions == 2)
    check(qualified.ordersTotal == 12)
    check(qualified.medianReturnBasisPoints == 90L)
    check(qualified.worstReturnBasisPoints == -80L)
    check(qualified.worstDrawdownBasisPoints == 230L)
    check(!qualified.productionMoneyMovementAuthorized)

    println("M20A revenue qualification gate: PASS")
}
