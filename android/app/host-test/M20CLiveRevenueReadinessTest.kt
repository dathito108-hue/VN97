package ai.vn97.app

private fun q(
    stage: VN97RevenueQualificationStage,
): VN97RevenueQualification =
    VN97RevenueQualification(
        stage = stage,
        paperQualified =
            stage == VN97RevenueQualificationStage.PAPER_QUALIFIED,
        sessionsConsidered = 3,
        evidenceConsidered = 129,
        profitableSessions = 2,
        ordersTotal = 9,
        medianReturnBasisPoints = 40L,
        worstReturnBasisPoints = -100L,
        worstDrawdownBasisPoints = 200L,
        blockers =
            if (stage == VN97RevenueQualificationStage.PAPER_QUALIFIED) {
                emptyList()
            } else {
                listOf("fixture")
            },
    )

private val campaignId = "a".repeat(64)
private val channel =
    VN97RevenueChannelEvidence(
        providerId = "fixture.broker",
        configured = true,
        credentialBackedByKeystore = true,
        authenticated = true,
        dryRunValidated = true,
        orderSubmissionAvailable = true,
    )
private val authority =
    VN97RevenueMoneyAuthority(
        approvalId = "b".repeat(32),
        m6Verified = true,
        issuedWallTimeMillis = 1_000L,
        expiresWallTimeMillis = 61_000L,
        maxOrderNotionalBasisPoints = 500,
        maxDailyLossBasisPoints = 100,
        maxConcurrentPositions = 3,
    )

fun main() {
    val lockedPaper =
        VN97RevenueLiveReadinessGate.evaluate(
            promotion =
                VN97RevenuePromotionEvidence(
                    campaignId = campaignId,
                    campaignState =
                        VN97RevenueCampaignState.REPAIR_RETURN,
                    qualification =
                        q(VN97RevenueQualificationStage.REPAIR_RETURN),
                ),
            channel = channel,
            authority = authority,
            nowWallTimeMillis = 2_000L,
        )
    check(
        lockedPaper.state ==
            VN97RevenueLiveReadinessState.LOCKED
    )
    check("paper_campaign_not_qualified" in lockedPaper.blockers)

    val lockedChannel =
        VN97RevenueLiveReadinessGate.evaluate(
            promotion =
                VN97RevenuePromotionEvidence(
                    campaignId = campaignId,
                    campaignState =
                        VN97RevenueCampaignState.PAPER_QUALIFIED,
                    qualification =
                        q(VN97RevenueQualificationStage.PAPER_QUALIFIED),
                ),
            channel =
                channel.copy(authenticated = false),
            authority = authority,
            nowWallTimeMillis = 2_000L,
        )
    check("revenue_channel_not_authenticated" in lockedChannel.blockers)

    val lockedAuthority =
        VN97RevenueLiveReadinessGate.evaluate(
            promotion =
                VN97RevenuePromotionEvidence(
                    campaignId = campaignId,
                    campaignState =
                        VN97RevenueCampaignState.PAPER_QUALIFIED,
                    qualification =
                        q(VN97RevenueQualificationStage.PAPER_QUALIFIED),
                ),
            channel = channel,
            authority = null,
            nowWallTimeMillis = 2_000L,
        )
    check("explicit_money_authority_missing" in lockedAuthority.blockers)
    check(!lockedAuthority.productionMoneyMovementAuthorized)

    val ready =
        VN97RevenueLiveReadinessGate.evaluate(
            promotion =
                VN97RevenuePromotionEvidence(
                    campaignId = campaignId,
                    campaignState =
                        VN97RevenueCampaignState.PAPER_QUALIFIED,
                    qualification =
                        q(VN97RevenueQualificationStage.PAPER_QUALIFIED),
                ),
            channel = channel,
            authority = authority,
            nowWallTimeMillis = 2_000L,
        )
    check(ready.state == VN97RevenueLiveReadinessState.READY)
    check(ready.readyForM6Execution)
    check(!ready.productionMoneyMovementAuthorized)

    val overRisk =
        VN97RevenueLiveReadinessGate.evaluate(
            promotion =
                VN97RevenuePromotionEvidence(
                    campaignId = campaignId,
                    campaignState =
                        VN97RevenueCampaignState.PAPER_QUALIFIED,
                    qualification =
                        q(VN97RevenueQualificationStage.PAPER_QUALIFIED),
                ),
            channel = channel,
            authority =
                authority.copy(
                    maxDailyLossBasisPoints = 101,
                ),
            nowWallTimeMillis = 2_000L,
        )
    check(
        "daily_loss_authority_exceeds_policy" in
            overRisk.blockers
    )

    println("M20C live revenue readiness gate: PASS")
}
