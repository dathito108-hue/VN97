package ai.vn97.app

private fun qualification(
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
        medianReturnBasisPoints = 30L,
        worstReturnBasisPoints = -100L,
        worstDrawdownBasisPoints = 250L,
        blockers =
            if (stage == VN97RevenueQualificationStage.PAPER_QUALIFIED) {
                emptyList()
            } else {
                listOf("fixture")
            },
    )

fun main() {
    val policy = VN97RevenueCampaignPolicy()
    check(policy.episodesPerSession * policy.targetSessions >= 128)

    val more =
        VN97RevenueCampaignPlanner.afterCompletedSession(
            qualification(
                VN97RevenueQualificationStage.BUILD_EVIDENCE
            ),
            completedSessions = 1,
            policy = policy,
        )
    check(
        more ==
            VN97RevenueCampaignDirective.START_NEXT_SESSION
    )

    val exhausted =
        VN97RevenueCampaignPlanner.afterCompletedSession(
            qualification(
                VN97RevenueQualificationStage.BUILD_EVIDENCE
            ),
            completedSessions = policy.maxSessions,
            policy = policy,
        )
    check(
        exhausted ==
            VN97RevenueCampaignDirective
                .COMPLETE_EVIDENCE_EXHAUSTED
    )

    val qualified =
        VN97RevenueCampaignPlanner.afterCompletedSession(
            qualification(
                VN97RevenueQualificationStage.PAPER_QUALIFIED
            ),
            completedSessions = 3,
            policy = policy,
        )
    check(
        VN97RevenueCampaignPlanner.stateFor(qualified) ==
            VN97RevenueCampaignState.PAPER_QUALIFIED
    )

    val repairReturn =
        VN97RevenueCampaignPlanner.afterCompletedSession(
            qualification(
                VN97RevenueQualificationStage.REPAIR_RETURN
            ),
            completedSessions = 3,
            policy = policy,
        )
    check(
        VN97RevenueCampaignPlanner.stateFor(repairReturn) ==
            VN97RevenueCampaignState.REPAIR_RETURN
    )

    val repairRisk =
        VN97RevenueCampaignPlanner.afterCompletedSession(
            qualification(
                VN97RevenueQualificationStage.REPAIR_RISK
            ),
            completedSessions = 3,
            policy = policy,
        )
    check(
        VN97RevenueCampaignPlanner.stateFor(repairRisk) ==
            VN97RevenueCampaignState.REPAIR_RISK
    )

    check(
        runCatching {
            VN97RevenueCampaignPolicy(
                episodesPerSession = 32,
                targetSessions = 3,
            )
        }.isFailure
    )

    println("M20B revenue campaign planner: PASS")
}
