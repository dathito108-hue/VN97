package ai.vn97.app

data class VN97RevenueCampaignPolicy(
    val episodesPerSession: Int = 43,
    val targetSessions: Int = 3,
    val maxSessions: Int = 5,
    val intervalMillis: Long = 90_000L,
) {
    init {
        require(episodesPerSession in 32..256)
        require(targetSessions in 3..8)
        require(maxSessions in targetSessions..8)
        require(
            intervalMillis in 15_000L..21_600_000L
        )
        require(
            Math.multiplyExact(
                episodesPerSession.toLong(),
                targetSessions.toLong(),
            ) >= 128L
        ) {
            "revenue campaign must collect at least 128 target observations"
        }
    }
}

enum class VN97RevenueCampaignState {
    RUNNING,
    PAPER_QUALIFIED,
    REPAIR_RETURN,
    REPAIR_RISK,
    EVIDENCE_EXHAUSTED,
    FAILED,
    STOPPED,
}

enum class VN97RevenueCampaignDirective {
    START_NEXT_SESSION,
    COMPLETE_PAPER_QUALIFIED,
    COMPLETE_REPAIR_RETURN,
    COMPLETE_REPAIR_RISK,
    COMPLETE_EVIDENCE_EXHAUSTED,
}

object VN97RevenueCampaignPlanner {
    fun afterCompletedSession(
        qualification: VN97RevenueQualification,
        completedSessions: Int,
        policy: VN97RevenueCampaignPolicy,
    ): VN97RevenueCampaignDirective {
        require(completedSessions in 1..policy.maxSessions)
        return when (qualification.stage) {
            VN97RevenueQualificationStage.PAPER_QUALIFIED ->
                VN97RevenueCampaignDirective.COMPLETE_PAPER_QUALIFIED

            VN97RevenueQualificationStage.REPAIR_RETURN ->
                VN97RevenueCampaignDirective.COMPLETE_REPAIR_RETURN

            VN97RevenueQualificationStage.REPAIR_RISK ->
                VN97RevenueCampaignDirective.COMPLETE_REPAIR_RISK

            VN97RevenueQualificationStage.BUILD_EVIDENCE ->
                if (completedSessions < policy.maxSessions) {
                    VN97RevenueCampaignDirective.START_NEXT_SESSION
                } else {
                    VN97RevenueCampaignDirective
                        .COMPLETE_EVIDENCE_EXHAUSTED
                }
        }
    }

    fun stateFor(
        directive: VN97RevenueCampaignDirective,
    ): VN97RevenueCampaignState =
        when (directive) {
            VN97RevenueCampaignDirective.START_NEXT_SESSION ->
                VN97RevenueCampaignState.RUNNING
            VN97RevenueCampaignDirective.COMPLETE_PAPER_QUALIFIED ->
                VN97RevenueCampaignState.PAPER_QUALIFIED
            VN97RevenueCampaignDirective.COMPLETE_REPAIR_RETURN ->
                VN97RevenueCampaignState.REPAIR_RETURN
            VN97RevenueCampaignDirective.COMPLETE_REPAIR_RISK ->
                VN97RevenueCampaignState.REPAIR_RISK
            VN97RevenueCampaignDirective.COMPLETE_EVIDENCE_EXHAUSTED ->
                VN97RevenueCampaignState.EVIDENCE_EXHAUSTED
        }
}
