package ai.vn97.app

data class VN97RevenueLivePolicy(
    val maxOrderNotionalBasisPoints: Int = 500,
    val maxDailyLossBasisPoints: Int = 100,
    val maxConcurrentPositions: Int = 3,
    val maxAuthorityTtlMillis: Long = 24L * 60L * 60L * 1000L,
) {
    init {
        require(maxOrderNotionalBasisPoints in 1..2_500)
        require(maxDailyLossBasisPoints in 1..1_000)
        require(maxConcurrentPositions in 1..16)
        require(maxAuthorityTtlMillis in 60_000L..86_400_000L)
    }
}

data class VN97RevenuePromotionEvidence(
    val campaignId: String,
    val campaignState: VN97RevenueCampaignState,
    val qualification: VN97RevenueQualification?,
) {
    init {
        require(
            campaignId.length == 64 &&
                campaignId.all { it in "0123456789abcdef" }
        )
    }
}

data class VN97RevenueChannelEvidence(
    val providerId: String,
    val configured: Boolean,
    val credentialBackedByKeystore: Boolean,
    val authenticated: Boolean,
    val dryRunValidated: Boolean,
    val orderSubmissionAvailable: Boolean,
) {
    init {
        require(
            providerId.matches(
                Regex("^[A-Za-z0-9][A-Za-z0-9._:-]{0,63}$")
            )
        )
    }
}

data class VN97RevenueMoneyAuthority(
    val approvalId: String,
    val m6Verified: Boolean,
    val issuedWallTimeMillis: Long,
    val expiresWallTimeMillis: Long,
    val maxOrderNotionalBasisPoints: Int,
    val maxDailyLossBasisPoints: Int,
    val maxConcurrentPositions: Int,
) {
    init {
        require(
            approvalId.length == 32 &&
                approvalId.all { it in "0123456789abcdef" }
        )
        require(issuedWallTimeMillis >= 0L)
        require(expiresWallTimeMillis > issuedWallTimeMillis)
        require(maxOrderNotionalBasisPoints > 0)
        require(maxDailyLossBasisPoints > 0)
        require(maxConcurrentPositions > 0)
    }
}

enum class VN97RevenueLiveReadinessState {
    LOCKED,
    READY,
}

data class VN97RevenueLiveReadiness(
    val state: VN97RevenueLiveReadinessState,
    val blockers: List<String>,
) {
    init {
        require(
            (state == VN97RevenueLiveReadinessState.READY) ==
                blockers.isEmpty()
        )
    }

    val productionMoneyMovementAuthorized: Boolean
        get() = state == VN97RevenueLiveReadinessState.READY
}

object VN97RevenueLiveReadinessGate {
    fun evaluate(
        promotion: VN97RevenuePromotionEvidence,
        channel: VN97RevenueChannelEvidence,
        authority: VN97RevenueMoneyAuthority?,
        policy: VN97RevenueLivePolicy =
            VN97RevenueLivePolicy(),
        nowWallTimeMillis: Long,
    ): VN97RevenueLiveReadiness {
        require(nowWallTimeMillis >= 0L)
        val blockers = mutableListOf<String>()

        if (
            promotion.campaignState !=
                VN97RevenueCampaignState.PAPER_QUALIFIED ||
            promotion.qualification?.paperQualified != true
        ) {
            blockers += "paper_campaign_not_qualified"
        }

        if (!channel.configured) {
            blockers += "revenue_channel_not_configured"
        }
        if (!channel.credentialBackedByKeystore) {
            blockers += "revenue_credentials_not_keystore_backed"
        }
        if (!channel.authenticated) {
            blockers += "revenue_channel_not_authenticated"
        }
        if (!channel.dryRunValidated) {
            blockers += "revenue_channel_dry_run_not_validated"
        }
        if (!channel.orderSubmissionAvailable) {
            blockers += "revenue_order_submission_unavailable"
        }

        if (authority == null) {
            blockers += "explicit_money_authority_missing"
        } else {
            if (!authority.m6Verified) {
                blockers += "m6_money_authority_not_verified"
            }
            if (
                nowWallTimeMillis < authority.issuedWallTimeMillis ||
                nowWallTimeMillis >= authority.expiresWallTimeMillis
            ) {
                blockers += "money_authority_outside_validity_window"
            }
            val ttl =
                authority.expiresWallTimeMillis -
                    authority.issuedWallTimeMillis
            if (ttl > policy.maxAuthorityTtlMillis) {
                blockers += "money_authority_ttl_exceeds_policy"
            }
            if (
                authority.maxOrderNotionalBasisPoints >
                    policy.maxOrderNotionalBasisPoints
            ) {
                blockers += "order_notional_authority_exceeds_policy"
            }
            if (
                authority.maxDailyLossBasisPoints >
                    policy.maxDailyLossBasisPoints
            ) {
                blockers += "daily_loss_authority_exceeds_policy"
            }
            if (
                authority.maxConcurrentPositions >
                    policy.maxConcurrentPositions
            ) {
                blockers += "position_authority_exceeds_policy"
            }
        }

        return if (blockers.isEmpty()) {
            VN97RevenueLiveReadiness(
                state = VN97RevenueLiveReadinessState.READY,
                blockers = emptyList(),
            )
        } else {
            VN97RevenueLiveReadiness(
                state = VN97RevenueLiveReadinessState.LOCKED,
                blockers = blockers.toList(),
            )
        }
    }
}
