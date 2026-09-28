package ai.vn97.app

data class VN97RevenueQualificationPolicy(
    val minSessions: Int = 3,
    val minEvidencePerSession: Int = 32,
    val minTotalEvidence: Int = 128,
    val minProfitableSessions: Int = 2,
    val minOrdersTotal: Int = 8,
    val minMedianReturnBasisPoints: Long = 25L,
    val minWorstReturnBasisPoints: Long = -250L,
    val maxSessionDrawdownBasisPoints: Long = 500L,
) {
    init {
        require(minSessions in 1..32)
        require(minEvidencePerSession in 1..2_048)
        require(minTotalEvidence in 1..131_072)
        require(minProfitableSessions in 1..minSessions)
        require(minOrdersTotal in 1..100_000)
        require(minMedianReturnBasisPoints in -10_000L..100_000L)
        require(minWorstReturnBasisPoints in -10_000L..100_000L)
        require(maxSessionDrawdownBasisPoints in 0L..10_000L)
    }
}

enum class VN97RevenueQualificationStage {
    BUILD_EVIDENCE,
    REPAIR_RETURN,
    REPAIR_RISK,
    PAPER_QUALIFIED,
}

data class VN97RevenueQualification(
    val stage: VN97RevenueQualificationStage,
    val paperQualified: Boolean,
    val sessionsConsidered: Int,
    val evidenceConsidered: Int,
    val profitableSessions: Int,
    val ordersTotal: Int,
    val medianReturnBasisPoints: Long?,
    val worstReturnBasisPoints: Long?,
    val worstDrawdownBasisPoints: Long?,
    val blockers: List<String>,
) {
    init {
        require(sessionsConsidered >= 0)
        require(evidenceConsidered >= 0)
        require(profitableSessions in 0..sessionsConsidered)
        require(ordersTotal >= 0)
        require(
            paperQualified ==
                (stage == VN97RevenueQualificationStage.PAPER_QUALIFIED)
        )
        require(blockers.isEmpty() == paperQualified)
    }

    val productionMoneyMovementAuthorized: Boolean
        get() = false
}

object VN97RevenueQualificationGate {
    fun evaluate(
        aggregate: VN97PaperPerformanceAggregate,
        policy: VN97RevenueQualificationPolicy =
            VN97RevenueQualificationPolicy(),
    ): VN97RevenueQualification {
        val eligibleSessions =
            aggregate.sessions
                .filter { it.evidenceCount >= policy.minEvidencePerSession }
                .sortedBy { it.jobId }

        val evidence =
            eligibleSessions.sumOf { it.evidenceCount }
        val profitable =
            eligibleSessions.count { it.latestTotalPnlMicros > 0L }
        val orders =
            eligibleSessions.sumOf { it.orderCount }
        val returns =
            eligibleSessions
                .map { it.latestReturnBasisPoints }
                .sorted()
        val medianReturn =
            if (returns.isEmpty()) {
                null
            } else {
                returns[(returns.size - 1) / 2]
            }
        val worstReturn = returns.firstOrNull()
        val worstDrawdown =
            eligibleSessions
                .maxOfOrNull { it.maxDrawdownBasisPoints }

        val evidenceBlockers = mutableListOf<String>()
        if (eligibleSessions.size < policy.minSessions) {
            evidenceBlockers +=
                "qualified_sessions=" + eligibleSessions.size +
                    "<" + policy.minSessions
        }
        if (evidence < policy.minTotalEvidence) {
            evidenceBlockers +=
                "evidence=" + evidence +
                    "<" + policy.minTotalEvidence
        }
        if (orders < policy.minOrdersTotal) {
            evidenceBlockers +=
                "orders=" + orders +
                    "<" + policy.minOrdersTotal
        }
        if (evidenceBlockers.isNotEmpty()) {
            return decision(
                stage = VN97RevenueQualificationStage.BUILD_EVIDENCE,
                sessions = eligibleSessions.size,
                evidence = evidence,
                profitable = profitable,
                orders = orders,
                medianReturn = medianReturn,
                worstReturn = worstReturn,
                worstDrawdown = worstDrawdown,
                blockers = evidenceBlockers,
            )
        }

        val riskBlockers = mutableListOf<String>()
        if (
            worstDrawdown == null ||
            worstDrawdown > policy.maxSessionDrawdownBasisPoints
        ) {
            riskBlockers +=
                "max_drawdown_bps=" + (worstDrawdown ?: -1L) +
                    ">" + policy.maxSessionDrawdownBasisPoints
        }
        if (
            worstReturn == null ||
            worstReturn < policy.minWorstReturnBasisPoints
        ) {
            riskBlockers +=
                "worst_return_bps=" + (worstReturn ?: Long.MIN_VALUE) +
                    "<" + policy.minWorstReturnBasisPoints
        }
        if (riskBlockers.isNotEmpty()) {
            return decision(
                stage = VN97RevenueQualificationStage.REPAIR_RISK,
                sessions = eligibleSessions.size,
                evidence = evidence,
                profitable = profitable,
                orders = orders,
                medianReturn = medianReturn,
                worstReturn = worstReturn,
                worstDrawdown = worstDrawdown,
                blockers = riskBlockers,
            )
        }

        val returnBlockers = mutableListOf<String>()
        if (profitable < policy.minProfitableSessions) {
            returnBlockers +=
                "profitable_sessions=" + profitable +
                    "<" + policy.minProfitableSessions
        }
        if (
            medianReturn == null ||
            medianReturn < policy.minMedianReturnBasisPoints
        ) {
            returnBlockers +=
                "median_return_bps=" +
                    (medianReturn ?: Long.MIN_VALUE) +
                    "<" + policy.minMedianReturnBasisPoints
        }
        if (returnBlockers.isNotEmpty()) {
            return decision(
                stage = VN97RevenueQualificationStage.REPAIR_RETURN,
                sessions = eligibleSessions.size,
                evidence = evidence,
                profitable = profitable,
                orders = orders,
                medianReturn = medianReturn,
                worstReturn = worstReturn,
                worstDrawdown = worstDrawdown,
                blockers = returnBlockers,
            )
        }

        return decision(
            stage = VN97RevenueQualificationStage.PAPER_QUALIFIED,
            sessions = eligibleSessions.size,
            evidence = evidence,
            profitable = profitable,
            orders = orders,
            medianReturn = medianReturn,
            worstReturn = worstReturn,
            worstDrawdown = worstDrawdown,
            blockers = emptyList(),
        )
    }

    private fun decision(
        stage: VN97RevenueQualificationStage,
        sessions: Int,
        evidence: Int,
        profitable: Int,
        orders: Int,
        medianReturn: Long?,
        worstReturn: Long?,
        worstDrawdown: Long?,
        blockers: List<String>,
    ) = VN97RevenueQualification(
        stage = stage,
        paperQualified =
            stage == VN97RevenueQualificationStage.PAPER_QUALIFIED,
        sessionsConsidered = sessions,
        evidenceConsidered = evidence,
        profitableSessions = profitable,
        ordersTotal = orders,
        medianReturnBasisPoints = medianReturn,
        worstReturnBasisPoints = worstReturn,
        worstDrawdownBasisPoints = worstDrawdown,
        blockers = blockers.toList(),
    )
}
