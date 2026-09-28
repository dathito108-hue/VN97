package ai.vn97.app

data class VN97PaperSessionPerformanceSummary(
    val sessionId: String,
    val jobId: Int,
    val evidenceCount: Int,
    val firstEpisode: Int,
    val latestEpisode: Int,
    val latestEquityMicros: Long,
    val latestTotalPnlMicros: Long,
    val latestReturnBasisPoints: Long,
    val peakEquityMicros: Long,
    val maxDrawdownMicros: Long,
    val maxDrawdownBasisPoints: Long,
    val holdCount: Int,
    val orderCount: Int,
)

data class VN97PaperPerformanceAggregate(
    val evidenceCount: Int,
    val sessionCount: Int,
    val sessions: List<VN97PaperSessionPerformanceSummary>,
)
