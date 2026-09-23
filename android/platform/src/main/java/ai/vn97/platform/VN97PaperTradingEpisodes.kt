package ai.vn97.platform

data class VN97PaperTradingEpisodeLimits(
    val maxEpisodeAttempts: Int = 256,
    val minEpisodeSpacingNs: Long = 1_000_000_000L,
) {
    init {
        require(maxEpisodeAttempts in 1..10_000) {
            "paper trading episode-attempt bound is invalid"
        }
        require(minEpisodeSpacingNs in 0L..300_000_000_000L) {
            "paper trading episode spacing is outside bounds"
        }
    }
}

data class VN97PaperTradingEpisode<R>(
    val attempt: Int,
    val snapshotId: String,
    val observedNs: Long,
    val evaluatedNs: Long,
    val result: R,
)

class VN97PaperTradingEpisodeRunner<R>(
    private val source: VN97MarketDataSource,
    private val evaluator: (VN97MarketSnapshot, Long) -> R,
    val limits: VN97PaperTradingEpisodeLimits =
        VN97PaperTradingEpisodeLimits(),
) {
    private var attempts = 0
    private var lastSnapshotId: String? = null
    private var lastObservedNs = -1L
    private var lastAttemptNs = -1L

    @Synchronized
    fun runNext(nowNs: Long): VN97PaperTradingEpisode<R> {
        require(nowNs >= 0L) {
            "paper trading episode time must be non-negative"
        }
        check(attempts < limits.maxEpisodeAttempts) {
            "paper trading episode-attempt budget exhausted"
        }
        if (lastAttemptNs >= 0L) {
            require(nowNs >= lastAttemptNs) {
                "paper trading episode time moved backwards"
            }
            require(
                nowNs - lastAttemptNs >= limits.minEpisodeSpacingNs
            ) {
                "paper trading episode started before minimum spacing elapsed"
            }
        }

        val snapshot = source.fetch(nowNs)
        require(snapshot.observedNs <= nowNs) {
            "paper trading snapshot cannot be newer than episode time"
        }
        require(snapshot.snapshotId != lastSnapshotId) {
            "paper trading requires a fresh market snapshot"
        }
        require(snapshot.observedNs > lastObservedNs) {
            "paper trading market observation did not advance"
        }

        val attempt = Math.addExact(attempts, 1)
        attempts = attempt
        lastSnapshotId = snapshot.snapshotId
        lastObservedNs = snapshot.observedNs
        lastAttemptNs = nowNs

        val result = evaluator(snapshot, nowNs)
        return VN97PaperTradingEpisode(
            attempt = attempt,
            snapshotId = snapshot.snapshotId,
            observedNs = snapshot.observedNs,
            evaluatedNs = nowNs,
            result = result,
        )
    }

    @Synchronized
    fun attemptsUsed(): Int = attempts

    @Synchronized
    fun attemptsRemaining(): Int =
        limits.maxEpisodeAttempts - attempts
}
