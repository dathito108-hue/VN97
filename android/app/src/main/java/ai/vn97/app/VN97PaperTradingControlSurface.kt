package ai.vn97.app

import java.net.URI
import java.nio.charset.StandardCharsets

data class VN97PaperTradingControlSpec(
    val endpoint: String,
    val sourceId: String,
    val symbols: Set<String>,
    val userGoal: String,
    val intervalMillis: Long,
    val maxEpisodes: Int,
    val requiresBatteryNotLow: Boolean,
    val requiresCharging: Boolean,
)

object VN97PaperTradingControlSurface {
    private val sourceRe =
        Regex("^[A-Za-z0-9][A-Za-z0-9._:-]{0,63}$")
    private val symbolRe =
        Regex("^[A-Z][A-Z0-9._-]{0,23}$")

    fun parse(
        endpointText: String,
        sourceIdText: String,
        symbolsText: String,
        userGoalText: String,
        intervalSecondsText: String,
        maxEpisodesText: String,
        requiresBatteryNotLow: Boolean,
        requiresCharging: Boolean,
    ): VN97PaperTradingControlSpec {
        val endpoint = endpointText.trim()
        val sourceId = sourceIdText.trim()
        val userGoal = userGoalText.trim()
        val intervalSeconds =
            intervalSecondsText.trim().toLongOrNull()
                ?: throw IllegalArgumentException(
                    "paper interval must be whole seconds"
                )
        val maxEpisodes =
            maxEpisodesText.trim().toIntOrNull()
                ?: throw IllegalArgumentException(
                    "paper episode budget must be a whole number"
                )

        requireHttpsEndpoint(endpoint)
        require(sourceRe.matches(sourceId)) {
            "paper source ID is invalid"
        }
        require(userGoal.isNotBlank()) {
            "paper goal must not be blank"
        }
        require(
            userGoal.toByteArray(StandardCharsets.UTF_8).size <=
                MAX_GOAL_BYTES
        ) {
            "paper goal exceeds byte bound"
        }
        require(intervalSeconds in MIN_INTERVAL_SECONDS..MAX_INTERVAL_SECONDS) {
            "paper interval must be between 15 and 21600 seconds"
        }
        require(maxEpisodes in 1..MAX_EPISODES) {
            "paper episode budget must be between 1 and 2048"
        }

        val symbols = symbolsText
            .split(',', ';', ' ', 10.toChar(), 13.toChar(), 9.toChar())
            .map(String::trim)
            .filter(String::isNotEmpty)
            .map(String::uppercase)
            .distinct()
            .sorted()
        require(symbols.isNotEmpty()) {
            "paper symbol list must not be empty"
        }
        require(symbols.size <= MAX_SYMBOLS) {
            "paper symbol list exceeds 64 symbols"
        }
        require(symbols.all(symbolRe::matches)) {
            "paper symbol list contains an invalid symbol"
        }

        return VN97PaperTradingControlSpec(
            endpoint = endpoint,
            sourceId = sourceId,
            symbols = symbols.toSet(),
            userGoal = userGoal,
            intervalMillis = Math.multiplyExact(intervalSeconds, 1_000L),
            maxEpisodes = maxEpisodes,
            requiresBatteryNotLow = requiresBatteryNotLow,
            requiresCharging = requiresCharging,
        )
    }

    fun parseJobId(text: String): Int {
        val value =
            text.trim().toIntOrNull()
                ?: throw IllegalArgumentException(
                    "paper session job ID must be a positive integer"
                )
        require(value > 0) {
            "paper session job ID must be positive"
        }
        return value
    }

    fun formatReports(
        reports: List<VN97PaperTradingUiReport>,
    ): String {
        if (reports.isEmpty()) {
            return "Paper trading: no sessions."
        }
        return buildString {
            append("Paper trading sessions (simulation only):")
            reports
                .sortedByDescending { it.jobId }
                .take(MAX_RENDERED_REPORTS)
                .forEach { report ->
                    append(10.toChar())
                    append('#')
                    append(report.jobId)
                    append(' ')
                    append(report.state)
                    append(" episodes=")
                    append(report.episodesAttempted)
                    append('/')
                    append(report.maxEpisodes)
                    append(" wakes=")
                    append(report.wakeCount)
                    if (report.lastDecision.isNotBlank()) {
                        append(10.toChar())
                        append("  decision=")
                        append(singleLine(report.lastDecision, 320))
                    }
                    if (report.lastOutcome.isNotBlank()) {
                        append(10.toChar())
                        append("  outcome=")
                        append(singleLine(report.lastOutcome, 480))
                    }
                    if (report.terminalReason.isNotBlank()) {
                        append(10.toChar())
                        append("  terminal=")
                        append(singleLine(report.terminalReason, 320))
                    } else if (report.nextRunWallTimeMillis > 0L) {
                        append(10.toChar())
                        append("  next_run_ms=")
                        append(report.nextRunWallTimeMillis)
                    }
                }
        }
    }

    private fun requireHttpsEndpoint(value: String) {
        require(
            value.isNotEmpty() &&
                value.toByteArray(StandardCharsets.UTF_8).size <= 2_048
        ) {
            "paper HTTPS endpoint is missing or too long"
        }
        val uri = URI(value)
        require(uri.scheme?.equals("https", ignoreCase = true) == true) {
            "paper endpoint must use HTTPS"
        }
        require(!uri.host.isNullOrBlank()) {
            "paper endpoint host is missing"
        }
        require(uri.rawUserInfo == null && uri.rawFragment == null) {
            "paper endpoint contains forbidden URL components"
        }
    }

    private fun singleLine(
        value: String,
        maxChars: Int,
    ): String =
        value
            .replace(10.toChar(), ' ')
            .replace(13.toChar(), ' ')
            .take(maxChars)

    const val MIN_INTERVAL_SECONDS = 15L
    const val MAX_INTERVAL_SECONDS = 21_600L
    const val MAX_EPISODES = 2_048
    const val MAX_SYMBOLS = 64
    const val MAX_GOAL_BYTES = 8 * 1024
    const val MAX_RENDERED_REPORTS = 8
}

data class VN97PaperTradingUiReport(
    val jobId: Int,
    val state: String,
    val episodesAttempted: Int,
    val maxEpisodes: Int,
    val wakeCount: Int,
    val lastDecision: String,
    val lastOutcome: String,
    val terminalReason: String,
    val nextRunWallTimeMillis: Long,
)
