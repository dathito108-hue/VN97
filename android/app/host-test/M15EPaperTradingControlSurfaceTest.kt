package ai.vn97.app

private fun expectM15EFailure(
    label: String,
    block: () -> Unit,
) {
    check(runCatching(block).isFailure) {
        "expected M15E failure: $label"
    }
}

fun main() {
    val parsed = VN97PaperTradingControlSurface.parse(
        endpointText = " https://market.example/feed ",
        sourceIdText = "fixture.market",
        symbolsText = " xyz,ABC;abc ",
        userGoalText = " Paper simulation only. ",
        intervalSecondsText = "60",
        maxEpisodesText = "12",
        requiresBatteryNotLow = true,
        requiresCharging = false,
    )
    check(parsed.endpoint == "https://market.example/feed")
    check(parsed.symbols.toList() == listOf("ABC", "XYZ"))
    check(parsed.intervalMillis == 60_000L)
    check(parsed.maxEpisodes == 12)
    check(parsed.userGoal == "Paper simulation only.")
    check(VN97PaperTradingControlSurface.parseJobId("123") == 123)

    expectM15EFailure("cleartext endpoint") {
        VN97PaperTradingControlSurface.parse(
            "http://market.example/feed",
            "fixture.market",
            "ABC",
            "goal",
            "60",
            "4",
            true,
            false,
        )
    }
    expectM15EFailure("URL credentials") {
        VN97PaperTradingControlSurface.parse(
            "https://user:pass@market.example/feed",
            "fixture.market",
            "ABC",
            "goal",
            "60",
            "4",
            true,
            false,
        )
    }
    expectM15EFailure("empty symbols") {
        VN97PaperTradingControlSurface.parse(
            "https://market.example/feed",
            "fixture.market",
            " ",
            "goal",
            "60",
            "4",
            true,
            false,
        )
    }
    expectM15EFailure("short interval") {
        VN97PaperTradingControlSurface.parse(
            "https://market.example/feed",
            "fixture.market",
            "ABC",
            "goal",
            "14",
            "4",
            true,
            false,
        )
    }
    expectM15EFailure("episode overflow") {
        VN97PaperTradingControlSurface.parse(
            "https://market.example/feed",
            "fixture.market",
            "ABC",
            "goal",
            "60",
            "2049",
            true,
            false,
        )
    }
    expectM15EFailure("invalid job ID") {
        VN97PaperTradingControlSurface.parseJobId("0")
    }

    val rendered = VN97PaperTradingControlSurface.formatReports(
        listOf(
            VN97PaperTradingUiReport(
                jobId = 101,
                state = "SCHEDULED",
                episodesAttempted = 2,
                maxEpisodes = 8,
                wakeCount = 3,
                lastDecision = "HOLD",
                lastOutcome = "fresh snapshot accepted",
                terminalReason = "",
                nextRunWallTimeMillis = 123456L,
            ),
            VN97PaperTradingUiReport(
                jobId = 102,
                state = "STOPPED",
                episodesAttempted = 4,
                maxEpisodes = 8,
                wakeCount = 4,
                lastDecision = "ORDER|BUY|ABC|1000000|5",
                lastOutcome = "PAPER_FILL",
                terminalReason = "paper session stopped by user",
                nextRunWallTimeMillis = 0L,
            ),
        )
    )
    check("simulation only" in rendered)
    check("#102 STOPPED" in rendered)
    check("PAPER_FILL" in rendered)
    check("paper session stopped by user" in rendered)

    println("M15E paper control-surface contracts: PASS")
}
