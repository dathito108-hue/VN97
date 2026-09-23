package ai.vn97.app

import android.app.Activity
import android.os.Bundle
import android.view.Gravity
import android.view.ViewGroup
import android.widget.Button
import android.widget.CheckBox
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import java.util.concurrent.Executors

class VN97PaperTradingActivity : Activity() {
    private lateinit var endpointView: EditText
    private lateinit var sourceView: EditText
    private lateinit var symbolsView: EditText
    private lateinit var goalView: EditText
    private lateinit var intervalSecondsView: EditText
    private lateinit var maxEpisodesView: EditText
    private lateinit var batteryNotLowCheck: CheckBox
    private lateinit var chargingCheck: CheckBox
    private lateinit var jobIdView: EditText
    private lateinit var statusView: TextView
    private lateinit var startButton: Button
    private lateinit var pauseButton: Button
    private lateinit var resumeButton: Button
    private lateinit var stopButton: Button
    private lateinit var refreshButton: Button

    private val worker = Executors.newSingleThreadExecutor()

    private val app: VN97Application
        get() = application as VN97Application

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val density = resources.displayMetrics.density
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(
                (16 * density).toInt(),
                (16 * density).toInt(),
                (16 * density).toInt(),
                (24 * density).toInt(),
            )
        }
        val scroll = ScrollView(this).apply {
            isFillViewport = true
            addView(
                root,
                ViewGroup.LayoutParams(
                    ViewGroup.LayoutParams.MATCH_PARENT,
                    ViewGroup.LayoutParams.WRAP_CONTENT,
                ),
            )
        }

        root.addView(
            TextView(this).apply {
                text = "VN97 Paper Trading"
                textSize = 24f
                gravity = Gravity.CENTER_HORIZONTAL
            },
            fullWidth(),
        )
        root.addView(
            TextView(this).apply {
                text =
                    "Simulation only. This screen has no broker credentials, " +
                        "deposit/withdrawal route, or live-money order authority."
                setTextIsSelectable(true)
            },
            fullWidth(),
        )

        endpointView = EditText(this).apply {
            hint = "HTTPS VN97MKTFEED1 endpoint"
            maxLines = 2
        }
        root.addView(endpointView, fullWidth())

        val sourceSymbols = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
        }
        sourceView = EditText(this).apply {
            hint = "Source ID"
            maxLines = 1
        }
        symbolsView = EditText(this).apply {
            hint = "Symbols: ABC,XYZ"
            maxLines = 1
        }
        sourceSymbols.addView(sourceView, weighted())
        sourceSymbols.addView(symbolsView, weighted())
        root.addView(sourceSymbols, fullWidth())

        goalView = EditText(this).apply {
            hint = "Paper-only trading objective"
            maxLines = 4
            setText(
                "Paper trading simulation only; act only when bounded " +
                    "market evidence is sufficient."
            )
        }
        root.addView(goalView, fullWidth())

        val budgetRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
        }
        intervalSecondsView = EditText(this).apply {
            hint = "Interval seconds"
            maxLines = 1
            inputType = android.text.InputType.TYPE_CLASS_NUMBER
            setText("60")
        }
        maxEpisodesView = EditText(this).apply {
            hint = "Max episodes"
            maxLines = 1
            inputType = android.text.InputType.TYPE_CLASS_NUMBER
            setText("256")
        }
        budgetRow.addView(intervalSecondsView, weighted())
        budgetRow.addView(maxEpisodesView, weighted())
        root.addView(budgetRow, fullWidth())

        val powerRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
        }
        batteryNotLowCheck = CheckBox(this).apply {
            text = "Battery not low"
            isChecked = true
        }
        chargingCheck = CheckBox(this).apply {
            text = "Charging only"
            isChecked = false
        }
        powerRow.addView(batteryNotLowCheck, weighted())
        powerRow.addView(chargingCheck, weighted())
        root.addView(powerRow, fullWidth())

        startButton = Button(this).apply {
            text = "Start paper session"
            setOnClickListener { startPaperSession() }
        }
        root.addView(startButton, fullWidth())

        jobIdView = EditText(this).apply {
            hint = "Paper session Job ID"
            maxLines = 1
            inputType = android.text.InputType.TYPE_CLASS_NUMBER
        }
        root.addView(jobIdView, fullWidth())

        val lifecycleRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
        }
        pauseButton = Button(this).apply {
            text = "Pause"
            setOnClickListener { controlSession("pause") }
        }
        resumeButton = Button(this).apply {
            text = "Resume"
            setOnClickListener { controlSession("resume") }
        }
        stopButton = Button(this).apply {
            text = "Stop"
            setOnClickListener { controlSession("stop") }
        }
        lifecycleRow.addView(pauseButton, weighted())
        lifecycleRow.addView(resumeButton, weighted())
        lifecycleRow.addView(stopButton, weighted())
        root.addView(lifecycleRow, fullWidth())

        refreshButton = Button(this).apply {
            text = "Refresh sessions"
            setOnClickListener { refreshStatus() }
        }
        root.addView(refreshButton, fullWidth())

        statusView = TextView(this).apply {
            text = "Paper trading: no sessions."
            setTextIsSelectable(true)
        }
        root.addView(statusView, fullWidth())

        setContentView(scroll)
    }

    override fun onResume() {
        super.onResume()
        if (::statusView.isInitialized) {
            refreshStatus()
        }
    }

    override fun onDestroy() {
        worker.shutdownNow()
        super.onDestroy()
    }

    private fun startPaperSession() {
        val spec = try {
            VN97PaperTradingControlSurface.parse(
                endpointText = endpointView.text.toString(),
                sourceIdText = sourceView.text.toString(),
                symbolsText = symbolsView.text.toString(),
                userGoalText = goalView.text.toString(),
                intervalSecondsText =
                    intervalSecondsView.text.toString(),
                maxEpisodesText = maxEpisodesView.text.toString(),
                requiresBatteryNotLow =
                    batteryNotLowCheck.isChecked,
                requiresCharging = chargingCheck.isChecked,
            )
        } catch (exc: Throwable) {
            statusView.text =
                "Configuration rejected: " +
                    (exc.message ?: exc::class.java.simpleName)
            return
        }

        setControlsEnabled(false)
        statusView.text =
            "Starting bounded paper-trading simulation session…"
        worker.execute {
            try {
                val record = app.paperTrading.startSession(
                    VN97PaperTradingSessionConfig(
                        endpoint = spec.endpoint,
                        sourceId = spec.sourceId,
                        symbols = spec.symbols,
                        userGoal = spec.userGoal,
                        intervalMillis = spec.intervalMillis,
                        maxEpisodes = spec.maxEpisodes,
                        requiresBatteryNotLow =
                            spec.requiresBatteryNotLow,
                        requiresCharging = spec.requiresCharging,
                    )
                )
                runOnUiThread {
                    jobIdView.setText(record.jobId.toString())
                    statusView.text =
                        "Started paper-only session #" +
                            record.jobId +
                            ". No live-money authority exists."
                    setControlsEnabled(true)
                    refreshStatus()
                }
            } catch (exc: Throwable) {
                runOnUiThread {
                    statusView.text =
                        "Start failed: " +
                            (exc.message ?: exc::class.java.simpleName)
                    setControlsEnabled(true)
                }
            }
        }
    }

    private fun controlSession(action: String) {
        val jobId = try {
            VN97PaperTradingControlSurface.parseJobId(
                jobIdView.text.toString()
            )
        } catch (exc: Throwable) {
            statusView.text =
                "Control rejected: " +
                    (exc.message ?: exc::class.java.simpleName)
            return
        }

        setControlsEnabled(false)
        worker.execute {
            try {
                val record = when (action) {
                    "pause" -> app.paperTrading.pause(jobId)
                    "resume" -> app.paperTrading.resume(jobId)
                    "stop" -> app.paperTrading.stop(jobId)
                    else -> error("unknown paper trading UI action")
                }
                runOnUiThread {
                    statusView.text =
                        "Session #" +
                            record.jobId +
                            " is now " +
                            record.state.name +
                            "."
                    setControlsEnabled(true)
                    refreshStatus()
                }
            } catch (exc: Throwable) {
                runOnUiThread {
                    statusView.text =
                        action.replaceFirstChar { it.uppercase() } +
                            " failed: " +
                            (exc.message ?: exc::class.java.simpleName)
                    setControlsEnabled(true)
                }
            }
        }
    }

    private fun refreshStatus() {
        if (!::statusView.isInitialized) return
        refreshButton.isEnabled = false
        worker.execute {
            val result = runCatching {
                Pair(
                    app.paperTrading.listReports(),
                    app.paperTrading.performanceEvidence(),
                )
            }
            runOnUiThread {
                val (reports, performance) = result.getOrElse { exc ->
                    statusView.text =
                        "Status unavailable: " +
                            (exc.message ?: exc::class.java.simpleName)
                    refreshButton.isEnabled = true
                    return@runOnUiThread
                }
                val sessionText =
                    VN97PaperTradingControlSurface.formatReports(
                        reports.map { report ->
                            VN97PaperTradingUiReport(
                                jobId = report.jobId,
                                state = report.state.name,
                                episodesAttempted =
                                    report.episodesAttempted,
                                maxEpisodes = report.maxEpisodes,
                                wakeCount = report.wakeCount,
                                lastDecision = report.lastDecision,
                                lastOutcome = report.lastOutcome,
                                terminalReason = report.terminalReason,
                                nextRunWallTimeMillis =
                                    report.nextRunWallTimeMillis,
                            )
                        }
                    )
                statusView.text =
                    sessionText +
                        formatPerformanceEvidence(performance)
                if (
                    jobIdView.text.isNullOrBlank() &&
                    reports.isNotEmpty()
                ) {
                    val active = reports.filter {
                        it.state != VN97PaperTradingSessionState.STOPPED &&
                            it.state !=
                                VN97PaperTradingSessionState.COMPLETED &&
                            it.state != VN97PaperTradingSessionState.FAILED
                    }
                    val selected =
                        active.maxByOrNull { it.jobId }
                            ?: reports.maxByOrNull { it.jobId }
                    selected?.let {
                        jobIdView.setText(it.jobId.toString())
                    }
                }
                setControlsEnabled(true)
            }
        }
    }

    private fun formatPerformanceEvidence(
        aggregate: VN97PaperPerformanceAggregate,
    ): String {
        if (aggregate.evidenceCount == 0) {
            return "\n\nHistorical paper-performance evidence: none yet."
        }
        return buildString {
            append("\n\nHistorical paper-performance evidence ")
            append("(simulation only; not a forecast):")
            append("\nrecords=")
            append(aggregate.evidenceCount)
            append(" sessions=")
            append(aggregate.sessionCount)
            aggregate.sessions.take(8).forEach { summary ->
                append("\n#")
                append(summary.jobId)
                append(" evidence=")
                append(summary.evidenceCount)
                append(" episodes=")
                append(summary.firstEpisode)
                append("..")
                append(summary.latestEpisode)
                append(" pnl_micros=")
                append(summary.latestTotalPnlMicros)
                append(" return_bps=")
                append(summary.latestReturnBasisPoints)
                append(" max_drawdown_bps=")
                append(summary.maxDrawdownBasisPoints)
                append(" holds=")
                append(summary.holdCount)
                append(" orders=")
                append(summary.orderCount)
            }
        }
    }

    private fun setControlsEnabled(enabled: Boolean) {
        startButton.isEnabled = enabled
        pauseButton.isEnabled = enabled
        resumeButton.isEnabled = enabled
        stopButton.isEnabled = enabled
        refreshButton.isEnabled = enabled
    }

    private fun fullWidth() =
        LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.WRAP_CONTENT,
        )

    private fun weighted() =
        LinearLayout.LayoutParams(
            0,
            ViewGroup.LayoutParams.WRAP_CONTENT,
            1f,
        )
}
