package ai.vn97.app

import android.app.Activity
import android.os.Bundle
import android.view.Gravity
import android.view.ViewGroup
import android.view.WindowManager
import android.widget.Button
import android.widget.CheckBox
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import java.nio.charset.StandardCharsets
import java.util.concurrent.Executors

class VN97PaperTradingActivity : Activity() {
    private lateinit var exnessApiKeyView: EditText
    private lateinit var exnessAccountIdView: EditText
    private lateinit var exnessBaseUrlView: EditText
    private lateinit var exnessSecretView: EditText
    private lateinit var saveExnessCredentialsButton: Button
    private lateinit var clearExnessCredentialsButton: Button
    private lateinit var validateExnessButton: Button
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
    private lateinit var revenueCampaignButton: Button
    private lateinit var pauseButton: Button
    private lateinit var resumeButton: Button
    private lateinit var stopButton: Button
    private lateinit var refreshButton: Button

    private val worker = Executors.newSingleThreadExecutor()

    private val app: VN97Application
        get() = application as VN97Application

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.addFlags(WindowManager.LayoutParams.FLAG_SECURE)

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

        root.addView(
            TextView(this).apply {
                text = "Exness live channel credentials"
                textSize = 18f
            },
            fullWidth(),
        )
        root.addView(
            TextView(this).apply {
                text =
                    "Stored only on this device in no-backup app storage, " +
                        "encrypted by Android Keystore. Saving credentials " +
                        "does not authorize live trading."
            },
            fullWidth(),
        )

        exnessApiKeyView = EditText(this).apply {
            hint = "Exness API key"
            maxLines = 1
        }
        root.addView(exnessApiKeyView, fullWidth())

        exnessAccountIdView = EditText(this).apply {
            hint = "Exness account ID"
            maxLines = 1
            inputType = android.text.InputType.TYPE_CLASS_NUMBER
        }
        root.addView(exnessAccountIdView, fullWidth())

        exnessBaseUrlView = EditText(this).apply {
            hint = "Exness base URL"
            maxLines = 1
            setText("https://api.exness.com")
        }
        root.addView(exnessBaseUrlView, fullWidth())

        exnessSecretView = EditText(this).apply {
            hint = "Exness Secret Key (never shown again)"
            maxLines = 3
            inputType =
                android.text.InputType.TYPE_CLASS_TEXT or
                    android.text.InputType.TYPE_TEXT_VARIATION_PASSWORD
        }
        root.addView(exnessSecretView, fullWidth())

        val credentialButtons = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
        }
        saveExnessCredentialsButton = Button(this).apply {
            text = "Encrypt & save"
            setOnClickListener {
                saveExnessCredentials()
            }
        }
        clearExnessCredentialsButton = Button(this).apply {
            text = "Delete credentials"
            setOnClickListener {
                clearExnessCredentials()
            }
        }
        credentialButtons.addView(
            saveExnessCredentialsButton,
            weighted(),
        )
        credentialButtons.addView(
            clearExnessCredentialsButton,
            weighted(),
        )
        root.addView(credentialButtons, fullWidth())

        validateExnessButton = Button(this).apply {
            text = "Validate Exness read-only"
            setOnClickListener {
                validateExnessReadOnly()
            }
        }
        root.addView(validateExnessButton, fullWidth())

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

        revenueCampaignButton = Button(this).apply {
            text = "Start revenue qualification campaign"
            setOnClickListener { startRevenueCampaign() }
        }
        root.addView(revenueCampaignButton, fullWidth())

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

    private fun saveExnessCredentials() {
        val apiKey =
            exnessApiKeyView.text.toString().trim()
        val accountId =
            exnessAccountIdView.text.toString().trim()
        val baseUrl =
            exnessBaseUrlView.text.toString().trim()
        val secretBytes =
            exnessSecretView.text.toString()
                .toByteArray(StandardCharsets.UTF_8)
        exnessSecretView.text.clear()

        setControlsEnabled(false)
        statusView.text =
            "Encrypting Exness credentials locally…"
        worker.execute {
            try {
                app.exnessCredentialVault.store(
                    apiKey = apiKey,
                    accountId = accountId,
                    baseUrl = baseUrl,
                    privateKeySecret = secretBytes,
                )
                app.exnessConnection.invalidate()
                runOnUiThread {
                    exnessApiKeyView.text.clear()
                    statusView.text =
                        "Exness credentials encrypted in Android Keystore-backed vault. " +
                            "Live-money authority remains disabled."
                    setControlsEnabled(true)
                    refreshStatus()
                }
            } catch (exc: Throwable) {
                runOnUiThread {
                    statusView.text =
                        "Credential provisioning failed: " +
                            (exc.message ?: exc::class.java.simpleName)
                    setControlsEnabled(true)
                }
            } finally {
                secretBytes.fill(0)
            }
        }
    }

    private fun clearExnessCredentials() {
        setControlsEnabled(false)
        worker.execute {
            val result =
                runCatching {
                    app.exnessCredentialVault.clear()
                    app.exnessConnection.invalidate()
                }
            runOnUiThread {
                exnessApiKeyView.text.clear()
                exnessAccountIdView.text.clear()
                exnessSecretView.text.clear()
                statusView.text =
                    result.fold(
                        onSuccess = {
                            "Exness credentials deleted; live revenue channel locked."
                        },
                        onFailure = { exc ->
                            "Credential deletion failed: " +
                                (exc.message ?: exc::class.java.simpleName)
                        },
                    )
                setControlsEnabled(true)
                refreshStatus()
            }
        }
    }

    private fun validateExnessReadOnly() {
        setControlsEnabled(false)
        statusView.text =
            "Validating signed Exness connection with read-only requests…"
        worker.execute {
            val result =
                app.exnessConnection
                    .validateReadOnly()
            runOnUiThread {
                statusView.text =
                    if (
                        result.evidence.authenticated &&
                        result.evidence.dryRunValidated
                    ) {
                        "Exness read-only validation PASS; " +
                            "instruments=" +
                            result.instrumentCount +
                            ", access_point=" +
                            result.accessPointBaseUrl +
                            ". Live order submission remains disabled."
                    } else {
                        "Exness read-only validation LOCKED; error_class=" +
                            (result.errorClass ?: "unknown") +
                            ". No live order was sent."
                    }
                setControlsEnabled(true)
                refreshStatus()
            }
        }
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

    private fun startRevenueCampaign() {
        val spec = try {
            VN97PaperTradingControlSurface.parse(
                endpointText = endpointView.text.toString(),
                sourceIdText = sourceView.text.toString(),
                symbolsText = symbolsView.text.toString(),
                userGoalText = goalView.text.toString(),
                intervalSecondsText = "90",
                maxEpisodesText = "43",
                requiresBatteryNotLow =
                    batteryNotLowCheck.isChecked,
                requiresCharging = chargingCheck.isChecked,
            )
        } catch (exc: Throwable) {
            statusView.text =
                "Revenue campaign rejected: " +
                    (exc.message ?: exc::class.java.simpleName)
            return
        }

        setControlsEnabled(false)
        statusView.text =
            "Starting sequential paper-only revenue campaign…"
        worker.execute {
            try {
                val campaign =
                    app.revenueCampaign.start(
                        VN97RevenueCampaignRequest(
                            endpoint = spec.endpoint,
                            sourceId = spec.sourceId,
                            symbols = spec.symbols,
                            userGoal = spec.userGoal,
                            requiresBatteryNotLow =
                                spec.requiresBatteryNotLow,
                            requiresCharging =
                                spec.requiresCharging,
                        )
                    )
                runOnUiThread {
                    campaign.currentJobId?.let {
                        jobIdView.setText(it.toString())
                    }
                    statusView.text =
                        "Revenue campaign started: " +
                            campaign.campaignId.take(12) +
                            "…; paper evidence only, no live-money authority."
                    setControlsEnabled(true)
                    refreshStatus()
                }
            } catch (exc: Throwable) {
                runOnUiThread {
                    statusView.text =
                        "Revenue campaign start failed: " +
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
                Triple(
                    app.paperTrading.listReports(),
                    app.paperTrading.performanceEvidence(),
                    app.revenueCampaign.current(),
                )
            }
            runOnUiThread {
                val (reports, performance, campaign) =
                    result.getOrElse { exc ->
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
                        formatPerformanceEvidence(performance) +
                        formatRevenueQualification(performance) +
                        formatRevenueCampaign(campaign) +
                        formatLiveRevenueReadiness(campaign)
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

    private fun formatRevenueQualification(
        aggregate: VN97PaperPerformanceAggregate,
    ): String {
        val result =
            VN97RevenueQualificationGate.evaluate(aggregate)
        return buildString {
            append("\n\nRevenue qualification:")
            append("\nstage=")
            append(result.stage.name)
            append(" paper_qualified=")
            append(result.paperQualified)
            append("\nsessions=")
            append(result.sessionsConsidered)
            append(" evidence=")
            append(result.evidenceConsidered)
            append(" profitable_sessions=")
            append(result.profitableSessions)
            append(" orders=")
            append(result.ordersTotal)
            append("\nmedian_return_bps=")
            append(result.medianReturnBasisPoints ?: "n/a")
            append(" worst_return_bps=")
            append(result.worstReturnBasisPoints ?: "n/a")
            append(" worst_drawdown_bps=")
            append(result.worstDrawdownBasisPoints ?: "n/a")
            if (result.blockers.isNotEmpty()) {
                append("\nblockers=")
                append(result.blockers.joinToString(","))
            }
            append("\nproduction_money_movement_authorized=false")
        }
    }

    private fun formatRevenueCampaign(
        campaign: VN97RevenueCampaignSnapshot?,
    ): String {
        if (campaign == null) {
            return "\n\nRevenue campaign: none."
        }
        return buildString {
            append("\n\nRevenue campaign:")
            append("\nid=")
            append(campaign.campaignId.take(16))
            append(" state=")
            append(campaign.state.name)
            append("\nsessions_started=")
            append(campaign.sessionsStarted)
            append(" completed=")
            append(campaign.sessionsCompleted)
            campaign.currentJobId?.let {
                append(" current_job=")
                append(it)
            }
            campaign.qualification?.let { q ->
                append("\nqualification=")
                append(q.stage.name)
                append(" evidence=")
                append(q.evidenceConsidered)
                append(" median_return_bps=")
                append(q.medianReturnBasisPoints ?: "n/a")
                append(" max_drawdown_bps=")
                append(q.worstDrawdownBasisPoints ?: "n/a")
            }
            append("\nproduction_money_movement_authorized=false")
        }
    }

    private fun formatLiveRevenueReadiness(
        campaign: VN97RevenueCampaignSnapshot?,
    ): String {
        if (campaign == null) {
            return "\n\nLive revenue: LOCKED; no qualified campaign."
        }
        val readiness =
            VN97RevenueLiveReadinessGate.evaluate(
                promotion =
                    VN97RevenuePromotionEvidence(
                        campaignId = campaign.campaignId,
                        campaignState = campaign.state,
                        qualification = campaign.qualification,
                    ),
                channel =
                    app.exnessConnection
                        .currentEvidence(),
                authority = null,
                nowWallTimeMillis =
                    System.currentTimeMillis(),
            )
        return buildString {
            append("\n\nLive revenue readiness=")
            append(readiness.state.name)
            append("\nblockers=")
            append(readiness.blockers.joinToString(","))
            append("\nproduction_money_movement_authorized=")
            append(
                readiness.productionMoneyMovementAuthorized
            )
        }
    }

    private fun setControlsEnabled(enabled: Boolean) {
        saveExnessCredentialsButton.isEnabled = enabled
        clearExnessCredentialsButton.isEnabled = enabled
        validateExnessButton.isEnabled = enabled
        startButton.isEnabled = enabled
        revenueCampaignButton.isEnabled = enabled
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
