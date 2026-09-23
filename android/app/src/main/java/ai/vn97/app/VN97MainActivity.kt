package ai.vn97.app

import ai.vn97.avatar.AssistantMode
import ai.vn97.avatar.AvatarCommand
import ai.vn97.avatar.AvatarGesture
import ai.vn97.avatar.VN97AvatarView
import ai.vn97.platform.VN97AssistantTurnState
import android.Manifest
import android.app.Activity
import android.content.Intent
import android.content.pm.PackageManager
import android.media.projection.MediaProjectionManager
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.SystemClock
import android.provider.Settings
import android.view.Gravity
import android.view.View
import android.view.ViewGroup
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import java.io.File
import java.io.FileOutputStream
import java.nio.charset.StandardCharsets
import java.nio.file.Files
import java.nio.file.StandardCopyOption
import java.util.concurrent.Executors

class VN97MainActivity : Activity() {
    private lateinit var avatar: VN97AvatarView
    private lateinit var statusView: TextView
    private lateinit var floatingAssistantButton: Button
    private lateinit var voicePermissionButton: Button
    private lateinit var screenShareButton: Button
    private lateinit var screenAnalyzeButton: Button
    private lateinit var cameraAnalyzeButton: Button
    private lateinit var mobileEvidenceButton: Button
    private lateinit var gameAccessibilityButton: Button
    private lateinit var gameControlsContainer: LinearLayout
    private lateinit var gamePackageView: EditText
    private lateinit var gameGoalView: EditText
    private lateinit var gameStartButton: Button
    private lateinit var gameStopButton: Button
    private lateinit var gameStatusView: TextView
    private lateinit var transcriptView: TextView
    private lateinit var inputView: EditText
    private lateinit var sendButton: Button
    private lateinit var autonomousButton: Button
    private lateinit var cancelAutonomousButton: Button
    private lateinit var autonomousStatusView: TextView
    private lateinit var autonomousApprovalView: TextView
    private lateinit var autonomousApproveButton: Button
    private lateinit var autonomousRejectButton: Button
    private lateinit var approvalView: TextView
    private lateinit var approveButton: Button
    private lateinit var rejectButton: Button
    private lateinit var provisioningView: TextView
    private lateinit var importModelButton: Button
    private lateinit var advancedProvisioningContainer: LinearLayout
    private lateinit var choosePackageButton: Button
    private lateinit var chooseSignatureButton: Button
    private lateinit var choosePublisherKeyButton: Button
    private lateinit var reviewModelButton: Button
    private lateinit var activateModelButton: Button

    private var packageUri: Uri? = null
    private var signatureUri: Uri? = null
    private var publisherKeyUri: Uri? = null
    private var pendingFloatingAssistantEnable = false
    private var pendingCameraCapture = false
    private var latestCancellableAutonomousJobId: Int? = null
    private var autonomousApprovalActive = false

    private val worker = Executors.newSingleThreadExecutor()
    private var state = VN97AppState()
    private var avatarSequence = 1L

    private val app: VN97Application
        get() = application as VN97Application

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        app.platformRuntime

        val density = resources.displayMetrics.density
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(
                (16 * density).toInt(),
                (16 * density).toInt(),
                (16 * density).toInt(),
                (16 * density).toInt(),
            )
        }

        val title = TextView(this).apply {
            text = "VN97"
            textSize = 24f
            gravity = Gravity.CENTER_HORIZONTAL
        }
        root.addView(
            title,
            LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT,
            ),
        )

        avatar = VN97AvatarView(this).apply {
            publish(
                AvatarCommand(
                    sourceSequence = avatarSequence++,
                    mode = AssistantMode.IDLE,
                    gesture = AvatarGesture.NONE,
                    energy = 0.35f,
                )
            )
            setInteractionListener { interaction ->
                statusView.text =
                    "Avatar interaction: ${interaction.type.name.lowercase()}"
            }
        }
        root.addView(
            avatar,
            LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                (260 * density).toInt(),
            ),
        )

        statusView = TextView(this)
        root.addView(
            statusView,
            LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT,
            ),
        )

        floatingAssistantButton = Button(this).apply {
            setOnClickListener { toggleFloatingAssistant() }
        }
        root.addView(
            floatingAssistantButton,
            LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT,
            ),
        )

        voicePermissionButton = Button(this).apply {
            setOnClickListener { requestMicrophonePermissionIfNeeded() }
        }
        root.addView(
            voicePermissionButton,
            LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT,
            ),
        )

        screenShareButton = Button(this).apply {
            setOnClickListener { toggleScreenPerception() }
        }
        root.addView(
            screenShareButton,
            LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT,
            ),
        )

        screenAnalyzeButton = Button(this).apply {
            text = "Analyze shared screen"
            setOnClickListener { analyzeScreenPerception() }
        }
        root.addView(
            screenAnalyzeButton,
            LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT,
            ),
        )

        cameraAnalyzeButton = Button(this).apply {
            text = "Analyze camera"
            setOnClickListener { analyzeCameraPerception() }
        }
        root.addView(
            cameraAnalyzeButton,
            LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT,
            ),
        )

        mobileEvidenceButton = Button(this).apply {
            text = "Collect VN97 mobile evidence"
            visibility =
                if (BuildConfig.VN97_TURNKEY_REQUIRED) View.GONE else View.VISIBLE
            setOnClickListener { collectMobileEvidence() }
        }
        root.addView(
            mobileEvidenceButton,
            LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT,
            ),
        )

        gameControlsContainer = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            visibility = View.GONE
        }
        val gamePanelButton = Button(this).apply {
            text = "Game Agent"
            setOnClickListener {
                gameControlsContainer.visibility =
                    if (
                        gameControlsContainer.visibility ==
                            View.VISIBLE
                    ) {
                        View.GONE
                    } else {
                        View.VISIBLE
                    }
            }
        }
        root.addView(
            gamePanelButton,
            LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT,
            ),
        )

        gameAccessibilityButton = Button(this).apply {
            setOnClickListener {
                startActivity(
                    Intent(Settings.ACTION_ACCESSIBILITY_SETTINGS)
                )
            }
        }
        gameControlsContainer.addView(
            gameAccessibilityButton,
            LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT,
            ),
        )

        gamePackageView = EditText(this).apply {
            hint = "Game package (for example com.example.game)"
            maxLines = 1
        }
        gameControlsContainer.addView(
            gamePackageView,
            LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT,
            ),
        )

        gameGoalView = EditText(this).apply {
            hint = "Game objective for VN97"
            maxLines = 3
        }
        gameControlsContainer.addView(
            gameGoalView,
            LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT,
            ),
        )

        val gameActionRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
        }
        gameStartButton = Button(this).apply {
            text = "Start Game Agent"
            setOnClickListener { startGameAgent() }
        }
        gameStopButton = Button(this).apply {
            text = "Stop Game Agent"
            setOnClickListener { stopGameAgent() }
        }
        gameActionRow.addView(
            gameStartButton,
            LinearLayout.LayoutParams(
                0,
                ViewGroup.LayoutParams.WRAP_CONTENT,
                1f,
            ),
        )
        gameActionRow.addView(
            gameStopButton,
            LinearLayout.LayoutParams(
                0,
                ViewGroup.LayoutParams.WRAP_CONTENT,
                1f,
            ),
        )
        gameControlsContainer.addView(
            gameActionRow,
            LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT,
            ),
        )

        gameStatusView = TextView(this).apply {
            text = "Game Agent: idle"
            setTextIsSelectable(true)
        }
        gameControlsContainer.addView(
            gameStatusView,
            LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT,
            ),
        )
        root.addView(
            gameControlsContainer,
            LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT,
            ),
        )

        approvalView = TextView(this).apply {
            visibility = View.GONE
            setTextIsSelectable(true)
        }
        root.addView(
            approvalView,
            LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT,
            ),
        )

        val approvalRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.END
        }
        rejectButton = Button(this).apply {
            text = "Reject"
            visibility = View.GONE
            setOnClickListener { resolveApproval(false) }
        }
        approveButton = Button(this).apply {
            text = "Approve"
            visibility = View.GONE
            setOnClickListener { resolveApproval(true) }
        }
        approvalRow.addView(rejectButton)
        approvalRow.addView(approveButton)
        root.addView(
            approvalRow,
            LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT,
            ),
        )

        provisioningView = TextView(this).apply {
            text = if (BuildConfig.VN97_TURNKEY_REQUIRED) {
                "Preparing bundled VN97 intelligence…"
            } else {
                "No active VN97 model. A signed bundled model will activate automatically when present."
            }
            setTextIsSelectable(true)
        }
        root.addView(
            provisioningView,
            LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT,
            ),
        )

        importModelButton = Button(this).apply {
            text = "IMPORT VN97 MODEL"
            visibility =
                if (BuildConfig.VN97_TURNKEY_REQUIRED) View.GONE else View.VISIBLE
            setOnClickListener {
                if (BuildConfig.VN97_TURNKEY_REQUIRED) {
                    return@setOnClickListener
                }
                advancedProvisioningContainer.visibility =
                    if (advancedProvisioningContainer.visibility == View.VISIBLE) {
                        View.GONE
                    } else {
                        View.VISIBLE
                    }
            }
        }
        root.addView(
            importModelButton,
            LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT,
            ),
        )

        advancedProvisioningContainer = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            visibility = View.GONE
        }

        val provisioningRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
        }
        choosePackageButton = Button(this).apply {
            text = "Package"
            setOnClickListener { openDocument(REQUEST_PACKAGE) }
        }
        chooseSignatureButton = Button(this).apply {
            text = "Signature"
            setOnClickListener { openDocument(REQUEST_SIGNATURE) }
        }
        choosePublisherKeyButton = Button(this).apply {
            text = "Publisher Key"
            setOnClickListener { openDocument(REQUEST_PUBLISHER_KEY) }
        }
        provisioningRow.addView(choosePackageButton)
        provisioningRow.addView(chooseSignatureButton)
        provisioningRow.addView(choosePublisherKeyButton)
        advancedProvisioningContainer.addView(
            provisioningRow,
            LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT,
            ),
        )

        val provisioningActionRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.END
        }
        reviewModelButton = Button(this).apply {
            text = "Review"
            setOnClickListener { reviewProvisioning() }
        }
        activateModelButton = Button(this).apply {
            text = "Trust & Activate"
            isEnabled = false
            setOnClickListener { activateReviewedModel() }
        }
        provisioningActionRow.addView(reviewModelButton)
        provisioningActionRow.addView(activateModelButton)
        advancedProvisioningContainer.addView(
            provisioningActionRow,
            LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT,
            ),
        )
        root.addView(
            advancedProvisioningContainer,
            LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT,
            ),
        )

        transcriptView = TextView(this).apply {
            textSize = 16f
        }
        val transcriptScroll = ScrollView(this).apply {
            addView(
                transcriptView,
                ViewGroup.LayoutParams(
                    ViewGroup.LayoutParams.MATCH_PARENT,
                    ViewGroup.LayoutParams.WRAP_CONTENT,
                ),
            )
        }
        root.addView(
            transcriptScroll,
            LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                0,
                1f,
            ),
        )

        val inputRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
        }
        inputView = EditText(this).apply {
            hint = "Message VN97"
            maxLines = 4
        }
        inputRow.addView(
            inputView,
            LinearLayout.LayoutParams(
                0,
                ViewGroup.LayoutParams.WRAP_CONTENT,
                1f,
            ),
        )

        sendButton = Button(this).apply {
            text = "Send"
            setOnClickListener { submitTurn() }
        }
        inputRow.addView(
            sendButton,
            LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.WRAP_CONTENT,
                ViewGroup.LayoutParams.WRAP_CONTENT,
            ),
        )
        root.addView(
            inputRow,
            LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT,
            ),
        )

        val autonomousRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
        }
        autonomousButton = Button(this).apply {
            text = "Run autonomously"
            setOnClickListener { startAutonomousGoal() }
        }
        cancelAutonomousButton = Button(this).apply {
            text = "Cancel latest goal"
            isEnabled = false
            setOnClickListener { cancelLatestAutonomousGoal() }
        }
        autonomousRow.addView(
            autonomousButton,
            LinearLayout.LayoutParams(
                0,
                ViewGroup.LayoutParams.WRAP_CONTENT,
                1f,
            ),
        )
        autonomousRow.addView(
            cancelAutonomousButton,
            LinearLayout.LayoutParams(
                0,
                ViewGroup.LayoutParams.WRAP_CONTENT,
                1f,
            ),
        )
        root.addView(
            autonomousRow,
            LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT,
            ),
        )

        autonomousStatusView = TextView(this).apply {
            text = "Autonomous goals: none"
            setTextIsSelectable(true)
        }
        root.addView(
            autonomousStatusView,
            LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT,
            ),
        )

        autonomousApprovalView = TextView(this).apply {
            visibility = View.GONE
            setTextIsSelectable(true)
        }
        root.addView(
            autonomousApprovalView,
            LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT,
            ),
        )
        val autonomousApprovalRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.END
        }
        autonomousRejectButton = Button(this).apply {
            text = "Reject autonomous action"
            visibility = View.GONE
            setOnClickListener {
                resolveAutonomousApproval(false)
            }
        }
        autonomousApproveButton = Button(this).apply {
            text = "Approve autonomous action"
            visibility = View.GONE
            setOnClickListener {
                resolveAutonomousApproval(true)
            }
        }
        autonomousApprovalRow.addView(autonomousRejectButton)
        autonomousApprovalRow.addView(autonomousApproveButton)
        root.addView(
            autonomousApprovalRow,
            LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT,
            ),
        )

        setContentView(root)
        refreshFloatingAssistantButton()
        refreshVoicePermissionButton()
        refreshVisualButtons()
        refreshGameAgentStatus()
        render(state)
        refreshAutonomousStatus()
        attachTrustedModel()
        if (
            intent?.action ==
                VN97FloatingAssistantService.ACTION_REQUEST_MICROPHONE_PERMISSION
        ) {
            requestMicrophonePermissionIfNeeded()
        }
    }

    override fun onResume() {
        super.onResume()
        avatar.onAvatarResume()
        if (pendingFloatingAssistantEnable) {
            pendingFloatingAssistantEnable = false
            if (Settings.canDrawOverlays(this)) {
                VN97FloatingAssistantService.enable(this)
            }
        }
        refreshFloatingAssistantButton()
        refreshVoicePermissionButton()
        requestAutonomousNotificationPermissionIfNeeded()
        refreshVisualButtons()
        refreshGameAgentStatus()
        refreshAutonomousStatus()
    }

    override fun onNewIntent(intent: Intent?) {
        super.onNewIntent(intent)
        setIntent(intent)
        if (
            intent?.action ==
                VN97FloatingAssistantService.ACTION_REQUEST_MICROPHONE_PERMISSION
        ) {
            requestMicrophonePermissionIfNeeded()
        }
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray,
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_MICROPHONE_PERMISSION) {
            refreshVoicePermissionButton()
            statusView.text =
                if (hasMicrophonePermission()) {
                    "Microphone enabled for local VN97 voice capture."
                } else {
                    "Microphone permission was not granted."
                }
        }
        if (requestCode == REQUEST_AUTONOMOUS_NOTIFICATION_PERMISSION) {
            statusView.text =
                if (
                    Build.VERSION.SDK_INT <
                        Build.VERSION_CODES.TIRAMISU ||
                    checkSelfPermission(
                        Manifest.permission.POST_NOTIFICATIONS
                    ) == PackageManager.PERMISSION_GRANTED
                ) {
                    "Autonomous completion notifications enabled."
                } else {
                    "Autonomous notification permission was not granted."
                }
        }
        if (requestCode == REQUEST_CAMERA_PERMISSION) {
            val granted = hasCameraPermission()
            statusView.text =
                if (granted) {
                    "Camera enabled for explicit VN97 visual capture."
                } else {
                    "Camera permission was not granted."
                }
            if (granted && pendingCameraCapture) {
                pendingCameraCapture = false
                analyzeCameraPerception()
            } else {
                pendingCameraCapture = false
            }
            refreshVisualButtons()
        }
    }

    override fun onPause() {
        avatar.onAvatarPause()
        super.onPause()
    }

    override fun onDestroy() {
        runCatching {
            app.autonomousWork.releaseForegroundApprovalSession()
        }
        worker.shutdownNow()
        super.onDestroy()
    }

    private fun requestAutonomousNotificationPermissionIfNeeded() {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.TIRAMISU) {
            return
        }
        if (
            checkSelfPermission(
                Manifest.permission.POST_NOTIFICATIONS
            ) == PackageManager.PERMISSION_GRANTED
        ) {
            return
        }
        val preferences =
            getSharedPreferences(
                "vn97-autonomous-notifications",
                MODE_PRIVATE,
            )
        if (preferences.getBoolean("prompted", false)) {
            return
        }
        preferences.edit().putBoolean("prompted", true).apply()
        requestPermissions(
            arrayOf(Manifest.permission.POST_NOTIFICATIONS),
            REQUEST_AUTONOMOUS_NOTIFICATION_PERMISSION,
        )
    }

    private fun startGameAgent() {
        val packageName = gamePackageView.text.toString().trim()
        val objective = gameGoalView.text.toString().trim()
        if (packageName.isBlank() || objective.isBlank()) {
            statusView.text =
                "Game Agent requires a target package and objective."
            return
        }
        if (!app.screenCaptureBroker.isActive()) {
            statusView.text =
                "Enable shared-screen perception before starting Game Agent."
            return
        }
        if (!app.platformRuntime.gameSession.isAccessibilityReady()) {
            statusView.text =
                "Enable VN97 Game Control in Android Accessibility settings."
            return
        }
        val launch =
            packageManager.getLaunchIntentForPackage(packageName)
        if (launch == null) {
            statusView.text =
                "No launchable app found for package: $packageName"
            return
        }

        try {
            VN97GameAgentService.start(
                context = this,
                packageName = packageName,
                objective = objective,
            )
            startActivity(launch)
            statusView.text =
                "VN97 Game Agent starting for $packageName."
        } catch (exc: Throwable) {
            statusView.text =
                "Game Agent start failed: " +
                    exc::class.java.simpleName
        }
        refreshGameAgentStatus()
    }

    private fun stopGameAgent() {
        app.gameAgent.requestStop()
        runCatching {
            VN97GameAgentService.stop(this)
        }
        statusView.text = "VN97 Game Agent stop requested."
        refreshGameAgentStatus()
    }

    private fun refreshGameAgentStatus() {
        val accessibilityReady =
            app.platformRuntime.gameSession
                .isAccessibilityReady()
        gameAccessibilityButton.text =
            if (accessibilityReady) {
                "VN97 Game Control accessibility: enabled"
            } else {
                "Enable VN97 Game Control accessibility"
            }

        val game = app.gameAgent.status()
        gameStatusView.text = buildString {
            append("Game Agent: ")
            append(game.state.name)
            if (game.packageName.isNotBlank()) {
                append("\npackage=")
                append(game.packageName)
            }
            if (game.maxRounds > 0) {
                append("\nround=")
                append(game.round)
                append('/')
                append(game.maxRounds)
            }
            if (game.maxActions > 0) {
                append(" actions=")
                append(game.actionsUsed)
                append('/')
                append(game.maxActions)
            }
            if (game.lastVerification.isNotBlank()) {
                append("\nverify: ")
                append(
                    game.lastVerification
                        .replace('\n', ' ')
                        .take(240)
                )
            }
            if (game.finalMessage.isNotBlank()) {
                append("\nresult: ")
                append(
                    game.finalMessage
                        .replace('\n', ' ')
                        .take(240)
                )
            }
        }
        val active =
            game.state == VN97GameAgentState.STARTING ||
                game.state == VN97GameAgentState.RUNNING ||
                game.state ==
                    VN97GameAgentState.WAITING_APPROVAL
        gameStartButton.isEnabled = !active
        gameStopButton.isEnabled = active
    }

    private fun hasMicrophonePermission(): Boolean =
        checkSelfPermission(Manifest.permission.RECORD_AUDIO) ==
            PackageManager.PERMISSION_GRANTED

    private fun requestMicrophonePermissionIfNeeded() {
        if (hasMicrophonePermission()) {
            refreshVoicePermissionButton()
            return
        }
        requestPermissions(
            arrayOf(Manifest.permission.RECORD_AUDIO),
            REQUEST_MICROPHONE_PERMISSION,
        )
    }

    private fun refreshVoicePermissionButton() {
        voicePermissionButton.text =
            if (hasMicrophonePermission()) {
                "Microphone voice enabled"
            } else {
                "Enable microphone voice"
            }
        voicePermissionButton.isEnabled = !hasMicrophonePermission()
    }

    private fun hasCameraPermission(): Boolean =
        checkSelfPermission(Manifest.permission.CAMERA) ==
            PackageManager.PERMISSION_GRANTED

    private fun requestCameraPermissionIfNeeded(): Boolean {
        if (hasCameraPermission()) return true
        pendingCameraCapture = true
        requestPermissions(
            arrayOf(Manifest.permission.CAMERA),
            REQUEST_CAMERA_PERMISSION,
        )
        return false
    }

    private fun toggleScreenPerception() {
        if (app.screenCaptureBroker.isActive()) {
            VN97ScreenCaptureService.stop(this)
            statusView.text = "Screen perception stopped."
            refreshVisualButtons()
            return
        }
        if (!app.assistant.hasProductionVision()) {
            statusView.text =
                "Activated VN97 model has no production vision weights."
            return
        }
        val manager = getSystemService(
            MediaProjectionManager::class.java
        ) ?: run {
            statusView.text =
                "MediaProjectionManager unavailable."
            return
        }
        startActivityForResult(
            manager.createScreenCaptureIntent(),
            REQUEST_SCREEN_CAPTURE,
        )
    }

    private fun analyzeScreenPerception() {
        if (
            state.phase != VN97AppPhase.READY ||
            !state.inputEnabled
        ) {
            statusView.text =
                "VN97 must be READY before visual perception."
            return
        }
        if (!app.assistant.hasProductionVision()) {
            statusView.text =
                "Activated VN97 model has no production vision weights."
            return
        }
        if (!app.screenCaptureBroker.isActive()) {
            statusView.text =
                "Start user-approved screen sharing first."
            return
        }

        val goal = inputView.text.toString().trim()
        if (goal.isNotEmpty()) {
            inputView.text.clear()
        }
        render(
            VN97AppReducer.reduce(
                state,
                VN97AppEvent.TurnStarted,
            )
        )
        statusView.text =
            if (goal.isEmpty()) {
                "VN97 is perceiving the shared screen…"
            } else {
                "VN97 is perceiving the screen and planning the requested task…"
            }

        val after = SystemClock.elapsedRealtimeNanos()
        worker.execute {
            try {
                val frame = app.screenCaptureBroker.awaitFreshFrame(
                    afterElapsedRealtimeNs = after,
                )
                val visual = if (goal.isEmpty()) {
                    app.visualActions.observe(
                        VN97VisualSource.SCREEN,
                        frame.prepared,
                    )
                } else {
                    app.visualActions.runGoal(
                        VN97VisualSource.SCREEN,
                        frame.prepared,
                        goal,
                    )
                }
                runOnUiThread {
                    applyVisualResult(visual)
                }
            } catch (exc: Throwable) {
                runOnUiThread {
                    render(
                        VN97AppReducer.reduce(
                            state,
                            VN97AppEvent.VisualFailed(
                                "Screen perception failed: " +
                                    exc::class.java.simpleName
                            ),
                        )
                    )
                }
            }
        }
    }

    private fun analyzeCameraPerception() {
        if (
            state.phase != VN97AppPhase.READY ||
            !state.inputEnabled
        ) {
            statusView.text =
                "VN97 must be READY before visual perception."
            return
        }
        if (!app.assistant.hasProductionVision()) {
            statusView.text =
                "Activated VN97 model has no production vision weights."
            return
        }
        if (!requestCameraPermissionIfNeeded()) {
            statusView.text =
                "Camera permission is required for this explicit capture."
            return
        }

        val goal = inputView.text.toString().trim()
        if (goal.isNotEmpty()) {
            inputView.text.clear()
        }
        render(
            VN97AppReducer.reduce(
                state,
                VN97AppEvent.TurnStarted,
            )
        )
        statusView.text =
            if (goal.isEmpty()) {
                "Capturing one camera frame for local VN97 perception…"
            } else {
                "Capturing one camera frame for the requested VN97 task…"
            }

        VN97CameraCapture(applicationContext).capture { capture ->
            worker.execute {
                try {
                    val prepared = capture.getOrThrow()
                    val visual = if (goal.isEmpty()) {
                        app.visualActions.observe(
                            VN97VisualSource.CAMERA,
                            prepared,
                        )
                    } else {
                        app.visualActions.runGoal(
                            VN97VisualSource.CAMERA,
                            prepared,
                            goal,
                        )
                    }
                    runOnUiThread {
                        applyVisualResult(visual)
                    }
                } catch (exc: Throwable) {
                    runOnUiThread {
                        render(
                            VN97AppReducer.reduce(
                                state,
                                VN97AppEvent.VisualFailed(
                                    "Camera perception failed: " +
                                        exc::class.java.simpleName
                                ),
                            )
                        )
                    }
                }
            }
        }
    }

    private fun applyVisualResult(
        visual: VN97VisualActionResult,
    ) {
        val turn = visual.turn
        if (turn == null) {
            render(
                VN97AppReducer.reduce(
                    state,
                    VN97AppEvent.VisualObserved(
                        source = visual.source.name.lowercase(),
                        observation = visual.beforeObservation,
                    ),
                )
            )
            refreshVisualButtons()
            return
        }

        val extra = when {
            visual.verification != null &&
                visual.afterObservation != null ->
                "\n\nVisual verification: " +
                    visual.verification
            visual.verificationFailure != null ->
                "\n\n" + visual.verificationFailure
            else -> ""
        }
        val withVerification =
            if (
                turn.update.state ==
                    VN97AssistantTurnState.COMPLETED &&
                extra.isNotEmpty()
            ) {
                turn.copy(
                    update = turn.update.copy(
                        finalResponse =
                            turn.update.finalResponse + extra
                    )
                )
            } else {
                turn
            }
        applyTurnResult(withVerification)
        refreshVisualButtons()
    }

    private fun refreshVisualButtons() {
        val vision = app.assistant.hasProductionVision()
        val ready =
            vision &&
                state.phase == VN97AppPhase.READY &&
                state.inputEnabled
        val sharing = app.screenCaptureBroker.isActive()
        screenShareButton.text =
            if (sharing) {
                "Stop screen perception"
            } else {
                "Start screen perception"
            }
        screenShareButton.isEnabled = sharing || ready
        screenAnalyzeButton.isEnabled = ready && sharing
        cameraAnalyzeButton.isEnabled = ready
    }

    private fun toggleFloatingAssistant() {
        val permissionGranted = Settings.canDrawOverlays(this)
        if (
            permissionGranted &&
            VN97FloatingAssistantService.isEnabled(this)
        ) {
            VN97FloatingAssistantService.disable(this)
            refreshFloatingAssistantButton()
            return
        }

        if (permissionGranted) {
            VN97FloatingAssistantService.enable(this)
            refreshFloatingAssistantButton()
            return
        }

        pendingFloatingAssistantEnable = true
        startActivity(
            Intent(
                Settings.ACTION_MANAGE_OVERLAY_PERMISSION,
                Uri.parse("package:$packageName"),
            )
        )
    }

    private fun refreshFloatingAssistantButton() {
        val permissionGranted = Settings.canDrawOverlays(this)
        val enabled =
            permissionGranted &&
                VN97FloatingAssistantService.isEnabled(this)
        floatingAssistantButton.text = when {
            enabled -> "Hide floating 3D assistant"
            permissionGranted -> "Show floating 3D assistant"
            else -> "Enable floating 3D assistant"
        }
    }

    private fun attachTrustedModel() {
        worker.execute {
            try {
                app.provisioner.recoverPending()
                var active = app.assistant.openIfActivated()
                var bundled = false
                if (!active) {
                    bundled =
                        app.bundledBootstrap.activateIfPresent(
                            required = BuildConfig.VN97_TURNKEY_REQUIRED,
                        ) == VN97BundledBootstrapOutcome.ACTIVATED
                    if (bundled) {
                        active = app.assistant.openIfActivated()
                        check(active) {
                            "bundled VN97 model activated but could not be opened"
                        }
                    }
                }
                runOnUiThread {
                    if (active) {
                        var next = VN97AppReducer.reduce(
                            state,
                            VN97AppEvent.TrustedModelActivated,
                        )
                        if (app.assistant.pendingApproval() != null) {
                            next = VN97AppReducer.reduce(
                                next,
                                VN97AppEvent.ApprovalRequired,
                            )
                        }
                        render(next)
                        provisioningView.text =
                            if (bundled) {
                                "Bundled signed VN97 model verified and activated."
                            } else {
                                "VN97 model active."
                            }
                        advancedProvisioningContainer.visibility = View.GONE
                    } else {
                        render(state)
                        provisioningView.text =
                            "Developer build: no bundled VN97 model. Import a signed VN97 model to enable chat."
                    }
                }
            } catch (exc: Throwable) {
                runOnUiThread {
                    renderFailure(
                        "Trusted model activation failed: " +
                            exc::class.java.simpleName
                    )
                }
            }
        }
    }

    override fun onActivityResult(
        requestCode: Int,
        resultCode: Int,
        data: Intent?,
    ) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode == REQUEST_SCREEN_CAPTURE) {
            if (resultCode == RESULT_OK && data != null) {
                try {
                    VN97ScreenCaptureService.start(
                        this,
                        resultCode,
                        data,
                    )
                    statusView.text =
                        "Screen sharing approved. VN97 perception stays local on-device."
                } catch (exc: Throwable) {
                    statusView.text =
                        "Screen perception failed to start: " +
                            exc::class.java.simpleName
                }
            } else {
                statusView.text =
                    "Screen sharing was not approved."
            }
            refreshVisualButtons()
            return
        }
        if (resultCode != RESULT_OK) return
        val uri = data?.data ?: return
        try {
            contentResolver.takePersistableUriPermission(
                uri,
                Intent.FLAG_GRANT_READ_URI_PERMISSION,
            )
        } catch (_: SecurityException) {
            // Some document providers grant only the current read; review still reopens immediately.
        }
        when (requestCode) {
            REQUEST_PACKAGE -> packageUri = uri
            REQUEST_SIGNATURE -> signatureUri = uri
            REQUEST_PUBLISHER_KEY -> publisherKeyUri = uri
            else -> return
        }
        app.provisioner.clearReview()
        advancedProvisioningContainer.visibility = View.VISIBLE
        activateModelButton.isEnabled = false
        renderProvisioningSelection()
    }

    private fun openDocument(requestCode: Int) {
        if (!provisioningAllowed()) return
        val intent = Intent(Intent.ACTION_OPEN_DOCUMENT).apply {
            addCategory(Intent.CATEGORY_OPENABLE)
            type = "application/octet-stream"
            addFlags(
                Intent.FLAG_GRANT_READ_URI_PERMISSION or
                    Intent.FLAG_GRANT_PERSISTABLE_URI_PERMISSION
            )
        }
        startActivityForResult(intent, requestCode)
    }

    private fun reviewProvisioning() {
        if (!provisioningAllowed()) return
        val packageValue = packageUri ?: return
        val signatureValue = signatureUri ?: return
        val keyValue = publisherKeyUri ?: return
        setProvisioningControlsEnabled(false)
        provisioningView.text = "Verifying publisher and compatibility…"
        worker.execute {
            try {
                val review = app.provisioner.review(
                    packageUri = packageValue,
                    signatureUri = signatureValue,
                    publisherKeyUri = keyValue,
                )
                runOnUiThread {
                    renderProvisioningReview(review)
                    setProvisioningControlsEnabled(true)
                    activateModelButton.isEnabled = true
                }
            } catch (exc: Throwable) {
                app.provisioner.clearReview()
                runOnUiThread {
                    provisioningView.text =
                        "Provisioning review failed: " + exc::class.java.simpleName
                    setProvisioningControlsEnabled(true)
                    activateModelButton.isEnabled = false
                }
            }
        }
    }

    private fun activateReviewedModel() {
        if (!provisioningAllowed()) return
        if (app.provisioner.pendingReview() == null) return
        setProvisioningControlsEnabled(false)
        activateModelButton.isEnabled = false
        provisioningView.text = "Activating reviewed VN97 model…"
        worker.execute {
            try {
                val item = app.provisioner.activateReviewed()
                val opened = app.assistant.reloadActivatedModel()
                check(opened) { "activated model could not be reopened" }
                runOnUiThread {
                    provisioningView.text =
                        "Activated model v${item.capabilityVersion}\n" +
                            "Artifact SHA-256: ${item.artifactSha256}"
                    val next = when (state.phase) {
                        VN97AppPhase.MODEL_REQUIRED,
                        VN97AppPhase.ERROR,
                        -> VN97AppReducer.reduce(
                            state,
                            VN97AppEvent.TrustedModelActivated,
                        )
                        VN97AppPhase.READY -> state.copy(
                            status = "VN97 model activation updated.",
                            inputEnabled = true,
                        )
                        else -> state
                    }
                    render(next)
                    advancedProvisioningContainer.visibility = View.GONE
                    setProvisioningControlsEnabled(true)
                }
            } catch (exc: Throwable) {
                runOnUiThread {
                    provisioningView.text =
                        "Activation failed: " + exc::class.java.simpleName
                    setProvisioningControlsEnabled(true)
                }
            }
        }
    }

    private fun provisioningAllowed(): Boolean =
        !BuildConfig.VN97_TURNKEY_REQUIRED &&
            (
                state.phase == VN97AppPhase.MODEL_REQUIRED ||
                    state.phase == VN97AppPhase.READY ||
                    state.phase == VN97AppPhase.ERROR
            )

    private fun setProvisioningControlsEnabled(enabled: Boolean) {
        val allowed = enabled && provisioningAllowed()
        importModelButton.isEnabled = allowed
        choosePackageButton.isEnabled = allowed
        chooseSignatureButton.isEnabled = allowed
        choosePublisherKeyButton.isEnabled = allowed
        reviewModelButton.isEnabled = allowed &&
            packageUri != null &&
            signatureUri != null &&
            publisherKeyUri != null
        if (!allowed) activateModelButton.isEnabled = false
    }

    private fun renderProvisioningSelection() {
        provisioningView.text = buildString {
            append("Package: ")
            append(if (packageUri == null) "missing" else "selected")
            append(" | Signature: ")
            append(if (signatureUri == null) "missing" else "selected")
            append(" | Publisher key: ")
            append(if (publisherKeyUri == null) "missing" else "selected")
        }
        setProvisioningControlsEnabled(true)
    }

    private fun renderProvisioningReview(
        review: ai.vn97.runtime.VN97ModelProvisioningReview,
    ) {
        provisioningView.text = buildString {
            append("Publisher: ")
            append(review.publisherKeyId)
            append("\nKey SHA-256: ")
            append(review.publisherKeySha256)
            append("\nTrust status: ")
            append(
                if (review.publisherPreviouslyTrusted) {
                    "publisher key already enrolled"
                } else {
                    "NEW publisher key — explicit trust will be persisted"
                }
            )
            append("\nPackage SHA-256: ")
            append(review.packageSha256)
            append("\nCapability: ")
            append(review.capabilityId)
            append(" v")
            append(review.capabilityVersion)
            append("\nSource: ")
            append(review.sourceOrigin)
            append("\nLicense: ")
            append(review.sourceLicense)
            append("\nPlan SHA-256: ")
            append(review.planSha256)
            append("\n\nReview verified. Trust & Activate is an explicit user action.")
        }
    }

    private fun collectMobileEvidence() {
        if (BuildConfig.VN97_TURNKEY_REQUIRED) return
        if (
            state.phase != VN97AppPhase.READY ||
            !state.inputEnabled
        ) {
            statusView.text =
                "VN97 must be READY before collecting mobile evidence."
            return
        }

        mobileEvidenceButton.isEnabled = false
        sendButton.isEnabled = false
        statusView.text =
            "Collecting VN97 p50/p95 latency, PSS, thermal and energy evidence…"

        worker.execute {
            try {
                val evidence = app.assistant.collectMobileEvidence()
                val root =
                    getExternalFilesDir(null) ?: filesDir
                val target = File(root, "vn97-mobile-evidence.json")
                val temp = File(root, ".vn97-mobile-evidence.tmp")
                val bytes = evidence.toCanonicalJson().toByteArray(
                    StandardCharsets.UTF_8
                )
                FileOutputStream(temp, false).use { output ->
                    output.write(bytes)
                    output.flush()
                    output.fd.sync()
                }
                try {
                    Files.move(
                        temp.toPath(),
                        target.toPath(),
                        StandardCopyOption.REPLACE_EXISTING,
                    )
                } catch (exc: Throwable) {
                    temp.delete()
                    throw IllegalStateException(
                        "could not publish mobile evidence",
                        exc,
                    )
                }
                if (!target.readBytes().contentEquals(bytes)) {
                    throw IllegalStateException(
                        "mobile evidence post-write verification failed"
                    )
                }

                runOnUiThread {
                    statusView.text =
                        "VN97 mobile evidence saved: ${target.absolutePath}\n" +
                            "Model SHA-256: ${evidence.modelImageSha256}\n" +
                            "Text p95: ${evidence.textPrefill.p95Ms} ms prefill / " +
                            "${evidence.textDecodePerToken.p95Ms} ms per decode token"
                    mobileEvidenceButton.isEnabled = true
                    sendButton.isEnabled = state.inputEnabled
        if (!BuildConfig.VN97_TURNKEY_REQUIRED) {
            mobileEvidenceButton.isEnabled =
                state.phase == VN97AppPhase.READY &&
                    state.inputEnabled
        }
                }
            } catch (exc: Throwable) {
                runOnUiThread {
                    statusView.text =
                        "Mobile evidence failed: " +
                            exc::class.java.simpleName
                    mobileEvidenceButton.isEnabled = true
                    sendButton.isEnabled = state.inputEnabled
                }
            }
        }
    }

    private fun startAutonomousGoal() {
        if (autonomousApprovalActive) {
            statusView.text =
                "Resolve the pending autonomous approval first."
            return
        }
        if (
            state.phase != VN97AppPhase.READY ||
            !state.inputEnabled
        ) {
            statusView.text =
                "VN97 must be READY before starting an autonomous goal."
            return
        }
        val goal = inputView.text.toString().trim()
        if (goal.isEmpty()) {
            statusView.text =
                "Enter a goal before choosing Run autonomously."
            return
        }
        inputView.text.clear()
        autonomousButton.isEnabled = false
        sendButton.isEnabled = false
        statusView.text =
            "VN97 is building and persisting the autonomous plan…"

        worker.execute {
            try {
                val record = app.autonomousWork.startGoal(goal)
                runOnUiThread {
                    statusView.text =
                        if (
                            record.state ==
                                VN97AutonomousGoalState.SCHEDULED
                        ) {
                            "Autonomous goal scheduled as job " +
                                record.jobId +
                                ". It can continue after process death/reboot."
                        } else {
                            "Autonomous goal persisted but is " +
                                record.state.name.lowercase() +
                                ": " +
                                record.terminalReason
                        }
                    refreshAutonomousStatus()
                    autonomousButton.isEnabled =
                        state.phase == VN97AppPhase.READY &&
                            state.inputEnabled
                    sendButton.isEnabled = state.inputEnabled
                }
            } catch (exc: Throwable) {
                runOnUiThread {
                    statusView.text =
                        "Autonomous goal creation failed: " +
                            exc::class.java.simpleName
                    autonomousButton.isEnabled =
                        state.phase == VN97AppPhase.READY &&
                            state.inputEnabled
                    sendButton.isEnabled = state.inputEnabled
                    refreshAutonomousStatus()
                }
            }
        }
    }

    private fun cancelLatestAutonomousGoal() {
        val jobId = latestCancellableAutonomousJobId ?: return
        cancelAutonomousButton.isEnabled = false
        worker.execute {
            try {
                val record = app.autonomousWork.cancel(jobId)
                runOnUiThread {
                    statusView.text =
                        "Autonomous job " +
                            record.jobId +
                            " cancelled."
                    refreshAutonomousStatus()
                }
            } catch (exc: Throwable) {
                runOnUiThread {
                    statusView.text =
                        "Autonomous cancellation failed: " +
                            exc::class.java.simpleName
                    refreshAutonomousStatus()
                }
            }
        }
    }

    private fun refreshAutonomousStatus() {
        worker.execute {
            val result = runCatching {
                val records = app.autonomousWork.listGoals()
                val approval =
                    app.autonomousWork.pendingApprovalRequest()
                records to approval
            }
            runOnUiThread {
                result.onSuccess { (records, autonomousApproval) ->
                    val visible = records.take(5)
                    latestCancellableAutonomousJobId =
                        records.firstOrNull { !it.terminal }?.jobId
                    autonomousStatusView.text =
                        if (visible.isEmpty()) {
                            "Autonomous goals: none"
                        } else {
                            buildString {
                                append("Autonomous goals\n")
                                visible.forEach { record ->
                                    append("#")
                                    append(record.jobId)
                                    append(" ")
                                    append(record.state.name)
                                    append(" gen=")
                                    append(record.generation)
                                    append(" wakes=")
                                    append(record.wakeCount)
                                    append("\n")
                                    append(
                                        record.goal
                                            .replace('\n', ' ')
                                            .take(160)
                                    )
                                    val detail = when {
                                        record.finalResponse.isNotBlank() ->
                                            record.finalResponse
                                        record.terminalReason.isNotBlank() ->
                                            record.terminalReason
                                        else -> ""
                                    }
                                    if (detail.isNotBlank()) {
                                        append("\n↳ ")
                                        append(
                                            detail
                                                .replace('\n', ' ')
                                                .take(220)
                                        )
                                    }
                                    append("\n")
                                }
                            }.trimEnd()
                        }
                    cancelAutonomousButton.isEnabled =
                        latestCancellableAutonomousJobId != null
                    autonomousButton.isEnabled =
                        state.phase == VN97AppPhase.READY &&
                            state.inputEnabled
                    autonomousApprovalActive =
                        autonomousApproval != null
                    if (autonomousApproval == null) {
                        autonomousApprovalView.text = ""
                        autonomousApprovalView.visibility = View.GONE
                        autonomousApproveButton.visibility = View.GONE
                        autonomousRejectButton.visibility = View.GONE
                        autonomousApproveButton.isEnabled = false
                        autonomousRejectButton.isEnabled = false
                    } else {
                        autonomousApprovalView.text = buildString {
                            append("Autonomous approval • job #")
                            append(autonomousApproval.jobId)
                            append(" • gen ")
                            append(autonomousApproval.generation)
                            append("\ncapability=")
                            append(autonomousApproval.capabilityId)
                            append("\nscope=")
                            append(autonomousApproval.scopeDigest)
                            append("\n")
                            append(autonomousApproval.presentationJson)
                        }
                        autonomousApprovalView.visibility = View.VISIBLE
                        autonomousApproveButton.visibility = View.VISIBLE
                        autonomousRejectButton.visibility = View.VISIBLE
                        autonomousApproveButton.isEnabled = true
                        autonomousRejectButton.isEnabled = true
                    }
                    sendButton.isEnabled =
                        state.inputEnabled &&
                            !autonomousApprovalActive
                    autonomousButton.isEnabled =
                        state.phase == VN97AppPhase.READY &&
                            state.inputEnabled &&
                            !autonomousApprovalActive
                    refreshVisualButtons()
                }.onFailure { exc ->
                    autonomousStatusView.text =
                        "Autonomous status unavailable: " +
                            exc::class.java.simpleName
                    latestCancellableAutonomousJobId = null
                    cancelAutonomousButton.isEnabled = false
                    autonomousApprovalActive = false
                    autonomousApprovalView.text = ""
                    autonomousApprovalView.visibility = View.GONE
                    autonomousApproveButton.visibility = View.GONE
                    autonomousRejectButton.visibility = View.GONE
                    autonomousApproveButton.isEnabled = false
                    autonomousRejectButton.isEnabled = false
                    sendButton.isEnabled = state.inputEnabled
                    autonomousButton.isEnabled =
                        state.phase == VN97AppPhase.READY &&
                            state.inputEnabled
                    refreshVisualButtons()
                }
            }
        }
    }

    private fun resolveAutonomousApproval(
        approved: Boolean,
    ) {
        autonomousApproveButton.isEnabled = false
        autonomousRejectButton.isEnabled = false
        statusView.text =
            if (approved) {
                "Applying autonomous action approval…"
            } else {
                "Rejecting autonomous action…"
            }
        worker.execute {
            try {
                val record =
                    app.autonomousWork.resolvePendingApproval(
                        approved
                    )
                runOnUiThread {
                    statusView.text =
                        "Autonomous job #" +
                            record.jobId +
                            " is " +
                            record.state.name.lowercase() +
                            "."
                    refreshAutonomousStatus()
                }
            } catch (exc: Throwable) {
                runOnUiThread {
                    statusView.text =
                        "Autonomous approval failed: " +
                            exc::class.java.simpleName
                    refreshAutonomousStatus()
                }
            }
        }
    }

    private fun submitTurn() {
        if (autonomousApprovalActive) {
            statusView.text =
                "Resolve the pending autonomous approval first."
            return
        }
        if (!state.inputEnabled) return
        val message = inputView.text.toString()
        if (message.isBlank()) return
        inputView.text.clear()
        render(VN97AppReducer.reduce(state, VN97AppEvent.TurnStarted))

        worker.execute {
            try {
                val result = app.assistant.runTurn(message)
                runOnUiThread { applyTurnResult(result) }
            } catch (exc: Throwable) {
                runOnUiThread {
                    renderFailure("VN97 turn failed: " + exc::class.java.simpleName)
                }
            }
        }
    }

    private fun resolveApproval(approved: Boolean) {
        if (state.phase != VN97AppPhase.WAITING_APPROVAL) return
        render(
            state.copy(
                status = if (approved) {
                    "Applying explicit approval…"
                } else {
                    "Rejecting external action…"
                },
            )
        )
        worker.execute {
            try {
                if (app.visualActions.hasPendingVisualApproval()) {
                    val visual =
                        app.visualActions.resolvePendingApproval(
                            approved
                        )
                    runOnUiThread {
                        applyVisualResult(visual)
                    }
                } else {
                    val result =
                        app.assistant.resolvePendingApproval(approved)
                    runOnUiThread {
                        applyTurnResult(result)
                    }
                }
            } catch (exc: Throwable) {
                runOnUiThread {
                    renderFailure(
                        "Approval resolution failed: " +
                            exc::class.java.simpleName
                    )
                }
            }
        }
    }

    private fun applyTurnResult(result: VN97AppTurnResult) {
        when (result.update.state) {
            VN97AssistantTurnState.COMPLETED -> render(
                VN97AppReducer.reduce(
                    state,
                    VN97AppEvent.TurnCompleted(
                        user = result.userMessage,
                        assistant = result.update.finalResponse,
                    ),
                )
            )

            VN97AssistantTurnState.APPROVAL_REQUIRED -> render(
                VN97AppReducer.reduce(
                    state,
                    VN97AppEvent.ApprovalRequired,
                )
            )

            VN97AssistantTurnState.APPROVAL_REJECTED -> render(
                VN97AppReducer.reduce(
                    state,
                    VN97AppEvent.ApprovalRejected(result.userMessage),
                )
            )

            else -> renderFailure(
                "VN97 turn ended at ${result.update.state.name}."
            )
        }
    }

    private fun renderFailure(message: String) {
        render(
            VN97AppReducer.reduce(
                state,
                VN97AppEvent.Failed(message),
            )
        )
    }

    private fun render(next: VN97AppState) {
        state = next
        statusView.text = state.status
        inputView.isEnabled = state.inputEnabled
        sendButton.isEnabled =
            state.inputEnabled &&
                !autonomousApprovalActive
        autonomousButton.isEnabled =
            state.phase == VN97AppPhase.READY &&
                state.inputEnabled &&
                !autonomousApprovalActive
        transcriptView.text = state.transcript.joinToString("\n\n")

        val approval = if (state.phase == VN97AppPhase.WAITING_APPROVAL) {
            app.assistant.pendingApproval()
        } else {
            null
        }
        approvalView.text = approval?.presentationJson.orEmpty()
        val approvalVisibility =
            if (approval == null) View.GONE else View.VISIBLE
        approvalView.visibility = approvalVisibility
        approveButton.visibility = approvalVisibility
        rejectButton.visibility = approvalVisibility
        approveButton.isEnabled = approval != null
        rejectButton.isEnabled = approval != null
        setProvisioningControlsEnabled(true)
        val pendingReview = app.provisioner.pendingReview()
        if (pendingReview != null) {
            advancedProvisioningContainer.visibility = View.VISIBLE
            renderProvisioningReview(pendingReview)
            activateModelButton.isEnabled = provisioningAllowed()
        }

        val mode = when (state.phase) {
            VN97AppPhase.MODEL_REQUIRED -> AssistantMode.SLEEPING
            VN97AppPhase.READY -> AssistantMode.IDLE
            VN97AppPhase.RUNNING -> AssistantMode.THINKING
            VN97AppPhase.WAITING_APPROVAL -> AssistantMode.WAITING_APPROVAL
            VN97AppPhase.ERROR -> AssistantMode.ERROR
        }
        refreshVisualButtons()

        avatar.publish(
            AvatarCommand(
                sourceSequence = avatarSequence++,
                mode = mode,
                gesture = if (state.phase == VN97AppPhase.ERROR) {
                    AvatarGesture.ALERT
                } else {
                    AvatarGesture.NONE
                },
                energy = if (state.phase == VN97AppPhase.MODEL_REQUIRED) 0.15f else 0.5f,
            )
        )
    }

    companion object {
        private const val REQUEST_PACKAGE = 4101
        private const val REQUEST_SIGNATURE = 4102
        private const val REQUEST_PUBLISHER_KEY = 4103
        private const val REQUEST_MICROPHONE_PERMISSION = 4201
        private const val REQUEST_AUTONOMOUS_NOTIFICATION_PERMISSION = 4202
        private const val REQUEST_SCREEN_CAPTURE = 4301
        private const val REQUEST_CAMERA_PERMISSION = 4302
    }
}
