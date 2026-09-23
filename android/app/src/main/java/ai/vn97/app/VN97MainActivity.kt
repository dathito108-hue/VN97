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
    private lateinit var transcriptView: TextView
    private lateinit var inputView: EditText
    private lateinit var sendButton: Button
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

        setContentView(root)
        refreshFloatingAssistantButton()
        refreshVoicePermissionButton()
        refreshVisualButtons()
        render(state)
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
        refreshVisualButtons()
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
        worker.shutdownNow()
        super.onDestroy()
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

        val withVerification =
            if (
                turn.update.state ==
                    VN97AssistantTurnState.COMPLETED &&
                visual.verification != null &&
                visual.afterObservation != null
            ) {
                turn.copy(
                    update = turn.update.copy(
                        finalResponse =
                            turn.update.finalResponse +
                                "\n\nVisual verification: " +
                                visual.verification
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

    private fun submitTurn() {
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
        sendButton.isEnabled = state.inputEnabled
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
        private const val REQUEST_SCREEN_CAPTURE = 4301
        private const val REQUEST_CAMERA_PERMISSION = 4302
    }
}
