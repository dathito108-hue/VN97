package ai.vn97.app

import ai.vn97.avatar.AssistantMode
import ai.vn97.avatar.AvatarCommand
import ai.vn97.avatar.AvatarGesture
import ai.vn97.avatar.VN97AvatarView
import ai.vn97.platform.VN97AssistantTurnState
import ai.vn97.platform.VN97GameAccessibilityController
import ai.vn97.platform.VN97MobileEvidenceConfig
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
import android.widget.TextView
import java.io.File
import java.io.FileOutputStream
import java.nio.charset.StandardCharsets
import java.nio.file.Files
import java.nio.file.LinkOption
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
    private lateinit var gameControlStatusView: TextView
    private lateinit var gameAccessibilityButton: Button
    private lateinit var gameAuthorizeButton: Button
    private lateinit var gameRevokeButton: Button
    private lateinit var gameAgentStartButton: Button
    private lateinit var gameAgentStopButton: Button
    private lateinit var mobileEvidenceButton: Button
    private lateinit var r2OrtEvidenceButton: Button
    private lateinit var paperTradingButton: Button
    private lateinit var capabilityAcquisitionButton: Button
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

        avatar = VN97AvatarView(this).apply {
            enableTransparentOverlaySurface()
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

        statusView = TextView(this)

        floatingAssistantButton = Button(this).apply {
            setOnClickListener { toggleFloatingAssistant() }
        }

        voicePermissionButton = Button(this).apply {
            setOnClickListener { requestMicrophonePermissionIfNeeded() }
        }

        screenShareButton = Button(this).apply {
            setOnClickListener { toggleScreenPerception() }
        }

        screenAnalyzeButton = Button(this).apply {
            text = "Phân tích màn hình"
            setOnClickListener { analyzeScreenPerception() }
        }

        cameraAnalyzeButton = Button(this).apply {
            text = "Phân tích camera"
            setOnClickListener { analyzeCameraPerception() }
        }

        gameControlStatusView = TextView(this).apply {
            setTextIsSelectable(true)
        }

        gameAccessibilityButton = Button(this).apply {
            text = "Game accessibility"
            setOnClickListener { openGameAccessibilitySettings() }
        }
        gameAuthorizeButton = Button(this).apply {
            text = "Cho phép ứng dụng gần nhất"
            setOnClickListener { authorizeLastGamePackage() }
        }
        gameRevokeButton = Button(this).apply {
            text = "Dừng điều khiển"
            setOnClickListener { revokeGameControl() }
        }

        gameAgentStartButton = Button(this).apply {
            text = "Bắt đầu chơi"
            setOnClickListener { startGameAgent() }
        }
        gameAgentStopButton = Button(this).apply {
            text = "Dừng chơi"
            setOnClickListener { stopGameAgent() }
        }

        mobileEvidenceButton = Button(this).apply {
            text = "Kiểm tra thiết bị"
            visibility =
                if (BuildConfig.VN97_TURNKEY_REQUIRED) View.GONE else View.VISIBLE
            setOnClickListener { collectMobileEvidence() }
        }

        r2OrtEvidenceButton = Button(this).apply {
            text = "Đo hiệu năng mô hình"
            visibility =
                if (BuildConfig.VN97_TURNKEY_REQUIRED) View.GONE else View.VISIBLE
            setOnClickListener {
                startActivity(
                    Intent(
                        this@VN97MainActivity,
                        VN97R2OrtEvidenceActivity::class.java,
                    )
                )
            }
        }

        val int8ImportButton = Button(this).apply {
            text = "Nạp mô hình INT8 (.zip)"
            setOnClickListener {
                startActivity(Intent(this@VN97MainActivity, VN97Int8TrialActivity::class.java)
                    .putExtra("open_model_zip_picker", true))
            }
        }
        val int8TrialButton = Button(this).apply {
            text = "Mở mô hình INT8 đã nạp"
            setOnClickListener {
                startActivity(Intent(this@VN97MainActivity, VN97Int8TrialActivity::class.java))
            }
        }

        val digitalServicesButton = Button(this).apply {
            text = "Dịch vụ số"
            setOnClickListener {
                startActivity(Intent(this@VN97MainActivity, VN97DigitalServicesActivity::class.java))
            }
        }

        paperTradingButton = Button(this).apply {
            text = "Mở giao dịch mô phỏng"
            setOnClickListener {
                startActivity(
                    Intent(
                        this@VN97MainActivity,
                        VN97PaperTradingActivity::class.java,
                    )
                )
            }
        }

        capabilityAcquisitionButton = Button(this).apply {
            text = "Quản lý năng lực"
            setOnClickListener {
                startActivity(
                    Intent(
                        this@VN97MainActivity,
                        VN97CapabilityAcquisitionActivity::class.java,
                    )
                )
            }
        }

        approvalView = TextView(this).apply {
            visibility = View.GONE
            setTextIsSelectable(true)
        }

        rejectButton = Button(this).apply {
            text = "Từ chối"
            visibility = View.GONE
            setOnClickListener { resolveApproval(false) }
        }
        approveButton = Button(this).apply {
            text = "Phê duyệt"
            visibility = View.GONE
            setOnClickListener { resolveApproval(true) }
        }

        provisioningView = TextView(this).apply {
            text = if (BuildConfig.VN97_TURNKEY_REQUIRED) {
                "Preparing bundled VN97 intelligence…"
            } else {
                "No active VN97 model. A signed bundled model will activate automatically when present."
            }
            setTextIsSelectable(true)
        }

        transcriptView = TextView(this).apply {
            textSize = 16f
        }

        val inputRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
        }
        inputView = EditText(this).apply {
            hint = "Nhắn tin hoặc nhập mục tiêu…"
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
            text = "Gửi"
            setOnClickListener {
                if (state.phase in setOf(VN97AppPhase.YIELDED, VN97AppPhase.PAUSED)) resumeYieldedTurn() else submitTurn()
            }
        }
        inputRow.addView(
            sendButton,
            LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.WRAP_CONTENT,
                ViewGroup.LayoutParams.WRAP_CONTENT,
            ),
        )

        autonomousButton = Button(this).apply {
            text = "Bắt đầu nhiệm vụ"
            setOnClickListener { startAutonomousGoal() }
        }
        cancelAutonomousButton = Button(this).apply {
            text = "Dừng nhiệm vụ gần nhất"
            isEnabled = false
            setOnClickListener { cancelLatestAutonomousGoal() }
        }

        autonomousStatusView = TextView(this).apply {
            text = "Autonomous goals: none"
            setTextIsSelectable(true)
        }

        autonomousApprovalView = TextView(this).apply {
            visibility = View.GONE
            setTextIsSelectable(true)
        }
        autonomousRejectButton = Button(this).apply {
            text = "Từ chối hành động tự chủ"
            visibility = View.GONE
            setOnClickListener {
                resolveAutonomousApproval(false)
            }
        }
        autonomousApproveButton = Button(this).apply {
            text = "Phê duyệt hành động tự chủ"
            visibility = View.GONE
            setOnClickListener {
                resolveAutonomousApproval(true)
            }
        }

        // Reuse the existing controls and their authority/runtime listeners.
        val dashboard = VN97Dashboard(this)
        setContentView(dashboard.build(
            avatar = avatar,
            status = statusView,
            composer = inputRow,
            approvals = listOf(approvalView, rejectButton, approveButton,
                autonomousApprovalView, autonomousRejectButton, autonomousApproveButton),
            pages = listOf(
                listOf(
                    dashboard.card("Nạp mô hình từ tệp ZIP", "Thử nghiệm INT8: chọn ZIP mô hình, sau đó nạp tokenizer G08 và thử viết tiếp văn bản.",
                        int8ImportButton, int8TrialButton).apply {
                            visibility = if (BuildConfig.VN97_TURNKEY_REQUIRED) View.GONE else View.VISIBLE
                        },
                    dashboard.card("Trò chuyện", "Trao đổi và giao nhiệm vụ cho VN97.", transcriptView),
                    dashboard.card("Công việc tự chủ", "Nhập mục tiêu trong ô tin nhắn rồi bắt đầu.",
                        autonomousButton, cancelAutonomousButton, autonomousStatusView),
                    dashboard.card("Dịch vụ số", "Xử lý CSV, văn bản và danh mục bằng công cụ có sẵn.", digitalServicesButton),
                ),
                listOf(
                    dashboard.card("Trợ lý & nhận thức", "Nút bị mờ khi mô hình hoặc quyền cần thiết chưa sẵn sàng.",
                        floatingAssistantButton, voicePermissionButton, screenShareButton,
                        screenAnalyzeButton, cameraAnalyzeButton),
                    dashboard.card("Điều khiển trò chơi", "Cấp quyền cho ứng dụng đích trước khi chạy.",
                        gameAccessibilityButton, gameAuthorizeButton, gameRevokeButton,
                        gameAgentStartButton, gameAgentStopButton, gameControlStatusView),
                    dashboard.card("Giao dịch mô phỏng", "Theo dõi thử nghiệm bằng tiền mô phỏng.", paperTradingButton),
                ),
                listOf(
                    dashboard.card("Mô hình VN97", "Nạp ZIP INT8 thử nghiệm tại thẻ Nạp mô hình trên trang chính. Lõi chính thức: G06. Cần bản phân phối có runtime ONNX, tokenizer và bằng chứng tương thích. Nhập CAP/MI1 và checkpoint fast/slow đã ngừng; không dùng các nút nhập cũ để thay lõi G06.",
                        provisioningView),
                    dashboard.card("Phát triển năng lực", "Quản lý năng lực và các bản cải tiến có kiểm soát.",
                        capabilityAcquisitionButton),
                    dashboard.card("Chẩn đoán", "Thu thập bằng chứng chạy thực tế trên thiết bị.",
                        mobileEvidenceButton, r2OrtEvidenceButton),
                ),
            ),
        ))
        refreshFloatingAssistantButton()
        refreshVoicePermissionButton()
        refreshGameControlStatus()
        refreshVisualButtons()
        render(state)
        refreshAutonomousStatus()
        attachTrustedModel()
        if (
            intent?.action ==
                VN97FloatingAssistantService.ACTION_REQUEST_MICROPHONE_PERMISSION
        ) {
            requestMicrophonePermissionIfNeeded()
        }
        handleM19JDeveloperEvidenceIntent(intent)
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
        refreshGameControlStatus()
        refreshVisualButtons()
        refreshAutonomousStatus()
        if (!app.assistant.isOpen()) {
            attachTrustedModel()
        }
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
        handleM19JDeveloperEvidenceIntent(intent)
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

    private fun openGameAccessibilitySettings() {
        startActivity(
            Intent(Settings.ACTION_ACCESSIBILITY_SETTINGS)
        )
    }

    private fun authorizeLastGamePackage() {
        if (!VN97GameAccessibilityController.isConnected()) {
            statusView.text =
                "Enable VN97 Game Control in Android accessibility settings first."
            openGameAccessibilitySettings()
            return
        }
        val packageName =
            VN97GameAccessibilityController
                .lastExternalPackageName()
        if (packageName.isNullOrBlank() || packageName == this.packageName) {
            statusView.text =
                "Open the target game, then return to VN97 before authorizing it."
            refreshGameControlStatus()
            return
        }

        gameAuthorizeButton.isEnabled = false
        gameRevokeButton.isEnabled = false
        statusView.text =
            "Authorizing exact-package game control for $packageName…"
        worker.execute {
            try {
                val session =
                    app.platformRuntime.gameControlPolicy.authorize(
                        packageName
                    )
                val refreshed =
                    if (state.phase == VN97AppPhase.READY) {
                        app.assistant.reloadActivatedModel()
                    } else {
                        false
                    }
                runOnUiThread {
                    statusView.text =
                        "Game control authorized for " +
                            session.packageName +
                            " until " +
                            session.expiresAtWallTimeMillis +
                            if (refreshed) {
                                ". VN97 authority grants refreshed."
                            } else {
                                "."
                            }
                    refreshGameControlStatus()
                }
            } catch (exc: Throwable) {
                runOnUiThread {
                    statusView.text =
                        "Game authorization failed: " +
                            exc::class.java.simpleName
                    refreshGameControlStatus()
                }
            }
        }
    }

    private fun revokeGameControl() {
        gameAuthorizeButton.isEnabled = false
        gameRevokeButton.isEnabled = false
        worker.execute {
            try {
                app.platformRuntime.gameControlPolicy.revoke()
                if (state.phase == VN97AppPhase.READY) {
                    runCatching {
                        app.assistant.reloadActivatedModel()
                    }
                }
                runOnUiThread {
                    statusView.text =
                        "Game control authorization revoked."
                    refreshGameControlStatus()
                }
            } catch (exc: Throwable) {
                runOnUiThread {
                    statusView.text =
                        "Game control revoke failed: " +
                            exc::class.java.simpleName
                    refreshGameControlStatus()
                }
            }
        }
    }

    private fun startGameAgent() {
        if (
            state.phase != VN97AppPhase.READY ||
            !state.inputEnabled ||
            autonomousApprovalActive
        ) {
            statusView.text =
                "VN97 must be READY with no pending approval before game-agent start."
            return
        }
        val activeSession = runCatching {
            app.platformRuntime.gameControlPolicy
                .activeSessionOrNull()
        }.getOrNull()
        if (activeSession == null) {
            statusView.text =
                "Authorize the exact target game package first."
            refreshGameControlStatus()
            return
        }
        if (!VN97GameAccessibilityController.isConnected()) {
            statusView.text =
                "Enable VN97 Game Control accessibility first."
            return
        }
        if (!app.screenCaptureBroker.isActive()) {
            statusView.text =
                "Start user-approved screen sharing before the game agent."
            return
        }
        if (!app.assistant.hasProductionVision()) {
            statusView.text =
                "The activated VN97 model has no production vision weights."
            return
        }
        val goal = inputView.text.toString().trim()
        if (goal.isEmpty()) {
            statusView.text =
                "Enter the game episode goal before starting the agent."
            return
        }
        if (
            goal.toByteArray(Charsets.UTF_8).size >
                32 * 1024
        ) {
            statusView.text = "Game episode goal is too large."
            return
        }
        inputView.text.clear()
        VN97GameAgentService.start(this, goal)
        statusView.text =
            "Game agent started for " +
                activeSession.packageName +
                ". Open that game within 30 seconds."
        refreshGameControlStatus()
    }

    private fun stopGameAgent() {
        VN97GameAgentService.stop(this)
        statusView.text = "Stopping the VN97 game agent…"
        refreshGameControlStatus()
    }

    private fun refreshGameControlStatus() {
        val connected =
            VN97GameAccessibilityController.isConnected()
        val lastExternal =
            VN97GameAccessibilityController
                .lastExternalPackageName()
        val active =
            runCatching {
                app.platformRuntime.gameControlPolicy
                    .activeSessionOrNull()
            }.getOrNull()

        gameControlStatusView.text = buildString {
            append("Game control: ")
            append(
                when {
                    active != null -> "authorized"
                    connected -> "accessibility ready"
                    else -> "accessibility disabled"
                }
            )
            if (active != null) {
                append("\npackage=")
                append(active.packageName)
                append("\nexpires_ms=")
                append(active.expiresAtWallTimeMillis)
            }
            if (!lastExternal.isNullOrBlank()) {
                append("\nlast external app=")
                append(lastExternal)
            }
            append("\ncontrols=tap/swipe/multitouch/back")
            append(
                "\nScreen sharing must remain user-approved for visual game perception."
            )
        }
        gameAccessibilityButton.text =
            if (connected) {
                "Đã bật quyền trợ năng"
            } else {
                "Mở cài đặt trợ năng"
            }
        val agent = VN97GameAgentService.snapshot()
        gameControlStatusView.append(
            buildString {
                append("\nagent=")
                append(agent.state.name.lowercase())
                append(" actions=")
                append(agent.actionCount)
                if (agent.detail.isNotBlank()) {
                    append("\nagent_detail=")
                    append(
                        agent.detail
                            .replace('\n', ' ')
                            .take(512)
                    )
                }
            }
        )
        val agentActive =
            agent.state == VN97GameAgentState.WAITING_FOR_GAME ||
                agent.state == VN97GameAgentState.RUNNING
        gameAuthorizeButton.isEnabled =
            connected &&
                !lastExternal.isNullOrBlank() &&
                !agentActive
        gameRevokeButton.isEnabled =
            active != null && !agentActive
        gameAgentStartButton.isEnabled =
            active != null &&
                connected &&
                app.screenCaptureBroker.isActive() &&
                !agentActive
        gameAgentStopButton.isEnabled = agentActive
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
                "Đã cấp quyền micro"
            } else {
                "Cấp quyền micro"
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
                "Dừng chia sẻ màn hình"
            } else {
                "Chia sẻ màn hình"
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
            enabled -> "Ẩn trợ lý nổi"
            permissionGranted -> "Hiện trợ lý nổi"
            else -> "Cấp quyền trợ lý nổi"
        }
    }

    private fun attachTrustedModel() {
        worker.execute {
            try {
                app.mobileRecovery
                    .recover(
                        VN97MobileRecoveryTrigger
                            .EXECUTION_ENTRY
                    )
                    .requireActivationReady()
                app.runtimeResources
                    .requireRunnable(
                        VN97RuntimeExecutionClass
                            .INTERACTIVE
                    )
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
                        var next =
                            if (
                                state.phase ==
                                    VN97AppPhase
                                        .MODEL_REQUIRED ||
                                state.phase ==
                                    VN97AppPhase.ERROR
                            ) {
                                VN97AppReducer.reduce(
                                    state,
                                    VN97AppEvent
                                        .TrustedModelActivated,
                                )
                            } else {
                                state
                            }
                        if (
                            next.phase ==
                                VN97AppPhase.READY &&
                            app.assistant
                                .pendingApproval() !=
                                null
                        ) {
                            next = VN97AppReducer.reduce(
                                next,
                                VN97AppEvent
                                    .ApprovalRequired,
                            )
                        }
                        if (next.phase == VN97AppPhase.READY && app.assistant.hasYieldedTurn()) {
                            next = VN97AppReducer.reduce(next, if (app.assistant.hasPausedTurn()) VN97AppEvent.TurnPaused else VN97AppEvent.TurnYielded)
                        }
                        render(next)
                        refreshVisualButtons()
                        provisioningView.text =
                            if (bundled) {
                                "Bundled signed VN97 model verified and activated."
                            } else {
                                "VN97 model active."
                            }
                    } else {
                        render(state)
                        provisioningView.text =
                            "Chưa có bộ G06 đã kiểm chứng, nên chat chưa hoạt động. Cần runtime, tokenizer và hồ sơ tương thích G06; CAP/MI1 và fast/slow không còn được dùng. Các công cụ dịch vụ số vẫn hoạt động."
                    }
                }
            } catch (exc: Throwable) {
                runOnUiThread {
                    renderFailure(
                        "Không mở được G06: " + (exc.message ?: exc::class.java.simpleName).take(500)
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
        // Model deployment is G06-only and is installed by the signed APK path.
        // Legacy CAP/MI1 document selection was deliberately removed.
    }

    private fun handleM19JDeveloperEvidenceIntent(
        request: Intent?,
    ) {
        if (
            request?.action !=
                ACTION_M19J_COLLECT_MOBILE_EVIDENCE
        ) {
            return
        }
        if (BuildConfig.VN97_TURNKEY_REQUIRED) {
            statusView.text =
                "M19J developer evidence action is disabled in turnkey builds."
            return
        }

        val warmupRuns =
            request.getIntExtra(
                EXTRA_M19J_WARMUP_RUNS,
                1,
            )
        val measuredRuns =
            request.getIntExtra(
                EXTRA_M19J_MEASURED_RUNS,
                5,
            )
        val decodeTokens =
            request.getIntExtra(
                EXTRA_M19J_DECODE_TOKENS,
                16,
            )
        if (
            warmupRuns !in 0..10 ||
            measuredRuns !in 3..100 ||
            decodeTokens !in 1..256
        ) {
            statusView.text =
                "M19J developer evidence parameters are outside safe bounds."
            return
        }

        mobileEvidenceButton.isEnabled = false
        sendButton.isEnabled = false
        statusView.text =
            "M19J verifying the active G06 deployment and collecting physical-device evidence…"

        worker.execute {
            val stagingRoot =
                File(
                    filesDir,
                    M19J_STAGING_DIRECTORY,
                )
            val evidenceTarget =
                File(
                    stagingRoot,
                    M19J_EVIDENCE_FILE,
                )
            val errorTarget =
                File(
                    stagingRoot,
                    M19J_ERROR_FILE,
                )
            try {
                requireM19JOutputRoot(stagingRoot)
                evidenceTarget.delete()
                errorTarget.delete()

                check(app.assistant.openIfActivated()) {
                    "M19J requires the signed APK G06 deployment to be installed and valid"
                }

                val evidence =
                    app.assistant
                        .collectMobileEvidence(
                            VN97MobileEvidenceConfig(
                                warmupRuns =
                                    warmupRuns,
                                measuredRuns =
                                    measuredRuns,
                                decodeTokens =
                                    decodeTokens,
                            )
                        )
                val bytes =
                    evidence
                        .toCanonicalJson()
                        .toByteArray(
                            StandardCharsets.UTF_8
                        )
                publishM19JBytes(
                    evidenceTarget,
                    bytes,
                )
                runOnUiThread {
                    statusView.text =
                        "M19J physical evidence ready. Model SHA-256: " +
                            evidence.modelImageSha256
                    mobileEvidenceButton.isEnabled =
                        true
                    sendButton.isEnabled =
                        state.inputEnabled
                }
            } catch (exc: Throwable) {
                runCatching {
                    val detail =
                        (
                            exc::class.java
                                .simpleName +
                                ":" +
                                (
                                    exc.message
                                        ?: "unknown"
                                )
                        )
                            .replace(
                                Regex("[\\u0000-\\u001f\\u007f]"),
                                " ",
                            )
                            .take(1024)
                            .toByteArray(
                                StandardCharsets.UTF_8
                            )
                    publishM19JBytes(
                        errorTarget,
                        detail,
                    )
                }
                runOnUiThread {
                    statusView.text =
                        "M19J evidence failed: " +
                            exc::class.java.simpleName
                    mobileEvidenceButton.isEnabled =
                        true
                    sendButton.isEnabled =
                        state.inputEnabled
                }
            }
        }
    }

    private fun requireM19JOutputRoot(root: File) {
        if (!root.exists()) {
            check(root.mkdirs()) {
                "M19J could not create its app-private evidence directory"
            }
        }
        val rootPath =
            root.toPath()
        check(
            Files.isDirectory(
                rootPath,
                LinkOption.NOFOLLOW_LINKS,
            ) &&
                !Files.isSymbolicLink(
                    rootPath
                )
        ) {
            "M19J evidence root must be a real app-private directory"
        }
    }

    private fun publishM19JBytes(
        target: File,
        bytes: ByteArray,
    ) {
        val parent =
            checkNotNull(
                target.parentFile
            )
        check(
            Files.isDirectory(
                parent.toPath(),
                LinkOption.NOFOLLOW_LINKS,
            ) &&
                !Files.isSymbolicLink(
                    parent.toPath()
                )
        ) {
            "M19J output parent must be a real directory"
        }
        val temp =
            File(
                parent,
                "." +
                    target.name +
                    ".tmp",
            )
        try {
            FileOutputStream(
                temp,
                false,
            ).use {
                output ->
                output.write(bytes)
                output.flush()
                output.fd.sync()
            }
            Files.move(
                temp.toPath(),
                target.toPath(),
                StandardCopyOption.REPLACE_EXISTING,
            )
            if (
                !target
                    .readBytes()
                    .contentEquals(bytes)
            ) {
                throw IllegalStateException(
                    "M19J output post-write verification failed"
                )
            }
        } finally {
            temp.delete()
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
                            "G06 deployment SHA-256: ${evidence.modelImageSha256}\n" +
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

    private fun resumeYieldedTurn() {
        if (state.phase !in setOf(VN97AppPhase.YIELDED, VN97AppPhase.PAUSED) || autonomousApprovalActive) return
        render(VN97AppReducer.reduce(state, VN97AppEvent.TurnResumed))
        worker.execute {
            try {
                val result = app.assistant.continueYieldedTurn()
                runOnUiThread { if (!isDestroyed) applyTurnResult(result) }
            } catch (failure: Throwable) {
                runOnUiThread {
                    if (!isDestroyed) {
                        if (app.assistant.hasYieldedTurn()) {
                            render(VN97AppReducer.reduce(state, if (app.assistant.hasPausedTurn()) VN97AppEvent.TurnPaused else VN97AppEvent.TurnYielded))
                            statusView.text = "Chưa thể tiếp tục: " + failure::class.java.simpleName +
                                ". Lượt đang dở vẫn được giữ; có thể thử lại."
                        } else renderFailure("VN97 continuation failed: " + failure::class.java.simpleName)
                    }
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
            VN97AssistantTurnState.PAUSED -> render(
                VN97AppReducer.reduce(state, VN97AppEvent.TurnPaused)
            )

            VN97AssistantTurnState.YIELDED -> render(
                VN97AppReducer.reduce(state, VN97AppEvent.TurnYielded)
            )

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
        sendButton.text = if (state.phase in setOf(VN97AppPhase.YIELDED, VN97AppPhase.PAUSED)) "Tiếp tục" else "Gửi"
        sendButton.isEnabled =
            (state.inputEnabled || state.phase in setOf(VN97AppPhase.YIELDED, VN97AppPhase.PAUSED)) &&
                !autonomousApprovalActive
        autonomousButton.isEnabled =
            state.phase == VN97AppPhase.READY &&
                state.inputEnabled &&
                !autonomousApprovalActive
        transcriptView.text = state.transcript.joinToString("\n\n").ifBlank { "Cuộc trò chuyện sẽ hiển thị tại đây.\nNếu chưa có mô hình hoạt động, mở mục Hệ thống để xem trạng thái." }

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
        val mode = when (state.phase) {
            VN97AppPhase.MODEL_REQUIRED -> AssistantMode.SLEEPING
            VN97AppPhase.READY -> AssistantMode.IDLE
            VN97AppPhase.RUNNING -> AssistantMode.THINKING
            VN97AppPhase.YIELDED, VN97AppPhase.PAUSED -> AssistantMode.IDLE
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
        const val ACTION_M19J_COLLECT_MOBILE_EVIDENCE =
            "ai.vn97.app.action.M19J_COLLECT_MOBILE_EVIDENCE"
        const val EXTRA_M19J_WARMUP_RUNS =
            "ai.vn97.app.extra.M19J_WARMUP_RUNS"
        const val EXTRA_M19J_MEASURED_RUNS =
            "ai.vn97.app.extra.M19J_MEASURED_RUNS"
        const val EXTRA_M19J_DECODE_TOKENS =
            "ai.vn97.app.extra.M19J_DECODE_TOKENS"
        const val M19J_STAGING_DIRECTORY =
            "m19j-evidence"
        const val M19J_EVIDENCE_FILE =
            "vn97-mobile-evidence.json"
        const val M19J_ERROR_FILE =
            "vn97-mobile-evidence.error.txt"

        private const val REQUEST_MICROPHONE_PERMISSION = 4201
        private const val REQUEST_AUTONOMOUS_NOTIFICATION_PERMISSION = 4202
        private const val REQUEST_SCREEN_CAPTURE = 4301
        private const val REQUEST_CAMERA_PERMISSION = 4302
    }
}
