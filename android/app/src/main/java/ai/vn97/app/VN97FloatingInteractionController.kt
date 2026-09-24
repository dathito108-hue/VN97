package ai.vn97.app

import ai.vn97.avatar.AssistantMode
import ai.vn97.platform.VN97AssistantTurnState
import ai.vn97.runtime.NativeAudioModality
import android.content.Context
import android.graphics.Color
import android.graphics.PixelFormat
import android.graphics.drawable.GradientDrawable
import android.os.Build
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.view.Gravity
import android.view.View
import android.view.WindowManager
import android.view.inputmethod.InputMethodManager
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import java.util.concurrent.Executors
import kotlin.math.min

internal class VN97FloatingInteractionController(
    private val context: Context,
    private val windowManager: WindowManager,
    private val assistant: VN97AppAssistant,
    private val beforeAssistantOpen: () -> Unit,
    private val anchorProvider: () -> WindowManager.LayoutParams?,
    private val publishMode: (AssistantMode, Float) -> Unit,
    private val publishListeningLevel: (Float, Long) -> Unit,
    private val requestMicrophonePermission: () -> Boolean,
    private val setMicrophoneForegroundActive: (Boolean) -> Boolean,
) : AutoCloseable {
    private val mainHandler = Handler(Looper.getMainLooper())
    private val worker = Executors.newSingleThreadExecutor()

    private var closed = false
    private var panel: LinearLayout? = null
    private var panelLayoutParams: WindowManager.LayoutParams? = null
    private var statusView: TextView? = null
    private var transcriptView: TextView? = null
    private var transcriptScroll: ScrollView? = null
    private var approvalView: TextView? = null
    private var inputView: EditText? = null
    private var voiceButton: Button? = null
    private var sendButton: Button? = null
    private var approveButton: Button? = null
    private var rejectButton: Button? = null

    private var state = VN97AppState(
        status = "Opening trusted VN97 model…",
    )
    private var voiceAvailable = false
    private var lastVoiceLevelMillis = 0L

    private val voiceCapture = VN97VoiceCapture(
        context = context,
        onLevel = { level -> handleVoiceLevel(level) },
        onUtterance = { utterance -> handleVoiceUtterance(utterance) },
        onFailure = { message -> handleVoiceFailure(message) },
    )

    fun initialize() {
        render(state)
        worker.execute {
            try {
                beforeAssistantOpen()
                val active = assistant.openIfActivated()
                val hasApproval = active && assistant.pendingApproval() != null
                val hasVoice = active && assistant.hasProductionVoice()
                mainHandler.post {
                    if (closed) return@post
                    if (!active) {
                        render(
                            VN97AppState(
                                status = "Trusted VN97 model is not active.",
                            )
                        )
                        return@post
                    }

                    voiceAvailable = hasVoice
                    var next = VN97AppReducer.reduce(
                        VN97AppState(),
                        VN97AppEvent.TrustedModelActivated,
                    )
                    if (!hasVoice) {
                        next = next.copy(
                            status =
                                "VN97 native model ready. " +
                                    "Voice requires an M11 speech-enabled VN97MI1 package."
                        )
                    }
                    if (hasApproval) {
                        next = VN97AppReducer.reduce(
                            next,
                            VN97AppEvent.ApprovalRequired,
                        )
                    }
                    render(next)
                }
            } catch (exc: Throwable) {
                mainHandler.post {
                    if (!closed) {
                        renderFailure(
                            "VN97 overlay init failed: " +
                                exc::class.java.simpleName
                        )
                    }
                }
            }
        }
    }

    fun toggle() {
        if (closed) return
        if (panel == null) {
            showPanel()
        } else {
            removePanel()
        }
    }

    fun updatePosition() {
        if (closed) return
        val currentPanel = panel ?: return
        val params = panelLayoutParams ?: return
        positionPanel(params)
        try {
            windowManager.updateViewLayout(currentPanel, params)
        } catch (_: IllegalArgumentException) {
            // Window is already detached.
        }
    }

    private fun showPanel() {
        if (panel != null || closed) return

        val root = LinearLayout(context).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(12), dp(10), dp(12), dp(10))
            background = GradientDrawable().apply {
                shape = GradientDrawable.RECTANGLE
                cornerRadius = dp(18).toFloat()
                setColor(Color.argb(242, 22, 24, 30))
                setStroke(dp(1), Color.argb(210, 92, 107, 132))
            }
        }

        val header = LinearLayout(context).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
        }
        header.addView(
            TextView(context).apply {
                text = "VN97"
                textSize = 18f
                setTextColor(Color.WHITE)
            },
            LinearLayout.LayoutParams(
                0,
                LinearLayout.LayoutParams.WRAP_CONTENT,
                1f,
            ),
        )
        header.addView(
            Button(context).apply {
                text = "×"
                contentDescription = "Close floating VN97 interaction"
                minWidth = 0
                minimumWidth = 0
                setOnClickListener { removePanel() }
            },
            LinearLayout.LayoutParams(
                dp(48),
                LinearLayout.LayoutParams.WRAP_CONTENT,
            ),
        )
        root.addView(header)

        val status = TextView(context).apply {
            textSize = 13f
            setTextColor(Color.LTGRAY)
            setPadding(0, dp(4), 0, dp(6))
        }
        statusView = status
        root.addView(status)

        val transcript = TextView(context).apply {
            textSize = 14f
            setTextColor(Color.WHITE)
            setTextIsSelectable(true)
        }
        transcriptView = transcript
        val scroll = ScrollView(context).apply {
            isFillViewport = true
            addView(transcript)
        }
        transcriptScroll = scroll
        root.addView(
            scroll,
            LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                0,
                1f,
            ),
        )

        val approval = TextView(context).apply {
            textSize = 12f
            setTextColor(Color.rgb(255, 211, 120))
            setTextIsSelectable(true)
            maxLines = 6
            visibility = View.GONE
            setPadding(0, dp(6), 0, dp(4))
        }
        approvalView = approval
        root.addView(approval)

        val approvalActions = LinearLayout(context).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.END
        }
        val reject = Button(context).apply {
            text = "Reject"
            visibility = View.GONE
            setOnClickListener { resolveApproval(false) }
        }
        val approve = Button(context).apply {
            text = "Approve"
            visibility = View.GONE
            setOnClickListener { resolveApproval(true) }
        }
        rejectButton = reject
        approveButton = approve
        approvalActions.addView(
            reject,
            LinearLayout.LayoutParams(
                0,
                LinearLayout.LayoutParams.WRAP_CONTENT,
                1f,
            ),
        )
        approvalActions.addView(
            approve,
            LinearLayout.LayoutParams(
                0,
                LinearLayout.LayoutParams.WRAP_CONTENT,
                1f,
            ),
        )
        root.addView(approvalActions)

        val inputRow = LinearLayout(context).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            setPadding(0, dp(6), 0, 0)
        }
        val input = EditText(context).apply {
            hint = "Ask VN97…"
            maxLines = 3
            setTextColor(Color.WHITE)
            setHintTextColor(Color.GRAY)
        }
        val voice = Button(context).apply {
            text = "Mic"
            contentDescription = "Start or stop local VN97 voice capture"
            setOnClickListener { toggleVoiceCapture() }
        }
        val send = Button(context).apply {
            text = "Send"
            setOnClickListener { submitTurn() }
        }
        inputView = input
        voiceButton = voice
        sendButton = send
        inputRow.addView(
            input,
            LinearLayout.LayoutParams(
                0,
                LinearLayout.LayoutParams.WRAP_CONTENT,
                1f,
            ),
        )
        inputRow.addView(
            voice,
            LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.WRAP_CONTENT,
                LinearLayout.LayoutParams.WRAP_CONTENT,
            ),
        )
        inputRow.addView(
            send,
            LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.WRAP_CONTENT,
                LinearLayout.LayoutParams.WRAP_CONTENT,
            ),
        )
        root.addView(inputRow)

        val params = createPanelLayoutParams()
        panel = root
        panelLayoutParams = params
        positionPanel(params)
        windowManager.addView(root, params)
        render(state)
    }

    private fun removePanel() {
        val currentPanel = panel ?: return
        hideKeyboard(currentPanel)
        panel = null
        panelLayoutParams = null
        statusView = null
        transcriptView = null
        transcriptScroll = null
        approvalView = null
        inputView = null
        voiceButton = null
        sendButton = null
        approveButton = null
        rejectButton = null
        try {
            windowManager.removeView(currentPanel)
        } catch (_: IllegalArgumentException) {
            // Window is already detached.
        }
    }

    private fun toggleVoiceCapture() {
        if (
            closed ||
            state.phase != VN97AppPhase.READY ||
            !voiceAvailable
        ) return

        if (voiceCapture.isRecording) {
            voiceCapture.stop()
            voiceButton?.isEnabled = false
            return
        }

        if (!voiceCapture.hasPermission()) {
            val opened = requestMicrophonePermission()
            render(
                state.copy(
                    status = if (opened) {
                        "Grant microphone permission in VN97, then tap Mic again."
                    } else {
                        "Microphone permission is required for local voice capture."
                    },
                    inputEnabled = true,
                )
            )
            return
        }

        if (!setMicrophoneForegroundActive(true)) {
            render(
                state.copy(
                    status = "Android blocked background microphone access. Open VN97 and try Mic again.",
                    inputEnabled = true,
                )
            )
            return
        }

        lastVoiceLevelMillis = SystemClock.elapsedRealtime()
        render(
            state.copy(
                status = "Listening locally… tap Stop when finished.",
                inputEnabled = false,
            )
        )
        if (!voiceCapture.start()) {
            setMicrophoneForegroundActive(false)
            render(
                state.copy(
                    status = "Microphone capture could not start.",
                    inputEnabled = true,
                )
            )
        }
    }

    private fun handleVoiceLevel(level: Float) {
        if (closed) return
        val now = SystemClock.elapsedRealtime()
        val delta = (now - lastVoiceLevelMillis).coerceIn(0L, 1000L)
        lastVoiceLevelMillis = now
        publishListeningLevel(level, delta)
        voiceButton?.apply {
            text = if (voiceCapture.isRecording) "Stop" else "Mic"
            isEnabled = true
        }
    }

    private fun handleVoiceUtterance(utterance: VN97VoiceUtterance) {
        if (closed) return
        setMicrophoneForegroundActive(false)
        val running = VN97AppReducer.reduce(
            state,
            VN97AppEvent.TurnStarted,
        ).copy(
            status =
                "VN97 is understanding " +
                    utterance.durationMillis +
                    " ms of local voice…"
        )
        render(running)

        worker.execute {
            try {
                val prepared =
                    NativeAudioModality.preparePcm16(utterance.pcm16)
                val result = assistant.runVoiceTurn(prepared)
                mainHandler.post {
                    if (!closed) applyTurnResult(result)
                }
            } catch (exc: Throwable) {
                mainHandler.post {
                    if (!closed) {
                        render(
                            state.copy(
                                phase = VN97AppPhase.READY,
                                status =
                                    "VN97 voice turn failed: " +
                                        exc::class.java.simpleName,
                                inputEnabled = true,
                            )
                        )
                    }
                }
            }
        }
    }

    private fun handleVoiceFailure(message: String) {
        if (closed) return
        setMicrophoneForegroundActive(false)
        render(
            state.copy(
                status = message,
                inputEnabled = state.phase == VN97AppPhase.READY,
            )
        )
    }

    private fun submitTurn() {
        if (!state.inputEnabled || closed) return
        val input = inputView ?: return
        val message = input.text.toString()
        if (message.isBlank()) return

        input.text.clear()
        render(
            VN97AppReducer.reduce(
                state,
                VN97AppEvent.TurnStarted,
            )
        )

        worker.execute {
            try {
                val result = assistant.runTurn(message)
                mainHandler.post {
                    if (!closed) applyTurnResult(result)
                }
            } catch (exc: Throwable) {
                mainHandler.post {
                    if (!closed) {
                        renderFailure(
                            "VN97 turn failed: " +
                                exc::class.java.simpleName
                        )
                    }
                }
            }
        }
    }

    private fun resolveApproval(approved: Boolean) {
        if (state.phase != VN97AppPhase.WAITING_APPROVAL || closed) return
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
                val result = assistant.resolvePendingApproval(approved)
                mainHandler.post {
                    if (!closed) applyTurnResult(result)
                }
            } catch (exc: Throwable) {
                mainHandler.post {
                    if (!closed) {
                        renderFailure(
                            "Approval resolution failed: " +
                                exc::class.java.simpleName
                        )
                    }
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
                "VN97 turn ended at " +
                    result.update.state.name +
                    "."
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
        statusView?.text = next.status
        transcriptView?.text =
            if (next.transcript.isEmpty()) {
                "Chat and task control use the same native VN97 assistant runtime."
            } else {
                next.transcript.joinToString("\n\n")
            }
        transcriptScroll?.post {
            transcriptScroll?.fullScroll(View.FOCUS_DOWN)
        }

        val approval = if (next.phase == VN97AppPhase.WAITING_APPROVAL) {
            assistant.pendingApproval()
        } else {
            null
        }
        approvalView?.apply {
            text = approval?.presentationJson.orEmpty()
            visibility = if (approval == null) View.GONE else View.VISIBLE
        }
        approveButton?.apply {
            visibility = if (approval == null) View.GONE else View.VISIBLE
            isEnabled = approval != null
        }
        rejectButton?.apply {
            visibility = if (approval == null) View.GONE else View.VISIBLE
            isEnabled = approval != null
        }
        inputView?.isEnabled = next.inputEnabled
        sendButton?.isEnabled = next.inputEnabled
        voiceButton?.apply {
            text = if (voiceCapture.isRecording) "Stop" else "Mic"
            isEnabled =
                if (voiceCapture.isRecording) {
                    true
                } else {
                    next.phase == VN97AppPhase.READY &&
                        voiceAvailable
                }
        }

        val mode = when (next.phase) {
            VN97AppPhase.MODEL_REQUIRED -> AssistantMode.SLEEPING
            VN97AppPhase.READY -> AssistantMode.IDLE
            VN97AppPhase.RUNNING -> AssistantMode.THINKING
            VN97AppPhase.WAITING_APPROVAL -> AssistantMode.WAITING_APPROVAL
            VN97AppPhase.ERROR -> AssistantMode.ERROR
        }
        val energy = when (next.phase) {
            VN97AppPhase.MODEL_REQUIRED -> 0.15f
            VN97AppPhase.READY -> 0.46f
            VN97AppPhase.RUNNING -> 0.72f
            VN97AppPhase.WAITING_APPROVAL -> 0.60f
            VN97AppPhase.ERROR -> 0.28f
        }
        publishMode(mode, energy)
    }

    private fun createPanelLayoutParams(): WindowManager.LayoutParams {
        val bounds = displayBounds()
        val margin = dp(16)
        val availableWidth = (bounds.first - margin * 2).coerceAtLeast(1)
        val availableHeight = (bounds.second - margin * 2).coerceAtLeast(1)
        return WindowManager.LayoutParams(
            min(dp(PANEL_WIDTH_DP), availableWidth),
            min(dp(PANEL_HEIGHT_DP), availableHeight),
            WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY,
            WindowManager.LayoutParams.FLAG_LAYOUT_IN_SCREEN,
            PixelFormat.TRANSLUCENT,
        ).apply {
            gravity = Gravity.TOP or Gravity.START
            softInputMode = WindowManager.LayoutParams.SOFT_INPUT_ADJUST_RESIZE
        }
    }

    private fun positionPanel(params: WindowManager.LayoutParams) {
        val anchor = anchorProvider() ?: return
        val bounds = displayBounds()
        val gap = dp(8)
        val leftCandidate = anchor.x - params.width - gap
        val rightCandidate = anchor.x + anchor.width + gap
        val preferredX = if (leftCandidate >= 0) leftCandidate else rightCandidate
        val maxX = (bounds.first - params.width).coerceAtLeast(0)
        val maxY = (bounds.second - params.height).coerceAtLeast(0)
        params.x = preferredX.coerceIn(0, maxX)
        params.y = anchor.y.coerceIn(0, maxY)
    }

    private fun hideKeyboard(view: View) {
        val inputMethodManager =
            context.getSystemService(InputMethodManager::class.java) ?: return
        inputMethodManager.hideSoftInputFromWindow(view.windowToken, 0)
    }

    private fun dp(value: Int): Int =
        (value * context.resources.displayMetrics.density)
            .toInt()
            .coerceAtLeast(1)

    @Suppress("DEPRECATION")
    private fun displayBounds(): Pair<Int, Int> =
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            val bounds = windowManager.currentWindowMetrics.bounds
            Pair(bounds.width(), bounds.height())
        } else {
            val metrics = context.resources.displayMetrics
            Pair(metrics.widthPixels, metrics.heightPixels)
        }

    override fun close() {
        if (closed) return
        closed = true
        voiceCapture.close()
        setMicrophoneForegroundActive(false)
        removePanel()
        worker.shutdownNow()
    }

    companion object {
        private const val PANEL_WIDTH_DP = 320
        private const val PANEL_HEIGHT_DP = 390
    }
}
