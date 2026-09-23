package ai.vn97.app

import ai.vn97.avatar.AssistantMode
import ai.vn97.avatar.AvatarCommand
import ai.vn97.avatar.AvatarGesture
import ai.vn97.avatar.VN97AvatarView
import ai.vn97.platform.VN97AssistantTurnState
import android.app.Activity
import android.os.Bundle
import android.view.Gravity
import android.view.View
import android.view.ViewGroup
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import java.util.concurrent.Executors

class VN97MainActivity : Activity() {
    private lateinit var avatar: VN97AvatarView
    private lateinit var statusView: TextView
    private lateinit var transcriptView: TextView
    private lateinit var inputView: EditText
    private lateinit var sendButton: Button
    private lateinit var approvalView: TextView
    private lateinit var approveButton: Button
    private lateinit var rejectButton: Button

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
        render(state)
        attachTrustedModel()
    }

    override fun onResume() {
        super.onResume()
        avatar.onAvatarResume()
    }

    override fun onPause() {
        avatar.onAvatarPause()
        super.onPause()
    }

    override fun onDestroy() {
        worker.shutdownNow()
        super.onDestroy()
    }

    private fun attachTrustedModel() {
        worker.execute {
            try {
                val active = app.assistant.openIfActivated()
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
                    } else {
                        render(state)
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
                val result = app.assistant.resolvePendingApproval(approved)
                runOnUiThread { applyTurnResult(result) }
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

        val mode = when (state.phase) {
            VN97AppPhase.MODEL_REQUIRED -> AssistantMode.SLEEPING
            VN97AppPhase.READY -> AssistantMode.IDLE
            VN97AppPhase.RUNNING -> AssistantMode.THINKING
            VN97AppPhase.WAITING_APPROVAL -> AssistantMode.WAITING_APPROVAL
            VN97AppPhase.ERROR -> AssistantMode.ERROR
        }
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
}
