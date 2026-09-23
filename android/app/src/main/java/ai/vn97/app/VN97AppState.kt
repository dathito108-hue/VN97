package ai.vn97.app

enum class VN97AppPhase {
    MODEL_REQUIRED,
    READY,
    RUNNING,
    WAITING_APPROVAL,
    ERROR,
}

data class VN97AppState(
    val phase: VN97AppPhase = VN97AppPhase.MODEL_REQUIRED,
    val status: String = "Activate a trusted VN97 model to begin.",
    val inputEnabled: Boolean = false,
    val transcript: List<String> = emptyList(),
) {
    init {
        require(transcript.size <= 512) { "transcript entry bound exceeded" }
        require(status.length <= 4096) { "status text bound exceeded" }
        if (phase == VN97AppPhase.MODEL_REQUIRED) {
            require(!inputEnabled) {
                "chat input must stay disabled until a trusted model is active"
            }
        }
        if (phase == VN97AppPhase.RUNNING || phase == VN97AppPhase.WAITING_APPROVAL) {
            require(!inputEnabled) {
                "chat input must be single-flight while a turn is active"
            }
        }
    }
}

sealed interface VN97AppEvent {
    data object TrustedModelActivated : VN97AppEvent
    data object TurnStarted : VN97AppEvent
    data object ApprovalRequired : VN97AppEvent
    data class ApprovalRejected(val user: String) : VN97AppEvent
    data class TurnCompleted(val user: String, val assistant: String) : VN97AppEvent
    data class Failed(val message: String) : VN97AppEvent
    data object ResetModel : VN97AppEvent
}

object VN97AppReducer {
    fun reduce(state: VN97AppState, event: VN97AppEvent): VN97AppState = when (event) {
        VN97AppEvent.TrustedModelActivated -> {
            check(state.phase == VN97AppPhase.MODEL_REQUIRED || state.phase == VN97AppPhase.ERROR)
            state.copy(
                phase = VN97AppPhase.READY,
                status = "VN97 native model ready.",
                inputEnabled = true,
            )
        }

        VN97AppEvent.TurnStarted -> {
            check(state.phase == VN97AppPhase.READY)
            state.copy(
                phase = VN97AppPhase.RUNNING,
                status = "VN97 is thinking…",
                inputEnabled = false,
            )
        }

        VN97AppEvent.ApprovalRequired -> {
            check(
                state.phase == VN97AppPhase.READY ||
                    state.phase == VN97AppPhase.RUNNING ||
                    state.phase == VN97AppPhase.WAITING_APPROVAL
            )
            state.copy(
                phase = VN97AppPhase.WAITING_APPROVAL,
                status = "External action requires explicit approval.",
                inputEnabled = false,
            )
        }

        is VN97AppEvent.ApprovalRejected -> {
            check(state.phase == VN97AppPhase.WAITING_APPROVAL)
            require(event.user.isNotBlank()) {
                "rejected turn user text must not be blank"
            }
            state.copy(
                phase = VN97AppPhase.READY,
                status = "External action rejected.",
                inputEnabled = true,
                transcript = boundedTranscript(
                    state.transcript +
                        "You: ${event.user}" +
                        "System: External action rejected."
                ),
            )
        }

        is VN97AppEvent.TurnCompleted -> {
            check(
                state.phase == VN97AppPhase.RUNNING ||
                    state.phase == VN97AppPhase.WAITING_APPROVAL
            )
            require(event.user.isNotBlank()) { "completed turn user text must not be blank" }
            require(event.assistant.isNotEmpty()) {
                "completed turn assistant response must not be empty"
            }
            state.copy(
                phase = VN97AppPhase.READY,
                status = "Ready.",
                inputEnabled = true,
                transcript = boundedTranscript(
                    state.transcript +
                        "You: ${event.user}" +
                        "VN97: ${event.assistant}"
                ),
            )
        }

        is VN97AppEvent.Failed -> {
            require(event.message.isNotBlank()) { "failure message must not be blank" }
            state.copy(
                phase = VN97AppPhase.ERROR,
                status = event.message.take(4096),
                inputEnabled = false,
            )
        }

        VN97AppEvent.ResetModel -> VN97AppState()
    }

    private fun boundedTranscript(values: List<String>): List<String> =
        if (values.size <= 512) values else values.takeLast(512)
}
