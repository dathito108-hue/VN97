import ai.vn97.app.*

private inline fun expectFailure(block: () -> Unit) {
    var failed = false
    try {
        block()
    } catch (_: IllegalStateException) {
        failed = true
    }
    check(failed)
}

fun main() {
    val initial = VN97AppState()
    check(initial.phase == VN97AppPhase.MODEL_REQUIRED)
    check(!initial.inputEnabled)

    val ready = VN97AppReducer.reduce(
        initial,
        VN97AppEvent.TrustedModelActivated,
    )
    check(ready.phase == VN97AppPhase.READY)
    check(ready.inputEnabled)

    val running = VN97AppReducer.reduce(ready, VN97AppEvent.TurnStarted)
    check(running.phase == VN97AppPhase.RUNNING)
    check(!running.inputEnabled)

    val waiting = VN97AppReducer.reduce(
        running,
        VN97AppEvent.ApprovalRequired,
    )
    check(waiting.phase == VN97AppPhase.WAITING_APPROVAL)
    check(!waiting.inputEnabled)

    val completed = VN97AppReducer.reduce(
        waiting,
        VN97AppEvent.TurnCompleted(
            user = "hello",
            assistant = "world",
        ),
    )
    check(completed.phase == VN97AppPhase.READY)
    check(completed.inputEnabled)
    check(completed.transcript == listOf("You: helloVN97: world"))

    expectFailure {
        VN97AppReducer.reduce(initial, VN97AppEvent.TurnStarted)
    }

    val reset = VN97AppReducer.reduce(completed, VN97AppEvent.ResetModel)
    check(reset.phase == VN97AppPhase.MODEL_REQUIRED)
    check(!reset.inputEnabled)

    println("M10A_APP_STATE_PASS")
}
