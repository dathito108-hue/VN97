import ai.vn97.app.*

fun main() {
    val ready = VN97AppReducer.reduce(
        VN97AppState(),
        VN97AppEvent.TrustedModelActivated,
    )
    val running = VN97AppReducer.reduce(
        ready,
        VN97AppEvent.TurnStarted,
    )
    val waiting = VN97AppReducer.reduce(
        running,
        VN97AppEvent.ApprovalRequired,
    )
    check(waiting.phase == VN97AppPhase.WAITING_APPROVAL)
    check(!waiting.inputEnabled)

    val chained = VN97AppReducer.reduce(
        waiting,
        VN97AppEvent.ApprovalRequired,
    )
    check(chained.phase == VN97AppPhase.WAITING_APPROVAL)

    val rejected = VN97AppReducer.reduce(
        waiting,
        VN97AppEvent.ApprovalRejected("copy this"),
    )
    check(rejected.phase == VN97AppPhase.READY)
    check(rejected.inputEnabled)
    check(
        rejected.transcript ==
            listOf(
                "You: copy this",
                "System: External action rejected.",
            )
    )

    println("M10C_APPROVAL_STATE_PASS")
}
