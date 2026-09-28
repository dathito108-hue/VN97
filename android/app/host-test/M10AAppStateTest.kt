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

    val visualDone = VN97AppReducer.reduce(
        running,
        VN97AppEvent.VisualObserved(
            source = "screen",
            observation = "settings screen",
        ),
    )
    check(visualDone.phase == VN97AppPhase.READY)
    check(visualDone.inputEnabled)
    check(
        visualDone.transcript.last() ==
            "VN97 screen: settings screen"
    )

    val visualRunning = VN97AppReducer.reduce(
        ready,
        VN97AppEvent.TurnStarted,
    )
    val visualFailed = VN97AppReducer.reduce(
        visualRunning,
        VN97AppEvent.VisualFailed(
            "camera capture unavailable"
        ),
    )
    check(visualFailed.phase == VN97AppPhase.READY)
    check(visualFailed.inputEnabled)

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
    check(completed.transcript == listOf("You: hello", "VN97: world"))

    expectFailure {
        VN97AppReducer.reduce(initial, VN97AppEvent.TurnStarted)
    }

    val reset = VN97AppReducer.reduce(completed, VN97AppEvent.ResetModel)
    check(reset.phase == VN97AppPhase.MODEL_REQUIRED)
    check(!reset.inputEnabled)

    val yielded = VN97AppReducer.reduce(running, VN97AppEvent.TurnYielded)
    check(yielded.phase == VN97AppPhase.YIELDED && !yielded.inputEnabled)
    check(yielded.transcript == running.transcript)
    expectFailure { VN97AppReducer.reduce(yielded, VN97AppEvent.TurnStarted) }
    val resumed = VN97AppReducer.reduce(yielded, VN97AppEvent.TurnResumed)
    check(resumed.phase == VN97AppPhase.RUNNING && !resumed.inputEnabled)
    val yieldedAgain = VN97AppReducer.reduce(resumed, VN97AppEvent.TurnYielded)
    check(yieldedAgain.transcript == yielded.transcript)
    val afterResume = VN97AppReducer.reduce(
        VN97AppReducer.reduce(yieldedAgain, VN97AppEvent.TurnResumed),
        VN97AppEvent.TurnCompleted("original question", "final answer"),
    )
    check(afterResume.transcript == listOf("You: original question", "VN97: final answer"))
    expectFailure { VN97AppReducer.reduce(waiting, VN97AppEvent.TurnResumed) }
    expectFailure { VN97AppReducer.reduce(ready, VN97AppEvent.TurnResumed) }
    // Recreated UI can recover the still-live yielded turn without resubmission.
    check(VN97AppReducer.reduce(ready, VN97AppEvent.TurnYielded).phase == VN97AppPhase.YIELDED)
    check(VN97AppReducer.reduce(
        VN97AppReducer.reduce(resumed, VN97AppEvent.ApprovalRequired),
        VN97AppEvent.TurnYielded,
    ).phase == VN97AppPhase.YIELDED)

    data class Slice(val turnId: Int, val offset: Int, val yielded: Boolean)
    val advances = mutableListOf<Int>()
    val first = Slice(97, 1, true)
    val paused = advanceVN97BoundedTurn(first, 2, { it.yielded }) {
        check(it.turnId == 97)
        advances += it.offset
        Slice(it.turnId, it.offset + 1, true)
    }
    check(paused.offset == 2 && advances == listOf(1))
    val finalSlice = advanceVN97BoundedTurn(paused, 8, { it.yielded }) {
        check(it.turnId == 97)
        advances += it.offset
        Slice(it.turnId, it.offset + 1, false)
    }
    check(finalSlice.offset == 3 && advances == listOf(1, 2))
    check(advanceVN97BoundedTurn(first, 1, { it.yielded }) {
        error("budget of one cannot advance again")
    } === first)
    check(advanceVN97BoundedTurn(finalSlice, 8, { it.yielded }) {
        error("completed/approval boundaries must not be advanced")
    } === finalSlice)
    check(runCatching { advanceVN97BoundedTurn(first, 0, { it.yielded }, { it }) }.isFailure)
    val paused = VN97AppReducer.reduce(running, VN97AppEvent.TurnPaused)
    check(paused.phase == VN97AppPhase.PAUSED && !paused.inputEnabled)
    expectFailure { VN97AppReducer.reduce(paused, VN97AppEvent.TurnStarted) }
    val resumedPause = VN97AppReducer.reduce(paused, VN97AppEvent.TurnResumed)
    check(resumedPause.phase == VN97AppPhase.RUNNING && !resumedPause.inputEnabled)
    val waitingAfterResume = VN97AppReducer.reduce(resumedPause, VN97AppEvent.ApprovalRequired)
    check(waitingAfterResume.phase == VN97AppPhase.WAITING_APPROVAL)
    expectFailure { VN97AppReducer.reduce(waitingAfterResume, VN97AppEvent.TurnResumed) }
    println("M10A_APP_STATE_PASS: yielded/paused continuation and bounded driver")
}
