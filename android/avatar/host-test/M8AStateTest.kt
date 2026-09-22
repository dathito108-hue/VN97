import ai.vn97.avatar.*

fun main() {
    val bridge = AvatarStateBridge()
    check(bridge.snapshot().mode == AssistantMode.IDLE)
    val speaking = bridge.publish(
        AvatarCommand(
            sourceSequence = 1,
            mode = AssistantMode.SPEAKING,
            gesture = AvatarGesture.NOD,
            speakingLevel = 1f,
            blink = 0.25f,
            gazeX = 0.5f,
            gazeY = -0.25f,
            energy = 0.8f,
        )
    )
    check(speaking.mode == AssistantMode.SPEAKING)
    check(bridge.snapshot().sourceSequence == 1L)
    try {
        bridge.publish(AvatarCommand(1, AssistantMode.IDLE))
        error("stale avatar sequence must fail")
    } catch (_: IllegalArgumentException) {
    }
    try {
        AvatarCommand(2, AssistantMode.IDLE, speakingLevel = Float.NaN)
        error("non-finite parameter must fail")
    } catch (_: IllegalArgumentException) {
    }

    val filter = AvatarMotionFilter(responseHz = 12f)
    val first = filter.reset(AvatarFrameState.initial())
    check(first.speakingLevel == 0f)
    val moved = filter.step(speaking, 1f / 60f)
    check(moved.speakingLevel in 0f..1f)
    check(moved.speakingLevel > 0f && moved.speakingLevel < 1f)
    val converged = generateSequence(moved) { filter.step(speaking, 1f / 60f) }
        .drop(120)
        .first()
    check(converged.speakingLevel > 0.99f)
    check(converged.gazeX > 0.49f)

    val tap = AvatarInteraction(AvatarInteractionType.TAP, 0.5f, 0.25f, 80)
    check(tap.type == AvatarInteractionType.TAP)
    try {
        AvatarInteraction(AvatarInteractionType.DRAG, -0.1f, 0.2f, 1)
        error("out-of-range interaction must fail")
    } catch (_: IllegalArgumentException) {
    }

    check(AvatarFramePolicy.targetFps(AssistantMode.SLEEPING) == 5)
    check(AvatarFramePolicy.targetFps(AssistantMode.IDLE) == 15)
    check(AvatarFramePolicy.targetFps(AssistantMode.THINKING) == 30)
    check(AvatarFramePolicy.targetFps(AssistantMode.SPEAKING) == 60)
    check(AvatarFramePolicy.intervalNanos(AssistantMode.SPEAKING) > 0L)

    println("M8A_AVATAR_STATE_PASS")
}
