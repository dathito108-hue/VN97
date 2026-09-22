package ai.vn97.avatar

object AvatarFramePolicy {
    fun targetFps(mode: AssistantMode): Int = when (mode) {
        AssistantMode.SLEEPING -> 5
        AssistantMode.IDLE -> 15
        AssistantMode.THINKING, AssistantMode.ERROR -> 30
        AssistantMode.LISTENING,
        AssistantMode.SPEAKING,
        AssistantMode.WAITING_APPROVAL,
        AssistantMode.EXECUTING -> 60
    }

    fun intervalNanos(mode: AssistantMode): Long = 1_000_000_000L / targetFps(mode)
}
