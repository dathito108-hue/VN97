package ai.vn97.avatar

enum class AvatarInteractionType {
    TAP,
    LONG_PRESS,
    DRAG,
}

data class AvatarInteraction(
    val type: AvatarInteractionType,
    val x: Float,
    val y: Float,
    val durationMillis: Long,
) {
    init {
        require(x.isFinite() && x in 0f..1f) { "x must be finite and in 0..1" }
        require(y.isFinite() && y in 0f..1f) { "y must be finite and in 0..1" }
        require(durationMillis >= 0) { "durationMillis must be non-negative" }
    }
}

fun interface AvatarInteractionListener {
    fun onAvatarInteraction(interaction: AvatarInteraction)
}
