package ai.vn97.avatar

import kotlin.math.sin

data class AvatarRigBinding(
    val jointCount: Int,
    val rootJoint: Int,
    val headJoint: Int? = null,
    val jawJoint: Int? = null,
    val leftArmJoint: Int? = null,
    val rightArmJoint: Int? = null,
) {
    init {
        require(jointCount in 1..128) { "jointCount must be in 1..128" }
        val values = listOfNotNull(rootJoint, headJoint, jawJoint, leftArmJoint, rightArmJoint)
        require(values.all { it in 0 until jointCount }) { "rig binding joint index is out of range" }
        require(values.distinct().size == values.size) { "rig binding joint indices must be unique" }
    }
}

data class AvatarRigPose(
    val headPitchDegrees: Float,
    val headYawDegrees: Float,
    val jawDegrees: Float,
    val leftArmRollDegrees: Float,
    val rightArmRollDegrees: Float,
    val browRaise: Float,
    val smile: Float,
    val mouthWide: Float,
    val lipRound: Float,
) {
    init {
        listOf(headPitchDegrees, headYawDegrees).forEach {
            require(it.isFinite() && it in -45f..45f) { "head rotation must be finite and in -45..45" }
        }
        require(jawDegrees.isFinite() && jawDegrees in 0f..45f) { "jawDegrees must be finite and in 0..45" }
        listOf(leftArmRollDegrees, rightArmRollDegrees).forEach {
            require(it.isFinite() && it in -120f..120f) { "arm rotation must be finite and in -120..120" }
        }
        listOf(browRaise, smile, mouthWide, lipRound).forEach {
            require(it.isFinite() && it in 0f..1f) { "face controls must be finite and in 0..1" }
        }
    }
}

class AvatarRigAnimator {
    fun sample(frame: AvatarFrameState, elapsedSeconds: Float): AvatarRigPose {
        require(elapsedSeconds.isFinite() && elapsedSeconds >= 0f) {
            "elapsedSeconds must be finite and non-negative"
        }
        val t = elapsedSeconds.coerceAtMost(86_400f).toDouble()
        val nod = if (frame.gesture == AvatarGesture.NOD) sin(t * 7.0).toFloat() * 10f else 0f
        val shake = if (frame.gesture == AvatarGesture.SHAKE) sin(t * 7.5).toFloat() * 13f else 0f
        val wave = if (frame.gesture == AvatarGesture.WAVE) sin(t * 8.0).toFloat() * 42f else 0f
        val confirm = if (frame.gesture == AvatarGesture.CONFIRM) 0.28f else 0f
        val alert = if (frame.gesture == AvatarGesture.ALERT || frame.mode == AssistantMode.ERROR) 0.8f else 0f
        val listeningBrow = if (frame.mode == AssistantMode.LISTENING) frame.listeningLevel * 0.5f else 0f
        val approvalBrow = if (frame.mode == AssistantMode.WAITING_APPROVAL) 0.55f else 0f

        return AvatarRigPose(
            headPitchDegrees = (-frame.gazeY * 8f + nod).coerceIn(-45f, 45f),
            headYawDegrees = (frame.gazeX * 11f + shake).coerceIn(-45f, 45f),
            jawDegrees = maxOf(frame.jawOpen * 32f, frame.speakingLevel * 18f).coerceIn(0f, 45f),
            leftArmRollDegrees = (if (frame.gesture == AvatarGesture.WAVE) 32f + wave else 0f).coerceIn(-120f, 120f),
            rightArmRollDegrees = (if (frame.gesture == AvatarGesture.CONFIRM) -18f else 0f).coerceIn(-120f, 120f),
            browRaise = maxOf(alert, listeningBrow, approvalBrow).coerceIn(0f, 1f),
            smile = (frame.mouthWide * 0.55f + confirm).coerceIn(0f, 1f),
            mouthWide = frame.mouthWide,
            lipRound = frame.lipRound,
        )
    }
}
