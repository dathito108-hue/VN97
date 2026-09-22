import ai.vn97.avatar.*

private fun fail(block: () -> Unit) {
    try { block(); error("expected failure") } catch (_: IllegalArgumentException) {}
}

fun main() {
    val binding = AvatarRigBinding(8, rootJoint = 0, headJoint = 1, jawJoint = 2, leftArmJoint = 3, rightArmJoint = 4)
    check(binding.headJoint == 1)
    fail { AvatarRigBinding(2, 0, headJoint = 0) }
    fail { AvatarRigBinding(129, 0) }

    val frame = AvatarFrameState(
        sourceSequence = 10,
        mode = AssistantMode.SPEAKING,
        gesture = AvatarGesture.WAVE,
        speakingLevel = 0.9f,
        listeningLevel = 0f,
        jawOpen = 0.8f,
        mouthWide = 0.7f,
        lipRound = 0.2f,
        blink = 0f,
        gazeX = 0.5f,
        gazeY = -0.25f,
        energy = 0.8f,
    )
    val animator = AvatarRigAnimator()
    val pose = animator.sample(frame, 0.1f)
    check(pose.headYawDegrees > 0f)
    check(pose.headPitchDegrees > 0f)
    check(pose.jawDegrees > 20f)
    check(pose.leftArmRollDegrees != 0f)
    check(pose.mouthWide == 0.7f)

    val alert = animator.sample(frame.copy(mode = AssistantMode.ERROR, gesture = AvatarGesture.ALERT), 1f)
    check(alert.browRaise >= 0.8f)
    val approval = animator.sample(frame.copy(mode = AssistantMode.WAITING_APPROVAL, gesture = AvatarGesture.NONE), 1f)
    check(approval.browRaise >= 0.55f)
    fail { animator.sample(frame, Float.NaN) }

    println("M8C_RIG_POSE_PASS")
}
