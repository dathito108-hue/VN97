import ai.vn97.avatar.*

private fun expectFailure(block: () -> Unit) {
    try {
        block()
        error("expected failure")
    } catch (_: IllegalArgumentException) {
    }
}

fun main() {
    val timeline = SpeechTimeline(
        listOf(
            VisemeCue(0, 80, Viseme.CLOSED),
            VisemeCue(80, 120, Viseme.OPEN),
            VisemeCue(200, 100, Viseme.ROUND, 0.75f),
        )
    )
    check(timeline.cueCount == 3)
    check(timeline.poseAt(10).jawOpen < 0.1f)
    check(timeline.poseAt(100).jawOpen > 0.8f)
    check(timeline.poseAt(250).lipRound > 0.6f)
    check(timeline.poseAt(500) == MouthPose.REST)
    expectFailure {
        SpeechTimeline(listOf(VisemeCue(10, 100, Viseme.OPEN), VisemeCue(50, 20, Viseme.CLOSED)))
    }

    val envelope = AudioLevelEnvelope(20f, 200f)
    val up = envelope.update(1f, 20)
    check(up > 0.5f && up < 1f)
    val down = envelope.update(0f, 20)
    check(down in 0f..<up)
    expectFailure { envelope.update(Float.NaN, 10) }

    check(AudioLevelMeter.rmsPcm16(shortArrayOf(0, 0, 0, 0)) == 0f)
    val pcmLevel = AudioLevelMeter.rmsPcm16(shortArrayOf(Short.MAX_VALUE, Short.MIN_VALUE))
    check(pcmLevel > 0.99f)
    expectFailure { AudioLevelMeter.rmsPcm16(shortArrayOf(1, 2), offset = 2, count = 0) }

    expectFailure { SpeechTimeline(emptyList(), maxTimelineMillis = 3_600_001L) }
    expectFailure { AudioLevelMeter.rmsPcm16(ShortArray(262_145)) }

    check(CognitionPresentationState.REASONING.toAssistantMode() == AssistantMode.THINKING)
    check(CognitionPresentationState.RESPONDING.toAssistantMode() == AssistantMode.SPEAKING)

    val bridge = AvatarStateBridge()
    val sync = SpeechAvatarSynchronizer(bridge)
    val listening = sync.publish(
        SpeechAvatarInput(
            sourceSequence = 1,
            cognitionState = CognitionPresentationState.LISTENING,
            inputLevel = 1f,
            gazeX = 0.25f,
        ),
        deltaMillis = 60,
    )
    check(listening.mode == AssistantMode.LISTENING)
    check(listening.listeningLevel > 0.7f)
    check(listening.speakingLevel == 0f)
    check(listening.jawOpen == 0f)

    val speaking = sync.publish(
        SpeechAvatarInput(
            sourceSequence = 2,
            cognitionState = CognitionPresentationState.RESPONDING,
            outputLevel = 1f,
            playbackPositionMillis = 100,
            timeline = timeline,
            energy = 0.8f,
        ),
        deltaMillis = 40,
    )
    check(speaking.mode == AssistantMode.SPEAKING)
    check(speaking.speakingLevel > 0.8f)
    check(speaking.jawOpen > 0.65f)
    check(speaking.mouthWide > 0.2f)
    check(speaking.lipRound < 0.1f)

    val rounded = sync.publish(
        SpeechAvatarInput(
            sourceSequence = 3,
            cognitionState = CognitionPresentationState.RESPONDING,
            outputLevel = 1f,
            playbackPositionMillis = 250,
            timeline = timeline,
        ),
        deltaMillis = 20,
    )
    check(rounded.lipRound > 0.5f)

    expectFailure {
        sync.publish(
            SpeechAvatarInput(
                sourceSequence = 3,
                cognitionState = CognitionPresentationState.READY,
            ),
            deltaMillis = 10,
        )
    }

    val gatedBridge = AvatarStateBridge()
    val gatedSync = SpeechAvatarSynchronizer(gatedBridge)
    gatedSync.publish(
        SpeechAvatarInput(
            sourceSequence = 1,
            cognitionState = CognitionPresentationState.READY,
            inputLevel = 1f,
            outputLevel = 1f,
        ),
        deltaMillis = 100,
    )
    val gatedSpeaking = gatedSync.publish(
        SpeechAvatarInput(
            sourceSequence = 2,
            cognitionState = CognitionPresentationState.RESPONDING,
            outputLevel = 0f,
        ),
        deltaMillis = 0,
    )
    check(gatedSpeaking.speakingLevel == 0f)
    check(gatedSpeaking.jawOpen == 0f)

    val filter = AvatarMotionFilter(12f)
    filter.reset(AvatarFrameState.initial())
    val filtered = filter.step(rounded, 1f / 60f)
    check(filtered.jawOpen in 0f..1f)
    check(filtered.lipRound in 0f..1f)
    check(filtered.speakingLevel in 0f..1f)

    println("M8B_SPEECH_VISEME_SYNC_PASS")
}
