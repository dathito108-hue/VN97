package ai.vn97.avatar

import kotlin.math.exp
import kotlin.math.sqrt

enum class Viseme {
    REST,
    CLOSED,
    LABIODENTAL,
    DENTAL,
    ALVEOLAR,
    VELAR,
    SIBILANT,
    RHOTIC,
    OPEN,
    FRONT,
    ROUND,
}

data class MouthPose(
    val jawOpen: Float,
    val mouthWide: Float,
    val lipRound: Float,
) {
    init {
        listOf(jawOpen, mouthWide, lipRound).forEach { value ->
            require(value.isFinite() && value in 0f..1f) { "mouth pose values must be finite and in 0..1" }
        }
    }

    fun scaled(weight: Float): MouthPose {
        require(weight.isFinite() && weight in 0f..1f) { "weight must be finite and in 0..1" }
        return MouthPose(jawOpen * weight, mouthWide * weight, lipRound * weight)
    }

    companion object {
        val REST = MouthPose(0f, 0f, 0f)
    }
}

data class VisemeCue(
    val startMillis: Long,
    val durationMillis: Long,
    val viseme: Viseme,
    val weight: Float = 1f,
) {
    init {
        require(startMillis >= 0) { "startMillis must be non-negative" }
        require(durationMillis in 1..10_000) { "durationMillis must be in 1..10000" }
        require(weight.isFinite() && weight in 0f..1f) { "weight must be finite and in 0..1" }
        require(startMillis <= Long.MAX_VALUE - durationMillis) { "viseme cue end overflows" }
    }

    val endMillis: Long get() = startMillis + durationMillis
}

class SpeechTimeline(
    cues: List<VisemeCue>,
    private val maxTimelineMillis: Long = 10 * 60 * 1000L,
) {
    private val cues: List<VisemeCue> = cues.toList()

    init {
        require(maxTimelineMillis in 1..3_600_000L) { "maxTimelineMillis must be in 1..3600000" }
        require(this.cues.size <= 4096) { "speech timeline exceeds 4096 cues" }
        var previousEnd = 0L
        this.cues.forEachIndexed { index, cue ->
            require(cue.startMillis >= previousEnd) { "viseme cues must be sorted and non-overlapping" }
            require(cue.endMillis <= maxTimelineMillis) { "viseme cue exceeds timeline bound" }
            if (index > 0) {
                require(cue.startMillis >= this.cues[index - 1].endMillis) {
                    "viseme cues must be sorted and non-overlapping"
                }
            }
            previousEnd = cue.endMillis
        }
    }

    fun poseAt(positionMillis: Long): MouthPose {
        require(positionMillis >= 0) { "positionMillis must be non-negative" }
        if (cues.isEmpty()) return MouthPose.REST
        var low = 0
        var high = cues.lastIndex
        while (low <= high) {
            val mid = (low + high).ushr(1)
            val cue = cues[mid]
            when {
                positionMillis < cue.startMillis -> high = mid - 1
                positionMillis >= cue.endMillis -> low = mid + 1
                else -> return cue.viseme.pose().scaled(cue.weight)
            }
        }
        return MouthPose.REST
    }

    val cueCount: Int get() = cues.size
}

private fun Viseme.pose(): MouthPose = when (this) {
    Viseme.REST -> MouthPose.REST
    Viseme.CLOSED -> MouthPose(0.04f, 0.10f, 0.02f)
    Viseme.LABIODENTAL -> MouthPose(0.16f, 0.48f, 0.02f)
    Viseme.DENTAL -> MouthPose(0.22f, 0.38f, 0.02f)
    Viseme.ALVEOLAR -> MouthPose(0.28f, 0.46f, 0.04f)
    Viseme.VELAR -> MouthPose(0.38f, 0.24f, 0.08f)
    Viseme.SIBILANT -> MouthPose(0.18f, 0.68f, 0.02f)
    Viseme.RHOTIC -> MouthPose(0.32f, 0.18f, 0.38f)
    Viseme.OPEN -> MouthPose(0.90f, 0.32f, 0.04f)
    Viseme.FRONT -> MouthPose(0.48f, 0.78f, 0.02f)
    Viseme.ROUND -> MouthPose(0.46f, 0.08f, 0.92f)
}

class AudioLevelEnvelope(
    private val attackMillis: Float,
    private val releaseMillis: Float,
) {
    init {
        require(attackMillis.isFinite() && attackMillis > 0f) { "attackMillis must be finite and positive" }
        require(releaseMillis.isFinite() && releaseMillis > 0f) { "releaseMillis must be finite and positive" }
    }

    private var value = 0f

    fun reset(level: Float = 0f): Float {
        requireLevel(level)
        value = level
        return value
    }

    fun update(level: Float, deltaMillis: Long): Float {
        requireLevel(level)
        require(deltaMillis >= 0) { "deltaMillis must be non-negative" }
        val boundedDelta = deltaMillis.coerceAtMost(1000L).toFloat()
        if (boundedDelta == 0f) return value
        val tau = if (level > value) attackMillis else releaseMillis
        val alpha = (1.0 - exp((-boundedDelta / tau).toDouble())).toFloat().coerceIn(0f, 1f)
        value += (level - value) * alpha
        return value.coerceIn(0f, 1f).also { value = it }
    }

    private fun requireLevel(level: Float) {
        require(level.isFinite() && level in 0f..1f) { "audio level must be finite and in 0..1" }
    }
}

object AudioLevelMeter {
    fun rmsPcm16(samples: ShortArray, offset: Int = 0, count: Int = samples.size - offset): Float {
        require(offset >= 0 && count > 0 && offset <= samples.size - count) {
            "offset/count must select a non-empty PCM16 range"
        }
        require(count <= 262_144) { "PCM16 meter window exceeds 262144 samples" }
        var sumSquares = 0.0
        val end = offset + count
        for (index in offset until end) {
            val normalized = samples[index].toDouble() / 32768.0
            sumSquares += normalized * normalized
        }
        return sqrt(sumSquares / count.toDouble()).toFloat().coerceIn(0f, 1f)
    }
}

enum class CognitionPresentationState {
    READY,
    LISTENING,
    REASONING,
    RESPONDING,
    WAITING_APPROVAL,
    EXECUTING,
    ERROR,
    SLEEPING,
}

fun CognitionPresentationState.toAssistantMode(): AssistantMode = when (this) {
    CognitionPresentationState.READY -> AssistantMode.IDLE
    CognitionPresentationState.LISTENING -> AssistantMode.LISTENING
    CognitionPresentationState.REASONING -> AssistantMode.THINKING
    CognitionPresentationState.RESPONDING -> AssistantMode.SPEAKING
    CognitionPresentationState.WAITING_APPROVAL -> AssistantMode.WAITING_APPROVAL
    CognitionPresentationState.EXECUTING -> AssistantMode.EXECUTING
    CognitionPresentationState.ERROR -> AssistantMode.ERROR
    CognitionPresentationState.SLEEPING -> AssistantMode.SLEEPING
}

data class SpeechAvatarInput(
    val sourceSequence: Long,
    val cognitionState: CognitionPresentationState,
    val inputLevel: Float = 0f,
    val outputLevel: Float = 0f,
    val playbackPositionMillis: Long = 0,
    val timeline: SpeechTimeline? = null,
    val gesture: AvatarGesture = AvatarGesture.NONE,
    val blink: Float = 0f,
    val gazeX: Float = 0f,
    val gazeY: Float = 0f,
    val energy: Float = 0.5f,
) {
    init {
        require(sourceSequence >= 0) { "sourceSequence must be non-negative" }
        require(inputLevel.isFinite() && inputLevel in 0f..1f) { "inputLevel must be finite and in 0..1" }
        require(outputLevel.isFinite() && outputLevel in 0f..1f) { "outputLevel must be finite and in 0..1" }
        require(playbackPositionMillis >= 0) { "playbackPositionMillis must be non-negative" }
        require(blink.isFinite() && blink in 0f..1f) { "blink must be finite and in 0..1" }
        require(gazeX.isFinite() && gazeX in -1f..1f) { "gazeX must be finite and in -1..1" }
        require(gazeY.isFinite() && gazeY in -1f..1f) { "gazeY must be finite and in -1..1" }
        require(energy.isFinite() && energy in 0f..1f) { "energy must be finite and in 0..1" }
    }
}

class SpeechAvatarSynchronizer(
    private val bridge: AvatarStateBridge,
    listeningAttackMillis: Float = 45f,
    listeningReleaseMillis: Float = 180f,
    speakingAttackMillis: Float = 20f,
    speakingReleaseMillis: Float = 90f,
) {
    private val listeningEnvelope = AudioLevelEnvelope(listeningAttackMillis, listeningReleaseMillis)
    private val speakingEnvelope = AudioLevelEnvelope(speakingAttackMillis, speakingReleaseMillis)

    @Synchronized
    fun publish(input: SpeechAvatarInput, deltaMillis: Long): AvatarFrameState {
        require(deltaMillis >= 0) { "deltaMillis must be non-negative" }
        val mode = input.cognitionState.toAssistantMode()
        val listening = listeningEnvelope.update(
            if (mode == AssistantMode.LISTENING) input.inputLevel else 0f,
            deltaMillis,
        )
        val speaking = speakingEnvelope.update(
            if (mode == AssistantMode.SPEAKING) input.outputLevel else 0f,
            deltaMillis,
        )
        val rawPose = if (mode == AssistantMode.SPEAKING) {
            input.timeline?.poseAt(input.playbackPositionMillis) ?: MouthPose.REST
        } else {
            MouthPose.REST
        }
        val pose = rawPose.scaled(speaking)
        return bridge.publish(
            AvatarCommand(
                sourceSequence = input.sourceSequence,
                mode = mode,
                gesture = input.gesture,
                speakingLevel = if (mode == AssistantMode.SPEAKING) speaking else 0f,
                listeningLevel = if (mode == AssistantMode.LISTENING) listening else 0f,
                jawOpen = pose.jawOpen,
                mouthWide = pose.mouthWide,
                lipRound = pose.lipRound,
                blink = input.blink,
                gazeX = input.gazeX,
                gazeY = input.gazeY,
                energy = input.energy,
            )
        )
    }

    @Synchronized
    fun resetAudioEnvelopes() {
        listeningEnvelope.reset()
        speakingEnvelope.reset()
    }
}
