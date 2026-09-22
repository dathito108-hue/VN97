package ai.vn97.avatar

import java.util.concurrent.atomic.AtomicReference
import kotlin.math.exp

private fun requireFinite(value: Float, label: String) {
    require(value.isFinite()) { "$label must be finite" }
}

enum class AssistantMode {
    IDLE,
    LISTENING,
    THINKING,
    SPEAKING,
    WAITING_APPROVAL,
    EXECUTING,
    ERROR,
    SLEEPING,
}

enum class AvatarGesture {
    NONE,
    NOD,
    SHAKE,
    WAVE,
    CONFIRM,
    ALERT,
}

data class AvatarCommand(
    val sourceSequence: Long,
    val mode: AssistantMode,
    val gesture: AvatarGesture = AvatarGesture.NONE,
    val speakingLevel: Float = 0f,
    val blink: Float = 0f,
    val gazeX: Float = 0f,
    val gazeY: Float = 0f,
    val energy: Float = 0.5f,
) {
    init {
        require(sourceSequence >= 0) { "sourceSequence must be non-negative" }
        listOf(
            speakingLevel to "speakingLevel",
            blink to "blink",
            energy to "energy",
        ).forEach { (value, label) ->
            requireFinite(value, label)
            require(value in 0f..1f) { "$label must be in 0..1" }
        }
        listOf(gazeX to "gazeX", gazeY to "gazeY").forEach { (value, label) ->
            requireFinite(value, label)
            require(value in -1f..1f) { "$label must be in -1..1" }
        }
    }
}

data class AvatarFrameState(
    val sourceSequence: Long,
    val mode: AssistantMode,
    val gesture: AvatarGesture,
    val speakingLevel: Float,
    val blink: Float,
    val gazeX: Float,
    val gazeY: Float,
    val energy: Float,
) {
    companion object {
        fun initial(): AvatarFrameState = AvatarFrameState(
            sourceSequence = 0,
            mode = AssistantMode.IDLE,
            gesture = AvatarGesture.NONE,
            speakingLevel = 0f,
            blink = 0f,
            gazeX = 0f,
            gazeY = 0f,
            energy = 0.5f,
        )
    }
}

class AvatarStateBridge(initial: AvatarFrameState = AvatarFrameState.initial()) {
    private val state = AtomicReference(initial)
    private val publishLock = Any()

    fun snapshot(): AvatarFrameState = state.get()

    fun publish(command: AvatarCommand): AvatarFrameState = synchronized(publishLock) {
        val current = state.get()
        require(command.sourceSequence > current.sourceSequence) {
            "avatar command sourceSequence must increase monotonically"
        }
        val next = AvatarFrameState(
            sourceSequence = command.sourceSequence,
            mode = command.mode,
            gesture = command.gesture,
            speakingLevel = command.speakingLevel,
            blink = command.blink,
            gazeX = command.gazeX,
            gazeY = command.gazeY,
            energy = command.energy,
        )
        state.set(next)
        next
    }
}

data class AvatarRenderState(
    val speakingLevel: Float,
    val blink: Float,
    val gazeX: Float,
    val gazeY: Float,
    val energy: Float,
)

class AvatarMotionFilter(
    private val responseHz: Float = 10f,
) {
    init {
        require(responseHz.isFinite() && responseHz > 0f) { "responseHz must be finite and positive" }
    }

    private var initialized = false
    private var value = AvatarRenderState(0f, 0f, 0f, 0f, 0.5f)

    fun reset(target: AvatarFrameState): AvatarRenderState {
        value = target.toRenderState()
        initialized = true
        return value
    }

    fun step(target: AvatarFrameState, deltaSeconds: Float): AvatarRenderState {
        require(deltaSeconds.isFinite() && deltaSeconds >= 0f) { "deltaSeconds must be finite and non-negative" }
        if (!initialized) return reset(target)
        val dt = deltaSeconds.coerceAtMost(0.1f)
        val alpha = (1.0 - exp((-responseHz * dt).toDouble())).toFloat().coerceIn(0f, 1f)
        val targetState = target.toRenderState()
        value = AvatarRenderState(
            speakingLevel = mix(value.speakingLevel, targetState.speakingLevel, alpha),
            blink = mix(value.blink, targetState.blink, alpha),
            gazeX = mix(value.gazeX, targetState.gazeX, alpha),
            gazeY = mix(value.gazeY, targetState.gazeY, alpha),
            energy = mix(value.energy, targetState.energy, alpha),
        )
        return value
    }

    private fun AvatarFrameState.toRenderState() = AvatarRenderState(
        speakingLevel = speakingLevel,
        blink = blink,
        gazeX = gazeX,
        gazeY = gazeY,
        energy = energy,
    )

    private fun mix(a: Float, b: Float, t: Float): Float = a + (b - a) * t
}
