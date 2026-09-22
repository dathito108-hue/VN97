package ai.vn97.avatar

import android.content.Context
import android.opengl.GLSurfaceView
import android.util.AttributeSet
import android.view.Choreographer
import android.view.MotionEvent
import android.view.ViewConfiguration
import kotlin.math.hypot

class VN97AvatarView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
) : GLSurfaceView(context, attrs), Choreographer.FrameCallback {
    val stateBridge = AvatarStateBridge()

    private val avatarRenderer = AvatarRenderer(stateBridge)
    private val touchSlop = ViewConfiguration.get(context).scaledTouchSlop.toFloat()
    @Volatile private var interactionListener: AvatarInteractionListener? = null
    private var frameLoopRunning = false
    private var lastRenderRequestNanos = 0L
    private var downX = 0f
    private var downY = 0f
    private var downMillis = 0L
    private var dragging = false

    init {
        setEGLContextClientVersion(3)
        setRenderer(avatarRenderer)
        renderMode = RENDERMODE_WHEN_DIRTY
        preserveEGLContextOnPause = true
        isFocusable = true
        importantForAccessibility = IMPORTANT_FOR_ACCESSIBILITY_YES
        contentDescription = "VN97 interactive 3D assistant"
    }

    fun publish(command: AvatarCommand): AvatarFrameState {
        val state = stateBridge.publish(command)
        requestRender()
        return state
    }

    fun setInteractionListener(listener: AvatarInteractionListener?) {
        interactionListener = listener
    }

    fun onAvatarResume() {
        super.onResume()
        startFrameLoop()
    }

    fun onAvatarPause() {
        stopFrameLoop()
        super.onPause()
    }

    override fun onAttachedToWindow() {
        super.onAttachedToWindow()
        startFrameLoop()
    }

    override fun onDetachedFromWindow() {
        stopFrameLoop()
        super.onDetachedFromWindow()
    }

    override fun doFrame(frameTimeNanos: Long) {
        if (!frameLoopRunning) return
        val interval = AvatarFramePolicy.intervalNanos(stateBridge.snapshot().mode)
        if (
            lastRenderRequestNanos == 0L ||
            frameTimeNanos - lastRenderRequestNanos >= interval
        ) {
            requestRender()
            lastRenderRequestNanos = frameTimeNanos
        }
        Choreographer.getInstance().postFrameCallback(this)
    }

    override fun onTouchEvent(event: MotionEvent): Boolean {
        when (event.actionMasked) {
            MotionEvent.ACTION_DOWN -> {
                downX = event.x
                downY = event.y
                downMillis = event.eventTime
                dragging = false
                return true
            }
            MotionEvent.ACTION_MOVE -> {
                if (hypot(event.x - downX, event.y - downY) > touchSlop) {
                    dragging = true
                }
                return true
            }
            MotionEvent.ACTION_UP -> {
                val duration = (event.eventTime - downMillis).coerceAtLeast(0L)
                val type = when {
                    dragging -> AvatarInteractionType.DRAG
                    duration >= ViewConfiguration.getLongPressTimeout() -> AvatarInteractionType.LONG_PRESS
                    else -> AvatarInteractionType.TAP
                }
                val normalizedX = if (width > 0) (event.x / width).coerceIn(0f, 1f) else 0.5f
                val normalizedY = if (height > 0) (event.y / height).coerceIn(0f, 1f) else 0.5f
                interactionListener?.onAvatarInteraction(
                    AvatarInteraction(type, normalizedX, normalizedY, duration)
                )
                performClick()
                return true
            }
            MotionEvent.ACTION_CANCEL -> {
                dragging = false
                return true
            }
        }
        return super.onTouchEvent(event)
    }

    override fun performClick(): Boolean {
        super.performClick()
        return true
    }

    private fun startFrameLoop() {
        if (frameLoopRunning) return
        frameLoopRunning = true
        lastRenderRequestNanos = 0L
        Choreographer.getInstance().postFrameCallback(this)
    }

    private fun stopFrameLoop() {
        if (!frameLoopRunning) return
        frameLoopRunning = false
        Choreographer.getInstance().removeFrameCallback(this)
    }
}
