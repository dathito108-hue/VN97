package ai.vn97.avatar

import android.opengl.GLES30
import android.opengl.GLSurfaceView
import android.opengl.Matrix
import java.nio.ByteBuffer
import java.nio.ByteOrder
import javax.microedition.khronos.egl.EGLConfig
import javax.microedition.khronos.opengles.GL10
import kotlin.math.sin

internal class AvatarRenderer(
    private val stateBridge: AvatarStateBridge,
) : GLSurfaceView.Renderer {
    private var program = 0
    private var vertexBuffer = 0
    private var mvpLocation = -1
    private var colorLocation = -1
    private val projection = FloatArray(16)
    private val view = FloatArray(16)
    private val model = FloatArray(16)
    private val viewModel = FloatArray(16)
    private val mvp = FloatArray(16)
    private val filter = AvatarMotionFilter()
    private var lastFrameNanos = 0L
    private var startNanos = 0L

    override fun onSurfaceCreated(gl: GL10?, config: EGLConfig?) {
        program = linkProgram(VERTEX_SHADER, FRAGMENT_SHADER)
        mvpLocation = GLES30.glGetUniformLocation(program, "uMvp")
        colorLocation = GLES30.glGetUniformLocation(program, "uColor")
        check(mvpLocation >= 0 && colorLocation >= 0) { "avatar shader uniforms unavailable" }

        val buffers = IntArray(1)
        GLES30.glGenBuffers(1, buffers, 0)
        vertexBuffer = buffers[0]
        check(vertexBuffer != 0) { "avatar vertex buffer allocation failed" }

        val data = ByteBuffer.allocateDirect(CUBE_VERTICES.size * Float.SIZE_BYTES)
            .order(ByteOrder.nativeOrder())
            .asFloatBuffer()
            .put(CUBE_VERTICES)
        data.position(0)
        GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, vertexBuffer)
        GLES30.glBufferData(
            GLES30.GL_ARRAY_BUFFER,
            CUBE_VERTICES.size * Float.SIZE_BYTES,
            data,
            GLES30.GL_STATIC_DRAW,
        )
        GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, 0)
        GLES30.glEnable(GLES30.GL_DEPTH_TEST)
        GLES30.glClearColor(0f, 0f, 0f, 0f)
        startNanos = System.nanoTime()
        lastFrameNanos = 0L
    }

    override fun onSurfaceChanged(gl: GL10?, width: Int, height: Int) {
        GLES30.glViewport(0, 0, width, height)
        val aspect = width.toFloat() / height.coerceAtLeast(1).toFloat()
        Matrix.perspectiveM(projection, 0, 42f, aspect, 0.1f, 100f)
        Matrix.setLookAtM(view, 0, 0f, 0.2f, 5.2f, 0f, 0f, 0f, 0f, 1f, 0f)
    }

    override fun onDrawFrame(gl: GL10?) {
        val now = System.nanoTime()
        val delta = if (lastFrameNanos == 0L) 0f else (now - lastFrameNanos) / 1_000_000_000f
        lastFrameNanos = now
        val target = stateBridge.snapshot()
        val animated = filter.step(target, delta)
        val t = (now - startNanos).coerceAtLeast(0L) / 1_000_000_000.0

        GLES30.glClear(GLES30.GL_COLOR_BUFFER_BIT or GLES30.GL_DEPTH_BUFFER_BIT)
        GLES30.glUseProgram(program)
        GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, vertexBuffer)
        GLES30.glEnableVertexAttribArray(0)
        GLES30.glVertexAttribPointer(0, 3, GLES30.GL_FLOAT, false, 3 * Float.SIZE_BYTES, 0)

        val accent = accentColor(target.mode)
        val breathing = sin(t * (1.4 + animated.energy)).toFloat() * 0.025f * animated.energy
        val listeningLift = if (target.mode == AssistantMode.LISTENING) {
            animated.listeningLevel * 0.045f
        } else {
            0f
        }
        val thinking = if (target.mode == AssistantMode.THINKING) sin(t * 1.7).toFloat() * 5f else 0f
        val gestureWave = if (target.gesture == AvatarGesture.WAVE) sin(t * 8.0).toFloat() * 35f else 0f
        val nod = if (target.gesture == AvatarGesture.NOD) sin(t * 7.0).toFloat() * 8f else 0f
        val shake = if (target.gesture == AvatarGesture.SHAKE) sin(t * 7.5).toFloat() * 10f else 0f

        drawPart(0f, -0.45f + breathing, 0f, 1.05f, 1.25f, 0.52f, 0f, 0f, 0f, accent)
        drawPart(
            0f,
            0.68f + breathing + listeningLift,
            0f,
            0.88f,
            0.78f,
            0.68f,
            nod - animated.gazeY * 7f,
            thinking + shake + animated.gazeX * 9f,
            0f,
            floatArrayOf(0.72f, 0.78f, 0.88f),
        )

        val eyeHeight = (0.055f * (1f - animated.blink * 0.92f)).coerceAtLeast(0.006f)
        val eyeX = animated.gazeX * 0.035f
        val eyeY = animated.gazeY * 0.025f
        drawPart(-0.22f + eyeX, 0.76f + eyeY + breathing, 0.36f, 0.12f, eyeHeight, 0.04f, 0f, 0f, 0f, floatArrayOf(0.08f, 0.12f, 0.18f))
        drawPart(0.22f + eyeX, 0.76f + eyeY + breathing, 0.36f, 0.12f, eyeHeight, 0.04f, 0f, 0f, 0f, floatArrayOf(0.08f, 0.12f, 0.18f))

        val jaw = maxOf(animated.speakingLevel * 0.55f, animated.jawOpen)
        val mouthHeight = 0.025f + jaw * 0.12f
        val mouthWidth = (
            0.24f + animated.mouthWide * 0.18f - animated.lipRound * 0.10f
        ).coerceIn(0.12f, 0.42f)
        val mouthDepth = 0.035f + animated.lipRound * 0.025f
        drawPart(
            0f,
            0.49f + breathing,
            0.365f,
            mouthWidth,
            mouthHeight,
            mouthDepth,
            0f,
            0f,
            0f,
            accent,
        )

        drawPart(-0.72f, -0.36f + breathing, 0f, 0.2f, 0.9f, 0.22f, 0f, 0f, 12f + gestureWave, accent)
        drawPart(0.72f, -0.36f + breathing, 0f, 0.2f, 0.9f, 0.22f, 0f, 0f, -12f, accent)

        GLES30.glDisableVertexAttribArray(0)
        GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, 0)
    }

    private fun drawPart(
        tx: Float,
        ty: Float,
        tz: Float,
        sx: Float,
        sy: Float,
        sz: Float,
        rx: Float,
        ry: Float,
        rz: Float,
        color: FloatArray,
    ) {
        Matrix.setIdentityM(model, 0)
        Matrix.translateM(model, 0, tx, ty, tz)
        Matrix.rotateM(model, 0, rz, 0f, 0f, 1f)
        Matrix.rotateM(model, 0, ry, 0f, 1f, 0f)
        Matrix.rotateM(model, 0, rx, 1f, 0f, 0f)
        Matrix.scaleM(model, 0, sx, sy, sz)
        Matrix.multiplyMM(viewModel, 0, view, 0, model, 0)
        Matrix.multiplyMM(mvp, 0, projection, 0, viewModel, 0)
        GLES30.glUniformMatrix4fv(mvpLocation, 1, false, mvp, 0)
        GLES30.glUniform3f(colorLocation, color[0], color[1], color[2])
        GLES30.glDrawArrays(GLES30.GL_TRIANGLES, 0, CUBE_VERTICES.size / 3)
    }

    private fun accentColor(mode: AssistantMode): FloatArray = when (mode) {
        AssistantMode.IDLE, AssistantMode.SLEEPING -> floatArrayOf(0.32f, 0.48f, 0.72f)
        AssistantMode.LISTENING -> floatArrayOf(0.20f, 0.72f, 0.82f)
        AssistantMode.THINKING -> floatArrayOf(0.48f, 0.42f, 0.82f)
        AssistantMode.SPEAKING -> floatArrayOf(0.20f, 0.78f, 0.58f)
        AssistantMode.WAITING_APPROVAL -> floatArrayOf(0.88f, 0.64f, 0.18f)
        AssistantMode.EXECUTING -> floatArrayOf(0.28f, 0.68f, 0.92f)
        AssistantMode.ERROR -> floatArrayOf(0.88f, 0.24f, 0.28f)
    }

    private fun linkProgram(vertex: String, fragment: String): Int {
        val vertexShader = compileShader(GLES30.GL_VERTEX_SHADER, vertex)
        val fragmentShader = compileShader(GLES30.GL_FRAGMENT_SHADER, fragment)
        val linked = GLES30.glCreateProgram()
        check(linked != 0) { "avatar program allocation failed" }
        GLES30.glAttachShader(linked, vertexShader)
        GLES30.glAttachShader(linked, fragmentShader)
        GLES30.glLinkProgram(linked)
        val status = IntArray(1)
        GLES30.glGetProgramiv(linked, GLES30.GL_LINK_STATUS, status, 0)
        GLES30.glDeleteShader(vertexShader)
        GLES30.glDeleteShader(fragmentShader)
        if (status[0] == 0) {
            val log = GLES30.glGetProgramInfoLog(linked)
            GLES30.glDeleteProgram(linked)
            error("avatar program link failed: $log")
        }
        return linked
    }

    private fun compileShader(type: Int, source: String): Int {
        val shader = GLES30.glCreateShader(type)
        check(shader != 0) { "avatar shader allocation failed" }
        GLES30.glShaderSource(shader, source)
        GLES30.glCompileShader(shader)
        val status = IntArray(1)
        GLES30.glGetShaderiv(shader, GLES30.GL_COMPILE_STATUS, status, 0)
        if (status[0] == 0) {
            val log = GLES30.glGetShaderInfoLog(shader)
            GLES30.glDeleteShader(shader)
            error("avatar shader compile failed: $log")
        }
        return shader
    }

    companion object {
        private const val VERTEX_SHADER = """#version 300 es
            layout(location = 0) in vec3 aPosition;
            uniform mat4 uMvp;
            void main() {
                gl_Position = uMvp * vec4(aPosition, 1.0);
            }
        """

        private const val FRAGMENT_SHADER = """#version 300 es
            precision mediump float;
            uniform vec3 uColor;
            out vec4 outColor;
            void main() {
                outColor = vec4(uColor, 1.0);
            }
        """

        private val CUBE_VERTICES = floatArrayOf(
            -0.5f,-0.5f, 0.5f,  0.5f,-0.5f, 0.5f,  0.5f, 0.5f, 0.5f,
            -0.5f,-0.5f, 0.5f,  0.5f, 0.5f, 0.5f, -0.5f, 0.5f, 0.5f,
             0.5f,-0.5f,-0.5f, -0.5f,-0.5f,-0.5f, -0.5f, 0.5f,-0.5f,
             0.5f,-0.5f,-0.5f, -0.5f, 0.5f,-0.5f,  0.5f, 0.5f,-0.5f,
            -0.5f,-0.5f,-0.5f, -0.5f,-0.5f, 0.5f, -0.5f, 0.5f, 0.5f,
            -0.5f,-0.5f,-0.5f, -0.5f, 0.5f, 0.5f, -0.5f, 0.5f,-0.5f,
             0.5f,-0.5f, 0.5f,  0.5f,-0.5f,-0.5f,  0.5f, 0.5f,-0.5f,
             0.5f,-0.5f, 0.5f,  0.5f, 0.5f,-0.5f,  0.5f, 0.5f, 0.5f,
            -0.5f, 0.5f, 0.5f,  0.5f, 0.5f, 0.5f,  0.5f, 0.5f,-0.5f,
            -0.5f, 0.5f, 0.5f,  0.5f, 0.5f,-0.5f, -0.5f, 0.5f,-0.5f,
            -0.5f,-0.5f,-0.5f,  0.5f,-0.5f,-0.5f,  0.5f,-0.5f, 0.5f,
            -0.5f,-0.5f,-0.5f,  0.5f,-0.5f, 0.5f, -0.5f,-0.5f, 0.5f,
        )
    }
}
