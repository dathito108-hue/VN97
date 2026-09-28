package ai.vn97.avatar

import android.opengl.GLES30
import android.opengl.GLSurfaceView
import android.opengl.Matrix
import java.nio.ByteBuffer
import java.nio.ByteOrder
import javax.microedition.khronos.egl.EGLConfig
import javax.microedition.khronos.opengles.GL10
import kotlin.math.sin
import kotlin.math.cos
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.sqrt
import kotlin.math.sign

internal class AvatarRenderer(
    private val stateBridge: AvatarStateBridge,
) : GLSurfaceView.Renderer {
    private var program = 0
    private var vertexBuffer = 0
    private var mvpLocation = -1
    private var colorLocation = -1
    private var modelLocation = -1
    private var materialLocation = -1
    private val parent = FloatArray(16)
    private val worldModel = FloatArray(16)
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
        modelLocation = GLES30.glGetUniformLocation(program, "uModel")
        materialLocation = GLES30.glGetUniformLocation(program, "uMaterial")
        check(mvpLocation >= 0 && colorLocation >= 0 && modelLocation >= 0 && materialLocation >= 0) { "avatar shader uniforms unavailable" }

        val buffers = IntArray(1)
        GLES30.glGenBuffers(1, buffers, 0)
        vertexBuffer = buffers[0]
        check(vertexBuffer != 0) { "avatar vertex buffer allocation failed" }

        val data = ByteBuffer.allocateDirect(ROUNDED_VERTICES.size * Float.SIZE_BYTES)
            .order(ByteOrder.nativeOrder())
            .asFloatBuffer()
            .put(ROUNDED_VERTICES)
        data.position(0)
        GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, vertexBuffer)
        GLES30.glBufferData(
            GLES30.GL_ARRAY_BUFFER,
            ROUNDED_VERTICES.size * Float.SIZE_BYTES,
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

        // Pearl ceramic shell, graphite joints and a luminous inset core.
        Matrix.setIdentityM(parent, 0)
        Matrix.translateM(parent, 0, 0f, breathing, 0f)
        Matrix.rotateM(parent, 0, -8f, 0f, 1f, 0f)
        drawPart(0f, -0.46f, 0f, 0.90f, 0.88f, 0.58f, 0f, 0f, 0f, PEARL)
        drawPart(0f, -0.45f, 0.30f, 0.61f, 0.49f, 0.09f, 0f, 0f, 0f, GRAPHITE)
        drawPart(0f, -0.43f, 0.36f, 0.19f, 0.19f, 0.035f, 0f, 0f, 45f, accent, 0.8f)
        drawPart(0f, -0.70f, 0.345f, 0.30f, 0.025f, 0.02f, 0f, 0f, 0f, accent, 0.5f)
        drawPart(0f, -0.98f, 0f, 0.55f, 0.11f, 0.41f, 0f, 0f, 0f, GRAPHITE)
        drawPart(0f, -1.06f, 0f, 0.38f, 0.04f, 0.28f, 0f, 0f, 0f, accent, 0.75f)
        drawPart(0f, 0.12f, 0f, 0.30f, 0.26f, 0.30f, 0f, 0f, 0f, GRAPHITE)
        drawPart(-0.56f, -0.24f, 0f, 0.22f, 0.26f, 0.32f, 0f, 0f, 15f, GRAPHITE)
        drawPart(0.56f, -0.24f, 0f, 0.22f, 0.26f, 0.32f, 0f, 0f, -15f, GRAPHITE)
        drawPart(-0.68f, -0.48f, 0f, 0.23f, 0.52f, 0.31f, 0f, 0f, 14f + gestureWave, PEARL)
        drawPart(0.68f, -0.48f, 0f, 0.23f, 0.52f, 0.31f, 0f, 0f, -14f, PEARL)

        // Head, faceplate and expressions share one transform: no detached eyes while nodding.
        Matrix.setIdentityM(parent, 0)
        Matrix.translateM(parent, 0, 0f, 0.70f + breathing + listeningLift, 0f)
        Matrix.rotateM(parent, 0, -8f + thinking + shake + animated.gazeX * 9f, 0f, 1f, 0f)
        Matrix.rotateM(parent, 0, nod - animated.gazeY * 7f, 1f, 0f, 0f)
        drawPart(0f, 0f, 0f, 1.46f, 1.02f, 0.80f, 0f, 0f, 0f, PEARL)
        drawPart(-0.76f, 0f, 0f, 0.12f, 0.35f, 0.35f, 0f, 0f, 0f, GRAPHITE)
        drawPart(0.76f, 0f, 0f, 0.12f, 0.35f, 0.35f, 0f, 0f, 0f, GRAPHITE)
        drawPart(-0.825f, 0f, 0f, 0.02f, 0.18f, 0.20f, 0f, 0f, 0f, accent, 0.55f)
        drawPart(0.825f, 0f, 0f, 0.02f, 0.18f, 0.20f, 0f, 0f, 0f, accent, 0.55f)
        drawPart(0f, 0f, 0.385f, 1.29f, 0.76f, 0.15f, 0f, 0f, 0f, METAL)
        drawPart(0f, 0f, 0.435f, 1.19f, 0.65f, 0.10f, 0f, 0f, 0f, GLASS)
        val eyeHeight = (0.22f * (1f - animated.blink * 0.92f)).coerceAtLeast(0.018f)
        val eyeX = animated.gazeX * 0.045f
        val eyeY = animated.gazeY * 0.03f
        drawPart(-0.27f + eyeX, 0.055f + eyeY, 0.496f, 0.14f, eyeHeight, 0.018f, 0f, 0f, 0f, accent, 1f)
        drawPart(0.27f + eyeX, 0.055f + eyeY, 0.496f, 0.14f, eyeHeight, 0.018f, 0f, 0f, 0f, accent, 1f)
        val jaw = maxOf(animated.speakingLevel * 0.55f, animated.jawOpen)
        val mouthWidth = (0.24f + animated.mouthWide * 0.13f - animated.lipRound * 0.08f).coerceIn(0.12f, 0.38f)
        drawPart(0f, -0.19f, 0.495f, mouthWidth, 0.028f + jaw * 0.09f, 0.015f, 0f, 0f, 0f, accent, 0.8f)

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
        emission: Float = 0f,
    ) {
        Matrix.setIdentityM(model, 0)
        Matrix.translateM(model, 0, tx, ty, tz)
        Matrix.rotateM(model, 0, rz, 0f, 0f, 1f)
        Matrix.rotateM(model, 0, ry, 0f, 1f, 0f)
        Matrix.rotateM(model, 0, rx, 1f, 0f, 0f)
        Matrix.scaleM(model, 0, sx, sy, sz)
        Matrix.multiplyMM(worldModel, 0, parent, 0, model, 0)
        Matrix.multiplyMM(viewModel, 0, view, 0, worldModel, 0)
        Matrix.multiplyMM(mvp, 0, projection, 0, viewModel, 0)
        GLES30.glUniformMatrix4fv(mvpLocation, 1, false, mvp, 0)
        GLES30.glUniformMatrix4fv(modelLocation, 1, false, worldModel, 0)
        GLES30.glUniform2f(materialLocation, 0.32f, emission)
        GLES30.glUniform3f(colorLocation, color[0], color[1], color[2])
        GLES30.glDrawArrays(GLES30.GL_TRIANGLES, 0, ROUNDED_VERTICES.size / 3)
    }

    private fun accentColor(mode: AssistantMode): FloatArray = when (mode) {
        AssistantMode.IDLE, AssistantMode.SLEEPING -> floatArrayOf(0.18f, 0.90f, 0.83f)
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
            uniform mat4 uModel;
            out mediump vec3 vNormal;
            void main() {
                gl_Position = uMvp * vec4(aPosition, 1.0);
                vNormal = transpose(inverse(mat3(uModel))) * normalize(aPosition * aPosition * aPosition);
            }
        """

        private const val FRAGMENT_SHADER = """#version 300 es
            precision mediump float;
            uniform vec3 uColor;
            uniform vec2 uMaterial;
            in mediump vec3 vNormal;
            out vec4 outColor;
            void main() {
                vec3 normal = normalize(vNormal);
                vec3 light = normalize(vec3(-0.5, 0.8, 1.2));
                float diffuse = max(dot(normal, light), 0.0);
                float rim = pow(1.0 - abs(normal.z), 3.0);
                float specular = pow(max(dot(normal, normalize(light + vec3(0.0, 0.0, 1.0))), 0.0), 48.0);
                vec3 shaded = uColor * (0.48 + 0.52 * diffuse)
                    + vec3(0.78, 0.88, 1.0) * specular * uMaterial.x
                    + vec3(0.04, 0.09, 0.11) * rim;
                outColor = vec4(mix(shaded, uColor, uMaterial.y), 1.0);
            }
        """

        private val PEARL = floatArrayOf(0.86f, 0.91f, 0.96f)
        private val GRAPHITE = floatArrayOf(0.055f, 0.085f, 0.12f)
        private val METAL = floatArrayOf(0.22f, 0.32f, 0.40f)
        private val GLASS = floatArrayOf(0.008f, 0.020f, 0.030f)

        // Shared rounded-box superellipsoid mesh: generated once, uploaded once, no frame allocations.
        private val ROUNDED_VERTICES: FloatArray = buildList<Float> {
            val rings = 16
            val segments = 24
            fun vertex(ring: Int, segment: Int) {
                val latitude = PI * ring / rings
                val longitude = 2.0 * PI * segment / segments
                fun rounded(value: Double) = sign(value) * sqrt(abs(value))
                add((0.5 * rounded(sin(latitude)) * rounded(cos(longitude))).toFloat())
                add((0.5 * rounded(cos(latitude))).toFloat())
                add((0.5 * rounded(sin(latitude)) * rounded(sin(longitude))).toFloat())
            }
            for (ring in 0 until rings) for (segment in 0 until segments) {
                vertex(ring, segment)
                vertex(ring + 1, segment)
                vertex(ring + 1, segment + 1)
                vertex(ring, segment)
                vertex(ring + 1, segment + 1)
                vertex(ring, segment + 1)
            }
        }.toFloatArray()
    }
}
