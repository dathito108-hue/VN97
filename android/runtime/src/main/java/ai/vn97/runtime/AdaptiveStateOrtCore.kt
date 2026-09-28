package ai.vn97.runtime

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.annotation.SuppressLint
import org.json.JSONObject
import java.io.File
import java.nio.FloatBuffer
import java.nio.LongBuffer
import java.nio.file.Files
import java.security.MessageDigest

/** Experimental PC/Android core. Not wired to the canonical assistant or activation inventory. */
class AdaptiveStateOrtCore(
    directory: File,
    private val expectedManifestSha256: String,
    private val threads: Int = 1,
) : AutoCloseable {
    data class State(val manifestSha256: String, val position: Long, val fast: FloatArray, val slow: FloatArray)
    data class Step(val logits: FloatArray, val state: State)
    private val root = directory.canonicalFile
    private val environment = OrtEnvironment.getEnvironment()
    private val shape: LongArray
    private val elements: Int
    private var session: OrtSession? = null
    private var chunk = 0
    private var closed = false

    init {
        require(threads in 1..16)
        require(expectedManifestSha256.matches(Regex("[0-9a-f]{64}")))
        val manifestFile = regular("manifest.json")
        require(manifestFile.length() in 1..65536 && hash(manifestFile) == expectedManifestSha256) {
            "adaptive core manifest identity mismatch"
        }
        val manifest = JSONObject(manifestFile.readText(Charsets.UTF_8))
        require(manifest.getString("schema") == "VN97ASCORT1")
        require(manifest.getString("dtype") == "float32")
        require(manifest.getString("tokenizer") == "utf8-bytes-256")
        require(!manifest.getBoolean("production_activation_authorized"))
        val states = manifest.getJSONArray("state_shape")
        require(states.length() == 3)
        shape = LongArray(3) { states.getLong(it) }
        require(shape[0] in 1L..8L && shape[1] == 1L && shape[2] in 1L..1024L)
        elements = (shape[0] * shape[2]).toInt()
        val graphs = manifest.getJSONObject("graphs")
        require(graphs.keys().asSequence().toSet() == setOf("1", "8", "32"))
        for (size in listOf(1, 8, 32)) require(graphs.getString("$size") == "chunk-$size.onnx")
        val files = manifest.getJSONObject("files")
        require(files.keys().asSequence().toSet() == setOf("chunk-1.onnx", "chunk-8.onnx", "chunk-32.onnx", "weights.bin"))
        for (name in files.keys()) {
            val file = regular(name)
            val identity = files.getJSONObject(name)
            require(file.length() in 1..(128L * 1024 * 1024))
            require(file.length() == identity.getLong("bytes") && hash(file) == identity.getString("sha256")) {
                "adaptive core file identity mismatch: $name"
            }
        }
    }

    @Synchronized
    fun advance(ids: LongArray, prior: State? = null): Step {
        check(!closed)
        require(ids.size in listOf(1, 8, 32) && ids.all { it in 0L..255L })
        val state = prior ?: State(expectedManifestSha256, 0, FloatArray(elements), FloatArray(elements))
        require(state.manifestSha256 == expectedManifestSha256 && state.position >= 0)
        require(state.position <= Long.MAX_VALUE - ids.size)
        require(state.fast.size == elements && state.slow.size == elements)
        require(state.fast.all { it.isFinite() } && state.slow.all { it.isFinite() })
        if (chunk != ids.size) {
            session?.close(); session = null; chunk = 0
            OrtSession.SessionOptions().use { options ->
                options.setIntraOpNumThreads(threads)
                session = environment.createSession(regular("chunk-${ids.size}.onnx").path, options)
            }
            chunk = ids.size
        }
        OnnxTensor.createTensor(environment, LongBuffer.wrap(ids.copyOf()), longArrayOf(1, ids.size.toLong())).use { tokens ->
            OnnxTensor.createTensor(environment, FloatBuffer.wrap(state.fast.copyOf()), shape).use { fast ->
                OnnxTensor.createTensor(environment, FloatBuffer.wrap(state.slow.copyOf()), shape).use { slow ->
                    checkNotNull(session).run(mapOf("input_ids" to tokens, "fast_state" to fast, "slow_state" to slow)).use { result ->
                        fun read(name: String, expectedShape: LongArray, count: Int): FloatArray {
                            val tensor = result.get(name).orElseThrow { IllegalStateException("missing $name") } as OnnxTensor
                            require(tensor.info.shape.contentEquals(expectedShape))
                            val values = FloatArray(count)
                            tensor.floatBuffer.get(values)
                            require(values.all { it.isFinite() })
                            return values
                        }
                        val logits = read("logits", longArrayOf(1, ids.size.toLong(), 256), ids.size * 256)
                        val nextFast = read("next_fast_state", shape, elements)
                        val nextSlow = read("next_slow_state", shape, elements)
                        return Step(logits, State(expectedManifestSha256, state.position + ids.size, nextFast, nextSlow))
                    }
                }
            }
        }
    }

    @Synchronized
    override fun close() {
        if (!closed) { closed = true; session?.close(); session = null }
    }

    private fun regular(name: String): File = File(root, name).also {
        require(it.isFile && !Files.isSymbolicLink(it.toPath()) && it.canonicalFile.parentFile == root)
    }

    private fun hash(file: File): String {
        val digest = MessageDigest.getInstance("SHA-256")
        file.inputStream().use { source ->
            val buffer = ByteArray(64 * 1024)
            while (true) { val n = source.read(buffer); if (n < 0) break; check(n > 0); digest.update(buffer, 0, n) }
        }
        return digest.digest().joinToString("") { "%02x".format(it.toInt() and 0xff) }
    }
}
