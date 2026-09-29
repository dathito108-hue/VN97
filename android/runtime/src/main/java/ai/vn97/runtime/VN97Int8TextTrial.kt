package ai.vn97.runtime

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import org.json.JSONArray
import org.json.JSONObject

object VN97Int8TextTrial {
    const val TOKENIZER_ID = "27ce0a2f005befaa98ec4ee05d5d83a71aac1bcc5e4dd00032e33a17615ca575"
    fun tokenizer(root: File): VN97GptNeoXTokenizer {
        val info = Mamba2TokenizerPackage.load(root)
        require(info.tokenizerId == TOKENIZER_ID) { "Token hóa không khớp ứng viên INT8" }
        return VN97GptNeoXTokenizer(info)
    }
    fun run(root: File, tokenizerRoot: File, prompt: String, cancelled: () -> Boolean,
            decodeGraph: File? = null, disablePrepacking: Boolean = false, memorySnapshot: (() -> VN97TrialMemory.Sample)? = null, checkpoint: (String) -> Unit): String {
        require(prompt.length in 1..2048) { "Câu nhập dài tối đa 2048 ký tự." }
        val tokenizer = tokenizer(tokenizerRoot)
        val tokens = tokenizer.encode(prompt)
        require(tokens.size in 1..VN97Int8TextLoop.MAX_PROMPT_TOKENS) { "Giới hạn 128 token đầu vào; hãy rút ngắn câu." }
        val report = JSONObject().put("schema", "VN97_INT8_TEXT_TRIAL_1")
            .put("candidate_id", "331dcb61ad1547667fc24cb000fabe202d8380937b57859e0aa209814e4ab8fd")
            .put("tokenizer_id", TOKENIZER_ID).put("device", android.os.Build.FINGERPRINT)
            .put("quality_qualified", false).put("production_activation_authorized", false)
            .put("execution_passed", false).put("status", "running")
            .put("prompt_tokens", tokens.size).put("max_new_tokens", VN97Int8TextLoop.MAX_NEW_TOKENS)
            .put("generated_text", "").put("sampling", "greedy").put("cpu_threads", 2)
        report.put("decode_mode", if (decodeGraph == null) "chunk8" else "specialized_valid1")
            .put("prefill_chunk", if (decodeGraph == null) 8 else 1)
        val steps = JSONArray(); report.put("steps", steps)
        val owned = mutableListOf<AutoCloseable>()
        fun tensor(shape: LongArray, count: Int) = Mamba2OrtTensorStorage.create(
            OrtEnvironment.getEnvironment(), Mamba2OrtValueDtype.FLOAT16, shape, count,
        ).also { owned.add(it); it.zero() }
        fun checkMemory() {
            val sample = memorySnapshot?.invoke() ?: return
            report.put("system_memory", JSONObject()
                .put("available_bytes", sample.availableBytes).put("threshold_bytes", sample.thresholdBytes)
                .put("total_bytes", sample.totalBytes).put("low_memory", sample.lowMemory)
                .put("reserve_bytes", VN97TrialMemory.RESERVE_BYTES))
            checkpoint(report.toString(2))
            if (VN97TrialMemory.underPressure(sample)) throw VN97TrialMemory.Pressure()
        }
        try {
            checkpoint(report.toString(2))
            val env = OrtEnvironment.getEnvironment()
            report.put("ort_version", env.version)
            OrtSession.SessionOptions().use { options ->
                // Avoid retaining a large arena and allocating a learned memory pattern on run 2.
                options.setMemoryPatternOptimization(false)
                options.setCPUArenaAllocator(false)
                if (disablePrepacking) options.addConfigEntry("session.disable_prepacking", "1")
                report.put("disable_prepacking", disablePrepacking)
                report.put("memory_pattern", false).put("cpu_arena", false)
                    .put("phase", "session_load")
                checkpoint(report.toString(2))
                options.setIntraOpNumThreads(2); options.setInterOpNumThreads(1)
                if (cancelled()) throw java.util.concurrent.CancellationException()
                checkMemory()
                val loadStart = System.nanoTime()
                env.createSession((decodeGraph ?: File(root, "candidate.onnx")).absolutePath, options).use { session ->
                    report.put("session_load_ms", (System.nanoTime() - loadStart) / 1e6)
                    val conv = Array(2) { tensor(longArrayOf(64, 1, 5376, 4), 1376256) }
                    val ssm = Array(2) { tensor(longArrayOf(64, 1, 80, 64, 128), 41943040) }
                    val logits = tensor(longArrayOf(1, 8, 50288), 402304)
                    val ids = ByteBuffer.allocateDirect(64).order(ByteOrder.nativeOrder()).asLongBuffer()
                    val valid = ByteBuffer.allocateDirect(8).order(ByteOrder.nativeOrder()).asLongBuffer()
                    OnnxTensor.createTensor(env, ids, longArrayOf(1, 8)).use { input ->
                        OnnxTensor.createTensor(env, valid, longArrayOf(1)).use { length ->
                            var from = 0
                            val result = VN97Int8TextLoop.generate(tokens, tokenizer.packageInfo.eosTokenId, cancelled, { chunk ->
                                val to = 1 - from
                                for (i in 0 until 8) ids.put(i, if (i < chunk.size) chunk[i].toLong() else 0L)
                                valid.put(0, chunk.size.toLong())
                                report.put("phase", "inference").put("pending_valid_length", chunk.size)
                                    .put("pss_before_inference_kib", android.os.Debug.getPss())
                                checkpoint(report.toString(2))
                                checkMemory()
                                val start = System.nanoTime()
                                val inputs = mutableMapOf("input_ids" to input,
                                    "conv_state" to conv[from].tensor, "ssm_state" to ssm[from].tensor)
                                if (decodeGraph == null) inputs["valid_length"] = length
                                session.run(inputs,
                                    mapOf("logits" to logits.tensor, "next_conv_state" to conv[to].tensor,
                                        "next_ssm_state" to ssm[to].tensor)).use { }
                                report.put("phase", "validate_outputs")
                                val ms = (System.nanoTime() - start) / 1e6
                                check(logits.allFinite() && conv[to].allFinite() && ssm[to].allFinite()) { "NaN/Inf trong kết quả" }
                                from = to
                                steps.put(JSONObject().put("valid_length", chunk.size).put("inference_ms", ms)
                                    .put("pss_kib", android.os.Debug.getPss()))
                                checkpoint(report.toString(2))
                                logits.readFloatRange((chunk.size - 1) * 50288, 50288)
                            }, { generated ->
                                report.put("generated_tokens", generated.size)
                                    .put("generated_text", tokenizer.decode(generated, skipEos = true))
                                checkpoint(report.toString(2))
                            }, prefillChunk = if (decodeGraph == null) 8 else 1)
                            report.put("generated_tokens", result.size)
                                .put("stop_reason", if (result.size == VN97Int8TextLoop.MAX_NEW_TOKENS) "token_limit" else "eos")
                        }
                    }
                }
            }
            report.put("phase", "completed").put("status", "completed").put("execution_passed", true).put("status", "completed")
        } catch (e: Exception) {
            report.put("status", when (e) { is VN97TrialMemory.Pressure -> "memory_pressure"; is java.util.concurrent.CancellationException -> "cancelled"; else -> "failed" })
                .put("error_class", e.javaClass.simpleName).put("error", e.message ?: "Không hoàn tất")
        } finally { owned.asReversed().forEach { it.close() } }
        return report.toString(2).also(checkpoint)
    }
}
