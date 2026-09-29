package ai.vn97.runtime

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import org.json.JSONArray
import org.json.JSONObject

/** Diagnostic only: synthetic token IDs, no tokenizer and no activation authority. */
object VN97Int8DeviceTrial {
    fun run(root: File, cancelled: () -> Boolean = { false }, memorySnapshot: (() -> VN97TrialMemory.Sample)? = null, checkpoint: (String) -> Unit): String {
        val env = OrtEnvironment.getEnvironment()
        val owned = mutableListOf<AutoCloseable>()
        fun tensor(shape: LongArray, count: Int) = Mamba2OrtTensorStorage.create(
            env, Mamba2OrtValueDtype.FLOAT16, shape, count,
        ).also { owned.add(it); it.zero() }
        val report = JSONObject().put("kind", "VN97_INT8_DEVICE_TRIAL_1")
            .put("production_activation_authorized", false).put("quality_qualified", false)
            .put("status", "running").put("synthetic_tokens", true).put("device", android.os.Build.FINGERPRINT)
            .put("ort_version", env.version).put("execution_passed", false)
        val cases = JSONArray()
        report.put("cases", cases)
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
            OrtSession.SessionOptions().use { options ->
                // Avoid retaining a large arena and allocating a learned memory pattern on run 2.
                options.setMemoryPatternOptimization(false)
                options.setCPUArenaAllocator(false)
                report.put("memory_pattern", false).put("cpu_arena", false)
                    .put("phase", "session_load")
                checkpoint(report.toString(2))
                options.setIntraOpNumThreads(2)
                options.setInterOpNumThreads(1)
                checkMemory()
                val loadStart = System.nanoTime()
                env.createSession(File(root, "candidate.onnx").absolutePath, options).use { session ->
                    report.put("session_load_ms", (System.nanoTime() - loadStart) / 1e6)
                    val conv = Array(2) { tensor(longArrayOf(64, 1, 5376, 4), 1376256) }
                    val ssm = Array(2) { tensor(longArrayOf(64, 1, 80, 64, 128), 41943040) }
                    val logits = tensor(longArrayOf(1, 8, 50288), 402304)
                    val ids = ByteBuffer.allocateDirect(64).order(ByteOrder.nativeOrder()).asLongBuffer()
                    val valid = ByteBuffer.allocateDirect(8).order(ByteOrder.nativeOrder()).asLongBuffer()
                    OnnxTensor.createTensor(env, ids, longArrayOf(1, 8)).use { idsTensor ->
                        OnnxTensor.createTensor(env, valid, longArrayOf(1)).use { validTensor ->
                            intArrayOf(1, 8, 1).forEachIndexed { index, length ->
                                if (cancelled()) throw java.util.concurrent.CancellationException("Đã dừng")
                                val from = if (index == 2) 1 else 0
                                val to = 1 - from
                                if (index < 2) { conv[from].zero(); ssm[from].zero() }
                                for (i in 0 until 8) ids.put(i, if (i < length) (100 + i).toLong() else 0L)
                                valid.put(0, length.toLong())
                                report.put("phase", "inference").put("pending_valid_length", length)
                                    .put("pss_before_inference_kib", android.os.Debug.getPss())
                                checkpoint(report.toString(2))
                                checkMemory()
                                val start = System.nanoTime()
                                session.run(mapOf("input_ids" to idsTensor, "valid_length" to validTensor,
                                    "conv_state" to conv[from].tensor, "ssm_state" to ssm[from].tensor),
                                    mapOf("logits" to logits.tensor, "next_conv_state" to conv[to].tensor,
                                        "next_ssm_state" to ssm[to].tensor)).use { }
                                report.put("phase", "validate_outputs")
                                val elapsed = (System.nanoTime() - start) / 1e6
                                check(logits.allFinite() && conv[to].allFinite() && ssm[to].allFinite()) { "NaN/Inf trong đầu ra" }
                                if (length < 8) check(logits.readFloatRange(length * 50288, (8 - length) * 50288).all { it == 0f })
                                val row = logits.readFloatRange((length - 1) * 50288, 50277)
                                cases.put(JSONObject().put("valid_length", length).put("reset", index < 2)
                                    .put("inference_ms", elapsed).put("pss_kib", android.os.Debug.getPss())
                                    .put("top1_token_id", row.indices.maxByOrNull { row[it] }).put("finite", true))
                                checkpoint(report.toString(2))
                            }
                        }
                    }
                }
            }
            report.put("phase", "completed").put("status", "completed").put("execution_passed", true)
            return report.toString(2)
        } catch (e: VN97TrialMemory.Pressure) {
            report.put("status", "memory_pressure").put("error", e.message)
            return report.toString(2).also(checkpoint)
        } finally { owned.asReversed().forEach { it.close() } }
    }
}
