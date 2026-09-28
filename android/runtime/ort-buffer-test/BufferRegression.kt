package ai.vn97.runtime

import ai.onnxruntime.OnnxJavaType
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.nio.ByteBuffer
import java.nio.ByteOrder

private fun direct(bytes: Int) =
    ByteBuffer.allocateDirect(bytes).order(ByteOrder.nativeOrder())

fun main(args: Array<String>) {
    val env = OrtEnvironment.getEnvironment()
    val shape = longArrayOf(1L, 2L, 3L)
    for (dtype in Mamba2OrtValueDtype.entries) {
        val half = dtype == Mamba2OrtValueDtype.FLOAT16
        val model = args.single() + if (half) "/float16.onnx" else "/float32.onnx"
        OrtSession.SessionOptions().use { options ->
            env.createSession(model, options).use { session ->
                val expectedType = if (half) OnnxJavaType.FLOAT16 else OnnxJavaType.FLOAT
                val inputBytes = direct(if (half) 12 else 24)
                val shorts = inputBytes.asShortBuffer()
                val floats = inputBytes.asFloatBuffer()
                if (half) {
                    // Exactly representable IEEE binary16: 0, 1, -2, 0.5, 4, -8.
                    intArrayOf(0, 0x3c00, 0xc000, 0x3800, 0x4400, 0xc800)
                        .forEachIndexed { i, v -> shorts.put(i, v.toShort()) }
                } else {
                    floats.put(floatArrayOf(0f, 1f, -2f, 0.5f, 4f, -8f)).rewind()
                }
                val input = if (half) {
                    OnnxTensor.createTensor(env, shorts, shape, OnnxJavaType.FLOAT16)
                } else {
                    OnnxTensor.createTensor(env, floats, shape)
                }
                input.use {
                    val ids = direct(2 * Long.SIZE_BYTES).asLongBuffer()
                    val valid = direct(Long.SIZE_BYTES).asLongBuffer()
                    Mamba2GraphBuffers(
                        inputBuffer = ids,
                        validLengthBuffer = valid,
                        inputTensor = OnnxTensor.createTensor(env, ids, longArrayOf(1L, 2L)),
                        validLengthTensor = OnnxTensor.createTensor(env, valid, longArrayOf(1L)),
                        logits = Mamba2OrtTensorStorage.create(env, dtype, shape, 6),
                        maxChunkSize = 2,
                        vocabSize = 3,
                    ).use { buffers ->
                        check(buffers.logits.tensor.info.type == expectedType)
                        buffers.put(intArrayOf(5, 6))
                        buffers.put(intArrayOf(7))
                        check(ids.get(0) == 7L && ids.get(1) == 0L && valid.get(0) == 1L)
                        session.run(mapOf("input" to input), mapOf("output" to buffers.logits.tensor)).use { }
                        check(buffers.finalLogits(1).contentEquals(floatArrayOf(0f, 1f, -2f)))
                        check(buffers.finalLogits(2).contentEquals(floatArrayOf(0.5f, 4f, -8f)))
                        check(runCatching { buffers.finalLogits(0) }.isFailure)
                        check(runCatching { buffers.finalLogits(3) }.isFailure)
                        Mamba2StateSlot(
                            Mamba2OrtTensorStorage.create(env, dtype, shape, 6),
                            Mamba2OrtTensorStorage.create(env, dtype, shape, 6),
                        ).use { state ->
                            check(state.conv.tensor.info.type == expectedType)
                            check(state.ssm.tensor.info.type == expectedType)
                            // Reuse pinned output directly as next recurrent input, without F32 conversion.
                            session.run(mapOf("input" to buffers.logits.tensor), mapOf("output" to state.conv.tensor)).use { }
                            session.run(mapOf("input" to state.conv.tensor), mapOf("output" to state.ssm.tensor)).use { }
                            check(state.ssm.readFloatRange(3, 3).contentEquals(buffers.finalLogits(2)))
                            state.zero()
                            check(state.conv.readFloatRange(0, 6).all { it == 0f })
                            check(state.ssm.readFloatRange(0, 6).all { it == 0f })
                            // A second run overwrites the same pinned logits allocation.
                            session.run(mapOf("input" to state.conv.tensor), mapOf("output" to buffers.logits.tensor)).use { }
                            check(buffers.finalLogits(2).all { it == 0f })
                        }
                    }
                }
            }
        }
        println("$dtype pinned output, recurrent reuse, reset, final-logit slice: PASS")
    }
}
