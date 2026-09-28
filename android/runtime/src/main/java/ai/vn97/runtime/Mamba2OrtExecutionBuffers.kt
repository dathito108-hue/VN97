package ai.vn97.runtime

import ai.onnxruntime.OnnxTensor
import java.nio.LongBuffer

internal data class Mamba2StateSlot(
    val conv: Mamba2OrtTensorStorage,
    val ssm: Mamba2OrtTensorStorage,
) : AutoCloseable {
    fun zero() {
        conv.zero()
        ssm.zero()
    }

    override fun close() {
        conv.close()
        ssm.close()
    }
}

internal data class Mamba2GraphBuffers(
    val inputBuffer: LongBuffer,
    val validLengthBuffer: LongBuffer,
    val inputTensor: OnnxTensor,
    val validLengthTensor: OnnxTensor,
    val logits: Mamba2OrtTensorStorage,
    val maxChunkSize: Int,
    val vocabSize: Int,
) : AutoCloseable {
    fun put(tokens: IntArray) {
        require(tokens.isNotEmpty())
        require(tokens.size <= maxChunkSize)
        for (index in 0 until maxChunkSize) {
            inputBuffer.put(
                index,
                if (index < tokens.size) tokens[index].toLong() else 0L,
            )
        }
        validLengthBuffer.put(0, tokens.size.toLong())
    }

    fun finalLogits(validLength: Int): FloatArray {
        require(validLength in 1..maxChunkSize)
        val offset = Math.multiplyExact(validLength - 1, vocabSize)
        return logits.readFloatRange(offset, vocabSize)
    }

    override fun close() {
        inputTensor.close()
        validLengthTensor.close()
        logits.close()
    }
}

