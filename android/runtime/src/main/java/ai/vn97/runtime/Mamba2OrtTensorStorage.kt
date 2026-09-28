package ai.vn97.runtime

import ai.onnxruntime.OnnxJavaType
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.nio.ShortBuffer

internal enum class Mamba2OrtValueDtype {
    FLOAT16,
    FLOAT32;

    companion object {
        fun fromRuntime(value: String): Mamba2OrtValueDtype =
            when (value) {
                "float16" -> FLOAT16
                "float32" -> FLOAT32
                else -> error("unsupported Mamba-2 ORT dtype: $value")
            }
    }
}

internal class Mamba2OrtTensorStorage private constructor(
    private val dtype: Mamba2OrtValueDtype,
    private val floatBuffer: FloatBuffer?,
    private val shortBuffer: ShortBuffer?,
    val tensor: OnnxTensor,
    val elementCount: Int,
) : AutoCloseable {
    companion object {
        fun create(
            environment: OrtEnvironment,
            dtype: Mamba2OrtValueDtype,
            shape: LongArray,
            elementCount: Int,
        ): Mamba2OrtTensorStorage {
            require(elementCount > 0)
            return when (dtype) {
                Mamba2OrtValueDtype.FLOAT32 -> {
                    val buffer = ByteBuffer.allocateDirect(
                        Math.multiplyExact(
                            elementCount,
                            Float.SIZE_BYTES,
                        )
                    ).order(ByteOrder.nativeOrder()).asFloatBuffer()
                    Mamba2OrtTensorStorage(
                        dtype = dtype,
                        floatBuffer = buffer,
                        shortBuffer = null,
                        tensor = OnnxTensor.createTensor(
                            environment,
                            buffer,
                            shape,
                        ),
                        elementCount = elementCount,
                    )
                }
                Mamba2OrtValueDtype.FLOAT16 -> {
                    val buffer = ByteBuffer.allocateDirect(
                        Math.multiplyExact(
                            elementCount,
                            Short.SIZE_BYTES,
                        )
                    ).order(ByteOrder.nativeOrder()).asShortBuffer()
                    Mamba2OrtTensorStorage(
                        dtype = dtype,
                        floatBuffer = null,
                        shortBuffer = buffer,
                        tensor = OnnxTensor.createTensor(
                            environment,
                            buffer,
                            shape,
                            OnnxJavaType.FLOAT16,
                        ),
                        elementCount = elementCount,
                    )
                }
            }
        }
    }

    fun zero() {
        when (dtype) {
            Mamba2OrtValueDtype.FLOAT32 -> {
                val buffer = requireNotNull(floatBuffer)
                for (index in 0 until buffer.capacity()) {
                    buffer.put(index, 0.0f)
                }
            }
            Mamba2OrtValueDtype.FLOAT16 -> {
                val buffer = requireNotNull(shortBuffer)
                for (index in 0 until buffer.capacity()) {
                    buffer.put(index, 0)
                }
            }
        }
    }

    fun readFloatRange(
        offset: Int,
        count: Int,
    ): FloatArray {
        require(offset >= 0)
        require(count > 0)
        require(offset <= elementCount - count)
        val source = tensor.floatBuffer
            ?: error("ORT tensor cannot be converted to FloatBuffer")
        val result = FloatArray(count)
        for (index in 0 until count) {
            result[index] = source.get(offset + index)
        }
        return result
    }

    fun allFinite(): Boolean {
        val source = tensor.floatBuffer
            ?: error("ORT tensor cannot be converted to FloatBuffer")
        for (index in 0 until source.capacity()) {
            if (!source.get(index).isFinite()) return false
        }
        return true
    }

    override fun close() {
        tensor.close()
    }
}
