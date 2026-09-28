package ai.vn97.runtime

import ai.onnxruntime.OnnxJavaType
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.platform.Fp16Conversions
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
    private val byteBuffer: ByteBuffer,
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
                    val bytes = ByteBuffer.allocateDirect(
                        Math.multiplyExact(
                            elementCount,
                            Float.SIZE_BYTES,
                        )
                    ).order(ByteOrder.nativeOrder())
                    val buffer = bytes.asFloatBuffer()
                    Mamba2OrtTensorStorage(
                        dtype = dtype,
                        byteBuffer = bytes,
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
                    val bytes = ByteBuffer.allocateDirect(
                        Math.multiplyExact(
                            elementCount,
                            Short.SIZE_BYTES,
                        )
                    ).order(ByteOrder.nativeOrder())
                    val buffer = bytes.asShortBuffer()
                    Mamba2OrtTensorStorage(
                        dtype = dtype,
                        byteBuffer = bytes,
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

    val rawByteSize: Int
        get() = byteBuffer.capacity()

    fun copyRawBytes(maxBytes: Long): ByteArray {
        require(maxBytes >= rawByteSize.toLong()) {
            "G0.6 state snapshot exceeds byte budget"
        }
        return ByteArray(rawByteSize).also { output ->
            byteBuffer.duplicate().apply { clear() }.get(output)
        }
    }

    fun writeRawBytes(bytes: ByteArray) {
        require(bytes.size == rawByteSize) {
            "G0.6 state snapshot byte length mismatch"
        }
        byteBuffer.duplicate().apply { clear() }.put(bytes)
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
        // These direct buffers back the pinned tensor. getFloatBuffer() would
        // copy/convert the entire chunk even when only its final row is needed.
        val result = FloatArray(count)
        when (dtype) {
            Mamba2OrtValueDtype.FLOAT32 -> {
                val source = requireNotNull(floatBuffer)
                for (index in 0 until count) {
                    result[index] = source.get(offset + index)
                }
            }
            Mamba2OrtValueDtype.FLOAT16 -> {
                val source = requireNotNull(shortBuffer)
                for (index in 0 until count) {
                    result[index] = Fp16Conversions.mlasFp16ToFloat(
                        source.get(offset + index)
                    )
                }
            }
        }
        return result
    }

    fun allFinite(): Boolean = when (dtype) {
        Mamba2OrtValueDtype.FLOAT32 -> {
            val source = requireNotNull(floatBuffer)
            (0 until elementCount).all { source.get(it).isFinite() }
        }
        Mamba2OrtValueDtype.FLOAT16 -> {
            val source = requireNotNull(shortBuffer)
            // An all-one binary16 exponent identifies both infinity and NaN.
            (0 until elementCount).all {
                (source.get(it).toInt() and 0x7c00) != 0x7c00
            }
        }
    }

    override fun close() {
        tensor.close()
    }
}
