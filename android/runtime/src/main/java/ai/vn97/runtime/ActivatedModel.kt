package ai.vn97.runtime

import java.nio.charset.StandardCharsets

enum class NativeModelStatus(val code: Int) {
    OK(0),
    NULL_ARGUMENT(1),
    INVALID_HANDLE(2),
    INVALID_FD(3),
    INVALID_RANGE(4),
    TOO_LARGE(5),
    MAP_FAILED(6),
    IDENTITY_MISMATCH(7),
    BAD_MAGIC(8),
    UNSUPPORTED_VERSION(9),
    INVALID_HEADER(10),
    INVALID_SECTION(11),
    INVALID_CONFIG(12),
    INVALID_PACKED_TERNARY(13),
    INVALID_TOKENIZER(14),
    INVALID_MODEL(15),
    OUT_OF_MEMORY(16),
    IO_ERROR(17),
    OUTPUT_TOO_SMALL(18);

    companion object {
        internal fun fromCode(code: Int): NativeModelStatus =
            entries.firstOrNull { it.code == code }
                ?: throw IllegalStateException("unknown native model status: $code")
    }
}

class NativeModelException(
    val status: NativeModelStatus,
    operation: String,
) : IllegalStateException("$operation failed: ${status.name} (${status.code})")

enum class NativeEmbeddingKind(val code: Int) {
    FULL_F32(0),
    FACTORIZED_F32(1);

    companion object {
        internal fun fromCode(code: Int): NativeEmbeddingKind =
            entries.firstOrNull { it.code == code }
                ?: throw IllegalStateException("unknown embedding kind: $code")
    }
}

data class NativeActivatedModelInfo(
    val modelId: ByteArray,
    val vocabSize: Int,
    val dModel: Int,
    val layers: Int,
    val dState: Int,
    val embeddingKind: NativeEmbeddingKind,
    val embeddingRank: Int,
    val hasTokenizer: Boolean,
    val imageBytes: Long,
)

/**
 * Trusted descriptor produced only after M9 activation has committed an artifact.
 * The native loader duplicates [fd], hashes exactly [offset, offset + length), and
 * requires that hash to equal [artifactSha256] before any model pointer is exposed.
 */
class TrustedActivatedModelArtifact internal constructor(
    internal val fd: Int,
    internal val offset: Long,
    internal val length: Long,
    artifactSha256: ByteArray,
) {
    internal val artifactSha256: ByteArray = artifactSha256.copyOf()

    init {
        require(fd >= 0) { "activated artifact fd must be non-negative" }
        require(offset >= 0L) { "activated artifact offset must be non-negative" }
        require(length > 0L) { "activated artifact length must be positive" }
        require(this.artifactSha256.size == 32) { "M9 artifact SHA-256 must be exactly 32 bytes" }
        require(this.artifactSha256.any { it.toInt() != 0 }) { "M9 artifact SHA-256 must be nonzero" }
    }
}

internal object M9ActivatedModelBridge {
    fun committedArtifact(
        fd: Int,
        offset: Long,
        length: Long,
        artifactSha256: ByteArray,
    ): TrustedActivatedModelArtifact = TrustedActivatedModelArtifact(
        fd = fd,
        offset = offset,
        length = length,
        artifactSha256 = artifactSha256,
    )
}

class NativeActivatedModel private constructor(
    private var handle: Long,
    val info: NativeActivatedModelInfo,
) : AutoCloseable {
    private val lock = Any()

    companion object {
        internal fun open(artifact: TrustedActivatedModelArtifact): NativeActivatedModel {
            val out = LongArray(1)
            checkModelStatus(
                NativeRuntimeBindings.nativeModelOpen(
                    artifact.fd,
                    artifact.offset,
                    artifact.length,
                    artifact.artifactSha256,
                    out,
                ),
                "activated model open",
            )
            check(out[0] != 0L) { "native model loader returned a zero handle" }
            val handle = out[0]
            return try {
                NativeActivatedModel(handle, readInfo(handle))
            } catch (exc: Throwable) {
                NativeRuntimeBindings.nativeModelDestroy(handle)
                throw exc
            }
        }

        private fun readInfo(handle: Long): NativeActivatedModelInfo {
            val ints = IntArray(7)
            val longs = LongArray(1)
            val modelId = ByteArray(32)
            checkModelStatus(
                NativeRuntimeBindings.nativeModelInfo(handle, ints, longs, modelId),
                "activated model info",
            )
            require(ints[0] > 1 && ints[1] > 0 && ints[2] > 0 && ints[3] > 0) {
                "native model info contains invalid geometry"
            }
            require(longs[0] > 0L) { "native model image length must be positive" }
            return NativeActivatedModelInfo(
                modelId = modelId,
                vocabSize = ints[0],
                dModel = ints[1],
                layers = ints[2],
                dState = ints[3],
                embeddingKind = NativeEmbeddingKind.fromCode(ints[4]),
                embeddingRank = ints[5],
                hasTokenizer = ints[6] != 0,
                imageBytes = longs[0],
            )
        }
    }

    internal inline fun <T> withHandle(block: (Long) -> T): T = synchronized(lock) {
        check(handle != 0L) { "native activated model is closed" }
        block(handle)
    }

    fun encodeUtf8(
        text: String,
        addBos: Boolean = true,
        addText: Boolean = true,
        addEos: Boolean = false,
    ): IntArray {
        check(info.hasTokenizer) { "activated model has no VN97TK1 tokenizer" }
        val input = text.toByteArray(StandardCharsets.UTF_8)
        val controlCount = (if (addBos) 1 else 0) + (if (addText) 1 else 0) + (if (addEos) 1 else 0)
        val capacity = checkedAddArrayCount(input.size, controlCount, "encoded token capacity")
        val output = IntArray(capacity)
        val countOut = IntArray(1)
        val flags = (if (addBos) 1 else 0) or
            (if (addText) 2 else 0) or
            (if (addEos) 4 else 0)
        withHandle<Unit> { h ->
            checkModelStatus(
                NativeRuntimeBindings.nativeModelEncode(h, input, flags, output, countOut),
                "VN97TK1 encode",
            )
        }
        require(countOut[0] in 0..output.size) { "native tokenizer returned invalid token count" }
        return output.copyOf(countOut[0])
    }

    internal fun decodeBytes(tokenIds: IntArray, skipControl: Boolean = true): ByteArray {
        check(info.hasTokenizer) { "activated model has no VN97TK1 tokenizer" }
        require(tokenIds.all { it >= 0 }) { "token IDs must be non-negative" }
        val sizeOut = LongArray(1)
        withHandle<Unit> { h ->
            checkModelStatus(
                NativeRuntimeBindings.nativeModelDecodedSize(h, tokenIds, skipControl, sizeOut),
                "VN97TK1 decoded size",
            )
        }
        val size = checkedArrayCount(sizeOut[0], "decoded byte size")
        val bytes = ByteArray(size)
        val writtenOut = LongArray(1)
        withHandle<Unit> { h ->
            checkModelStatus(
                NativeRuntimeBindings.nativeModelDecode(h, tokenIds, skipControl, bytes, writtenOut),
                "VN97TK1 decode",
            )
        }
        check(writtenOut[0] == sizeOut[0]) { "native tokenizer decoded size changed" }
        return bytes
    }

    fun decodeUtf8(tokenIds: IntArray, skipControl: Boolean = true): String =
        decodeBytes(tokenIds, skipControl).toString(StandardCharsets.UTF_8)

    override fun close() {
        val old = synchronized(lock) {
            val value = handle
            handle = 0L
            value
        }
        if (old != 0L) {
            val status = NativeModelStatus.fromCode(NativeRuntimeBindings.nativeModelDestroy(old))
            if (status != NativeModelStatus.OK && status != NativeModelStatus.INVALID_HANDLE) {
                throw NativeModelException(status, "activated model destroy")
            }
        }
    }
}

internal fun checkModelStatus(code: Int, operation: String) {
    val status = NativeModelStatus.fromCode(code)
    if (status != NativeModelStatus.OK) {
        throw NativeModelException(status, operation)
    }
}

internal fun checkedAddArrayCount(a: Int, b: Int, label: String): Int {
    require(a >= 0 && b >= 0 && a <= Int.MAX_VALUE - b) { "$label does not fit a JVM array" }
    return a + b
}
