package ai.vn97.avatar

enum class AvatarAssetStatus(val code: Int) {
    OK(0),
    NULL_ARGUMENT(1),
    BLOB_TOO_SHORT(2),
    BAD_MAGIC(3),
    UNSUPPORTED_VERSION(4),
    INVALID_HEADER(5),
    INVALID_GEOMETRY(6),
    INVALID_RIG(7),
    CHECKSUM_MISMATCH(8),
    SIZE_OVERFLOW(9),
    INDEX_OUT_OF_RANGE(10),
    NON_FINITE(11);

    companion object {
        fun fromCode(code: Int): AvatarAssetStatus =
            entries.firstOrNull { it.code == code }
                ?: error("unknown native avatar asset status: $code")
    }
}

class AvatarAssetException(
    val status: AvatarAssetStatus,
) : IllegalArgumentException("VN97AV1 validation failed: ${status.name} (${status.code})")

data class AvatarAssetMetadata(
    val vertexCount: Int,
    val indexCount: Int,
    val jointCount: Int,
) {
    init {
        require(vertexCount >= 3)
        require(indexCount >= 3 && indexCount % 3 == 0)
        require(jointCount in 0..128)
    }
}

class ValidatedAvatarAsset internal constructor(
    bytes: ByteArray,
    val metadata: AvatarAssetMetadata,
) {
    private val immutableBytes = bytes.copyOf()

    val size: Int get() = immutableBytes.size

    fun bytesCopy(): ByteArray = immutableBytes.copyOf()
}

internal object NativeAvatarAssetBindings {
    init {
        System.loadLibrary("vn97_avatar_jni")
    }

    external fun nativeValidate(blob: ByteArray, metadataOut: IntArray): Int
}

object NativeAvatarAsset {
    const val DEFAULT_MAX_ASSET_BYTES: Int = 64 * 1024 * 1024

    fun validate(
        blob: ByteArray,
        maxAssetBytes: Int = DEFAULT_MAX_ASSET_BYTES,
    ): ValidatedAvatarAsset {
        require(maxAssetBytes >= 80) { "maxAssetBytes must be at least one VN97AV1 header" }
        require(blob.size in 80..maxAssetBytes) { "VN97AV1 blob size is outside configured bounds" }
        val metadata = IntArray(3)
        val status = AvatarAssetStatus.fromCode(
            NativeAvatarAssetBindings.nativeValidate(blob, metadata)
        )
        if (status != AvatarAssetStatus.OK) throw AvatarAssetException(status)
        return ValidatedAvatarAsset(
            blob,
            AvatarAssetMetadata(metadata[0], metadata[1], metadata[2]),
        )
    }
}
