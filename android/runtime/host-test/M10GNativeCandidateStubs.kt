package ai.vn97.runtime

enum class NativeModelStatus(val code: Int) {
    OK(0), INVALID_HANDLE(2);
    companion object { fun fromCode(code: Int) = entries.first { it.code == code } }
}
class NativeModelException(val status: NativeModelStatus, operation: String) : IllegalStateException(operation)
enum class NativeEmbeddingKind(val code: Int) {
    FULL_F32(0);
    companion object { fun fromCode(code: Int) = entries.first { it.code == code } }
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
object NativeRuntimeBindings {
    fun nativeModelOpen(fd: Int, offset: Long, length: Long, expected: ByteArray, out: LongArray): Int {
        @Suppress("UNUSED_VARIABLE") val ignored = listOf(fd, offset, length, expected)
        out[0] = 1L
        return 0
    }
    fun nativeModelInfo(handle: Long, ints: IntArray, longs: LongArray, id: ByteArray): Int {
        @Suppress("UNUSED_VARIABLE") val ignored = handle
        ints[0]=10; ints[1]=4; ints[2]=2; ints[3]=3; ints[4]=0; ints[5]=0; ints[6]=1
        longs[0]=128
        id.fill(1)
        return 0
    }
    fun nativeModelDestroy(handle: Long): Int {
        @Suppress("UNUSED_VARIABLE") val ignored=handle
        return 0
    }
}
fun checkModelStatus(code: Int, operation: String) {
    if (code != 0) throw IllegalStateException(operation)
}
