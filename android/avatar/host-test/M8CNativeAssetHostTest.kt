import ai.vn97.avatar.*
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.zip.CRC32

private fun crc(bytes: ByteArray, offset: Int, length: Int): Int {
    val crc = CRC32()
    crc.update(bytes, offset, length)
    return crc.value.toInt()
}

private fun validAsset(): ByteArray {
    val vertexCount = 3
    val indexCount = 3
    val vertexOffset = 80
    val indexOffset = vertexOffset + vertexCount * 32
    val jointOffset = indexOffset + indexCount * 4
    val total = jointOffset
    val bytes = ByteArray(total)
    val b = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
    b.put(byteArrayOf('V'.code.toByte(),'N'.code.toByte(),'9'.code.toByte(),'7'.code.toByte(),'A'.code.toByte(),'V'.code.toByte(),'1'.code.toByte(),0))
    b.putShort(1); b.putShort(80); b.putInt(0)
    b.putInt(vertexCount); b.putInt(indexCount); b.putInt(0)
    b.putInt(32); b.putInt(48); b.putInt(0)
    b.putLong(vertexOffset.toLong()); b.putLong(indexOffset.toLong()); b.putLong(jointOffset.toLong()); b.putLong(total.toLong())
    b.putInt(0); b.putInt(0)
    val positions = arrayOf(floatArrayOf(-1f,0f,0f), floatArrayOf(1f,0f,0f), floatArrayOf(0f,1f,0f))
    repeat(vertexCount) { i ->
        b.position(vertexOffset + i * 32)
        positions[i].forEach(b::putFloat)
        b.putFloat(0f); b.putFloat(0f); b.putFloat(1f)
        repeat(8) { b.put(0) }
    }
    b.position(indexOffset); b.putInt(0); b.putInt(1); b.putInt(2)
    b.putInt(72, crc(bytes, 80, bytes.size - 80))
    b.putInt(76, crc(bytes, 0, 76))
    return bytes
}

fun main() {
    val source = validAsset()
    val asset = NativeAvatarAsset.validate(source, maxAssetBytes = 4096)
    check(asset.metadata.vertexCount == 3)
    check(asset.metadata.indexCount == 3)
    check(asset.metadata.jointCount == 0)
    check(asset.size == source.size)
    source[80] = (source[80].toInt() xor 1).toByte()
    check(asset.bytesCopy()[80] != source[80])

    val corrupt = asset.bytesCopy()
    corrupt[80] = (corrupt[80].toInt() xor 1).toByte()
    try {
        NativeAvatarAsset.validate(corrupt, 4096)
        error("corrupt asset must fail")
    } catch (exc: AvatarAssetException) {
        check(exc.status == AvatarAssetStatus.CHECKSUM_MISMATCH)
    }
    try {
        NativeAvatarAsset.validate(ByteArray(79), 4096)
        error("short asset must fail")
    } catch (_: IllegalArgumentException) {
    }
    println("M8C_NATIVE_ASSET_JNI_PASS")
}
