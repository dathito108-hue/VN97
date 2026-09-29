package ai.vn97.app

import android.content.res.AssetManager
import android.util.AtomicFile
import java.io.File
import java.security.MessageDigest
import java.util.zip.GZIPInputStream

/** Small graph only. External weights remain in the verified INT8 directory. */
internal object VN97Int8DecodeAsset {
    const val SHA256 = "873c862d50cfe7c10b47b5c02e11fc8dea42ad381431570e657a0eba99bdd88e"
    fun prepare(assets: AssetManager, root: File): File {
        val file = File(root, "decode.onnx")
        val atomic = AtomicFile(file)
        val output = atomic.startWrite()
        try {
            val digest = MessageDigest.getInstance("SHA-256")
            var total = 0L
            GZIPInputStream(assets.open("vn97-int8-decode.bin")).use { input ->
                val buffer = ByteArray(65536)
                while (true) {
                    val count = input.read(buffer)
                    if (count < 0) break
                    total += count
                    require(total <= 4906371L) { "Đồ thị giải mã quá lớn" }
                    digest.update(buffer, 0, count)
                    output.write(buffer, 0, count)
                }
            }
            require(total == 4906371L && digest.digest().joinToString("") { "%02x".format(it) } == SHA256) {
                "Đồ thị giải mã không khớp SHA-256"
            }
            atomic.finishWrite(output)
        } catch (e: Exception) {
            atomic.failWrite(output)
            throw e
        }
        return file
    }
}
