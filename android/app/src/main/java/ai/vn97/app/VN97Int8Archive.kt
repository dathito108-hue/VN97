package ai.vn97.app

import java.io.File
import java.io.InputStream
import java.security.MessageDigest
import java.util.zip.ZipInputStream

/** Pinned experimental payload; never installs into the production model directory. */
internal object VN97Int8Archive {
    const val ID = "331dcb61ad1547667fc24cb000fabe202d8380937b57859e0aa209814e4ab8fd"
    private val payload = mapOf(
        "candidate.onnx" to (11049650L to "403c934fcc683b55225664f2b6721e5b20fd5398b9cd356485f002bede5ee5b8"),
        "candidate.onnx.data" to (3000576000L to "f15fc9c45b508fcd28d0d6be724c1ab1c5f8694fcc6395a9167c495448e27493"),
    )
    fun verify(root: File, cancelled: () -> Boolean = { false }) {
        payload.forEach { (name, spec) ->
            val file = File(root, name)
            require(file.length() == spec.first) { "Sai kích thước $name" }
            val digest = MessageDigest.getInstance("SHA-256")
            file.inputStream().use { input ->
                val buffer = ByteArray(1024 * 1024)
                while (true) { if (cancelled()) throw java.util.concurrent.CancellationException("Đã dừng"); val n = input.read(buffer); if (n < 0) break; digest.update(buffer, 0, n) }
            }
            require(digest.digest().joinToString("") { "%02x".format(it) } == spec.second) { "Sai SHA-256 $name" }
        }
    }
    fun install(input: InputStream, parent: File, cancelled: () -> Boolean = { false }): File {
        parent.mkdirs()
        val target = File(parent, ID)
        require(!target.exists()) { "Đã có bản INT8. Có thể chạy thử hoặc xóa trước khi nhập lại." }
        val stage = File(parent, "staging")
        stage.deleteRecursively()
        check(stage.mkdirs())
        try {
            require(parent.usableSpace > 3_200_000_000L) { "Cần ít nhất 3,2 GB trống ngoài tệp ZIP." }
            val seen = mutableSetOf<String>()
            ZipInputStream(input).use { zip ->
                val buffer = ByteArray(1024 * 1024)
                while (true) {
                    val entry = zip.nextEntry ?: break
                    val limit = payload[entry.name]?.first ?: if (entry.name == "quantization.json") 65536L else error("Tệp không được phép: ${entry.name}")
                    require(!entry.isDirectory && seen.add(entry.name)) { "Tệp trùng hoặc thư mục không hợp lệ" }
                    var count = 0L
                    File(stage, entry.name).outputStream().use { out ->
                        while (true) {
                            if (cancelled()) throw java.util.concurrent.CancellationException("Đã dừng")
                            val n = zip.read(buffer); if (n < 0) break
                            count += n
                            require(count <= limit) { "Tệp vượt giới hạn" }
                            out.write(buffer, 0, n)
                        }
                    }
                }
            }
            require(seen == payload.keys + "quantization.json") { "Gói INT8 thiếu tệp" }
            verify(stage, cancelled)
            check(stage.renameTo(target)) { "Không thể hoàn tất nhập" }
            return target
        } finally { stage.deleteRecursively() }
    }
}
