package ai.vn97.app

import java.io.File
import java.io.FileOutputStream
import java.io.InputStream
import java.nio.file.Files
import java.util.zip.ZipInputStream

/** Transport envelope only. Trust still comes from the canonical CAP/SIG verifier. */
internal object VN97ModelImportArchive {
    private val limits = mapOf(
        "model.vn97cap" to 512L * 1024 * 1024,
        "model.vn97sig" to 16L * 1024,
        "publisher.ed25519" to 256L,
    )

    fun <T> read(input: InputStream, cacheRoot: File, review: (File, ByteArray, ByteArray) -> T): T {
        val directory = Files.createTempDirectory(cacheRoot.toPath(), "vn97-import-").toFile()
        try {
            val seen = HashSet<String>()
            ZipInputStream(input).use { zip ->
                while (true) {
                    val entry = zip.nextEntry ?: break
                    val limit = limits[entry.name]
                    require(!entry.isDirectory && limit != null && seen.add(entry.name)) {
                        "ZIP phải chứa đúng model.vn97cap, model.vn97sig, publisher.ed25519 tại thư mục gốc, không có tệp thừa hoặc trùng. Gói R2/checkpoint nghiên cứu không phải gói lõi để nhập tại đây."
                    }
                    val target = File(directory, entry.name)
                    var count = 0L
                    FileOutputStream(target).use { output ->
                        val buffer = ByteArray(64 * 1024)
                        while (true) {
                            val n = zip.read(buffer)
                            if (n < 0) break
                            check(n > 0) { "Không đọc được dữ liệu ZIP." }
                            count += n
                            require(count <= limit) { "Tệp ${entry.name} vượt giới hạn nhập." }
                            output.write(buffer, 0, n)
                        }
                    }
                    require(count > 0) { "Tệp ${entry.name} rỗng." }
                    zip.closeEntry()
                }
            }
            require(seen == limits.keys) { "Gói ZIP thiếu mô hình, chữ ký hoặc khóa công khai." }
            val model = File(directory, "model.vn97cap")
            val prefix = ByteArray(8)
            java.io.DataInputStream(model.inputStream()).use { it.readFully(prefix) }
            val magic = prefix.toString(Charsets.US_ASCII)
            require(magic == "VN97CAP1") { "Gói lõi không phải VN97CAP1. Không đổi đuôi GGUF, ONNX hoặc checkpoint thành gói VN97." }
            return review(model, File(directory, "model.vn97sig").readBytes(), File(directory, "publisher.ed25519").readBytes())
        } finally {
            directory.deleteRecursively()
        }
    }
}
