package ai.vn97.app

import java.io.File
import java.io.InputStream
import java.io.PushbackInputStream
import java.util.zip.ZipInputStream

/** Transport only. Callers must validate pinned identities before committing staging. */
internal object VN97TrialFiles {
    fun copy(input: InputStream, output: File, limit: Long, cancelled: () -> Boolean,
             progress: (String) -> Unit) {
        val buffer = ByteArray(1024 * 1024)
        var count = 0L
        var nextReport = 0L
        output.outputStream().use { out ->
            while (true) {
                if (cancelled()) throw java.util.concurrent.CancellationException("Đã dừng nhập")
                val n = input.read(buffer); if (n < 0) break
                count += n
                require(count <= limit) { "${output.name} vượt kích thước cho phép; kiểm tra đúng gói INT8/G08." }
                out.write(buffer, 0, n)
                if (count >= nextReport) {
                    progress("Đang nhập ${output.name}: ${count / 1048576} MB")
                    nextReport = count + 32L * 1048576
                }
            }
        }
    }
    fun requireComplete(seen: Set<String>, expected: Set<String>) {
        require(seen == expected) {
            "Thiếu tệp: ${(expected - seen).sorted().joinToString()}. Đã nhận: ${seen.sorted().joinToString().ifEmpty { "không có" }}."
        }
    }
    fun unzip(input: InputStream, stage: File, limits: Map<String, Long>,
              cancelled: () -> Boolean = { false }, progress: (String) -> Unit = {}) {
        val source = PushbackInputStream(input, 4)
        val header = ByteArray(4)
        var read = 0
        while (read < 4) { val n = source.read(header, read, 4 - read); if (n < 0) break; read += n }
        require(read == 4 && header[0] == 80.toByte() && header[1] == 75.toByte() &&
            ((header[2] == 3.toByte() && header[3] == 4.toByte()) ||
                (header[2] == 5.toByte() && header[3] == 6.toByte()))) {
            "Tệp đã chọn không phải ZIP đọc được. Nếu đã giải nén, dùng nút chọn các tệp; nếu tải chưa xong, chờ tải hoàn tất."
        }
        source.unread(header)
        val seen = mutableSetOf<String>()
        var commonParent: String? = null
        var entries = 0
        ZipInputStream(source).use { zip ->
            while (true) {
                val entry = zip.nextEntry ?: break
                require(++entries <= 32) { "ZIP có quá nhiều mục; hãy chọn đúng gói mô hình hoặc tokenizer." }
                val path = entry.name.removeSuffix("/")
                val parts = path.split('/')
                require(path.isNotEmpty() && !path.contains('\\') && parts.none { it.isEmpty() || it == "." || it == ".." || it.contains(':') }) { "Đường dẫn ZIP không hợp lệ" }
                if (entry.isDirectory) continue
                val name = parts.last()
                val limit = limits[name] ?: error("Sai gói hoặc tệp không được hỗ trợ: $name. Cần: ${limits.keys.joinToString()}.")
                val parent = parts.dropLast(1).joinToString("/")
                require(commonParent == null || commonParent == parent) { "Các tệp phải cùng một thư mục trong ZIP." }
                commonParent = parent
                require(seen.add(name)) { "ZIP chứa tệp trùng: $name" }
                copy(zip, File(stage, name), limit, cancelled, progress)
            }
        }
        requireComplete(seen, limits.keys)
    }
}
