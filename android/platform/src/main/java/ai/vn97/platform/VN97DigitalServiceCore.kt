package ai.vn97.platform

import java.security.MessageDigest

enum class VN97LocalDigitalService {
    TABLE_CLEANUP,
    DOCUMENT_FORMAT,
    CATALOG_PAGE,
}

data class VN97LocalDigitalDelivery(
    val fileName: String,
    val content: String,
    val report: String,
    val sourceSha256: String,
    val outputSha256: String,
)

/** Deterministic, bounded local transformations shared by the workbench and M6 tool. */
object VN97DigitalServiceCore {
    const val MAX_INPUT_BYTES = 128 * 1024
    const val MAX_OUTPUT_BYTES = 1024 * 1024
    private const val MAX_ROWS = 5_000
    private const val MAX_COLUMNS = 64

    fun produce(
        service: VN97LocalDigitalService,
        source: String,
    ): VN97LocalDigitalDelivery {
        require(source.isNotBlank()) { "Hãy nhập dữ liệu cần xử lý." }
        require(source.toByteArray(Charsets.UTF_8).size <= MAX_INPUT_BYTES) {
            "Dữ liệu vượt giới hạn 128 KiB. Hãy chia thành phần nhỏ hơn."
        }
        require(source.none { it.code < 32 && it !in "\r\n\t" }) {
            "Dữ liệu chứa ký tự điều khiển không được hỗ trợ."
        }
        val fileName: String
        val content: String
        val report: String
        when (service) {
            VN97LocalDigitalService.DOCUMENT_FORMAT -> {
                fileName = "document.txt"
                val lines = source.replace("\r\n", "\n").replace('\r', '\n')
                    .lines().map { it.trimEnd() }
                val output = mutableListOf<String>()
                for (line in lines) {
                    if (line.isNotEmpty() || output.lastOrNull()?.isNotEmpty() == true) {
                        output += line
                    }
                }
                while (output.lastOrNull()?.isEmpty() == true) output.removeAt(output.lastIndex)
                content = output.joinToString("\n") + "\n"
                report = "Chuẩn hóa xuống dòng, bỏ khoảng trắng cuối dòng và dòng trống thừa. " +
                    "Không viết lại, dịch hoặc kiểm chứng nội dung."
            }
            VN97LocalDigitalService.TABLE_CLEANUP,
            VN97LocalDigitalService.CATALOG_PAGE,
            -> {
                val parsed = parseCsv(source.removePrefix("\uFEFF"))
                val rows = parsed.map { row -> row.map { it.trim() } }
                    .filter { row -> row.any { it.isNotEmpty() } }
                require(rows.isNotEmpty()) { "CSV chưa có tiêu đề." }
                val header = rows.first()
                require(header.all { it.isNotEmpty() } && header.distinct().size == header.size) {
                    "Tên cột phải có nội dung và không trùng nhau."
                }
                require(rows.all { it.size == header.size }) {
                    "Số cột không nhất quán; không tự bỏ dữ liệu lỗi."
                }
                val data = rows.drop(1).distinct()
                val duplicates = rows.size - 1 - data.size
                if (service == VN97LocalDigitalService.TABLE_CLEANUP) {
                    fileName = "cleaned.csv"
                    var protectedCells = 0
                    content = (listOf(header) + data).joinToString("\r\n", postfix = "\r\n") { row ->
                        row.joinToString(",") { cell ->
                            val safe = if (cell.firstOrNull() in listOf('=', '+', '-', '@')) {
                                protectedCells++
                                "'$cell"
                            } else cell
                            "\"" + safe.replace("\"", "\"\"") + "\""
                        }
                    }
                    report = "${data.size} dòng dữ liệu; bỏ $duplicates dòng trùng và dòng trống. " +
                        "Cắt khoảng trắng ở đầu/cuối ô. $protectedCells ô được thêm dấu ' " +
                        "để không chạy như công thức khi mở bảng tính; kiểm tra cả số âm."
                } else {
                    fileName = "catalog.html"
                    content = buildString {
                        append("<!doctype html><html lang=\"vi\"><meta charset=\"utf-8\">")
                        append("<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">")
                        append("<title>Danh mục</title><style>body{font:16px system-ui;margin:24px;")
                        append("background:#f5f7fa;color:#17213a}article{background:white;padding:20px;")
                        append("margin:16px 0;border-radius:12px;overflow-wrap:anywhere}")
                        append("dt{font-weight:700}dd{margin:4px 0 16px;white-space:pre-wrap}</style>")
                        append("<main><h1>Danh mục</h1>")
                        for (row in data) {
                            require(length <= MAX_OUTPUT_BYTES) {
                                "Trang danh mục quá lớn; hãy chia nhỏ dữ liệu."
                            }
                            append("<article><dl>")
                            header.zip(row).forEach { (label, value) ->
                                append("<dt>${html(label)}</dt><dd>${html(value)}</dd>")
                            }
                            append("</dl></article>")
                        }
                        append("</main></html>")
                    }
                    report = "${data.size} mục; bỏ $duplicates dòng trùng. " +
                        "HTML độc lập, không script, không gửi dữ liệu ra mạng. Chưa được đăng lên web."
                }
            }
        }
        require(content.toByteArray(Charsets.UTF_8).size <= MAX_OUTPUT_BYTES) {
            "Sản phẩm vượt giới hạn 1 MiB; hãy chia nhỏ dữ liệu."
        }
        return VN97LocalDigitalDelivery(
            fileName,
            content,
            report,
            sha256(source),
            sha256(content),
        )
    }

    private fun html(value: String): String = value.replace("&", "&amp;")
        .replace("<", "&lt;").replace(">", "&gt;")
        .replace("\"", "&quot;").replace("'", "&#39;")

    private fun sha256(value: String): String = MessageDigest.getInstance("SHA-256")
        .digest(value.toByteArray(Charsets.UTF_8)).joinToString("") { "%02x".format(it) }

    private fun parseCsv(source: String): List<List<String>> {
        val rows = mutableListOf<List<String>>()
        val row = mutableListOf<String>()
        val cell = StringBuilder()
        var quoted = false
        var closedQuote = false
        var at = 0
        fun finishCell() {
            require(row.size < MAX_COLUMNS) { "CSV vượt giới hạn 64 cột." }
            row += cell.toString()
            cell.setLength(0)
            closedQuote = false
        }
        fun finishRow() {
            finishCell()
            require(rows.size < MAX_ROWS) { "CSV vượt giới hạn 5.000 dòng." }
            rows += row.toList()
            row.clear()
        }
        while (at < source.length) {
            val char = source[at]
            if (quoted) {
                if (char == '"') {
                    if (at + 1 < source.length && source[at + 1] == '"') {
                        cell.append('"')
                        at++
                    } else {
                        quoted = false
                        closedQuote = true
                    }
                } else cell.append(char)
            } else {
                when (char) {
                    '"' -> {
                        require(cell.isEmpty() && !closedQuote) { "Dấu ngoặc kép CSV không hợp lệ." }
                        quoted = true
                    }
                    ',' -> finishCell()
                    '\r', '\n' -> {
                        finishRow()
                        if (char == '\r' && at + 1 < source.length && source[at + 1] == '\n') at++
                    }
                    else -> {
                        require(!closedQuote) { "Có ký tự thừa sau ô CSV có ngoặc kép." }
                        cell.append(char)
                    }
                }
            }
            at++
        }
        require(!quoted) { "Ô CSV có ngoặc kép chưa đóng." }
        if (cell.isNotEmpty() || row.isNotEmpty() || closedQuote) finishRow()
        return rows
    }
}
