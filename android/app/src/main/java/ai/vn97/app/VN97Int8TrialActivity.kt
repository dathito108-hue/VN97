package ai.vn97.app

import android.app.Activity
import android.os.Bundle
import android.content.Intent
import android.content.ClipData
import android.content.ClipboardManager
import android.widget.Button
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import java.io.File
import java.util.concurrent.Executors
import ai.vn97.runtime.VN97Int8DeviceTrial
import ai.vn97.runtime.VN97Int8TextTrial
import ai.vn97.runtime.Mamba2TokenizerPackage
import android.widget.EditText
import android.provider.OpenableColumns
import java.util.concurrent.atomic.AtomicBoolean

/** Separate process confines native ORT failures and releases model memory on exit. */
class VN97Int8TrialActivity : Activity() {
    private val worker = Executors.newSingleThreadExecutor()
    private val buttons = mutableListOf<Button>()
    private lateinit var status: TextView
    private lateinit var readiness: TextView
    private lateinit var textRun: Button
    private lateinit var probeRun: Button
    private var busy = false
    private val cancelled = AtomicBoolean(false)
    private lateinit var prompt: EditText
    private lateinit var stopButton: Button
    private val tokenizerRoot get() = File(root, "tokenizer")
    private val root get() = File(noBackupFilesDir, "int8-device-trial")
    private val model get() = File(root, VN97Int8Archive.ID)
    private val receipt get() = File(root, "receipt.json")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        check(!BuildConfig.VN97_TURNKEY_REQUIRED)
        root.mkdirs()
        val layout = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; setPadding(24, 32, 24, 24) }
        layout.addView(TextView(this).apply {
            text = "VN97 — thử INT8 trên thiết bị\nGói 3,01 GB; cần thêm 3,2 GB trống để nhập. Đây là lượng tử hóa trọng số, không phải điện toán lượng tử. Có thể thử viết tiếp văn bản tối đa 16 token. Chưa chứng nhận chất lượng hoặc bật tự tiến hóa."
            textSize = 18f
        })
        readiness = TextView(this).apply { textSize = 17f }
        layout.addView(readiness)
        fun button(label: String, action: () -> Unit): Button {
            val b = Button(this).apply { text = label; setOnClickListener { action() } }
            buttons.add(b); layout.addView(b)
            return b
        }
        button("Nhập ZIP INT8") {
            chooseModelZip()
        }
        button("Hoặc nhập 3 tệp mô hình đã giải nén") {
            startActivityForResult(Intent(Intent.ACTION_OPEN_DOCUMENT).apply {
                type = "*/*"; addCategory(Intent.CATEGORY_OPENABLE)
                putExtra(Intent.EXTRA_ALLOW_MULTIPLE, true)
            }, 100)
        }
        button("Nhập ZIP tokenizer G08 (khoảng 1 MB)") {
            startActivityForResult(Intent(Intent.ACTION_OPEN_DOCUMENT).apply {
                type = "*/*"; addCategory(Intent.CATEGORY_OPENABLE)
            }, 99)
        }
        button("Hoặc chọn 6 tệp tokenizer đã giải nén") {
            startActivityForResult(Intent(Intent.ACTION_OPEN_DOCUMENT).apply {
                type = "*/*"; addCategory(Intent.CATEGORY_OPENABLE)
                putExtra(Intent.EXTRA_ALLOW_MULTIPLE, true)
            }, 98)
        }
        prompt = EditText(this).apply {
            hint = "Nhập đoạn văn để mô hình viết tiếp (tối đa 128 token)"
            minLines = 2
            filters = arrayOf(android.text.InputFilter.LengthFilter(2048))
        }
        layout.addView(prompt)
        textRun = button("Thử viết tiếp bằng INT8") {
            val text = prompt.text.toString()
            work {
                check(model.isDirectory) { "Hãy nhập ZIP INT8 trước." }
                saveReceipt("{\"status\":\"verifying\",\"execution_passed\":false}")
                VN97Int8Archive.verify(model) { cancelled.get() }
                VN97Int8TextTrial.run(model, tokenizerRoot, text, { cancelled.get() }) { value ->
                    saveReceipt(value)
                    runOnUiThread { status.text = value }
                }
            }
        }
        stopButton = Button(this).apply {
            text = "Dừng sau lượt đang chạy"; isEnabled = false
            setOnClickListener { cancelled.set(true); text = "Đã yêu cầu dừng; nút Quay lại đóng ngay phiên thử" }
        }
        layout.addView(stopButton)
        probeRun = button("Chạy kiểm tra 1 / 8 / 1 token") {
            work {
                check(model.isDirectory) { "Hãy nhập ZIP INT8 trước." }
                saveReceipt("{\"status\":\"running_or_interrupted\",\"production_activation_authorized\":false}")
                VN97Int8Archive.verify(model) { cancelled.get() }
                val result = VN97Int8DeviceTrial.run(model, { cancelled.get() }) { saveReceipt(it) }
                saveReceipt(result)
                result
            }
        }
        button("Sao chép kết quả") {
            (getSystemService(CLIPBOARD_SERVICE) as ClipboardManager).setPrimaryClip(ClipData.newPlainText("VN97 INT8", status.text))
        }
        button("Xóa gói thử nghiệm") { work { model.deleteRecursively(); receipt.delete(); "Đã xóa gói thử nghiệm." } }
        status = TextView(this).apply {
            text = if (receipt.exists()) android.util.AtomicFile(receipt).openRead().bufferedReader().use { it.readText() } else "Sẵn sàng nhập gói INT8. Kết quả là chẩn đoán thực thi, không phải chứng nhận chất lượng."
            setTextIsSelectable(true)
        }
        layout.addView(status)
        setContentView(ScrollView(this).apply { addView(layout) })
        refreshReadiness()
        if (savedInstanceState == null && intent.getBooleanExtra("open_model_zip_picker", false)) {
            chooseModelZip()
        }
    }
    private fun chooseModelZip() {
        startActivityForResult(Intent(Intent.ACTION_OPEN_DOCUMENT).apply {
            type = "*/*"; addCategory(Intent.CATEGORY_OPENABLE)
        }, 97)
    }
    private fun work(action: () -> String) {
        if (busy) return
        cancelled.set(false)
        stopButton.isEnabled = true; stopButton.text = "Dừng sau lượt đang chạy"
        prompt.isEnabled = false
        busy = true; buttons.forEach { it.isEnabled = false }; status.text = "Đang xử lý… giữ màn hình này mở."
        window.addFlags(android.view.WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        worker.execute {
            val result = try { action() } catch (e: Exception) {
                org.json.JSONObject().put("status", if (e is java.util.concurrent.CancellationException) "cancelled" else "failed").put("execution_passed", false)
                    .put("production_activation_authorized", false).put("error", e.message).toString(2)
                    .also { saveReceipt(it) }
            }
            runOnUiThread {
                stopButton.isEnabled = false; prompt.isEnabled = true
                busy = false; buttons.forEach { it.isEnabled = true }; status.text = result
                refreshReadiness()
                window.clearFlags(android.view.WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
            }
        }
    }
    @Deprecated("Android activity result bridge")
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode == 99 && resultCode == RESULT_OK) {
            val uri = data?.data ?: return
            work {
                require(!tokenizerRoot.exists()) { "Tokenizer đã có. Có thể dùng ngay." }
                val stage = File(root, "tokenizer-staging")
                stage.deleteRecursively(); check(stage.mkdirs())
                try {
                    val allowed = Mamba2TokenizerPackage.REQUIRED_ASSETS + Mamba2TokenizerPackage.FILENAME
                    contentResolver.openInputStream(uri)!!.use { input ->
                        VN97TrialFiles.unzip(input, stage, allowed.associateWith { 16L * 1024 * 1024 },
                            { cancelled.get() }, ::showProgress)
                    }
                    VN97Int8TextTrial.tokenizer(stage)
                    check(stage.renameTo(tokenizerRoot))
                    "Đã xác minh tokenizer G08. Có thể thử viết tiếp văn bản."
                } finally { stage.deleteRecursively() }
            }
            return
        }
        if (requestCode == 100 && resultCode == RESULT_OK) {
            val clip = data?.clipData
            val uris = if (clip != null) (0 until clip.itemCount).map { clip.getItemAt(it).uri }
                else listOfNotNull(data?.data)
            work {
                val selected = linkedMapOf<String, () -> java.io.InputStream>()
                uris.forEach { uri ->
                    val name = displayName(uri)
                    require(!selected.containsKey(name)) { "Tệp trùng: $name" }
                    selected[name] = { contentResolver.openInputStream(uri) ?: error("Không mở được $name") }
                }
                VN97Int8Archive.installFiles(selected, root, { cancelled.get() }, ::showProgress)
                "Đã xác minh và nhập INT8. Mô hình đã lưu thành công."
            }
            return
        }
        if (requestCode == 98 && resultCode == RESULT_OK) {
            val selected = data?.clipData
            val uris = if (selected != null) (0 until selected.itemCount).map { selected.getItemAt(it).uri }
                else listOfNotNull(data?.data)
            work { importTokenizer(uris); "Đã xác minh tokenizer G08. Có thể thử viết tiếp văn bản." }
            return
        }
        val uri = data?.data ?: return
        if (requestCode == 97 && resultCode == RESULT_OK) work {
            contentResolver.openInputStream(uri)!!.use { VN97Int8Archive.install(it, root, { cancelled.get() }, ::showProgress) }
            "Đã xác minh và nhập INT8. Nhấn Chạy kiểm tra để đo trên máy này."
        }
    }
    private fun showProgress(value: String) { runOnUiThread { status.text = value } }
    private fun displayName(uri: android.net.Uri): String =
        contentResolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)?.use {
            check(it.moveToFirst()); it.getString(0)
        } ?: error("Không đọc được tên tệp")
    private fun refreshReadiness() {
        val modelReady = model.isDirectory && VN97Int8Archive.fileLimits.keys.all { File(model, it).isFile }
        val tokenizerReady = tokenizerRoot.isDirectory &&
            (Mamba2TokenizerPackage.REQUIRED_ASSETS + Mamba2TokenizerPackage.FILENAME).all { File(tokenizerRoot, it).isFile }
        readiness.text = "Mô hình INT8: ${if (modelReady) "đã nhập" else "chưa nhập thành công"}\nTokenizer G08: ${if (tokenizerReady) "đã nhập" else "chưa nhập thành công"}"
        textRun.isEnabled = !busy && modelReady && tokenizerReady
        probeRun.isEnabled = !busy && modelReady
    }
    private fun saveReceipt(value: String) {
        val atomic = android.util.AtomicFile(receipt)
        val stream = atomic.startWrite()
        try { stream.write(value.toByteArray(Charsets.UTF_8)); atomic.finishWrite(stream) }
        catch (e: Exception) { atomic.failWrite(stream); throw e }
    }
    private fun importTokenizer(uris: List<android.net.Uri>) {
        val allowed = Mamba2TokenizerPackage.REQUIRED_ASSETS + Mamba2TokenizerPackage.FILENAME
        require(uris.size == allowed.size) { "Đây là cổng tokenizer: chọn đủ 6 tệp G08. Với candidate.onnx và candidate.onnx.data, dùng nút nhập 3 tệp mô hình." }
        require(!tokenizerRoot.exists()) { "Tokenizer đã có. Có thể dùng ngay." }
        val stage = File(root, "tokenizer-staging")
        stage.deleteRecursively(); check(stage.mkdirs())
        try {
            val seen = mutableSetOf<String>()
            for (uri in uris) {
                if (cancelled.get()) throw java.util.concurrent.CancellationException()
                val name = contentResolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)?.use {
                    check(it.moveToFirst()); it.getString(0)
                } ?: error("Không đọc được tên tệp")
                require(name in allowed && seen.add(name)) { "Tên tệp không hợp lệ hoặc bị trùng: $name" }
                contentResolver.openInputStream(uri)!!.use { input ->
                    File(stage, name).outputStream().use { out ->
                        val buffer = ByteArray(65536); var total = 0L
                        while (true) {
                            val n = input.read(buffer); if (n < 0) break
                            total += n; require(total <= 16L * 1024 * 1024) { "Tệp tokenizer quá lớn" }
                            out.write(buffer, 0, n)
                        }
                    }
                }
            }
            VN97Int8TextTrial.tokenizer(stage)
            check(stage.renameTo(tokenizerRoot))
        } finally { stage.deleteRecursively() }
    }
    override fun onDestroy() {
        worker.shutdownNow()
        super.onDestroy()
        // Only the dedicated :quant_trial process is terminated, never the main app.
        android.os.Process.killProcess(android.os.Process.myPid())
    }
}
