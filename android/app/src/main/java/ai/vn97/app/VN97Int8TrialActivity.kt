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

/** Separate process confines native ORT failures and releases model memory on exit. */
class VN97Int8TrialActivity : Activity() {
    private val worker = Executors.newSingleThreadExecutor()
    private val buttons = mutableListOf<Button>()
    private lateinit var status: TextView
    private var busy = false
    private val root get() = File(noBackupFilesDir, "int8-device-trial")
    private val model get() = File(root, VN97Int8Archive.ID)
    private val receipt get() = File(root, "receipt.json")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        check(!BuildConfig.VN97_TURNKEY_REQUIRED)
        root.mkdirs()
        val layout = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; setPadding(24, 32, 24, 24) }
        layout.addView(TextView(this).apply {
            text = "VN97 — thử INT8 trên thiết bị\nGói 3,01 GB; cần thêm 3,2 GB trống để nhập. Đây là lượng tử hóa trọng số, không phải điện toán lượng tử. Chưa bật trò chuyện hoặc tự tiến hóa."
            textSize = 18f
        })
        fun button(label: String, action: () -> Unit) {
            val b = Button(this).apply { text = label; setOnClickListener { action() } }
            buttons.add(b); layout.addView(b)
        }
        button("Nhập ZIP INT8") {
            startActivityForResult(Intent(Intent.ACTION_OPEN_DOCUMENT).apply {
                type = "*/*"; addCategory(Intent.CATEGORY_OPENABLE)
            }, 97)
        }
        button("Chạy kiểm tra 1 / 8 / 1 token") {
            work {
                check(model.isDirectory) { "Hãy nhập ZIP INT8 trước." }
                receipt.writeText("{\"status\":\"running_or_interrupted\",\"production_activation_authorized\":false}")
                VN97Int8Archive.verify(model)
                val result = VN97Int8DeviceTrial.run(model) { receipt.writeText(it) }
                receipt.writeText(result)
                result
            }
        }
        button("Sao chép kết quả") {
            (getSystemService(CLIPBOARD_SERVICE) as ClipboardManager).setPrimaryClip(ClipData.newPlainText("VN97 INT8", status.text))
        }
        button("Xóa gói thử nghiệm") { work { model.deleteRecursively(); receipt.delete(); "Đã xóa gói thử nghiệm." } }
        status = TextView(this).apply {
            text = if (receipt.exists()) receipt.readText() else "Sẵn sàng nhập gói INT8. Kết quả là chẩn đoán thực thi, không phải chứng nhận chất lượng."
            setTextIsSelectable(true)
        }
        layout.addView(status)
        setContentView(ScrollView(this).apply { addView(layout) })
    }
    private fun work(action: () -> String) {
        if (busy) return
        busy = true; buttons.forEach { it.isEnabled = false }; status.text = "Đang xử lý… giữ màn hình này mở."
        window.addFlags(android.view.WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        worker.execute {
            val result = try { action() } catch (e: Exception) { "Không hoàn tất: ${e.message}" }
            runOnUiThread {
                busy = false; buttons.forEach { it.isEnabled = true }; status.text = result
                window.clearFlags(android.view.WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
            }
        }
    }
    @Deprecated("Android activity result bridge")
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        val uri = data?.data ?: return
        if (requestCode == 97 && resultCode == RESULT_OK) work {
            contentResolver.openInputStream(uri)!!.use { VN97Int8Archive.install(it, root) }
            "Đã xác minh và nhập INT8. Nhấn Chạy kiểm tra để đo trên máy này."
        }
    }
    override fun onDestroy() {
        worker.shutdownNow()
        super.onDestroy()
        // Only the dedicated :quant_trial process is terminated, never the main app.
        android.os.Process.killProcess(android.os.Process.myPid())
    }
}
