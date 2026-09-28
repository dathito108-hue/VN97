package ai.vn97.app

import ai.vn97.runtime.Mamba2OrtRuntimePackage
import ai.vn97.runtime.VN97Mamba2OrtProviderProfiler
import android.app.Activity
import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.content.Intent
import android.database.Cursor
import android.net.Uri
import android.os.Bundle
import android.provider.OpenableColumns
import android.view.Gravity
import android.view.ViewGroup
import android.widget.Button
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import java.io.File
import java.io.FileOutputStream
import java.util.concurrent.Executors

class VN97R2OrtEvidenceActivity : Activity() {
    private val worker = Executors.newSingleThreadExecutor()
    private lateinit var status: TextView
    private lateinit var importButton: Button
    private lateinit var runButton: Button
    private lateinit var copyButton: Button
    private var latestReceipt: String = ""

    private val runtimeRoot: File
        get() = File(noBackupFilesDir, RUNTIME_DIR)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        check(!BuildConfig.VN97_TURNKEY_REQUIRED) {
            "K2Q developer evidence is disabled in turnkey builds"
        }

        val density = resources.displayMetrics.density
        val content = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(
                (16 * density).toInt(),
                (16 * density).toInt(),
                (16 * density).toInt(),
                (16 * density).toInt(),
            )
        }

        content.addView(
            TextView(this).apply {
                text = "VN97 R2 — Android ORT device evidence"
                textSize = 22f
                gravity = Gravity.CENTER_HORIZONTAL
            },
            LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT,
            ),
        )

        importButton = Button(this).apply {
            text = "Import verified G0.6 runtime files"
            setOnClickListener { chooseRuntimeFiles() }
        }
        content.addView(importButton)

        runButton = Button(this).apply {
            text = "Run Android ORT profile"
            setOnClickListener { runProfile() }
        }
        content.addView(runButton)

        copyButton = Button(this).apply {
            text = "Copy evidence JSON"
            isEnabled = false
            setOnClickListener { copyEvidence() }
        }
        content.addView(copyButton)

        status = TextView(this).apply {
            setTextIsSelectable(true)
            text = runtimeStatus()
        }
        content.addView(
            status,
            LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT,
            ),
        )

        val scroll = ScrollView(this)
        scroll.addView(content)
        setContentView(scroll)
    }

    override fun onDestroy() {
        worker.shutdownNow()
        super.onDestroy()
    }

    @Deprecated("Deprecated in Android framework; retained for minSdk compatibility.")
    override fun onActivityResult(
        requestCode: Int,
        resultCode: Int,
        data: Intent?,
    ) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode != REQUEST_RUNTIME_FILES || resultCode != RESULT_OK) {
            return
        }
        val uris = selectedUris(data)
        if (uris.isEmpty()) {
            status.text = "No runtime files selected."
            return
        }
        setBusy(true)
        worker.execute {
            val result = runCatching { installRuntimeFiles(uris) }
            runOnUiThread {
                setBusy(false)
                status.text = result.fold(
                    onSuccess = { runtime ->
                        "Imported and verified R2 runtime.\n" +
                            "runtime_id=" + runtime.runtimeId + "\n" +
                            "g05_manifest_id=" + runtime.g05ManifestId + "\n" +
                            "state_dtype=" + runtime.stateDtype + "\n" +
                            "graph=" + runtime.graphFilename
                    },
                    onFailure = { error ->
                        "Runtime import failed: " +
                            (error.message ?: error::class.java.simpleName)
                    },
                )
            }
        }
    }

    private fun chooseRuntimeFiles() {
        val intent = Intent(Intent.ACTION_OPEN_DOCUMENT).apply {
            addCategory(Intent.CATEGORY_OPENABLE)
            type = "*/*"
            putExtra(Intent.EXTRA_ALLOW_MULTIPLE, true)
        }
        startActivityForResult(intent, REQUEST_RUNTIME_FILES)
    }

    private fun selectedUris(data: Intent?): List<Uri> {
        if (data == null) return emptyList()
        val result = LinkedHashSet<Uri>()
        data.data?.let(result::add)
        val clip = data.clipData
        if (clip != null) {
            for (index in 0 until clip.itemCount) {
                result += clip.getItemAt(index).uri
            }
        }
        return result.toList()
    }

    private fun installRuntimeFiles(
        uris: List<Uri>,
    ): Mamba2OrtRuntimePackage {
        require(uris.isNotEmpty())
        val staging = File(noBackupFilesDir, STAGING_DIR)
        val backup = File(noBackupFilesDir, BACKUP_DIR)
        deletePrivate(staging)
        check(staging.mkdirs()) {
            "failed to create K2Q staging directory"
        }

        var totalBytes = 0L
        val seen = HashSet<String>()
        try {
            for (uri in uris) {
                val name = displayName(uri)
                require(SAFE_NAME.matches(name)) {
                    "unsafe runtime filename: " + name
                }
                require(seen.add(name)) {
                    "duplicate runtime filename: " + name
                }
                val target = File(staging, name)
                contentResolver.openInputStream(uri).use { input ->
                    requireNotNull(input) {
                        "cannot open selected runtime file: " + name
                    }
                    FileOutputStream(target).use { output ->
                        val buffer = ByteArray(1024 * 1024)
                        while (true) {
                            val count = input.read(buffer)
                            if (count < 0) break
                            check(count > 0) {
                                "runtime import made no progress"
                            }
                            totalBytes = Math.addExact(
                                totalBytes,
                                count.toLong(),
                            )
                            check(totalBytes <= MAX_IMPORT_BYTES) {
                                "runtime import exceeds 8 GiB safety bound"
                            }
                            output.write(buffer, 0, count)
                        }
                        output.fd.sync()
                    }
                }
                check(target.isFile && target.length() > 0L) {
                    "runtime file copied empty: " + name
                }
            }

            val verified = Mamba2OrtRuntimePackage.load(staging)
            require(verified.stateDtype == "float16") {
                "K2Q requires the K2P-verified FP16 G0.5 lineage"
            }

            deletePrivate(backup)
            if (runtimeRoot.exists()) {
                check(runtimeRoot.renameTo(backup)) {
                    "failed to stage prior K2Q runtime"
                }
            }
            try {
                check(staging.renameTo(runtimeRoot)) {
                    "failed to install K2Q runtime"
                }
                deletePrivate(backup)
            } catch (error: Throwable) {
                if (!runtimeRoot.exists() && backup.exists()) {
                    backup.renameTo(runtimeRoot)
                }
                throw error
            }
            return Mamba2OrtRuntimePackage.load(runtimeRoot)
        } catch (error: Throwable) {
            deletePrivate(staging)
            throw error
        }
    }

    private fun runProfile() {
        setBusy(true)
        status.text = "Profiling Android ONNX Runtime providers…"
        latestReceipt = ""
        copyButton.isEnabled = false

        worker.execute {
            val result = runCatching {
                val runtime = Mamba2OrtRuntimePackage.load(runtimeRoot)
                require(runtime.stateDtype == "float16") {
                    "K2Q requires FP16 runtime"
                }
                val report = VN97Mamba2OrtProviderProfiler().profile(
                    context = applicationContext,
                    runtimeRoot = runtimeRoot,
                    warmupIterations = 1,
                    steadyIterations = 5,
                )
                val output = File(
                    getExternalFilesDir(null) ?: filesDir,
                    EVIDENCE_FILENAME,
                )
                report.writeAtomic(output)
                val canonical = output.readText(Charsets.US_ASCII)
                require(canonical.isNotBlank())
                canonical
            }
            runOnUiThread {
                setBusy(false)
                result.fold(
                    onSuccess = { canonical ->
                        latestReceipt = canonical
                        copyButton.isEnabled = true
                        status.text =
                            "K2Q physical-device profile complete.\n" +
                                canonical
                    },
                    onFailure = { error ->
                        status.text =
                            "K2Q profile failed: " +
                                (error.message ?: error::class.java.simpleName)
                    },
                )
            }
        }
    }

    private fun copyEvidence() {
        if (latestReceipt.isBlank()) return
        val clipboard =
            getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
        clipboard.setPrimaryClip(
            ClipData.newPlainText(
                "VN97 K2Q device evidence",
                latestReceipt,
            )
        )
        status.text =
            "Evidence copied to clipboard.\n" + latestReceipt
    }

    private fun runtimeStatus(): String =
        runCatching {
            val runtime = Mamba2OrtRuntimePackage.load(runtimeRoot)
            "Runtime ready.\n" +
                "runtime_id=" + runtime.runtimeId + "\n" +
                "state_dtype=" + runtime.stateDtype
        }.getOrElse {
            "Import runtime.vn97m2g06.json and every recurrent-8.onnx payload " +
                "from the K2P G0.5 bundle."
        }

    private fun displayName(uri: Uri): String {
        var cursor: Cursor? = null
        try {
            cursor = contentResolver.query(
                uri,
                arrayOf(OpenableColumns.DISPLAY_NAME),
                null,
                null,
                null,
            )
            if (cursor != null && cursor.moveToFirst()) {
                val index = cursor.getColumnIndex(
                    OpenableColumns.DISPLAY_NAME
                )
                if (index >= 0) {
                    val value = cursor.getString(index)
                    if (!value.isNullOrBlank()) return value
                }
            }
        } finally {
            cursor?.close()
        }
        error("selected runtime file has no display name")
    }

    private fun setBusy(busy: Boolean) {
        importButton.isEnabled = !busy
        runButton.isEnabled = !busy
        if (busy) copyButton.isEnabled = false
    }

    private fun deletePrivate(file: File) {
        if (!file.exists()) return
        val root = noBackupFilesDir.canonicalFile
        val target = file.canonicalFile
        check(
            target.toPath().startsWith(root.toPath()) &&
                target != root
        ) {
            "refusing to delete outside app-private storage"
        }
        check(file.deleteRecursively()) {
            "failed to clean app-private runtime path"
        }
    }

    companion object {
        private const val REQUEST_RUNTIME_FILES = 9706
        private const val RUNTIME_DIR = "vn97-r2-k2q-g06"
        private const val STAGING_DIR = ".vn97-r2-k2q-g06.staging"
        private const val BACKUP_DIR = ".vn97-r2-k2q-g06.backup"
        private const val EVIDENCE_FILENAME =
            "vn97-r2-k2q-device-profile.json"
        private const val MAX_IMPORT_BYTES =
            8L * 1024L * 1024L * 1024L
        private val SAFE_NAME =
            Regex("^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    }
}
