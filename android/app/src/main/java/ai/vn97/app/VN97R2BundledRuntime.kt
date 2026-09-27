package ai.vn97.app

import android.content.res.AssetManager
import java.io.File
import java.io.FileOutputStream
import java.security.MessageDigest

/**
 * F6 crash-safe installer for APK-bundled R2 runtime assets.
 *
 * Release APKs must carry vn97-r2/. The installer copies into an app-private
 * staging directory, computes a deterministic tree identity, fsyncs files,
 * then atomically swaps the directory. F1/F2 perform semantic/hash validation
 * before any inference session can be opened.
 */
class VN97R2BundledRuntime(
    private val application: VN97Application,
) {
    private val lock = Any()

    fun installIfPresent(required: Boolean): Boolean =
        synchronized(lock) {
            val entries =
                application.assets
                    .list(ASSET_ROOT)
                    ?.toList()
                    .orEmpty()
            if (entries.isEmpty()) {
                check(!required) {
                    "turnkey APK is missing bundled VN97 R2 runtime"
                }
                return@synchronized false
            }

            val target =
                File(application.noBackupFilesDir, TARGET_DIR)
            val staging =
                File(application.noBackupFilesDir, STAGING_DIR)
            val backup =
                File(application.noBackupFilesDir, BACKUP_DIR)

            deleteRecursivelySafe(staging)
            check(staging.mkdirs()) {
                "failed to create VN97 R2 staging directory"
            }

            val digest = MessageDigest.getInstance("SHA-256")
            var totalBytes = 0L
            val files = ArrayList<String>()
            try {
                copyTree(
                    assets = application.assets,
                    assetPath = ASSET_ROOT,
                    relative = "",
                    destination = staging,
                    digest = digest,
                    files = files,
                    byteCounter = { count ->
                        totalBytes = Math.addExact(totalBytes, count)
                        check(totalBytes <= MAX_TOTAL_BYTES) {
                            "bundled VN97 R2 runtime exceeds byte bound"
                        }
                    },
                )
                requireRequiredLayout(staging)
                val treeId =
                    digest.digest().joinToString("") {
                        "%02x".format(it.toInt() and 0xff)
                    }
                File(staging, IDENTITY_FILE)
                    .writeText(treeId + "\n", Charsets.US_ASCII)

                val existingId =
                    File(target, IDENTITY_FILE)
                        .takeIf { it.isFile }
                        ?.readText(Charsets.US_ASCII)
                        ?.trim()
                if (existingId == treeId) {
                    deleteRecursivelySafe(staging)
                    return@synchronized false
                }

                deleteRecursivelySafe(backup)
                if (target.exists()) {
                    check(target.renameTo(backup)) {
                        "failed to stage prior VN97 R2 runtime for migration"
                    }
                }
                try {
                    check(staging.renameTo(target)) {
                        "failed to atomically install VN97 R2 runtime"
                    }
                    deleteRecursivelySafe(backup)
                } catch (error: Throwable) {
                    if (!target.exists() && backup.exists()) {
                        backup.renameTo(target)
                    }
                    throw error
                }
                true
            } catch (error: Throwable) {
                deleteRecursivelySafe(staging)
                throw error
            }
        }

    private fun copyTree(
        assets: AssetManager,
        assetPath: String,
        relative: String,
        destination: File,
        digest: MessageDigest,
        files: MutableList<String>,
        byteCounter: (Long) -> Unit,
    ) {
        val children = assets.list(assetPath)?.toList().orEmpty()
        if (children.isEmpty()) {
            require(relative.isNotEmpty()) {
                "VN97 R2 asset root is empty"
            }
            requireSafeRelative(relative)
            val out = File(destination, relative)
            check(out.parentFile?.mkdirs() != false)
            digest.update(relative.toByteArray(Charsets.UTF_8))
            digest.update(0.toByte())
            assets.open(assetPath).use { input ->
                FileOutputStream(out).use { output ->
                    val buffer = ByteArray(64 * 1024)
                    while (true) {
                        val count = input.read(buffer)
                        if (count < 0) break
                        check(count > 0) {
                            "VN97 R2 asset read made no progress"
                        }
                        byteCounter(count.toLong())
                        digest.update(buffer, 0, count)
                        output.write(buffer, 0, count)
                    }
                    output.fd.sync()
                }
            }
            files += relative
            return
        }

        children.sorted().forEach { child ->
            requireSafeName(child)
            val nextRelative =
                if (relative.isEmpty()) child
                else "$relative/$child"
            copyTree(
                assets = assets,
                assetPath = "$assetPath/$child",
                relative = nextRelative,
                destination = destination,
                digest = digest,
                files = files,
                byteCounter = byteCounter,
            )
        }
    }

    private fun requireRequiredLayout(root: File) {
        val required = listOf(
            "binding.vn97r2f2.json",
            "tuning.vn97r2e4.json",
            "runtime/runtime.vn97ort1.json",
            "runtime/step.onnx",
        )
        required.forEach { relative ->
            val file = File(root, relative)
            check(file.isFile && file.length() > 0L) {
                "bundled VN97 R2 required asset missing: $relative"
            }
        }
        val chunks =
            File(root, "runtime")
                .listFiles()
                ?.filter {
                    it.isFile &&
                        Regex("^chunk-[1-9][0-9]*\\.onnx$")
                            .matches(it.name) &&
                        it.length() > 0L
                }
                .orEmpty()
        check(chunks.isNotEmpty()) {
            "bundled VN97 R2 runtime has no chunk graph"
        }
    }

    private fun deleteRecursivelySafe(file: File) {
        if (!file.exists()) return
        check(
            file.canonicalPath.startsWith(
                application.noBackupFilesDir.canonicalPath +
                    File.separator
            )
        )
        file.deleteRecursively()
        check(!file.exists()) {
            "failed to clean VN97 R2 runtime path"
        }
    }

    private fun requireSafeRelative(value: String) {
        require(
            value.split('/').all {
                it.isNotEmpty() &&
                    it != "." &&
                    it != ".." &&
                    it.all { ch ->
                        ch.isLetterOrDigit() ||
                            ch == '.' ||
                            ch == '_' ||
                            ch == '-'
                    }
            }
        )
    }

    private fun requireSafeName(value: String) {
        requireSafeRelative(value)
        require('/' !in value)
    }

    companion object {
        private const val ASSET_ROOT = "vn97-r2"
        private const val TARGET_DIR = "vn97-r2"
        private const val STAGING_DIR = ".vn97-r2.staging"
        private const val BACKUP_DIR = ".vn97-r2.backup"
        private const val IDENTITY_FILE = ".apk-assets.sha256"
        private const val MAX_TOTAL_BYTES =
            768L * 1024L * 1024L
    }
}
