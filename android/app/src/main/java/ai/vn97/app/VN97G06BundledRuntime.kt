package ai.vn97.app

import android.content.res.AssetManager
import java.io.File
import java.io.FileOutputStream
import java.security.MessageDigest

/**
 * G06 crash-safe installer for APK-bundled G06 runtime assets.
 *
 * Release APKs must carry vn97-g06/. The installer copies into an app-private
 * staging directory, computes a deterministic tree identity, fsyncs files,
 * then atomically swaps the directory. G06/G09/G10 perform semantic/hash validation
 * before any inference session can be opened.
 */
class VN97G06BundledRuntime(
    private val application: VN97Application,
    private val validateDeployment: (File) -> Unit,
) {
    private val lock = Any()
    private var installedAssetId: String? = null

    fun installIfPresent(required: Boolean): Boolean =
        synchronized(lock) {
            val target = File(application.noBackupFilesDir, TARGET_DIR)
            val backup = File(application.noBackupFilesDir, BACKUP_DIR)
            // Process death can occur after target -> backup but before staging ->
            // target. Recover before touching assets or deleting any prior bytes.
            // A completed swap keeps its target; never roll it back just because
            // the process died before removing the old backup.
            if (!target.exists() && backup.exists()) {
                requireRequiredLayout(backup)
                validateDeployment(backup)
                check(backup.renameTo(target)) {
                    "failed to recover prior VN97 G06 runtime"
                }
                installedAssetId = null
            }
            val entries =
                application.assets
                    .list(ASSET_ROOT)
                    ?.toList()
                    .orEmpty()
            if (entries.isEmpty()) {
                check(!required) {
                    "turnkey APK is missing bundled VN97 G06 runtime"
                }
                return@synchronized false
            }

            // APK assets are immutable for this process. Runtime opening still verifies
            // every declared file; this only avoids a multi-GB staging copy on reopen.
            val cached = installedAssetId
            if (cached != null && File(target, IDENTITY_FILE).takeIf { it.isFile }
                    ?.readText(Charsets.US_ASCII)?.trim() == cached &&
                runCatching { requireRequiredLayout(target) }.isSuccess) {
                return@synchronized false
            }
            val staging =
                File(application.noBackupFilesDir, STAGING_DIR)

            deleteRecursivelySafe(staging)
            check(staging.mkdirs()) {
                "failed to create VN97 G06 staging directory"
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
                            "bundled VN97 G06 runtime exceeds byte bound"
                        }
                    },
                )
                requireRequiredLayout(staging)
                validateDeployment(staging)
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
                if (existingId == treeId && runCatching {
                        requireRequiredLayout(target)
                        validateDeployment(target)
                    }.isSuccess) {
                    // Do not trust a stale marker over corrupt/missing installed bytes.
                    installedAssetId = treeId
                    deleteRecursivelySafe(staging)
                    return@synchronized false
                }

                deleteRecursivelySafe(backup)
                if (target.exists()) {
                    check(target.renameTo(backup)) {
                        "failed to stage prior VN97 G06 runtime for migration"
                    }
                }
                try {
                    check(staging.renameTo(target)) {
                        "failed to atomically install VN97 G06 runtime"
                    }
                    deleteRecursivelySafe(backup)
                } catch (error: Throwable) {
                    if (!target.exists() && backup.exists()) {
                        backup.renameTo(target)
                    }
                    throw error
                }
                installedAssetId = treeId
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
                "VN97 G06 asset root is empty"
            }
            requireSafeRelative(relative)
            val out = File(destination, relative)
            val parent = checkNotNull(out.parentFile)
            check(parent.isDirectory || parent.mkdirs()) {
                "failed to create VN97 G06 asset directory"
            }
            digest.update(relative.toByteArray(Charsets.UTF_8))
            digest.update(0.toByte())
            assets.open(assetPath).use { input ->
                FileOutputStream(out).use { output ->
                    val buffer = ByteArray(64 * 1024)
                    while (true) {
                        val count = input.read(buffer)
                        if (count < 0) break
                        check(count > 0) {
                            "VN97 G06 asset read made no progress"
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
            "binding.vn97m2g09.json",
            "promotion.vn97m2g10.json",
            "tokenizer/tokenizer.vn97m2g08.json",
            "tokenizer/vocab.json",
            "tokenizer/merges.txt",
            "tuning.vn97m2g07.json",
            "runtime/runtime.vn97m2g06.json",
        )
        required.forEach { relative ->
            val file = File(root, relative)
            check(file.isFile && file.length() > 0L) {
                "bundled VN97 G06 required asset missing: $relative"
            }
        }
        val chunks =
            File(root, "runtime")
                .listFiles()
                ?.filter {
                    it.isFile &&
                        Regex("^recurrent-(8|16|32)\\.onnx$")
                            .matches(it.name) &&
                        it.length() > 0L
                }
                .orEmpty()
        check(chunks.isNotEmpty()) {
            "bundled VN97 G06 runtime has no chunk graph"
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
            "failed to clean VN97 G06 runtime path"
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
        private const val ASSET_ROOT = "vn97-g06"
        private const val TARGET_DIR = "vn97-g06"
        private const val STAGING_DIR = ".vn97-g06.staging"
        private const val BACKUP_DIR = ".vn97-g06.backup"
        private const val IDENTITY_FILE = ".apk-assets.sha256"
        private const val MAX_TOTAL_BYTES =
            8L * 1024L * 1024L * 1024L
    }
}
