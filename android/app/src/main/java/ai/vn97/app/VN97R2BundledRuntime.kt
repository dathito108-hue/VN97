package ai.vn97.app

import ai.vn97.runtime.NativeActivatedModel
import ai.vn97.runtime.VN97R2AssistantBinding
import ai.vn97.runtime.VN97R2CognitionInference
import java.io.File
import java.io.FileOutputStream
import java.nio.charset.StandardCharsets
import java.nio.file.Files
import java.security.MessageDigest
import org.json.JSONArray
import org.json.JSONObject

internal data class VN97R2BundledAsset(
    val name: String,
    val bytes: Long,
    val sha256: String,
)

internal data class VN97R2BundledIndex(
    val packageId: String,
    val bindingId: String,
    val runtimeId: String,
    val bundleId: String,
    val checkpointSha256: String,
    val tokenizerModelSha256: String,
    val tuningId: String,
    val files: List<VN97R2BundledAsset>,
    val totalBytes: Long,
) {
    init {
        requireSha256R2F6(packageId, "packageId")
        requireSha256R2F6(bindingId, "bindingId")
        requireSha256R2F6(runtimeId, "runtimeId")
        requireSha256R2F6(bundleId, "bundleId")
        requireSha256R2F6(checkpointSha256, "checkpointSha256")
        requireSha256R2F6(tokenizerModelSha256, "tokenizerModelSha256")
        requireSha256R2F6(tuningId, "tuningId")
        require(files.size >= 5)
        require(files.map { it.name }.distinct().size == files.size)
        require(totalBytes == files.sumOf { it.bytes })
        require(totalBytes in 1L..MAX_TOTAL_BYTES)
    }

    companion object {
        const val SCHEMA = "VN97R2APK1"
        const val INDEX_FILENAME = "assets.vn97r2apk1.json"
        const val ASSET_ROOT = "vn97-r2"
        const val MAX_INDEX_BYTES = 64 * 1024
        const val MAX_FILE_BYTES = 768L * 1024L * 1024L
        const val MAX_TOTAL_BYTES = 1536L * 1024L * 1024L

        fun parse(bytes: ByteArray): VN97R2BundledIndex {
            require(bytes.isNotEmpty() && bytes.size <= MAX_INDEX_BYTES)
            val text = bytes.toString(StandardCharsets.UTF_8)
            require(text.endsWith("\n"))
            val root = JSONObject(text)
            val fields = setOf(
                "schema",
                "binding_id",
                "runtime_id",
                "bundle_id",
                "checkpoint_sha256",
                "tokenizer_model_sha256",
                "tuning_id",
                "files",
                "total_bytes",
                "legacy_inference_fallback",
                "package_id",
            )
            require(root.keys().asSequence().toSet() == fields)
            require(root.getString("schema") == SCHEMA)
            require(!root.getBoolean("legacy_inference_fallback"))
            val packageId = root.getString("package_id")
            val body = JSONObject(root.toString())
            body.remove("package_id")
            val prefix = "VN97R2APK1" + String(charArrayOf(0.toChar()))
            require(
                packageId == sha256R2F6(
                    prefix.toByteArray(StandardCharsets.UTF_8) +
                        canonicalJsonR2F6(body)
                            .toByteArray(StandardCharsets.UTF_8)
                )
            ) {
                "R2 F6 package identity mismatch"
            }
            val rawFiles = root.getJSONArray("files")
            val files = buildList {
                for (index in 0 until rawFiles.length()) {
                    val item = rawFiles.getJSONObject(index)
                    require(
                        item.keys().asSequence().toSet() ==
                            setOf("bytes", "name", "sha256")
                    )
                    val name = item.getString("name")
                    require(
                        name.isNotBlank() &&
                            !name.contains('/') &&
                            !name.contains('\\') &&
                            name != INDEX_FILENAME
                    )
                    val size = item.getLong("bytes")
                    require(size in 1L..MAX_FILE_BYTES)
                    val sha = item.getString("sha256")
                    requireSha256R2F6(sha, "asset sha256")
                    add(VN97R2BundledAsset(name, size, sha))
                }
            }
            require(
                files.map { it.name } == files.map { it.name }.sorted()
            )
            return VN97R2BundledIndex(
                packageId = packageId,
                bindingId = root.getString("binding_id"),
                runtimeId = root.getString("runtime_id"),
                bundleId = root.getString("bundle_id"),
                checkpointSha256 = root.getString("checkpoint_sha256"),
                tokenizerModelSha256 =
                    root.getString("tokenizer_model_sha256"),
                tuningId = root.getString("tuning_id"),
                files = files,
                totalBytes = root.getLong("total_bytes"),
            )
        }
    }
}

class VN97R2BundledRuntime(
    private val application: VN97Application,
) {
    @Synchronized
    fun ensureInstalled(
        model: NativeActivatedModel,
        required: Boolean,
    ): File {
        val indexBytes =
            try {
                application.assets.open(
                    VN97R2BundledIndex.ASSET_ROOT + "/" +
                        VN97R2BundledIndex.INDEX_FILENAME
                ).use { it.readBytes() }
            } catch (exc: java.io.FileNotFoundException) {
                if (!required) {
                    val existing = currentRoot()
                    require(existing.isDirectory) {
                        "developer R2 runtime is not installed"
                    }
                    return existing
                }
                throw IllegalStateException(
                    "turnkey APK is missing bundled R2 runtime index",
                    exc,
                )
            }
        val index = VN97R2BundledIndex.parse(indexBytes)
        require(
            model.info.modelId.toHexR2F6() ==
                index.tokenizerModelSha256
        ) {
            "bundled R2 runtime belongs to another activated VN97 model"
        }

        val current = currentRoot()
        if (current.isDirectory) {
            val currentIndex = File(
                current,
                VN97R2BundledIndex.INDEX_FILENAME,
            )
            if (
                currentIndex.isFile &&
                runCatching {
                    VN97R2BundledIndex
                        .parse(currentIndex.readBytes())
                        .packageId
                }.getOrNull() == index.packageId
            ) {
                verifyInstalled(current, index, model)
                return current
            }
        }

        val root = storageRoot()
        ensureDirectory(root)
        val staging = File(root, ".stage-" + index.packageId)
        val previous = File(root, "previous")
        deleteTree(staging)
        check(staging.mkdir()) {
            "failed to create R2 runtime staging directory"
        }
        try {
            index.files.forEach { asset ->
                copyVerifiedAsset(asset, staging)
            }
            File(
                staging,
                VN97R2BundledIndex.INDEX_FILENAME,
            ).writeBytes(indexBytes)
            verifyInstalled(staging, index, model)

            deleteTree(previous)
            if (current.exists()) {
                check(current.renameTo(previous)) {
                    "failed to preserve previous R2 runtime"
                }
            }
            if (!staging.renameTo(current)) {
                if (previous.exists() && !current.exists()) {
                    previous.renameTo(current)
                }
                error("failed to atomically activate bundled R2 runtime")
            }
            deleteTree(previous)
            verifyInstalled(current, index, model)
            return current
        } catch (exc: Throwable) {
            deleteTree(staging)
            throw exc
        }
    }

    private fun verifyInstalled(
        root: File,
        index: VN97R2BundledIndex,
        model: NativeActivatedModel,
    ) {
        require(root.isDirectory && !Files.isSymbolicLink(root.toPath()))
        val actual = root.listFiles()?.map { it.name }?.toSet().orEmpty()
        val expected = index.files.map { it.name }.toSet() +
            VN97R2BundledIndex.INDEX_FILENAME
        require(actual == expected) {
            "installed R2 runtime asset set mismatch"
        }
        index.files.forEach { asset ->
            val file = File(root, asset.name)
            require(
                file.isFile &&
                    !Files.isSymbolicLink(file.toPath()) &&
                    file.length() == asset.bytes &&
                    sha256FileR2F6(file) == asset.sha256
            ) {
                "installed R2 asset integrity mismatch: " + asset.name
            }
        }
        val binding = VN97R2AssistantBinding.load(
            File(root, VN97R2AssistantBinding.FILENAME)
        )
        require(binding.bindingId == index.bindingId)
        require(binding.runtimeId == index.runtimeId)
        require(binding.bundleId == index.bundleId)
        require(binding.checkpointSha256 == index.checkpointSha256)
        require(
            binding.tokenizerModelSha256 ==
                index.tokenizerModelSha256
        )
        require(
            model.info.modelId.toHexR2F6() ==
                binding.tokenizerModelSha256
        )
    }

    private fun copyVerifiedAsset(
        asset: VN97R2BundledAsset,
        staging: File,
    ) {
        val target = File(staging, asset.name)
        val digest = MessageDigest.getInstance("SHA-256")
        var total = 0L
        application.assets.open(
            VN97R2BundledIndex.ASSET_ROOT + "/" + asset.name
        ).use { input ->
            FileOutputStream(target).use { output ->
                val buffer = ByteArray(1024 * 1024)
                while (true) {
                    val count = input.read(buffer)
                    if (count < 0) break
                    check(count > 0)
                    total = Math.addExact(total, count.toLong())
                    check(total <= asset.bytes)
                    digest.update(buffer, 0, count)
                    output.write(buffer, 0, count)
                }
                output.fd.sync()
            }
        }
        require(total == asset.bytes)
        require(
            digest.digest().joinToString("") {
                "%02x".format(it.toInt() and 0xff)
            } == asset.sha256
        )
    }

    private fun storageRoot(): File =
        File(application.noBackupFilesDir, "vn97-r2")

    private fun currentRoot(): File =
        File(storageRoot(), "current")

    private fun ensureDirectory(root: File) {
        if (root.exists()) {
            require(root.isDirectory && !Files.isSymbolicLink(root.toPath()))
        } else {
            check(root.mkdirs())
        }
    }

    private fun deleteTree(file: File) {
        if (!file.exists()) return
        require(!Files.isSymbolicLink(file.toPath()))
        if (file.isDirectory) {
            file.listFiles()?.forEach(::deleteTree)
        }
        check(file.delete()) {
            "failed to delete stale R2 runtime path"
        }
    }
}

private fun requireSha256R2F6(value: String, label: String) {
    require(
        value.length == 64 &&
            value.all { it in '0'..'9' || it in 'a'..'f' }
    ) {
        "$label must be lowercase SHA-256"
    }
}

private fun sha256R2F6(bytes: ByteArray): String =
    MessageDigest.getInstance("SHA-256")
        .digest(bytes)
        .joinToString("") { "%02x".format(it.toInt() and 0xff) }

private fun sha256FileR2F6(file: File): String {
    val digest = MessageDigest.getInstance("SHA-256")
    file.inputStream().buffered().use { input ->
        val buffer = ByteArray(1024 * 1024)
        while (true) {
            val count = input.read(buffer)
            if (count < 0) break
            check(count > 0)
            digest.update(buffer, 0, count)
        }
    }
    return digest.digest()
        .joinToString("") { "%02x".format(it.toInt() and 0xff) }
}

private fun ByteArray.toHexR2F6(): String =
    joinToString("") { "%02x".format(it.toInt() and 0xff) }

private fun canonicalJsonR2F6(value: Any?): String = when (value) {
    JSONObject.NULL, null -> "null"
    is JSONObject -> value.keys().asSequence().toList().sorted().joinToString(
        prefix = "{",
        postfix = "}",
        separator = ",",
    ) { key ->
        JSONObject.quote(key) + ":" + canonicalJsonR2F6(value.get(key))
    }
    is JSONArray -> (0 until value.length()).joinToString(
        prefix = "[",
        postfix = "]",
        separator = ",",
    ) { index -> canonicalJsonR2F6(value.get(index)) }
    is String -> JSONObject.quote(value)
    is Boolean -> if (value) "true" else "false"
    is Int, is Long, is Short, is Byte -> value.toString()
    else -> error(
        "unsupported R2 F6 canonical JSON type: " +
            value::class.java.name
    )
}
