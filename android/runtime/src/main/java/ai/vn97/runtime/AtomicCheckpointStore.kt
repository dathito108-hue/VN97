package ai.vn97.runtime

import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.nio.file.AtomicMoveNotSupportedException
import java.nio.file.Files
import java.nio.file.LinkOption
import java.nio.file.StandardCopyOption

class AtomicCheckpointStore(
    private val root: File,
    fileName: String = "runtime.vn97run1",
    private val maxCheckpointBytes: Int = 512 * 1024 * 1024 + 64,
) {
    private val target: File

    init {
        require(fileName.isNotBlank()) { "fileName must not be blank" }
        require('/' !in fileName && '\\' !in fileName && fileName != "." && fileName != "..") {
            "fileName must be one trusted path component"
        }
        require(maxCheckpointBytes >= 64) { "maxCheckpointBytes must be at least one VN97RUN1 header" }
        Files.createDirectories(root.toPath())
        require(Files.isDirectory(root.toPath(), LinkOption.NOFOLLOW_LINKS)) { "checkpoint root must be a directory" }
        require(!Files.isSymbolicLink(root.toPath())) { "checkpoint root must not be a symlink" }
        target = File(root, fileName)
    }

    @Synchronized
    fun save(checkpoint: ByteArray) {
        require(checkpoint.size in 64..maxCheckpointBytes) { "checkpoint size is outside configured bounds" }
        val temp = Files.createTempFile(root.toPath(), ".vn97-checkpoint-", ".tmp").toFile()
        try {
            FileOutputStream(temp).use { stream ->
                stream.write(checkpoint)
                stream.flush()
                stream.fd.sync()
            }
            try {
                Files.move(
                    temp.toPath(),
                    target.toPath(),
                    StandardCopyOption.ATOMIC_MOVE,
                    StandardCopyOption.REPLACE_EXISTING,
                )
            } catch (exc: AtomicMoveNotSupportedException) {
                throw IOException("checkpoint filesystem does not support atomic replace", exc)
            }
            val errno = NativeRuntimeBindings.nativeFsyncDirectory(root.absolutePath)
            if (errno != 0) {
                throw IOException("checkpoint directory fsync failed: errno=$errno")
            }
        } finally {
            Files.deleteIfExists(temp.toPath())
        }
    }

    @Synchronized
    fun loadOrNull(): ByteArray? {
        if (!Files.exists(target.toPath(), LinkOption.NOFOLLOW_LINKS)) return null
        if (Files.isSymbolicLink(target.toPath())) {
            throw IOException("checkpoint target must not be a symlink")
        }
        val size = Files.size(target.toPath())
        if (size !in 64L..maxCheckpointBytes.toLong()) {
            throw IOException("checkpoint size is outside configured bounds")
        }
        val bytes = Files.readAllBytes(target.toPath())
        if (bytes.size.toLong() != size) {
            throw IOException("checkpoint changed while being read")
        }
        return bytes
    }

    @Synchronized
    fun delete() {
        Files.deleteIfExists(target.toPath())
        val errno = NativeRuntimeBindings.nativeFsyncDirectory(root.absolutePath)
        if (errno != 0) {
            throw IOException("checkpoint directory fsync failed: errno=$errno")
        }
    }
}
