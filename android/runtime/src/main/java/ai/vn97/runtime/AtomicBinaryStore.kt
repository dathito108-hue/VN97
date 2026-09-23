package ai.vn97.runtime

import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.nio.file.AtomicMoveNotSupportedException
import java.nio.file.Files
import java.nio.file.LinkOption
import java.nio.file.StandardCopyOption

internal class AtomicBinaryStore(
    root: File,
    fileName: String,
    private val minBytes: Int,
    private val maxBytes: Int,
    private val tempPrefix: String,
) {
    private val root = root
    private val target: File

    init {
        require(fileName.isNotBlank())
        require('/' !in fileName && '\\' !in fileName && fileName != "." && fileName != "..")
        require(minBytes > 0 && maxBytes >= minBytes)
        require(tempPrefix.startsWith('.') && tempPrefix.length >= 3)
        Files.createDirectories(root.toPath())
        require(Files.isDirectory(root.toPath(), LinkOption.NOFOLLOW_LINKS))
        require(!Files.isSymbolicLink(root.toPath()))
        target = File(root, fileName)
    }

    @Synchronized
    fun save(bytes: ByteArray) {
        require(bytes.size in minBytes..maxBytes) { "binary checkpoint size is outside configured bounds" }
        val temp = Files.createTempFile(root.toPath(), tempPrefix, ".tmp").toFile()
        try {
            FileOutputStream(temp).use { stream ->
                stream.write(bytes)
                stream.flush()
                stream.fd.sync()
            }
            try {
                Files.move(temp.toPath(), target.toPath(), StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING)
            } catch (exc: AtomicMoveNotSupportedException) {
                throw IOException("checkpoint filesystem does not support atomic replace", exc)
            }
            fsyncDirectory()
        } finally {
            Files.deleteIfExists(temp.toPath())
        }
    }

    @Synchronized
    fun loadOrNull(): ByteArray? {
        if (!Files.exists(target.toPath(), LinkOption.NOFOLLOW_LINKS)) return null
        if (Files.isSymbolicLink(target.toPath())) throw IOException("checkpoint target must not be a symlink")
        val size = Files.size(target.toPath())
        if (size !in minBytes.toLong()..maxBytes.toLong()) throw IOException("checkpoint size is outside configured bounds")
        val bytes = Files.readAllBytes(target.toPath())
        if (bytes.size.toLong() != size) throw IOException("checkpoint changed while being read")
        return bytes
    }

    @Synchronized
    fun delete() {
        Files.deleteIfExists(target.toPath())
        fsyncDirectory()
    }

    private fun fsyncDirectory() {
        val errno = NativeRuntimeBindings.nativeFsyncDirectory(root.absolutePath)
        if (errno != 0) throw IOException("checkpoint directory fsync failed: errno=$errno")
    }
}
