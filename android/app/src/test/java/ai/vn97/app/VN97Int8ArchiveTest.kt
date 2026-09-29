package ai.vn97.app

import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import java.io.File
import java.nio.file.Files
import java.util.zip.ZipEntry
import java.util.zip.ZipOutputStream
import org.junit.Assert.*
import org.junit.Test

class VN97Int8ArchiveTest {
    private fun reject(name: String, bytes: ByteArray) {
        val root = Files.createTempDirectory("int8-test").toFile()
        try {
            val prior = File(root, "unrelated").apply { writeText("keep") }
            val archive = ByteArrayOutputStream()
            ZipOutputStream(archive).use { it.putNextEntry(ZipEntry(name)); it.write(bytes); it.closeEntry() }
            assertThrows(Exception::class.java) { VN97Int8Archive.install(ByteArrayInputStream(archive.toByteArray()), root) }
            assertFalse(File(root, "staging").exists())
            assertFalse(File(root, VN97Int8Archive.ID).exists())
            assertEquals("keep", prior.readText())
        } finally { root.deleteRecursively() }
    }
    @Test fun traversalRejected() = reject("../escape", byteArrayOf(1))
    @Test fun metadataBombRejected() = reject("quantization.json", ByteArray(65537))
    @Test fun incompletePayloadRejected() = reject("candidate.onnx", byteArrayOf(1))
    @Test fun mixedModelRejected() = reject("mixed.onnx", byteArrayOf(1))
}
