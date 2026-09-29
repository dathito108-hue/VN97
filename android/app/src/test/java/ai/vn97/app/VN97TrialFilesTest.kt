package ai.vn97.app

import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import java.io.File
import java.nio.file.Files
import java.util.zip.ZipEntry
import java.util.zip.ZipOutputStream
import org.junit.Assert.*
import org.junit.Test

class VN97TrialFilesTest {
    private val limits = mapOf("candidate.onnx" to 4L, "candidate.onnx.data" to 4L, "quantization.json" to 4L)
    private fun zip(names: List<String>): ByteArray {
        val bytes = ByteArrayOutputStream()
        ZipOutputStream(bytes).use { out -> names.forEach {
            out.putNextEntry(ZipEntry(it)); if (!it.endsWith('/')) out.write(byteArrayOf(1, 2)); out.closeEntry()
        } }
        return bytes.toByteArray()
    }
    private fun run(bytes: ByteArray, check: (File, Exception?) -> Unit) {
        val root = Files.createTempDirectory("trial-zip").toFile()
        try {
            val error = try { VN97TrialFiles.unzip(ByteArrayInputStream(bytes), root, limits); null } catch (e: Exception) { e }
            check(root, error)
        } finally { root.deleteRecursively() }
    }
    @Test fun flatArchiveAccepted() = run(zip(limits.keys.toList())) { root, e ->
        assertNull(e); assertEquals(limits.keys, root.listFiles()!!.map { it.name }.toSet())
    }
    @Test fun wrapperDirectoriesAccepted() = run(zip(listOf("download/", "download/int8/") + limits.keys.map { "download/int8/$it" })) { root, e ->
        assertNull(e); assertEquals(3, root.listFiles()!!.size)
    }
    @Test fun ordinaryFileIsNotMisreportedAsMissingArchiveFiles() = run("not a zip".toByteArray()) { _, e ->
        assertTrue(e!!.message!!.contains("không phải ZIP"))
    }
    @Test fun missingNamesAreReported() = run(zip(listOf("candidate.onnx"))) { _, e ->
        assertTrue(e!!.message!!.contains("candidate.onnx.data")); assertTrue(e.message!!.contains("quantization.json"))
    }
    @Test fun differentDirectoriesRejected() = run(zip(listOf("a/candidate.onnx", "b/candidate.onnx.data", "a/quantization.json"))) { _, e -> assertNotNull(e) }
    @Test fun traversalRejected() = run(zip(listOf("../candidate.onnx"))) { _, e -> assertNotNull(e) }
    @Test fun emptyZipRejected() = run(zip(emptyList())) { _, e -> assertNotNull(e) }
    @Test fun wrongPackageRejected() = run(zip(listOf("tokenizer.json"))) { _, e -> assertTrue(e!!.message!!.contains("Sai gói")) }
    @Test fun duplicateBasenameRejected() = run(zip(listOf("a/candidate.onnx", "b/candidate.onnx"))) { _, e -> assertNotNull(e) }
    @Test fun fileSetRequiresAllThreeBeforeOpeningStreams() {
        val root = Files.createTempDirectory("trial-files").toFile()
        try {
            val error = assertThrows(IllegalArgumentException::class.java) {
                VN97Int8Archive.installFiles(mapOf("candidate.onnx" to { error("must not open") }), root)
            }
            assertTrue(error.message!!.contains("candidate.onnx.data"))
            assertEquals(0, root.listFiles()!!.size)
        } finally { root.deleteRecursively() }
    }
}
