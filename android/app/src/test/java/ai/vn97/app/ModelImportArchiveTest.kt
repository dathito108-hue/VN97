package ai.vn97.app

import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import java.io.File
import java.nio.file.Files
import java.util.zip.ZipEntry
import java.util.zip.ZipOutputStream
import org.junit.Assert.*
import org.junit.Test

class ModelImportArchiveTest {
    private fun entries() = linkedMapOf(
        "model.vn97cap" to "VN97CAP1payload".toByteArray(),
        "model.vn97sig" to "signature".toByteArray(),
        "publisher.ed25519" to ByteArray(32) { 7 },
    )
    private fun zip(entries: Map<String, ByteArray>): ByteArray {
        val bytes = ByteArrayOutputStream()
        ZipOutputStream(bytes).use { out ->
            entries.forEach { (name, data) ->
                out.putNextEntry(ZipEntry(name)); out.write(data); out.closeEntry()
            }
        }
        return bytes.toByteArray()
    }
    private fun root(block: (File) -> Unit) {
        val dir = Files.createTempDirectory("import-test-").toFile()
        try { block(dir) } finally { dir.deleteRecursively() }
    }
    private fun reject(entries: Map<String, ByteArray>) = root { dir ->
        var reviewed = false
        assertThrows(Exception::class.java) {
            VN97ModelImportArchive.read(ByteArrayInputStream(zip(entries)), dir) { _, _, _ -> reviewed = true }
        }
        assertFalse(reviewed)
        assertEquals(0, dir.listFiles()!!.size)
    }
    @Test fun validTransportDelegatesToCanonicalReviewAndCleansTemporaryFiles() = root { dir ->
        val result = VN97ModelImportArchive.read(ByteArrayInputStream(zip(entries())), dir) { model, sig, key ->
            assertEquals("VN97CAP1payload", model.readText())
            assertArrayEquals("signature".toByteArray(), sig)
            assertArrayEquals(ByteArray(32) { 7 }, key)
            "review-result"
        }
        assertEquals("review-result", result)
        assertEquals(0, dir.listFiles()!!.size)
    }
    @Test fun missingSignatureRejected() = reject(entries().apply { remove("model.vn97sig") })
    @Test fun traversalRejected() = reject(entries().apply { put("../escaped", byteArrayOf(1)) })
    @Test fun runtimeExtrasRejected() = reject(entries().apply { put("runtime/step.onnx", byteArrayOf(1)) })
    @Test fun compressedOversizedSignatureRejectedByExpandedByteCount() = reject(entries().apply {
        put("model.vn97sig", ByteArray(16 * 1024 + 1))
    })
    @Test fun renamedCheckpointRejected() = reject(entries().apply { put("model.vn97cap", "GGUFnot-a-capability".toByteArray()) })
    @Test fun emptyKeyRejected() = reject(entries().apply { put("publisher.ed25519", byteArrayOf()) })
    @Test fun verificationFailurePreservesUnrelatedFilesAndCleansImport() = root { dir ->
        val prior = File(dir, "prior-model").apply { writeText("keep") }
        assertThrows(IllegalArgumentException::class.java) {
            VN97ModelImportArchive.read(ByteArrayInputStream(zip(entries())), dir) { _, _, _ ->
                throw IllegalArgumentException("signature rejected by canonical verifier")
            }
        }
        assertEquals("keep", prior.readText())
        assertEquals(listOf("prior-model"), dir.listFiles()!!.map { it.name })
    }
    @Test fun duplicateNameRejected() = root { dir ->
        // Equal-length ZIP names allow construction of a duplicate without writer normalization.
        val data = zip(entries().apply { put("model.vn97sih", "second".toByteArray()) })
        val old = "model.vn97sih".toByteArray(); val replacement = "model.vn97sig".toByteArray()
        for (i in 0..data.size - old.size) {
            if (old.indices.all { data[i + it] == old[it] }) replacement.copyInto(data, i)
        }
        assertThrows(IllegalArgumentException::class.java) {
            VN97ModelImportArchive.read(ByteArrayInputStream(data), dir) { _, _, _ -> fail("must not review") }
        }
        assertEquals(0, dir.listFiles()!!.size)
    }
}
