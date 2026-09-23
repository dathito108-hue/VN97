package ai.vn97.platform

import ai.vn97.runtime.NativeCompositeContinuity
import ai.vn97.runtime.NativePlanController
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.charset.CodingErrorAction
import java.nio.charset.StandardCharsets
import java.nio.file.AtomicMoveNotSupportedException
import java.nio.file.Files
import java.nio.file.LinkOption
import java.nio.file.StandardCopyOption
import java.security.MessageDigest

private val ACB_MAGIC = "VN97ACB1".toByteArray(StandardCharsets.US_ASCII)
private const val ACB_VERSION = 1
private const val ACB_HEADER_BYTES = 48
private const val ACB_FIXED_PAYLOAD_BYTES = 64
private const val ACB_MAX_PRINCIPAL_BYTES = 256
private const val ACB_MAX_BYTES = ACB_HEADER_BYTES + ACB_FIXED_PAYLOAD_BYTES + ACB_MAX_PRINCIPAL_BYTES

class VN97AssistantContinuationException(message: String, cause: Throwable? = null) :
    IllegalStateException(message, cause)

data class VN97AssistantContinuationBinding(
    val principal: String,
    val planId: String,
    val modelIdHex: String,
) {
    init {
        validatePrincipal(principal)
        requireHex64(planId, "planId")
        requireHex64(modelIdHex, "modelIdHex")
    }

    fun requireModelId(modelId: ByteArray) {
        if (modelId.size != 32 || modelId.hex() != modelIdHex) {
            fail("activated model identity does not match assistant continuation binding")
        }
    }
}

class VN97RestoredAssistantContinuation internal constructor(
    val controller: NativePlanController,
    val binding: VN97AssistantContinuationBinding,
    val epoch: Long,
    internal val continuity: NativeCompositeContinuity,
)

/** App-private immutable M6 principal / plan / model binding for persisted assistant jobs. */
class VN97AssistantContinuationBindingStore(
    root: File,
    fileName: String = "assistant.vn97acb1",
) {
    private val rootPath = root.toPath()
    private val target = root.resolve(fileName).toPath()

    init {
        require(safeFileName(fileName)) { "assistant binding file name is invalid" }
        Files.createDirectories(rootPath)
        if (!Files.isDirectory(rootPath, LinkOption.NOFOLLOW_LINKS) || Files.isSymbolicLink(rootPath)) {
            fail("assistant continuation root must be a real directory")
        }
        if (Files.exists(target, LinkOption.NOFOLLOW_LINKS) && Files.isSymbolicLink(target)) {
            fail("assistant continuation binding must not be a symlink")
        }
    }

    @Synchronized
    fun bind(binding: VN97AssistantContinuationBinding): VN97AssistantContinuationBinding {
        val existing = loadOrNull()
        if (existing != null) {
            if (existing != binding) fail("assistant continuation binding is immutable")
            return existing
        }
        val temp = Files.createTempFile(rootPath, ".vn97-acb-", ".tmp").toFile()
        try {
            FileOutputStream(temp).use { out ->
                out.write(encode(binding)); out.flush(); out.fd.sync()
            }
            try {
                Files.move(temp.toPath(), target, StandardCopyOption.ATOMIC_MOVE)
            } catch (exc: AtomicMoveNotSupportedException) {
                throw VN97AssistantContinuationException("assistant binding requires atomic create", exc)
            }
            forceDirectory()
        } catch (exc: VN97AssistantContinuationException) {
            throw exc
        } catch (exc: Exception) {
            throw VN97AssistantContinuationException("assistant binding persistence failed", exc)
        } finally {
            Files.deleteIfExists(temp.toPath())
        }
        return binding
    }

    @Synchronized
    fun require(expected: VN97AssistantContinuationBinding): VN97AssistantContinuationBinding {
        val durable = loadOrNull() ?: fail("assistant continuation binding is missing")
        if (durable != expected) fail("JobScheduler binding does not match durable assistant binding")
        return durable
    }

    @Synchronized
    fun loadOrNull(): VN97AssistantContinuationBinding? {
        if (!Files.exists(target, LinkOption.NOFOLLOW_LINKS)) return null
        if (Files.isSymbolicLink(target)) fail("assistant continuation binding must not be a symlink")
        val size = try { Files.size(target) } catch (exc: Exception) {
            throw VN97AssistantContinuationException("assistant binding size read failed", exc)
        }
        if (size !in ACB_HEADER_BYTES.toLong()..ACB_MAX_BYTES.toLong()) {
            fail("assistant continuation binding size is outside bounds")
        }
        val bytes = try { Files.readAllBytes(target) } catch (exc: Exception) {
            throw VN97AssistantContinuationException("assistant binding read failed", exc)
        }
        if (bytes.size.toLong() != size) fail("assistant continuation binding changed while reading")
        return decode(bytes)
    }

    @Synchronized
    fun delete() {
        val deleted = try { Files.deleteIfExists(target) } catch (exc: Exception) {
            throw VN97AssistantContinuationException("assistant binding delete failed", exc)
        }
        if (deleted) forceDirectory()
    }

    private fun forceDirectory() {
        try {
            java.nio.channels.FileChannel.open(rootPath, java.nio.file.StandardOpenOption.READ)
                .use { it.force(true) }
        } catch (exc: IOException) {
            throw VN97AssistantContinuationException("assistant binding directory fsync failed", exc)
        }
    }
}

internal fun restoreAssistantContinuation(
    expected: VN97AssistantContinuationBinding,
    continuity: NativeCompositeContinuity,
): VN97RestoredAssistantContinuation {
    val manifest = continuity.manifest
    if (manifest.planId != expected.planId || manifest.modelId.hex() != expected.modelIdHex) {
        fail("VN97CNT1 does not match durable assistant continuation binding")
    }
    if (continuity.planner.plan.isTerminal()) {
        fail("terminal planner must not be restored as background continuation")
    }
    return VN97RestoredAssistantContinuation(
        continuity.planner,
        expected,
        manifest.epoch,
        continuity,
    )
}

private fun encode(binding: VN97AssistantContinuationBinding): ByteArray {
    val principal = binding.principal.toByteArray(StandardCharsets.UTF_8)
    val payload = ByteBuffer.allocate(ACB_FIXED_PAYLOAD_BYTES + principal.size)
        .order(ByteOrder.LITTLE_ENDIAN)
        .apply {
            put(hexBytes(binding.planId)); put(hexBytes(binding.modelIdHex)); put(principal)
        }.array()
    val digest = MessageDigest.getInstance("SHA-256").digest(payload)
    return ByteBuffer.allocate(ACB_HEADER_BYTES + payload.size).order(ByteOrder.LITTLE_ENDIAN)
        .apply {
            put(ACB_MAGIC); putInt(ACB_VERSION); putInt(principal.size); put(digest); put(payload)
        }.array()
}

private fun decode(bytes: ByteArray): VN97AssistantContinuationBinding {
    if (bytes.size < ACB_HEADER_BYTES + ACB_FIXED_PAYLOAD_BYTES + 1) fail("VN97ACB1 is too short")
    val h = ByteBuffer.wrap(bytes, 0, ACB_HEADER_BYTES).order(ByteOrder.LITTLE_ENDIAN)
    val magic = ByteArray(8).also(h::get)
    if (!magic.contentEquals(ACB_MAGIC)) fail("bad VN97ACB1 magic")
    if (h.int != ACB_VERSION) fail("unsupported VN97ACB1 version")
    val principalSize = h.int
    if (principalSize !in 1..ACB_MAX_PRINCIPAL_BYTES ||
        bytes.size != ACB_HEADER_BYTES + ACB_FIXED_PAYLOAD_BYTES + principalSize) {
        fail("VN97ACB1 length is invalid")
    }
    val expected = ByteArray(32).also(h::get)
    val payload = bytes.copyOfRange(ACB_HEADER_BYTES, bytes.size)
    if (!MessageDigest.getInstance("SHA-256").digest(payload).contentEquals(expected)) {
        fail("VN97ACB1 SHA-256 mismatch")
    }
    val p = ByteBuffer.wrap(payload).order(ByteOrder.LITTLE_ENDIAN)
    val planId = ByteArray(32).also(p::get).hex()
    val modelId = ByteArray(32).also(p::get).hex()
    val principal = strictUtf8(ByteArray(principalSize).also(p::get))
    return VN97AssistantContinuationBinding(principal, planId, modelId)
}

private fun validatePrincipal(value: String) {
    val bytes = value.toByteArray(StandardCharsets.UTF_8)
    val allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._:-/".toSet()
    require(value.isNotEmpty() && bytes.size <= ACB_MAX_PRINCIPAL_BYTES && value.all { it in allowed }) {
        "principal is invalid"
    }
}

private fun requireHex64(value: String, label: String) {
    require(value.length == 64 && value.all { it in "0123456789abcdef" }) {
        "$label must be lowercase SHA-256 hex"
    }
}

private fun strictUtf8(bytes: ByteArray): String = try {
    StandardCharsets.UTF_8.newDecoder()
        .onMalformedInput(CodingErrorAction.REPORT)
        .onUnmappableCharacter(CodingErrorAction.REPORT)
        .decode(ByteBuffer.wrap(bytes)).toString()
} catch (exc: Exception) {
    throw VN97AssistantContinuationException("VN97ACB1 principal is not UTF-8", exc)
}

private fun safeFileName(value: String): Boolean =
    value.isNotEmpty() && value.length <= 128 && value != "." && value != ".." &&
        value.all { it.isLetterOrDigit() || it == '.' || it == '_' || it == '-' }

private fun hexBytes(value: String): ByteArray {
    requireHex64(value, "hex value")
    return ByteArray(32) { i -> value.substring(i * 2, i * 2 + 2).toInt(16).toByte() }
}

private fun ByteArray.hex(): String = joinToString("") { "%02x".format(it.toInt() and 0xff) }
private fun fail(message: String): Nothing = throw VN97AssistantContinuationException(message)
