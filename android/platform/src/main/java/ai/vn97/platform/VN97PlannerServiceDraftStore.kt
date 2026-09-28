package ai.vn97.platform

import android.content.Context
import android.util.AtomicFile
import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import java.io.DataInputStream
import java.io.DataOutputStream
import java.io.EOFException
import java.io.File

data class VN97PlannerServiceDraft(
    val service: VN97LocalDigitalService,
    val source: String,
    val delivery: VN97LocalDigitalDelivery,
    val createdAtEpochMs: Long,
) {
    init {
        require(createdAtEpochMs > 0L) { "draft time must be positive" }
        require(source.toByteArray(Charsets.UTF_8).size <= VN97DigitalServiceCore.MAX_INPUT_BYTES)
        require(delivery.content.toByteArray(Charsets.UTF_8).size <= VN97DigitalServiceCore.MAX_OUTPUT_BYTES)
    }
}

/** Single latest private draft. Atomic replacement prevents partial planner handoffs. */
class VN97PlannerServiceDraftStore(context: Context) {
    private val file = AtomicFile(
        File(context.applicationContext.noBackupFilesDir, FILE_NAME)
    )

    @Synchronized
    fun save(draft: VN97PlannerServiceDraft) {
        val bytes = encode(draft)
        val output = file.startWrite()
        try {
            output.write(bytes)
            file.finishWrite(output)
        } catch (failure: Throwable) {
            file.failWrite(output)
            throw failure
        }
    }

    @Synchronized
    fun loadOrNull(): VN97PlannerServiceDraft? = try {
        file.openRead().use { input ->
            val buffer = ByteArray(MAX_STORED_BYTES + 1)
            var count = 0
            while (count < buffer.size) {
                val read = input.read(buffer, count, buffer.size - count)
                if (read < 0) break
                count += read
            }
            require(count <= MAX_STORED_BYTES) { "planner service draft exceeds byte bound" }
            decode(buffer.copyOf(count))
        }
    } catch (_: java.io.FileNotFoundException) {
        null
    }

    private fun encode(draft: VN97PlannerServiceDraft): ByteArray =
        ByteArrayOutputStream().use { buffer ->
            DataOutputStream(buffer).use { output ->
                output.write(MAGIC)
                output.writeLong(draft.createdAtEpochMs)
                output.writeBounded(draft.service.name, 64)
                output.writeBounded(draft.source, VN97DigitalServiceCore.MAX_INPUT_BYTES)
                output.writeBounded(draft.delivery.fileName, 256)
                output.writeBounded(draft.delivery.content, VN97DigitalServiceCore.MAX_OUTPUT_BYTES)
                output.writeBounded(draft.delivery.report, 16 * 1024)
                output.writeBounded(draft.delivery.sourceSha256, 64)
                output.writeBounded(draft.delivery.outputSha256, 64)
            }
            buffer.toByteArray().also {
                require(it.size <= MAX_STORED_BYTES) { "planner service draft exceeds storage bound" }
            }
        }

    private fun decode(bytes: ByteArray): VN97PlannerServiceDraft = try {
        DataInputStream(ByteArrayInputStream(bytes)).use { input ->
            val magic = ByteArray(MAGIC.size).also(input::readFully)
            require(magic.contentEquals(MAGIC)) { "planner service draft magic mismatch" }
            val createdAtEpochMs = input.readLong()
            val service = VN97LocalDigitalService.valueOf(input.readBounded(64))
            val source = input.readBounded(VN97DigitalServiceCore.MAX_INPUT_BYTES)
            val delivery = VN97LocalDigitalDelivery(
                fileName = input.readBounded(256),
                content = input.readBounded(VN97DigitalServiceCore.MAX_OUTPUT_BYTES),
                report = input.readBounded(16 * 1024),
                sourceSha256 = input.readBounded(64),
                outputSha256 = input.readBounded(64),
            )
            require(input.read() == -1) { "planner service draft has trailing bytes" }
            val recomputed = VN97DigitalServiceCore.produce(service, source)
            require(recomputed == delivery) { "planner service draft content or identity mismatch" }
            VN97PlannerServiceDraft(service, source, delivery, createdAtEpochMs)
        }
    } catch (failure: EOFException) {
        throw IllegalArgumentException("planner service draft is truncated", failure)
    }

    private fun DataOutputStream.writeBounded(value: String, maxBytes: Int) {
        val bytes = value.toByteArray(Charsets.UTF_8)
        require(bytes.size <= maxBytes) { "draft field exceeds byte bound" }
        writeInt(bytes.size)
        write(bytes)
    }

    private fun DataInputStream.readBounded(maxBytes: Int): String {
        val size = readInt()
        require(size in 0..maxBytes) { "draft field length is invalid" }
        val bytes = ByteArray(size).also(::readFully)
        return String(bytes, Charsets.UTF_8).also {
            require(it.toByteArray(Charsets.UTF_8).contentEquals(bytes)) {
                "draft field is not canonical UTF-8"
            }
        }
    }

    companion object {
        private const val FILE_NAME = "vn97-planner-service-draft.vn97dsd1"
        private const val MAX_STORED_BYTES =
            VN97DigitalServiceCore.MAX_INPUT_BYTES +
                VN97DigitalServiceCore.MAX_OUTPUT_BYTES + 32 * 1024
        private val MAGIC = "VN97DSD1".toByteArray(Charsets.US_ASCII)
    }
}
