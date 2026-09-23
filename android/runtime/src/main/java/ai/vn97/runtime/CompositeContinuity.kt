package ai.vn97.runtime

import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.charset.CodingErrorAction
import java.nio.charset.StandardCharsets
import java.security.MessageDigest

private val CNT_MAGIC = "VN97CNT1".toByteArray(StandardCharsets.US_ASCII)
private const val CNT_VERSION = 1
private const val CNT_HEADER_BYTES = 48
private const val CNT_MAX_PAYLOAD = 16 * 1024
private const val CNT_MAX_RUNTIME = 512 * 1024 * 1024 + 100
private const val CNT_MAX_PLANNER = (16 shl 20) + 48

class NativeContinuityException(message: String, cause: Throwable? = null) : IllegalStateException(message, cause)

data class NativeRuntimeCheckpointSnapshot internal constructor(
    val checkpoint: ByteArray,
    val info: NativeRuntimeInfo,
    val modelBinding: RuntimeModelBinding,
)

data class NativeContinuityManifest internal constructor(
    val epoch: Long,
    val slot: String,
    val runtimeSha256: String,
    val plannerSha256: String,
    val modelId: ByteArray,
    val planId: String,
    val sequencePosition: Long,
)

data class NativeCompositeContinuity internal constructor(
    val manifest: NativeContinuityManifest,
    val runtimeCheckpoint: ByteArray,
    val planner: NativePlanController,
) {
    fun restoreRuntime(expectedConfig: NativeRuntimeConfig): NativeRuntimeSession {
        val session = NativeRuntimeSession.restore(runtimeCheckpoint)
        try {
            val info = session.info()
            if (info.config != expectedConfig) fail("VN97RUN configuration does not match requested runtime")
            if (info.lifecycle != RuntimeLifecycle.SUSPENDED) fail("composite runtime checkpoint must restore SUSPENDED")
            if (info.sequencePosition != manifest.sequencePosition) fail("runtime sequence position does not match continuity manifest")
            val binding = session.modelBinding()
            if (!binding.bound || !binding.modelId.contentEquals(manifest.modelId)) {
                fail("runtime model identity does not match continuity manifest")
            }
            return session
        } catch (exc: Throwable) {
            session.close()
            throw exc
        }
    }
}

class AtomicCompositeContinuityStore(
    private val root: File,
    manifestFileName: String = "continuity.vn97cnt1",
) {
    private val manifestStore = AtomicBinaryStore(
        root, manifestFileName, CNT_HEADER_BYTES, CNT_HEADER_BYTES + CNT_MAX_PAYLOAD, ".vn97-continuity-manifest-"
    )

    @Synchronized
    fun save(snapshot: NativeRuntimeCheckpointSnapshot, plan: NativePlan): NativeContinuityManifest {
        if (snapshot.info.lifecycle != RuntimeLifecycle.SUSPENDED) fail("runtime must be SUSPENDED before composite persistence")
        if (!snapshot.modelBinding.bound || snapshot.modelBinding.modelId.size != 32) fail("composite persistence requires exact bound model identity")
        if (snapshot.info.sequencePosition < 0) fail("runtime sequence position is invalid")
        val plannerBytes = NativePlannerCheckpoint.encode(plan)
        val previous = manifestStore.loadOrNull()?.let(ContinuityManifestCodec::decode)
        val epoch = if (previous == null) 1L else {
            if (previous.epoch == Long.MAX_VALUE) fail("continuity epoch overflow")
            previous.epoch + 1L
        }
        val slot = if (previous?.slot == "a") "b" else "a"
        val runtimeSha = sha256Hex(snapshot.checkpoint)
        val plannerSha = sha256Hex(plannerBytes)
        runtimeStore(slot).save(snapshot.checkpoint)
        plannerStore(slot).save(plannerBytes)
        val manifest = NativeContinuityManifest(
            epoch = epoch,
            slot = slot,
            runtimeSha256 = runtimeSha,
            plannerSha256 = plannerSha,
            modelId = snapshot.modelBinding.modelId.copyOf(),
            planId = plan.planId,
            sequencePosition = snapshot.info.sequencePosition,
        )
        manifestStore.save(ContinuityManifestCodec.encode(manifest))
        return manifest
    }

    @Synchronized
    fun loadOrNull(): NativeCompositeContinuity? {
        val manifestBytes = manifestStore.loadOrNull() ?: return null
        val manifest = ContinuityManifestCodec.decode(manifestBytes)
        val runtime = runtimeStore(manifest.slot).loadOrNull() ?: fail("committed runtime slot is missing")
        val plannerBytes = plannerStore(manifest.slot).loadOrNull() ?: fail("committed planner slot is missing")
        if (sha256Hex(runtime) != manifest.runtimeSha256) fail("runtime slot digest does not match continuity manifest")
        if (sha256Hex(plannerBytes) != manifest.plannerSha256) fail("planner slot digest does not match continuity manifest")
        val planner = NativePlannerCheckpoint.decode(plannerBytes)
        if (planner.plan.planId != manifest.planId) fail("planner identity does not match continuity manifest")
        return NativeCompositeContinuity(manifest, runtime, planner)
    }

    @Synchronized
    fun delete() {
        manifestStore.delete()
        runtimeStore("a").delete(); plannerStore("a").delete()
        runtimeStore("b").delete(); plannerStore("b").delete()
    }

    private fun runtimeStore(slot: String) = AtomicBinaryStore(
        root, "runtime-$slot.vn97run", 64, CNT_MAX_RUNTIME, ".vn97-runtime-$slot-"
    )
    private fun plannerStore(slot: String) = AtomicBinaryStore(
        root, "planner-$slot.vn97pln1", 48, CNT_MAX_PLANNER, ".vn97-planner-$slot-"
    )
}

private object ContinuityManifestCodec {
    fun encode(manifest: NativeContinuityManifest): ByteArray {
        validate(manifest)
        val payload = VnStrictJson.canonical(payload(manifest)).toByteArray(StandardCharsets.UTF_8)
        if (payload.size > CNT_MAX_PAYLOAD) fail("continuity manifest payload exceeds limit")
        val digest = MessageDigest.getInstance("SHA-256").digest(payload)
        return ByteBuffer.allocate(CNT_HEADER_BYTES + payload.size).order(ByteOrder.LITTLE_ENDIAN).apply {
            put(CNT_MAGIC); putInt(CNT_VERSION); putInt(payload.size); put(digest); put(payload)
        }.array()
    }

    fun decode(blob: ByteArray): NativeContinuityManifest {
        if (blob.size < CNT_HEADER_BYTES) fail("continuity manifest shorter than header")
        val header = ByteBuffer.wrap(blob, 0, CNT_HEADER_BYTES).order(ByteOrder.LITTLE_ENDIAN)
        val magic = ByteArray(8).also(header::get)
        if (!magic.contentEquals(CNT_MAGIC)) fail("bad VN97CNT1 magic")
        if (header.int != CNT_VERSION) fail("unsupported VN97CNT1 version")
        val size = header.int
        if (size < 0 || size > CNT_MAX_PAYLOAD || blob.size != CNT_HEADER_BYTES + size) fail("continuity manifest length is invalid")
        val digest = ByteArray(32).also(header::get)
        val payload = blob.copyOfRange(CNT_HEADER_BYTES, blob.size)
        if (!MessageDigest.getInstance("SHA-256").digest(payload).contentEquals(digest)) fail("continuity manifest SHA-256 mismatch")
        val text = strictUtf8(payload)
        val obj = try {
            VnStrictJson.parseObject(text, VnJsonLimits(CNT_MAX_PAYLOAD, 8, 128, 1024))
        } catch (exc: RuntimeException) {
            throw NativeContinuityException("continuity manifest JSON is invalid", exc)
        }
        val keys = setOf("epoch","slot","runtime_sha256","planner_sha256","model_id","plan_id","sequence_position")
        if (obj.values.keys != keys) fail("continuity manifest keys mismatch")
        val manifest = NativeContinuityManifest(
            epoch = positiveLong(obj, "epoch"),
            slot = string(obj, "slot"),
            runtimeSha256 = digestString(obj, "runtime_sha256"),
            plannerSha256 = digestString(obj, "planner_sha256"),
            modelId = parseHex(string(obj, "model_id"), 32, "model_id"),
            planId = digestString(obj, "plan_id"),
            sequencePosition = nonnegativeLong(obj, "sequence_position"),
        )
        validate(manifest)
        if (VnStrictJson.canonical(payload(manifest)) != text) {
            fail("continuity manifest JSON is not canonical")
        }
        return manifest
    }

    private fun payload(manifest: NativeContinuityManifest): VnJsonObject = VnStrictJson.objectOf(
        "epoch" to VnStrictJson.long(manifest.epoch),
        "slot" to VnStrictJson.string(manifest.slot),
        "runtime_sha256" to VnStrictJson.string(manifest.runtimeSha256),
        "planner_sha256" to VnStrictJson.string(manifest.plannerSha256),
        "model_id" to VnStrictJson.string(hex(manifest.modelId)),
        "plan_id" to VnStrictJson.string(manifest.planId),
        "sequence_position" to VnStrictJson.long(manifest.sequencePosition),
    )

    private fun validate(m: NativeContinuityManifest) {
        if (m.epoch <= 0) fail("continuity epoch must be positive")
        if (m.slot != "a" && m.slot != "b") fail("continuity slot must be a or b")
        if (m.modelId.size != 32) fail("continuity model ID must be 32 bytes")
        if (m.sequencePosition < 0) fail("continuity sequence position must be non-negative")
        requireHex64(m.runtimeSha256, "runtime_sha256"); requireHex64(m.plannerSha256, "planner_sha256"); requireHex64(m.planId, "plan_id")
    }

    private fun string(o: VnJsonObject, key: String) = (o.values[key] as? VnJsonString)?.value ?: fail("$key must be string")
    private fun positiveLong(o: VnJsonObject, key: String): Long = nonnegativeLong(o, key).also { if (it == 0L) fail("$key must be positive") }
    private fun nonnegativeLong(o: VnJsonObject, key: String): Long {
        val raw = (o.values[key] as? VnJsonNumber)?.canonical ?: fail("$key must be integer")
        if (raw.any { it == '.' || it == 'e' || it == 'E' }) fail("$key must be integer")
        return (raw.toLongOrNull() ?: fail("$key is outside Long range")).also { if (it < 0) fail("$key must be non-negative") }
    }
    private fun digestString(o: VnJsonObject, key: String) = string(o, key).also { requireHex64(it, key) }
    private fun requireHex64(value: String, label: String) { if (value.length != 64 || value.any { it !in "0123456789abcdef" }) fail("$label must be lowercase SHA-256 hex") }
    private fun parseHex(value: String, bytes: Int, label: String): ByteArray {
        if (value.length != bytes * 2 || value.any { it !in "0123456789abcdef" }) fail("$label has invalid hex")
        return ByteArray(bytes) { i -> value.substring(i * 2, i * 2 + 2).toInt(16).toByte() }
    }
    private fun hex(bytes: ByteArray) = bytes.joinToString("") { "%02x".format(it.toInt() and 0xff) }
    private fun strictUtf8(bytes: ByteArray) = try {
        StandardCharsets.UTF_8.newDecoder().onMalformedInput(CodingErrorAction.REPORT).onUnmappableCharacter(CodingErrorAction.REPORT).decode(ByteBuffer.wrap(bytes)).toString()
    } catch (exc: Exception) { throw NativeContinuityException("continuity manifest is not UTF-8", exc) }
}

private fun sha256Hex(bytes: ByteArray) = MessageDigest.getInstance("SHA-256").digest(bytes).joinToString("") { "%02x".format(it.toInt() and 0xff) }
private fun fail(message: String): Nothing = throw NativeContinuityException(message)
