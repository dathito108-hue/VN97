package ai.vn97.runtime

import java.nio.ByteBuffer
import java.nio.charset.CodingErrorAction
import java.nio.charset.StandardCharsets

private const val VN97_INVENTORY_MAX_BYTES = 4 * 1024 * 1024
private const val VN97_INVENTORY_MAX_STACK_DEPTH = 64
private const val VN97_INVENTORY_MAX_HISTORY = 4096

class NativeActivatedInventoryException(
    message: String,
    cause: Throwable? = null,
) : IllegalStateException(message, cause)

data class VN97ActiveModelEvidence(
    val activationId: String,
    val capabilityId: String,
    val backendId: String,
    val artifactSha256: String,
    val runtimeRevision: String,
    val generation: Long,
)

object VN97ActivatedModelInventoryEvidence {
    const val DEFAULT_CAPABILITY_ID = "model.language"
    const val DEFAULT_BACKEND_ID = "vn97.model_image"

    fun parse(
        bytes: ByteArray,
        capabilityId: String = DEFAULT_CAPABILITY_ID,
        backendId: String = DEFAULT_BACKEND_ID,
    ): VN97ActiveModelEvidence? {
        requireId(capabilityId, "capabilityId")
        requireId(backendId, "backendId")
        if (bytes.isEmpty() || bytes.size > VN97_INVENTORY_MAX_BYTES) {
            throw NativeActivatedInventoryException(
                "VN97INV1 inventory size is outside bounds"
            )
        }
        val text = strictInventoryUtf8(bytes)
        val root = try {
            VnStrictJson.parseObject(
                text,
                VnJsonLimits(
                    maxInputUtf8Bytes = VN97_INVENTORY_MAX_BYTES,
                    maxDepth = 64,
                    maxNodes = 512 * 1024,
                    maxStringUtf8Bytes = 4096,
                ),
            )
        } catch (exc: RuntimeException) {
            throw NativeActivatedInventoryException(
                "VN97INV1 is not strict JSON",
                exc,
            )
        }
        if (VnStrictJson.canonical(root) != text) {
            throw NativeActivatedInventoryException(
                "VN97INV1 must be canonical JSON"
            )
        }
        requireExactKeys(
            root,
            setOf("generation", "history", "pending", "schema", "stacks"),
            "inventory",
        )
        if (root.string("schema") != "VN97INV1") {
            throw NativeActivatedInventoryException("unsupported inventory schema")
        }
        val generation = root.nonnegativeLong("generation")
        if (root.values["pending"] !== VnJsonNull) {
            throw NativeActivatedInventoryException(
                "VN97INV1 has unresolved activation transaction"
            )
        }

        validateHistory(root.array("history"), generation)
        val stacks = root.array("stacks")
        var previousCapability: String? = null
        var target: VN97ActiveModelEvidence? = null
        val seen = HashSet<String>()
        for (raw in stacks.values) {
            val stack = raw as? VnJsonObject
                ?: fail("inventory stack must be object")
            requireExactKeys(stack, setOf("capability_id", "records"), "stack")
            val id = requireId(stack.string("capability_id"), "stack capability_id")
            if (!seen.add(id)) fail("duplicate inventory capability stack")
            if (previousCapability != null && id <= previousCapability) {
                fail("inventory stacks must be sorted by capability_id")
            }
            previousCapability = id
            val records = stack.array("records").values
            if (records.isEmpty() || records.size > VN97_INVENTORY_MAX_STACK_DEPTH) {
                fail("inventory stack depth is outside bounds")
            }
            var activeRecord: VN97ActiveModelEvidence? = null
            records.forEachIndexed { index, item ->
                val record = parseRecord(item, generation)
                if (record.capabilityId != id) {
                    fail("activation record capability does not match stack")
                }
                if (index == records.lastIndex) activeRecord = record
            }
            if (id == capabilityId) {
                val current = checkNotNull(activeRecord)
                if (current.backendId != backendId) {
                    throw NativeActivatedInventoryException(
                        "active model capability uses unexpected backend"
                    )
                }
                target = current
            }
        }
        return target
    }

    private fun parseRecord(
        raw: VnJsonValue,
        generation: Long,
    ): VN97ActiveModelEvidence {
        val obj = raw as? VnJsonObject ?: fail("activation record must be object")
        requireExactKeys(
            obj,
            setOf(
                "activation_id",
                "artifact_sha256",
                "backend_id",
                "backend_token",
                "capability_id",
                "capability_version",
                "package_sha256",
                "plan_sha256",
                "profile_id",
                "profile_sha256",
                "publisher_key_id",
                "runtime_api_version",
                "runtime_revision",
                "signature_sha256",
                "source_license",
                "source_origin",
                "source_sha256",
            ),
            "activation record",
        )
        val activationId = requireSha(obj.string("activation_id"), "activation_id")
        val artifact = requireSha(obj.string("artifact_sha256"), "artifact_sha256")
        val capability = requireId(obj.string("capability_id"), "capability_id")
        val backend = requireId(obj.string("backend_id"), "backend_id")
        requireToken(obj.string("backend_token"), "backend_token")
        requirePositiveInt(obj, "capability_version")
        requireSha(obj.string("package_sha256"), "package_sha256")
        requireSha(obj.string("plan_sha256"), "plan_sha256")
        requireId(obj.string("profile_id"), "profile_id")
        requireSha(obj.string("profile_sha256"), "profile_sha256")
        requireId(obj.string("publisher_key_id"), "publisher_key_id")
        requirePositiveInt(obj, "runtime_api_version")
        val revision = requireToken(obj.string("runtime_revision"), "runtime_revision")
        requireSha(obj.string("signature_sha256"), "signature_sha256")
        requireText(obj.string("source_license"), "source_license", 128)
        requireText(obj.string("source_origin"), "source_origin", 1024)
        requireSha(obj.string("source_sha256"), "source_sha256")
        return VN97ActiveModelEvidence(
            activationId = activationId,
            capabilityId = capability,
            backendId = backend,
            artifactSha256 = artifact,
            runtimeRevision = revision,
            generation = generation,
        )
    }

    private fun validateHistory(history: VnJsonArray, generation: Long) {
        if (history.values.size > VN97_INVENTORY_MAX_HISTORY) {
            fail("inventory history exceeds bound")
        }
        var previous = 0L
        history.values.forEach { raw ->
            val obj = raw as? VnJsonObject ?: fail("inventory event must be object")
            requireExactKeys(
                obj,
                setOf(
                    "action",
                    "activation_id",
                    "capability_id",
                    "generation",
                    "package_sha256",
                ),
                "inventory event",
            )
            val eventGeneration = obj.positiveLong("generation")
            if (eventGeneration <= previous || eventGeneration > generation) {
                fail("inventory history generation sequence is invalid")
            }
            previous = eventGeneration
            val action = obj.string("action")
            if (
                action != "activate" &&
                action != "rollback" &&
                action != "recover_activate" &&
                action != "recover_rollback"
            ) {
                fail("inventory event action is invalid")
            }
            requireSha(obj.string("activation_id"), "event activation_id")
            requireId(obj.string("capability_id"), "event capability_id")
            requireSha(obj.string("package_sha256"), "event package_sha256")
        }
    }

    private fun requireExactKeys(
        obj: VnJsonObject,
        keys: Set<String>,
        label: String,
    ) {
        if (obj.values.keys != keys) {
            fail("$label keys mismatch")
        }
    }

    private fun VnJsonObject.string(key: String): String =
        (values[key] as? VnJsonString)?.value ?: fail("$key must be string")

    private fun VnJsonObject.array(key: String): VnJsonArray =
        values[key] as? VnJsonArray ?: fail("$key must be array")

    private fun VnJsonObject.nonnegativeLong(key: String): Long {
        val raw = (values[key] as? VnJsonNumber)?.canonical
            ?: fail("$key must be integer")
        if (raw.any { it == '.' || it == 'e' || it == 'E' }) {
            fail("$key must be integer")
        }
        val value = raw.toLongOrNull() ?: fail("$key is outside Long range")
        if (value < 0L) fail("$key must be non-negative")
        return value
    }

    private fun VnJsonObject.positiveLong(key: String): Long =
        nonnegativeLong(key).also {
            if (it == 0L) fail("$key must be positive")
        }

    private fun requirePositiveInt(obj: VnJsonObject, key: String): Int {
        val value = obj.nonnegativeLong(key)
        if (value !in 1L..Int.MAX_VALUE.toLong()) {
            fail("$key must be positive Int")
        }
        return value.toInt()
    }

    private fun requireId(value: String, label: String): String {
        if (
            value.isEmpty() ||
            value.length > 128 ||
            value[0] !in 'a'..'z' ||
            value.any { ch ->
                ch !in 'a'..'z' &&
                    ch !in '0'..'9' &&
                    ch != '.' && ch != '_' && ch != '-'
            }
        ) {
            fail("$label is invalid")
        }
        return value
    }

    private fun requireSha(value: String, label: String): String {
        if (value.length != 64 || value.any { it !in "0123456789abcdef" }) {
            fail("$label is invalid")
        }
        return value
    }

    private fun requireToken(value: String, label: String): String {
        if (
            value.isEmpty() ||
            value.length > 255 ||
            value.any { ch ->
                ch !in 'A'..'Z' &&
                    ch !in 'a'..'z' &&
                    ch !in '0'..'9' &&
                    ch != '.' && ch != '_' && ch != ':' && ch != '-'
            }
        ) {
            fail("$label is invalid")
        }
        return value
    }

    private fun requireText(
        value: String,
        label: String,
        maxUtf8Bytes: Int,
    ): String {
        if (
            value.isEmpty() ||
            value.toByteArray(StandardCharsets.UTF_8).size > maxUtf8Bytes
        ) {
            fail("$label is invalid")
        }
        return value
    }

    private fun fail(message: String): Nothing =
        throw NativeActivatedInventoryException(message)
}

private fun strictInventoryUtf8(bytes: ByteArray): String = try {
    StandardCharsets.UTF_8.newDecoder()
        .onMalformedInput(CodingErrorAction.REPORT)
        .onUnmappableCharacter(CodingErrorAction.REPORT)
        .decode(ByteBuffer.wrap(bytes))
        .toString()
} catch (exc: Exception) {
    throw NativeActivatedInventoryException(
        "VN97INV1 is not strict UTF-8",
        exc,
    )
}
