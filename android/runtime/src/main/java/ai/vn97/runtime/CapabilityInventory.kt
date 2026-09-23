package ai.vn97.runtime

import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.charset.CodingErrorAction
import java.nio.charset.StandardCharsets
import java.nio.channels.FileChannel
import java.nio.file.AtomicMoveNotSupportedException
import java.nio.file.Files
import java.nio.file.LinkOption
import java.nio.file.StandardCopyOption
import java.nio.file.StandardOpenOption

const val VN97_INVENTORY_FILE = "inventory.vn97inv1.json"
private const val VN97_INVENTORY_LOCK = ".inventory.vn97inv1.lock"
private const val VN97_INVENTORY_MAX_BYTES = 4 * 1024 * 1024
private const val VN97_INVENTORY_MAX_HISTORY = 4096
private const val VN97_INVENTORY_MAX_STACK_DEPTH = 64

enum class VN97InventoryAction(val wire: String) {
    ACTIVATE("activate"),
    ROLLBACK("rollback"),
    RECOVER_ACTIVATE("recover_activate"),
    RECOVER_ROLLBACK("recover_rollback"),
}

data class VN97CapabilityInventoryItem(
    val activationId: String,
    val capabilityId: String,
    val capabilityVersion: Long,
    val packageSha256: String,
    val publisherKeyId: String,
    val signatureSha256: String,
    val profileId: String,
    val profileSha256: String,
    val planSha256: String,
    val runtimeApiVersion: Long,
    val backendId: String,
    val artifactSha256: String,
    val runtimeRevision: String,
    val sourceOrigin: String,
    val sourceSha256: String,
    val sourceLicense: String,
) {
    init { validateInventoryItem(this) }
}

data class VN97InventoryEvent(
    val generation: Long,
    val action: VN97InventoryAction,
    val capabilityId: String,
    val activationId: String,
    val packageSha256: String,
) {
    init {
        if (generation <= 0L) inventoryFail("event generation must be positive")
        inventoryRequireId(capabilityId, "event capability_id")
        inventoryRequireSha(activationId, "event activation_id")
        inventoryRequireSha(packageSha256, "event package_sha256")
    }
}

data class VN97InventorySnapshot(
    val generation: Long,
    val active: List<VN97CapabilityInventoryItem>,
    val history: List<VN97InventoryEvent>,
    val pendingOperation: String?,
) {
    fun current(capabilityId: String): VN97CapabilityInventoryItem? =
        active.firstOrNull { it.capabilityId == capabilityId }
}

open class VN97CapabilityActivationException(
    message: String,
    cause: Throwable? = null,
) : IllegalStateException(message, cause)

class VN97InventoryCorruptionException(
    message: String,
    cause: Throwable? = null,
) : VN97CapabilityActivationException(message, cause)

internal data class VN97StoredActivation(
    val item: VN97CapabilityInventoryItem,
    val backendToken: String,
)

internal data class VN97ActivationIdentity(
    val capabilityId: String,
    val capabilityVersion: Long,
    val packageSha256: String,
    val planSha256: String,
    val profileId: String,
    val profileSha256: String,
    val publisherKeyId: String,
    val runtimeApiVersion: Long,
    val signatureSha256: String,
    val sourceLicense: String,
    val sourceOrigin: String,
    val sourceSha256: String,
) {
    init { validateActivationIdentity(this) }
}

internal data class VN97PendingTransaction(
    val txId: String,
    val operation: String,
    val phase: String,
    val capabilityId: String,
    val backendId: String,
    val generationBase: Long,
    val identity: VN97ActivationIdentity?,
    val backendToken: String?,
    val artifactSha256: String?,
    val activationId: String?,
) {
    init { validatePending(this) }
}

internal data class VN97InventoryState(
    var generation: Long,
    val stacks: MutableMap<String, MutableList<VN97StoredActivation>>,
    val history: MutableList<VN97InventoryEvent>,
    var pending: VN97PendingTransaction?,
) {
    fun snapshot(): VN97InventorySnapshot = VN97InventorySnapshot(
        generation = generation,
        active = stacks.toSortedMap().values.mapNotNull { it.lastOrNull()?.item },
        history = history.toList(),
        pendingOperation = pending?.operation,
    )
}

internal class VN97InventoryTransaction(
    private val store: VN97CapabilityInventoryStore,
    val state: VN97InventoryState,
) {
    fun persist() = store.writeLocked(state)
}

class VN97CapabilityInventoryStore(
    private val root: File,
    private val maxInventoryBytes: Int = VN97_INVENTORY_MAX_BYTES,
) {
    private val rootPath = root.toPath().toAbsolutePath().normalize()
    private val inventoryPath = rootPath.resolve(VN97_INVENTORY_FILE)
    private val lockPath = rootPath.resolve(VN97_INVENTORY_LOCK)

    init {
        if (maxInventoryBytes < 1024) {
            throw IllegalArgumentException("maxInventoryBytes is too small")
        }
        if (!Files.exists(rootPath, LinkOption.NOFOLLOW_LINKS) ||
            !Files.isDirectory(rootPath, LinkOption.NOFOLLOW_LINKS) ||
            Files.isSymbolicLink(rootPath)
        ) {
            throw IllegalArgumentException(
                "inventory root must be an existing absolute non-symlink directory"
            )
        }
        if (Files.exists(inventoryPath, LinkOption.NOFOLLOW_LINKS) &&
            Files.isSymbolicLink(inventoryPath)
        ) {
            inventoryFail("inventory file must not be a symlink")
        }
        if (Files.exists(lockPath, LinkOption.NOFOLLOW_LINKS) &&
            Files.isSymbolicLink(lockPath)
        ) {
            inventoryFail("inventory lock must not be a symlink")
        }
    }

    fun load(): VN97InventorySnapshot = locked { it.state.snapshot() }

    internal fun <T> locked(block: (VN97InventoryTransaction) -> T): T {
        if (Files.exists(lockPath, LinkOption.NOFOLLOW_LINKS) &&
            Files.isSymbolicLink(lockPath)
        ) {
            inventoryFail("inventory lock became a symlink")
        }
        FileChannel.open(
            lockPath,
            StandardOpenOption.CREATE,
            StandardOpenOption.WRITE,
            LinkOption.NOFOLLOW_LINKS,
        ).use { channel ->
            channel.lock().use {
                val tx = VN97InventoryTransaction(this, readLocked())
                return block(tx)
            }
        }
    }

    internal fun writeLocked(state: VN97InventoryState) {
        validateState(state)
        val data = VnStrictJson.canonical(stateToJson(state))
            .toByteArray(StandardCharsets.UTF_8)
        if (data.size > maxInventoryBytes) {
            throw VN97CapabilityActivationException("inventory exceeds byte limit")
        }
        if (Files.exists(inventoryPath, LinkOption.NOFOLLOW_LINKS) &&
            Files.isSymbolicLink(inventoryPath)
        ) {
            inventoryFail("inventory file became a symlink")
        }
        val temp = Files.createTempFile(rootPath, ".vn97inv-", ".tmp")
        try {
            FileOutputStream(temp.toFile()).use { output ->
                output.write(data)
                output.flush()
                output.fd.sync()
            }
            try {
                Files.move(
                    temp,
                    inventoryPath,
                    StandardCopyOption.ATOMIC_MOVE,
                    StandardCopyOption.REPLACE_EXISTING,
                )
            } catch (exc: AtomicMoveNotSupportedException) {
                throw VN97CapabilityActivationException(
                    "inventory filesystem does not support atomic replace",
                    exc,
                )
            }
            FileChannel.open(rootPath, StandardOpenOption.READ).use { it.force(true) }
        } finally {
            Files.deleteIfExists(temp)
        }
    }

    private fun readLocked(): VN97InventoryState {
        if (!Files.exists(inventoryPath, LinkOption.NOFOLLOW_LINKS)) {
            return VN97InventoryState(0L, linkedMapOf(), mutableListOf(), null)
        }
        if (!Files.isRegularFile(inventoryPath, LinkOption.NOFOLLOW_LINKS) ||
            Files.isSymbolicLink(inventoryPath)
        ) {
            inventoryFail("inventory file type is invalid")
        }
        val size = Files.size(inventoryPath)
        if (size <= 0L || size > maxInventoryBytes.toLong()) {
            inventoryFail("inventory file size is invalid")
        }
        val bytes = Files.readAllBytes(inventoryPath)
        if (bytes.size.toLong() != size) inventoryFail("inventory read was truncated")
        val text = strictInventoryUtf8(bytes)
        val root = try {
            VnStrictJson.parseObject(
                text,
                VnJsonLimits(
                    maxInputUtf8Bytes = maxInventoryBytes,
                    maxDepth = 64,
                    maxNodes = 512 * 1024,
                    maxStringUtf8Bytes = 4096,
                ),
            )
        } catch (exc: RuntimeException) {
            throw VN97InventoryCorruptionException("inventory is not strict JSON", exc)
        }
        if (VnStrictJson.canonical(root) != text) {
            inventoryFail("inventory must be canonical JSON")
        }
        if (root.values.keys != setOf("schema", "generation", "stacks", "history", "pending") ||
            root.string("schema") != "VN97INV1"
        ) {
            inventoryFail("inventory schema/keys mismatch")
        }
        val generation = root.nonnegativeLong("generation")
        val stacksArray = root.array("stacks")
        val historyArray = root.array("history")
        if (historyArray.values.size > VN97_INVENTORY_MAX_HISTORY) {
            inventoryFail("inventory history exceeds bound")
        }

        val stacks = linkedMapOf<String, MutableList<VN97StoredActivation>>()
        var previousCapability: String? = null
        for (raw in stacksArray.values) {
            val stackObj = raw as? VnJsonObject ?: inventoryFail("inventory stack must be object")
            if (stackObj.values.keys != setOf("capability_id", "records")) {
                inventoryFail("inventory stack keys mismatch")
            }
            val capabilityId = inventoryRequireId(stackObj.string("capability_id"), "stack capability_id")
            if (previousCapability != null && capabilityId <= previousCapability) {
                inventoryFail("inventory stacks must be sorted")
            }
            previousCapability = capabilityId
            val records = stackObj.array("records").values
            if (records.size !in 1..VN97_INVENTORY_MAX_STACK_DEPTH) {
                inventoryFail("inventory stack depth is invalid")
            }
            if (stacks.containsKey(capabilityId)) inventoryFail("duplicate inventory stack")
            val decoded = records.map { storedFromJson(it) }.toMutableList()
            if (decoded.any { it.item.capabilityId != capabilityId }) {
                inventoryFail("stack record capability mismatch")
            }
            stacks[capabilityId] = decoded
        }

        val history = historyArray.values.map { eventFromJson(it) }.toMutableList()
        var lastGeneration = 0L
        for (event in history) {
            if (event.generation <= lastGeneration || event.generation > generation) {
                inventoryFail("inventory history generation sequence invalid")
            }
            lastGeneration = event.generation
        }

        val pendingRaw = root.values["pending"]
        val pending = when (pendingRaw) {
            VnJsonNull -> null
            is VnJsonObject -> pendingFromJson(pendingRaw)
            else -> inventoryFail("pending transaction must be object or null")
        }
        if (pending != null && pending.generationBase != generation) {
            inventoryFail("pending generation does not match inventory generation")
        }
        return VN97InventoryState(generation, stacks, history, pending)
    }
}

private fun stateToJson(state: VN97InventoryState): VnJsonObject {
    val stacks = state.stacks.toSortedMap().entries
        .filter { it.value.isNotEmpty() }
        .map { (capabilityId, records) ->
            VnStrictJson.objectOf(
                "capability_id" to VnStrictJson.string(capabilityId),
                "records" to VnStrictJson.array(records.map(::storedToJson)),
            )
        }
    return VnStrictJson.objectOf(
        "generation" to VnJsonNumber(state.generation.toString()),
        "history" to VnStrictJson.array(
            state.history.takeLast(VN97_INVENTORY_MAX_HISTORY).map(::eventToJson)
        ),
        "pending" to (state.pending?.let(::pendingToJson) ?: VnJsonNull),
        "schema" to VnStrictJson.string("VN97INV1"),
        "stacks" to VnStrictJson.array(stacks),
    )
}

private fun storedToJson(stored: VN97StoredActivation): VnJsonObject {
    val i = stored.item
    return VnStrictJson.objectOf(
        "activation_id" to VnStrictJson.string(i.activationId),
        "artifact_sha256" to VnStrictJson.string(i.artifactSha256),
        "backend_id" to VnStrictJson.string(i.backendId),
        "backend_token" to VnStrictJson.string(stored.backendToken),
        "capability_id" to VnStrictJson.string(i.capabilityId),
        "capability_version" to VnJsonNumber(i.capabilityVersion.toString()),
        "package_sha256" to VnStrictJson.string(i.packageSha256),
        "plan_sha256" to VnStrictJson.string(i.planSha256),
        "profile_id" to VnStrictJson.string(i.profileId),
        "profile_sha256" to VnStrictJson.string(i.profileSha256),
        "publisher_key_id" to VnStrictJson.string(i.publisherKeyId),
        "runtime_api_version" to VnJsonNumber(i.runtimeApiVersion.toString()),
        "runtime_revision" to VnStrictJson.string(i.runtimeRevision),
        "signature_sha256" to VnStrictJson.string(i.signatureSha256),
        "source_license" to VnStrictJson.string(i.sourceLicense),
        "source_origin" to VnStrictJson.string(i.sourceOrigin),
        "source_sha256" to VnStrictJson.string(i.sourceSha256),
    )
}

private fun storedFromJson(raw: VnJsonValue): VN97StoredActivation {
    val obj = raw as? VnJsonObject ?: inventoryFail("activation record must be object")
    val keys = setOf(
        "activation_id", "artifact_sha256", "backend_id", "backend_token",
        "capability_id", "capability_version", "package_sha256", "plan_sha256",
        "profile_id", "profile_sha256", "publisher_key_id", "runtime_api_version",
        "runtime_revision", "signature_sha256", "source_license", "source_origin",
        "source_sha256",
    )
    if (obj.values.keys != keys) inventoryFail("activation record keys mismatch")
    val item = VN97CapabilityInventoryItem(
        activationId = obj.string("activation_id"),
        capabilityId = obj.string("capability_id"),
        capabilityVersion = obj.unsigned32("capability_version"),
        packageSha256 = obj.string("package_sha256"),
        publisherKeyId = obj.string("publisher_key_id"),
        signatureSha256 = obj.string("signature_sha256"),
        profileId = obj.string("profile_id"),
        profileSha256 = obj.string("profile_sha256"),
        planSha256 = obj.string("plan_sha256"),
        runtimeApiVersion = obj.positiveLong("runtime_api_version"),
        backendId = obj.string("backend_id"),
        artifactSha256 = obj.string("artifact_sha256"),
        runtimeRevision = obj.string("runtime_revision"),
        sourceOrigin = obj.string("source_origin"),
        sourceSha256 = obj.string("source_sha256"),
        sourceLicense = obj.string("source_license"),
    )
    return VN97StoredActivation(item, inventoryRequireToken(obj.string("backend_token"), "backend_token"))
}

private fun eventToJson(event: VN97InventoryEvent): VnJsonObject = VnStrictJson.objectOf(
    "action" to VnStrictJson.string(event.action.wire),
    "activation_id" to VnStrictJson.string(event.activationId),
    "capability_id" to VnStrictJson.string(event.capabilityId),
    "generation" to VnJsonNumber(event.generation.toString()),
    "package_sha256" to VnStrictJson.string(event.packageSha256),
)

private fun eventFromJson(raw: VnJsonValue): VN97InventoryEvent {
    val obj = raw as? VnJsonObject ?: inventoryFail("inventory event must be object")
    if (obj.values.keys != setOf("action", "activation_id", "capability_id", "generation", "package_sha256")) {
        inventoryFail("inventory event keys mismatch")
    }
    val actionWire = obj.string("action")
    val action = VN97InventoryAction.entries.firstOrNull { it.wire == actionWire }
        ?: inventoryFail("inventory event action is invalid")
    return VN97InventoryEvent(
        generation = obj.positiveLong("generation"),
        action = action,
        capabilityId = obj.string("capability_id"),
        activationId = obj.string("activation_id"),
        packageSha256 = obj.string("package_sha256"),
    )
}

private fun pendingToJson(p: VN97PendingTransaction): VnJsonObject = VnStrictJson.objectOf(
    "activation_id" to (p.activationId?.let(VnStrictJson::string) ?: VnJsonNull),
    "artifact_sha256" to (p.artifactSha256?.let(VnStrictJson::string) ?: VnJsonNull),
    "backend_id" to VnStrictJson.string(p.backendId),
    "backend_token" to (p.backendToken?.let(VnStrictJson::string) ?: VnJsonNull),
    "capability_id" to VnStrictJson.string(p.capabilityId),
    "generation_base" to VnJsonNumber(p.generationBase.toString()),
    "identity" to (p.identity?.let(::identityToJson) ?: VnStrictJson.objectOf()),
    "operation" to VnStrictJson.string(p.operation),
    "phase" to VnStrictJson.string(p.phase),
    "tx_id" to VnStrictJson.string(p.txId),
)

private fun pendingFromJson(obj: VnJsonObject): VN97PendingTransaction {
    val keys = setOf(
        "tx_id", "operation", "phase", "capability_id", "backend_id",
        "generation_base", "identity", "backend_token", "artifact_sha256", "activation_id",
    )
    if (obj.values.keys != keys) inventoryFail("pending keys mismatch")
    val operation = obj.string("operation")
    val identityObj = obj.obj("identity")
    val identity = if (operation == "activate") identityFromJson(identityObj) else {
        if (identityObj.values.isNotEmpty()) inventoryFail("rollback pending identity must be empty")
        null
    }
    return VN97PendingTransaction(
        txId = obj.string("tx_id"),
        operation = operation,
        phase = obj.string("phase"),
        capabilityId = obj.string("capability_id"),
        backendId = obj.string("backend_id"),
        generationBase = obj.nonnegativeLong("generation_base"),
        identity = identity,
        backendToken = obj.nullableString("backend_token"),
        artifactSha256 = obj.nullableString("artifact_sha256"),
        activationId = obj.nullableString("activation_id"),
    )
}

internal fun identityToJson(i: VN97ActivationIdentity): VnJsonObject = VnStrictJson.objectOf(
    "capability_id" to VnStrictJson.string(i.capabilityId),
    "capability_version" to VnJsonNumber(i.capabilityVersion.toString()),
    "package_sha256" to VnStrictJson.string(i.packageSha256),
    "plan_sha256" to VnStrictJson.string(i.planSha256),
    "profile_id" to VnStrictJson.string(i.profileId),
    "profile_sha256" to VnStrictJson.string(i.profileSha256),
    "publisher_key_id" to VnStrictJson.string(i.publisherKeyId),
    "runtime_api_version" to VnJsonNumber(i.runtimeApiVersion.toString()),
    "signature_sha256" to VnStrictJson.string(i.signatureSha256),
    "source_license" to VnStrictJson.string(i.sourceLicense),
    "source_origin" to VnStrictJson.string(i.sourceOrigin),
    "source_sha256" to VnStrictJson.string(i.sourceSha256),
)

private fun identityFromJson(obj: VnJsonObject): VN97ActivationIdentity {
    val keys = setOf(
        "capability_id", "capability_version", "package_sha256", "plan_sha256",
        "profile_id", "profile_sha256", "publisher_key_id", "runtime_api_version",
        "signature_sha256", "source_license", "source_origin", "source_sha256",
    )
    if (obj.values.keys != keys) inventoryFail("pending activation identity keys mismatch")
    return VN97ActivationIdentity(
        capabilityId = obj.string("capability_id"),
        capabilityVersion = obj.unsigned32("capability_version"),
        packageSha256 = obj.string("package_sha256"),
        planSha256 = obj.string("plan_sha256"),
        profileId = obj.string("profile_id"),
        profileSha256 = obj.string("profile_sha256"),
        publisherKeyId = obj.string("publisher_key_id"),
        runtimeApiVersion = obj.positiveLong("runtime_api_version"),
        signatureSha256 = obj.string("signature_sha256"),
        sourceLicense = obj.string("source_license"),
        sourceOrigin = obj.string("source_origin"),
        sourceSha256 = obj.string("source_sha256"),
    )
}

private fun validateState(state: VN97InventoryState) {
    if (state.generation < 0L) inventoryFail("inventory generation is invalid")
    if (state.history.size > VN97_INVENTORY_MAX_HISTORY) inventoryFail("inventory history exceeds bound")
    state.stacks.forEach { (id, stack) ->
        inventoryRequireId(id, "stack capability_id")
        if (stack.size !in 1..VN97_INVENTORY_MAX_STACK_DEPTH) inventoryFail("inventory stack depth is invalid")
        if (stack.any { it.item.capabilityId != id }) inventoryFail("stack record capability mismatch")
    }
    var previous = 0L
    state.history.forEach {
        if (it.generation <= previous || it.generation > state.generation) inventoryFail("inventory history sequence invalid")
        previous = it.generation
    }
    state.pending?.let {
        if (it.generationBase != state.generation) inventoryFail("pending generation mismatch")
    }
}

private fun validateInventoryItem(item: VN97CapabilityInventoryItem) {
    inventoryRequireSha(item.activationId, "activation_id")
    inventoryRequireId(item.capabilityId, "capability_id")
    if (item.capabilityVersion !in 1L..0xffff_ffffL) inventoryFail("capability_version is invalid")
    listOf(
        "package_sha256" to item.packageSha256,
        "signature_sha256" to item.signatureSha256,
        "profile_sha256" to item.profileSha256,
        "plan_sha256" to item.planSha256,
        "artifact_sha256" to item.artifactSha256,
        "source_sha256" to item.sourceSha256,
    ).forEach { inventoryRequireSha(it.second, it.first) }
    inventoryRequireId(item.publisherKeyId, "publisher_key_id")
    inventoryRequireId(item.profileId, "profile_id")
    inventoryRequireId(item.backendId, "backend_id")
    if (item.runtimeApiVersion <= 0L) inventoryFail("runtime_api_version is invalid")
    inventoryRequireRevision(item.runtimeRevision, "runtime_revision")
    inventoryRequireText(item.sourceOrigin, "source_origin", 1024)
    inventoryRequireText(item.sourceLicense, "source_license", 128)
}

private fun validateActivationIdentity(i: VN97ActivationIdentity) {
    inventoryRequireId(i.capabilityId, "pending identity capability_id")
    if (i.capabilityVersion !in 1L..0xffff_ffffL) inventoryFail("pending identity capability_version invalid")
    inventoryRequireSha(i.packageSha256, "pending identity package_sha256")
    inventoryRequireSha(i.planSha256, "pending identity plan_sha256")
    inventoryRequireId(i.profileId, "pending identity profile_id")
    inventoryRequireSha(i.profileSha256, "pending identity profile_sha256")
    inventoryRequireId(i.publisherKeyId, "pending identity publisher_key_id")
    if (i.runtimeApiVersion <= 0L) inventoryFail("pending identity runtime_api_version invalid")
    inventoryRequireSha(i.signatureSha256, "pending identity signature_sha256")
    inventoryRequireText(i.sourceLicense, "pending identity source_license", 128)
    inventoryRequireText(i.sourceOrigin, "pending identity source_origin", 1024)
    inventoryRequireSha(i.sourceSha256, "pending identity source_sha256")
}

private fun validatePending(p: VN97PendingTransaction) {
    inventoryRequireSha(p.txId, "pending tx_id")
    inventoryRequireId(p.capabilityId, "pending capability_id")
    inventoryRequireId(p.backendId, "pending backend_id")
    if (p.generationBase < 0L) inventoryFail("pending generation invalid")
    if (p.operation !in setOf("activate", "rollback") || p.phase !in setOf("reserved", "prepared")) {
        inventoryFail("pending operation/phase invalid")
    }
    if (p.operation == "activate") {
        val id = p.identity ?: inventoryFail("activation pending identity is missing")
        if (id.capabilityId != p.capabilityId) inventoryFail("pending identity capability mismatch")
        if (p.activationId != null) inventoryFail("activation pending must not predeclare activation_id")
    } else {
        if (p.identity != null || p.phase != "prepared") inventoryFail("rollback pending identity/phase invalid")
        inventoryRequireSha(p.activationId ?: inventoryFail("rollback pending activation_id missing"), "pending activation_id")
    }
    if (p.phase == "reserved") {
        if (p.backendToken != null || p.artifactSha256 != null || p.activationId != null) {
            inventoryFail("reserved transaction has prepared fields")
        }
    } else {
        inventoryRequireToken(p.backendToken ?: inventoryFail("pending token missing"), "pending token")
        inventoryRequireSha(p.artifactSha256 ?: inventoryFail("pending artifact missing"), "pending artifact_sha256")
    }
}

internal fun inventoryRequireId(value: String, label: String): String {
    if (value.isEmpty() || value.length > 128 || value[0] !in 'a'..'z' ||
        value.any { it !in 'a'..'z' && it !in '0'..'9' && it != '.' && it != '_' && it != '-' }
    ) inventoryFail("$label is invalid")
    return value
}

internal fun inventoryRequireSha(value: String, label: String): String {
    if (value.length != 64 || value.any { it !in "0123456789abcdef" }) inventoryFail("$label is invalid")
    return value
}

internal fun inventoryRequireToken(value: String, label: String): String {
    if (value.isEmpty() || value.length > 255 ||
        value.any { it !in 'A'..'Z' && it !in 'a'..'z' && it !in '0'..'9' && it != '.' && it != '_' && it != ':' && it != '-' }
    ) inventoryFail("$label is invalid")
    return value
}

internal fun inventoryRequireRevision(value: String, label: String): String =
    inventoryRequireToken(value, label)

private fun inventoryRequireText(value: String, label: String, maxBytes: Int): String {
    if (value.isEmpty() || value.toByteArray(StandardCharsets.UTF_8).size > maxBytes) inventoryFail("$label is invalid")
    return value
}

private fun strictInventoryUtf8(bytes: ByteArray): String = try {
    StandardCharsets.UTF_8.newDecoder()
        .onMalformedInput(CodingErrorAction.REPORT)
        .onUnmappableCharacter(CodingErrorAction.REPORT)
        .decode(ByteBuffer.wrap(bytes))
        .toString()
} catch (exc: Exception) {
    throw VN97InventoryCorruptionException("inventory is not strict UTF-8", exc)
}

private fun VnJsonObject.string(key: String): String =
    (values[key] as? VnJsonString)?.value ?: inventoryFail("$key must be string")
private fun VnJsonObject.obj(key: String): VnJsonObject =
    values[key] as? VnJsonObject ?: inventoryFail("$key must be object")
private fun VnJsonObject.array(key: String): VnJsonArray =
    values[key] as? VnJsonArray ?: inventoryFail("$key must be array")
private fun VnJsonObject.nullableString(key: String): String? = when (val raw = values[key]) {
    VnJsonNull -> null
    is VnJsonString -> raw.value
    else -> inventoryFail("$key must be string or null")
}
private fun VnJsonObject.nonnegativeLong(key: String): Long {
    val raw = (values[key] as? VnJsonNumber)?.canonical ?: inventoryFail("$key must be integer")
    if (raw.any { it == '.' || it == 'e' || it == 'E' }) inventoryFail("$key must be integer")
    return raw.toLongOrNull()?.also { if (it < 0L) inventoryFail("$key must be non-negative") }
        ?: inventoryFail("$key is outside Long range")
}
private fun VnJsonObject.positiveLong(key: String): Long = nonnegativeLong(key).also {
    if (it == 0L) inventoryFail("$key must be positive")
}
private fun VnJsonObject.unsigned32(key: String): Long = positiveLong(key).also {
    if (it > 0xffff_ffffL) inventoryFail("$key exceeds unsigned 32-bit range")
}

private fun inventoryFail(message: String): Nothing =
    throw VN97InventoryCorruptionException(message)
