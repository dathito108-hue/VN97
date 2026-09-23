package ai.vn97.platform

import ai.vn97.runtime.NativeExternalCapabilityView
import ai.vn97.runtime.NativeExternalIntent
import ai.vn97.runtime.NativeExternalIntentRequest
import ai.vn97.runtime.NativePlanController
import ai.vn97.runtime.NativePlanStatus
import ai.vn97.runtime.NativeStepKind
import ai.vn97.runtime.NativeStepStatus
import ai.vn97.runtime.NativeTypedCognitionAdapter
import java.nio.charset.StandardCharsets
import java.security.MessageDigest

class M6ExternalIntentException(message: String, cause: Throwable? = null) :
    IllegalStateException(message, cause)

class M6ExternalIntentContractException(message: String, cause: Throwable? = null) :
    IllegalArgumentException(message, cause)

data class M6ExternalIntentLimits(
    val maxCapabilities: Int = 32,
    val maxScopeEntries: Int = 16,
    val maxScopeValueUtf8Bytes: Int = 4096,
    val maxPayloadUtf8Bytes: Int = 64 * 1024,
    val maxGoalUtf8Bytes: Int = 32 * 1024,
    val maxObjectiveUtf8Bytes: Int = 4096,
) {
    init {
        require(
            listOf(
                maxCapabilities,
                maxScopeEntries,
                maxScopeValueUtf8Bytes,
                maxPayloadUtf8Bytes,
                maxGoalUtf8Bytes,
                maxObjectiveUtf8Bytes,
            ).all { it > 0 }
        ) { "all external intent limits must be positive" }
    }
}

data class M6CapabilityDescriptor(
    val capabilityId: String,
    val requiredScopeKeys: Set<String>,
    val optionalScopeKeys: Set<String> = emptySet(),
    val approvalRequired: Boolean = true,
    val maxPayloadUtf8Bytes: Int = 16 * 1024,
    val maxLeaseNs: Long = 300_000_000_000L,
    val maxLeaseUses: Int = 1,
) {
    init {
        externalHandoffValidateIdentifier(capabilityId, "capabilityId")
        require(requiredScopeKeys.isNotEmpty()) {
            "capability descriptor requires at least one scope key"
        }
        require(requiredScopeKeys.intersect(optionalScopeKeys).isEmpty()) {
            "required and optional scope keys must be disjoint"
        }
        (requiredScopeKeys + optionalScopeKeys).forEach {
            externalHandoffValidateIdentifier(it, "scope key")
        }
        require(maxPayloadUtf8Bytes > 0) { "maxPayloadUtf8Bytes must be positive" }
        require(maxLeaseNs > 0L) { "maxLeaseNs must be positive" }
        require(maxLeaseUses > 0) { "maxLeaseUses must be positive" }
    }

    internal fun view(limits: M6ExternalIntentLimits): NativeExternalCapabilityView =
        NativeExternalCapabilityView(
            capabilityId = capabilityId,
            requiredScopeKeys = requiredScopeKeys.sorted(),
            optionalScopeKeys = optionalScopeKeys.sorted(),
            approvalRequired = approvalRequired,
            maxPayloadUtf8Bytes = minOf(maxPayloadUtf8Bytes, limits.maxPayloadUtf8Bytes),
        )
}

class M6CapabilityScope private constructor(
    val entries: List<Pair<String, String>>,
) {
    val keys: Set<String> = entries.mapTo(linkedSetOf()) { it.first }

    val digest: String by lazy(LazyThreadSafetyMode.PUBLICATION) {
        sha256Hex(
            buildString {
                append("{\"scope\":[")
                entries.forEachIndexed { index, (key, value) ->
                    if (index != 0) append(',')
                    append('[')
                    appendJsonString(key)
                    append(',')
                    appendJsonString(value)
                    append(']')
                }
                append("]}")
            }.toByteArray(StandardCharsets.UTF_8)
        )
    }

    fun asMap(): Map<String, String> = linkedMapOf<String, String>().also { out ->
        entries.forEach { (key, value) -> out[key] = value }
    }

    companion object {
        fun fromMap(values: Map<String, String>): M6CapabilityScope {
            if (values.isEmpty()) {
                throw M6ExternalIntentContractException("capability scope must not be empty")
            }
            val normalized = values.entries.map { entry ->
                val key = entry.key
                val value = entry.value
                externalHandoffValidateIdentifier(key, "scope key")
                if (value.isEmpty() || utf8Size(value) > 4096) {
                    throw M6ExternalIntentContractException(
                        "scope values must be non-empty and bounded"
                    )
                }
                key to value
            }.sortedBy { it.first }
            if (normalized.map { it.first }.distinct().size != normalized.size) {
                throw M6ExternalIntentContractException("capability scope keys must be unique")
            }
            return M6CapabilityScope(normalized)
        }
    }
}

class M6ExternalActionRequest internal constructor(
    val planId: String,
    val stepId: Int,
    val objective: String,
    val capabilityId: String,
    val scope: M6CapabilityScope,
    val payloadJson: String,
) {
    init {
        require(planId.isNotEmpty()) { "planId must not be empty" }
        require(stepId > 0) { "stepId must be positive" }
        require(objective.isNotEmpty()) { "objective must not be empty" }
        externalHandoffValidateIdentifier(capabilityId, "capabilityId")
        require(
            payloadJson.startsWith('{') &&
                payloadJson.endsWith('}') &&
                payloadJson == payloadJson.trim()
        ) { "payloadJson must be one canonical JSON object" }
    }

    val requestDigest: String by lazy(LazyThreadSafetyMode.PUBLICATION) {
        sha256Hex(canonicalRequestJson().toByteArray(StandardCharsets.UTF_8))
    }

    internal fun canonicalRequestJson(): String = buildString {
        append("{\"capability_id\":")
        appendJsonString(capabilityId)
        append(",\"objective\":")
        appendJsonString(objective)
        append(",\"payload\":")
        append(payloadJson)
        append(",\"plan_id\":")
        appendJsonString(planId)
        append(",\"scope\":[")
        scope.entries.forEachIndexed { index, (key, value) ->
            if (index != 0) append(',')
            append('[')
            appendJsonString(key)
            append(',')
            appendJsonString(value)
            append(']')
        }
        append("],\"step_id\":")
        append(stepId)
        append('}')
    }

    fun approvalPresentationJson(): String = buildString {
        append("{\"capability_id\":")
        appendJsonString(capabilityId)
        append(",\"objective\":")
        appendJsonString(objective)
        append(",\"payload\":")
        append(payloadJson)
        append(",\"plan_id\":")
        appendJsonString(planId)
        append(",\"request_digest\":")
        appendJsonString(requestDigest)
        append(",\"scope\":{")
        scope.entries.forEachIndexed { index, (key, value) ->
            if (index != 0) append(',')
            appendJsonString(key)
            append(':')
            appendJsonString(value)
        }
        append("},\"step_id\":")
        append(stepId)
        append('}')
    }
}

class M6ExternalIntentBinder(
    descriptors: List<M6CapabilityDescriptor>,
    private val limits: M6ExternalIntentLimits = M6ExternalIntentLimits(),
) {
    private val descriptors: Map<String, M6CapabilityDescriptor>
    private val views: List<NativeExternalCapabilityView>

    init {
        require(descriptors.isNotEmpty()) { "at least one external capability must be allowed" }
        require(descriptors.size <= limits.maxCapabilities) {
            "allowed capability catalog exceeds limit"
        }
        require(descriptors.map { it.capabilityId }.distinct().size == descriptors.size) {
            "allowed capability IDs must be unique"
        }
        this.descriptors = descriptors.associateBy { it.capabilityId }
        this.views = descriptors.map { it.view(limits) }
    }

    val capabilities: List<NativeExternalCapabilityView>
        get() = views.toList()

    fun requestForBackend(controller: NativePlanController): NativeExternalIntentRequest {
        val plan = controller.plan
        if (plan.status != NativePlanStatus.WAITING_EXTERNAL) {
            throw M6ExternalIntentException("plan is not WAITING_EXTERNAL")
        }
        val waiting = plan.steps.filter {
            it.status == NativeStepStatus.WAITING_EXTERNAL
        }
        if (waiting.size != 1) {
            throw M6ExternalIntentException(
                "plan must contain exactly one waiting external step"
            )
        }
        val step = waiting.single()
        if (step.spec.kind != NativeStepKind.EXTERNAL) {
            throw M6ExternalIntentException("waiting step is not EXTERNAL")
        }
        if (utf8Size(plan.goal) > limits.maxGoalUtf8Bytes) {
            throw M6ExternalIntentException("goal exceeds external intent byte limit")
        }
        if (utf8Size(step.spec.objective) > limits.maxObjectiveUtf8Bytes) {
            throw M6ExternalIntentException(
                "objective exceeds external intent byte limit"
            )
        }
        return NativeExternalIntentRequest(
            planId = plan.planId,
            goal = plan.goal,
            stepId = step.stepId,
            objective = step.spec.objective,
            capabilities = views,
        )
    }

    fun bind(
        controller: NativePlanController,
        intent: NativeExternalIntent,
    ): M6ExternalActionRequest = bindSnapshot(
        requestForBackend(controller),
        intent,
    )

    fun proposeAndBind(
        controller: NativePlanController,
        backend: NativeTypedCognitionAdapter,
    ): M6ExternalActionRequest {
        val snapshot = requestForBackend(controller)
        val intent = try {
            backend.proposeExternalIntent(snapshot)
        } catch (exc: Exception) {
            throw M6ExternalIntentException(
                "external intent backend raised ${exc::class.java.simpleName}",
                exc,
            )
        }
        if (requestForBackend(controller) != snapshot) {
            throw M6ExternalIntentException(
                "waiting external step changed during intent proposal"
            )
        }
        return bindSnapshot(snapshot, intent)
    }

    fun descriptor(capabilityId: String): M6CapabilityDescriptor =
        descriptors[capabilityId]
            ?: throw M6ExternalIntentContractException(
                "capability is outside trusted catalog"
            )

    private fun bindSnapshot(
        snapshot: NativeExternalIntentRequest,
        intent: NativeExternalIntent,
    ): M6ExternalActionRequest {
        val descriptor = descriptors[intent.capabilityId]
            ?: throw M6ExternalIntentContractException(
                "backend selected capability outside trusted catalog"
            )
        if (intent.scope.size > limits.maxScopeEntries) {
            throw M6ExternalIntentContractException(
                "intent scope exceeds entry limit"
            )
        }
        intent.scope.forEach { (key, value) ->
            if (key.isEmpty() ||
                value.isEmpty() ||
                utf8Size(value) > limits.maxScopeValueUtf8Bytes
            ) {
                throw M6ExternalIntentContractException(
                    "intent scope contains an invalid key/value"
                )
            }
        }
        val scope = M6CapabilityScope.fromMap(intent.scope)
        if (!scope.keys.containsAll(descriptor.requiredScopeKeys)) {
            throw M6ExternalIntentContractException(
                "intent scope is missing required keys"
            )
        }
        val allowed = descriptor.requiredScopeKeys + descriptor.optionalScopeKeys
        if (!allowed.containsAll(scope.keys)) {
            throw M6ExternalIntentContractException(
                "intent scope contains unsupported keys"
            )
        }
        val payloadBytes = utf8Size(intent.payloadJson)
        if (
            payloadBytes > limits.maxPayloadUtf8Bytes ||
            payloadBytes > descriptor.maxPayloadUtf8Bytes
         ) {
            throw M6ExternalIntentContractException(
                "intent payload exceeds capability limit"
            )
        }
        return M6ExternalActionRequest(
            planId = snapshot.planId,
            stepId = snapshot.stepId,
            objective = snapshot.objective,
            capabilityId = intent.capabilityId,
            scope = scope,
            payloadJson = intent.payloadJson,
        )
    }
}

interface M6ApprovalControllerPort {
    fun createPrompt(
        requestDigest: String,
        principal: String,
        presentationJson: String,
        promptTtlNs: Long,
        nowNs: Long,
    ): M6ApprovalPrompt

    fun resolve(
        prompt: M6ApprovalPrompt,
        approved: Boolean,
        approvalTtlNs: Long,
        nowNs: Long,
    ): M6ApprovalToken?

    fun verify(
        token: M6ApprovalToken,
        requestDigest: String,
        principal: String,
        nowNs: Long,
    )
}

class M6ExternalApprovalHandoff(
    private val approvals: M6ApprovalControllerPort,
) {
    fun createPrompt(
        request: M6ExternalActionRequest,
        principal: String,
        promptTtlNs: Long,
        nowNs: Long,
    ): M6ApprovalPrompt = approvals.createPrompt(
        requestDigest = request.requestDigest,
        principal = principal,
        presentationJson = request.approvalPresentationJson(),
        promptTtlNs = promptTtlNs,
        nowNs = nowNs,
    )

    fun resolve(
        prompt: M6ApprovalPrompt,
        approved: Boolean,
        approvalTtlNs: Long,
        nowNs: Long,
    ): M6ApprovalToken? = approvals.resolve(
        prompt = prompt,
        approved = approved,
        approvalTtlNs = approvalTtlNs,
        nowNs = nowNs,
    )

    fun verify(
        token: M6ApprovalToken,
        request: M6ExternalActionRequest,
        principal: String,
        nowNs: Long,
    ) = approvals.verify(
        token = token,
        requestDigest = request.requestDigest,
        principal = principal,
        nowNs = nowNs,
    )
}

private val identifierChars =
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._:-/".toSet()

private fun externalHandoffValidateIdentifier(value: String, label: String) {
    if (value.isEmpty() || utf8Size(value) > 256 ||
        value.any { it !in identifierChars }
    ) {
        throw M6ExternalIntentContractException(
            "$label must be a bounded canonical identifier"
        )
    }
}

private fun utf8Size(value: String): Int =
    value.toByteArray(StandardCharsets.UTF_8).size

private fun sha256Hex(bytes: ByteArray): String =
    MessageDigest.getInstance("SHA-256").digest(bytes)
        .joinToString("") { "%02x".format(it.toInt() and 0xff) }

private fun StringBuilder.appendJsonString(value: String) {
    append('"')
    var index = 0
    while (index < value.length) {
        val ch = value[index]
        when (ch) {
            '"' -> append("\\\"")
            '\\' -> append("\\\\")
            '\b' -> append("\\b")
            '\u000C' -> append("\\f")
            '\n' -> append("\\n")
            '\r' -> append("\\r")
            '\t' -> append("\\t")
            else -> when {
                ch.code < 0x20 -> append("\\u%04x".format(ch.code))
                Character.isHighSurrogate(ch) -> {
                    if (index + 1 >= value.length ||
                        !Character.isLowSurrogate(value[index + 1])
                    ) {
                        throw M6ExternalIntentContractException(
                            "JSON string contains unpaired surrogate"
                        )
                    }
                    append(ch)
                    append(value[index + 1])
                    index += 1
                }
                Character.isLowSurrogate(ch) ->
                    throw M6ExternalIntentContractException(
                        "JSON string contains unpaired surrogate"
                    )
                else -> append(ch)
            }
        }
        index += 1
    }
    append('"')
}
