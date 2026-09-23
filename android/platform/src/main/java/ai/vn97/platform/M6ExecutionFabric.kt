package ai.vn97.platform

import ai.vn97.runtime.NativePlanController
import ai.vn97.runtime.NativePlanStatus
import ai.vn97.runtime.NativeStepKind
import ai.vn97.runtime.NativeStepStatus
import java.math.BigDecimal
import java.nio.charset.StandardCharsets
import java.security.MessageDigest
import java.security.SecureRandom
import kotlin.math.min

open class M6AuthorityException(message: String, cause: Throwable? = null) :
    IllegalStateException(message, cause)

class M6AuthorizationDeniedException(message: String) : M6AuthorityException(message)
class M6ApprovalRequiredException(message: String) : M6AuthorityException(message)
class M6InvalidApprovalException(message: String, cause: Throwable? = null) :
    M6AuthorityException(message, cause)
class M6LeaseException(message: String) : M6AuthorityException(message)
class M6CapabilityNotFoundException(message: String) : M6AuthorityException(message)
class M6ExternalExecutionException(message: String, cause: Throwable? = null) :
    M6AuthorityException(message, cause)

data class M6PolicyGrant(
    val principal: String,
    val capabilityId: String,
    val scopeDigest: String,
    val approvalRequired: Boolean? = null,
    val maxLeaseNs: Long = 300_000_000_000L,
    val maxLeaseUses: Int = 1,
) {
    init {
        validateAuthorityIdentifier(principal, "principal")
        validateAuthorityIdentifier(capabilityId, "capabilityId")
        validateSha256(scopeDigest, "scopeDigest")
        require(maxLeaseNs > 0L) { "maxLeaseNs must be positive" }
        require(maxLeaseUses > 0) { "maxLeaseUses must be positive" }
    }
}

data class M6CapabilityLease(
    val leaseId: String,
    val principal: String,
    val capabilityId: String,
    val scopeDigest: String,
    val issuedNs: Long,
    val expiresNs: Long,
    val maxUses: Int,
)

data class M6AuthorizedAction(
    val request: M6ExternalActionRequest,
    val principal: String,
    val lease: M6CapabilityLease,
    val approvalId: String = "",
)

data class M6ActionOutcome(
    val success: Boolean,
    val result: String,
    val confidence: Double = 1.0,
    val evidenceRecordIds: List<Long> = emptyList(),
    val retryable: Boolean = false,
) {
    init {
        require(result.isNotEmpty()) { "action outcome result must not be empty" }
        require(confidence.isFinite() && confidence in 0.0..1.0) {
            "action outcome confidence must be finite and in [0, 1]"
        }
        require(evidenceRecordIds.all { it > 0L }) {
            "evidence record IDs must be positive"
        }
        require(evidenceRecordIds.distinct().size == evidenceRecordIds.size) {
            "evidence record IDs must be unique"
        }
    }
}

fun interface M6CapabilityHandler {
    fun execute(action: M6AuthorizedAction): M6ActionOutcome
}

fun interface M6PayloadValidator {
    fun validate(request: M6ExternalActionRequest)
}

private data class RegisteredCapability(
    val descriptor: M6CapabilityDescriptor,
    val handler: M6CapabilityHandler,
    val validator: M6PayloadValidator?,
)

class M6TypedCapabilityRegistry {
    private val entries = LinkedHashMap<String, RegisteredCapability>()
    var sealed: Boolean = false
        private set

    @Synchronized
    fun register(
        descriptor: M6CapabilityDescriptor,
        handler: M6CapabilityHandler,
        validator: M6PayloadValidator? = null,
    ) {
        if (sealed) throw M6AuthorityException("capability registry is sealed")
        if (entries.containsKey(descriptor.capabilityId)) {
            throw M6AuthorityException("capability is already registered")
        }
        entries[descriptor.capabilityId] =
            RegisteredCapability(descriptor, handler, validator)
    }

    @Synchronized
    fun seal() {
        sealed = true
    }

    @Synchronized
    fun descriptor(capabilityId: String): M6CapabilityDescriptor =
        entries[capabilityId]?.descriptor
            ?: throw M6CapabilityNotFoundException(
                "unknown capability: $capabilityId"
            )

    @Synchronized
    fun validate(request: M6ExternalActionRequest): M6CapabilityDescriptor {
        val entry = entries[request.capabilityId]
            ?: throw M6CapabilityNotFoundException(
                "unknown capability: ${request.capabilityId}"
            )
        validateCapabilityRequest(entry.descriptor, request)
        entry.validator?.validate(request)
        return entry.descriptor
    }

    @Synchronized
    internal fun executeAuthorized(
        action: M6AuthorizedAction,
    ): M6ActionOutcome {
        val entry = entries[action.request.capabilityId]
            ?: throw M6CapabilityNotFoundException(
                "unknown capability: ${action.request.capabilityId}"
            )
        return entry.handler.execute(action)
    }


}

class M6DenyByDefaultAuthorityGate(
    grants: List<M6PolicyGrant>,
    private val approvals: M6ApprovalControllerPort? = null,
    private val random: SecureRandom = SecureRandom(),
) {
    private data class GrantKey(
        val principal: String,
        val capabilityId: String,
        val scopeDigest: String,
    )

    private val grants: Map<GrantKey, M6PolicyGrant>
    private val leases = LinkedHashMap<String, M6CapabilityLease>()
    private val uses = LinkedHashMap<String, Int>()

    init {
        val mapped = LinkedHashMap<GrantKey, M6PolicyGrant>()
        for (grant in grants) {
            val key = GrantKey(
                grant.principal,
                grant.capabilityId,
                grant.scopeDigest,
            )
            require(!mapped.containsKey(key)) {
                "duplicate authority policy grant"
            }
            mapped[key] = grant
        }
        this.grants = mapped
    }

    @Synchronized
    fun issueLease(
        descriptor: M6CapabilityDescriptor,
        request: M6ExternalActionRequest,
        principal: String,
        approval: M6ApprovalToken? = null,
        nowNs: Long,
    ): M6CapabilityLease {
        validateAuthorityIdentifier(principal, "principal")
        require(nowNs >= 0L) { "nowNs must be non-negative" }
        validateCapabilityRequest(descriptor, request)
        val key = GrantKey(
            principal,
            request.capabilityId,
            request.scope.digest,
        )
        val grant = grants[key]
            ?: throw M6AuthorizationDeniedException(
                "no matching authority policy grant"
            )
        val requiresApproval =
            grant.approvalRequired ?: descriptor.approvalRequired
        if (requiresApproval) {
            val token = approval
                ?: throw M6ApprovalRequiredException(
                    "capability requires explicit approval"
                )
            val approvalPort = approvals
                ?: throw M6InvalidApprovalException(
                    "approval authority is not configured"
                )
            try {
                approvalPort.verify(
                    token = token,
                    requestDigest = request.requestDigest,
                    principal = principal,
                    nowNs = nowNs,
                )
            } catch (exc: Exception) {
                throw M6InvalidApprovalException(
                    "approval verification failed",
                    exc,
                )
            }
        }
        val ttl = min(descriptor.maxLeaseNs, grant.maxLeaseNs)
        val maxUses = min(descriptor.maxLeaseUses, grant.maxLeaseUses)
        val expires = try {
            Math.addExact(nowNs, ttl)
        } catch (exc: ArithmeticException) {
            throw IllegalArgumentException("lease expiry overflows", exc)
        }
        var leaseId: String
        do {
            leaseId = ByteArray(16).also(random::nextBytes).toLowerHex()
        } while (leases.containsKey(leaseId))
        val lease = M6CapabilityLease(
            leaseId = leaseId,
            principal = principal,
            capabilityId = request.capabilityId,
            scopeDigest = request.scope.digest,
            issuedNs = nowNs,
            expiresNs = expires,
            maxUses = maxUses,
        )
        leases[leaseId] = lease
        uses[leaseId] = 0
        return lease
    }

    @Synchronized
    fun consume(
        lease: M6CapabilityLease,
        request: M6ExternalActionRequest,
        principal: String,
        nowNs: Long,
    ) {
        val issued = leases[lease.leaseId]
        if (issued == null || issued != lease) {
            throw M6LeaseException(
                "lease was not issued by this authority gate"
            )
        }
        if (lease.principal != principal) {
            throw M6LeaseException("lease principal mismatch")
        }
        if (lease.capabilityId != request.capabilityId) {
            throw M6LeaseException("lease capability mismatch")
        }
        if (lease.scopeDigest != request.scope.digest) {
            throw M6LeaseException("lease scope mismatch")
        }
        if (nowNs < lease.issuedNs || nowNs >= lease.expiresNs) {
            throw M6LeaseException("lease is outside its validity window")
        }
        val used = uses[lease.leaseId]
            ?: throw M6LeaseException("lease use state is missing")
        if (used >= lease.maxUses) {
            throw M6LeaseException("lease use budget exhausted")
        }
        uses[lease.leaseId] = used + 1
    }
}

enum class M6ReceiptStatus(val code: Int) {
    DENIED(1),
    SUCCEEDED(2),
    FAILED(3),
}

data class M6ActionReceipt(
    val sequence: Int,
    val timestampNs: Long,
    val previousReceiptId: String,
    val receiptId: String,
    val status: M6ReceiptStatus,
    val requestDigest: String,
    val planId: String,
    val stepId: Int,
    val capabilityId: String,
    val scopeDigest: String,
    val principal: String,
    val leaseId: String,
    val approvalId: String,
    val result: String,
    val confidence: Double,
    val evidenceRecordIds: List<Long>,
    val retryable: Boolean,
    val errorType: String,
)

interface M6ActionAudit {
    val receipts: List<M6ActionReceipt>

    fun append(
        status: M6ReceiptStatus,
        request: M6ExternalActionRequest,
        principal: String,
        leaseId: String = "",
        approvalId: String = "",
        outcome: M6ActionOutcome? = null,
        errorType: String = "",
        timestampNs: Long,
    ): M6ActionReceipt

    fun successfulReceipt(requestDigest: String): M6ActionReceipt?
}

class M6InMemoryActionAudit : M6ActionAudit {
    private val records = ArrayList<M6ActionReceipt>()

    override val receipts: List<M6ActionReceipt>
        @Synchronized get() = records.toList()

    @Synchronized
    override fun append(
        status: M6ReceiptStatus,
        request: M6ExternalActionRequest,
        principal: String,
        leaseId: String,
        approvalId: String,
        outcome: M6ActionOutcome?,
        errorType: String,
        timestampNs: Long,
    ): M6ActionReceipt {
        require(timestampNs >= 0L) { "timestampNs must be non-negative" }
        validateAuthorityIdentifier(principal, "principal")
        val previous = records.lastOrNull()?.receiptId.orEmpty()
        val base = M6ActionReceipt(
            sequence = records.size + 1,
            timestampNs = timestampNs,
            previousReceiptId = previous,
            receiptId = "",
            status = status,
            requestDigest = request.requestDigest,
            planId = request.planId,
            stepId = request.stepId,
            capabilityId = request.capabilityId,
            scopeDigest = request.scope.digest,
            principal = principal,
            leaseId = leaseId,
            approvalId = approvalId,
            result = outcome?.result.orEmpty(),
            confidence = outcome?.confidence ?: 0.0,
            evidenceRecordIds = outcome?.evidenceRecordIds?.toList()
                ?: emptyList(),
            retryable = outcome?.retryable ?: false,
            errorType = errorType,
        )
        val receipt = base.copy(
            receiptId = sha256Hex(
                canonicalReceiptPayload(base)
                    .toByteArray(StandardCharsets.UTF_8)
            )
        )
        records += receipt
        return receipt
    }

    @Synchronized
    override fun successfulReceipt(
        requestDigest: String,
    ): M6ActionReceipt? = records.asReversed().firstOrNull {
        it.requestDigest == requestDigest &&
            it.status == M6ReceiptStatus.SUCCEEDED
    }
}

data class M6ExecutionResult(
    val receipt: M6ActionReceipt,
    val replayed: Boolean,
)

class M6ExternalExecutionFabric(
    private val registry: M6TypedCapabilityRegistry,
    private val authority: M6DenyByDefaultAuthorityGate,
    private val audit: M6ActionAudit,
) {
    init {
        require(registry.sealed) {
            "capability registry must be sealed before execution"
        }
    }

    fun executeWaiting(
        controller: NativePlanController,
        request: M6ExternalActionRequest,
        principal: String,
        approval: M6ApprovalToken? = null,
        lease: M6CapabilityLease? = null,
        nowNs: Long,
    ): M6ExecutionResult {
        validateWaitingStep(controller, request)
        val descriptor = registry.validate(request)

        val prior = audit.successfulReceipt(request.requestDigest)
        if (prior != null) {
            controller.recordExternalResult(
                request.stepId,
                prior.result,
                prior.confidence,
                prior.evidenceRecordIds,
            )
            return M6ExecutionResult(prior, replayed = true)
        }

        var activeLease = lease
        try {
            if (activeLease == null) {
                activeLease = authority.issueLease(
                    descriptor = descriptor,
                    request = request,
                    principal = principal,
                    approval = approval,
                    nowNs = nowNs,
                )
            }
            authority.consume(
                lease = activeLease,
                request = request,
                principal = principal,
                nowNs = nowNs,
            )
        } catch (exc: M6AuthorityException) {
            audit.append(
                status = M6ReceiptStatus.DENIED,
                request = request,
                principal = principal,
                leaseId = activeLease?.leaseId.orEmpty(),
                approvalId = approval?.approvalId.orEmpty(),
                errorType = exc::class.java.simpleName,
                timestampNs = nowNs,
            )
            throw exc
        }

        val action = M6AuthorizedAction(
            request = request,
            principal = principal,
            lease = checkNotNull(activeLease),
            approvalId = approval?.approvalId.orEmpty(),
        )
        val outcome = try {
            registry.executeAuthorized(action)
        } catch (exc: Exception) {
            audit.append(
                status = M6ReceiptStatus.FAILED,
                request = request,
                principal = principal,
                leaseId = action.lease.leaseId,
                approvalId = action.approvalId,
                errorType = exc::class.java.simpleName,
                timestampNs = nowNs,
            )
            controller.failStep(
                request.stepId,
                "external capability failure: ${exc::class.java.simpleName}",
                retryable = false,
            )
            throw M6ExternalExecutionException(
                "external capability raised before producing a typed outcome",
                exc,
            )
        }

        if (!outcome.success) {
            val receipt = audit.append(
                status = M6ReceiptStatus.FAILED,
                request = request,
                principal = principal,
                leaseId = action.lease.leaseId,
                approvalId = action.approvalId,
                outcome = outcome,
                timestampNs = nowNs,
            )
            controller.failStep(
                request.stepId,
                outcome.result,
                retryable = outcome.retryable,
            )
            return M6ExecutionResult(receipt, replayed = false)
        }

        val receipt = audit.append(
            status = M6ReceiptStatus.SUCCEEDED,
            request = request,
            principal = principal,
            leaseId = action.lease.leaseId,
            approvalId = action.approvalId,
            outcome = outcome,
            timestampNs = nowNs,
        )
        controller.recordExternalResult(
            request.stepId,
            outcome.result,
            outcome.confidence,
            outcome.evidenceRecordIds,
        )
        return M6ExecutionResult(receipt, replayed = false)
    }

    private fun validateWaitingStep(
        controller: NativePlanController,
        request: M6ExternalActionRequest,
    ) {
        val plan = controller.plan
        if (plan.planId != request.planId) {
            throw M6ExternalExecutionException(
                "request planId does not match controller"
            )
        }
        if (plan.status != NativePlanStatus.WAITING_EXTERNAL) {
            throw M6ExternalExecutionException(
                "plan is not WAITING_EXTERNAL"
            )
        }
        val step = try {
            plan.step(request.stepId)
        } catch (exc: Exception) {
            throw M6ExternalExecutionException(
                "request stepId is invalid",
                exc,
            )
        }
        if (step.status != NativeStepStatus.WAITING_EXTERNAL) {
            throw M6ExternalExecutionException(
                "step is not WAITING_EXTERNAL"
            )
        }
        if (step.spec.kind != NativeStepKind.EXTERNAL) {
            throw M6ExternalExecutionException(
                "waiting step is not EXTERNAL"
            )
        }
        if (step.spec.objective != request.objective) {
            throw M6ExternalExecutionException(
                "request objective does not match immutable plan step"
            )
        }
    }
}

private fun validateCapabilityRequest(
    descriptor: M6CapabilityDescriptor,
    request: M6ExternalActionRequest,
) {
    if (request.capabilityId != descriptor.capabilityId) {
        throw M6AuthorizationDeniedException(
            "request capability does not match descriptor"
        )
    }
    if (!request.scope.keys.containsAll(descriptor.requiredScopeKeys)) {
        throw M6AuthorizationDeniedException(
            "request scope is missing required keys"
        )
    }
    val allowed = descriptor.requiredScopeKeys + descriptor.optionalScopeKeys
    if (!allowed.containsAll(request.scope.keys)) {
        throw M6AuthorizationDeniedException(
            "request scope contains unsupported keys"
        )
    }
    if (
        request.payloadJson.toByteArray(StandardCharsets.UTF_8).size >
        descriptor.maxPayloadUtf8Bytes
    ) {
        throw M6AuthorizationDeniedException(
            "request payload exceeds capability limit"
        )
    }
}

private val authorityIdentifierChars =
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._:-/".toSet()

private fun validateAuthorityIdentifier(value: String, label: String) {
    require(value.isNotEmpty()) { "$label must not be empty" }
    require(value.toByteArray(StandardCharsets.UTF_8).size <= 256) {
        "$label exceeds byte bound"
    }
    require(value.all { it in authorityIdentifierChars }) {
        "$label contains unsupported characters"
    }
}

private fun validateSha256(value: String, label: String) {
    require(value.length == 64 && value.all { it in "0123456789abcdef" }) {
        "$label must be lowercase SHA-256 hex"
    }
}

private fun ByteArray.toLowerHex(): String =
    joinToString("") { "%02x".format(it.toInt() and 0xff) }

private fun sha256Hex(bytes: ByteArray): String =
    MessageDigest.getInstance("SHA-256").digest(bytes).toLowerHex()

internal fun canonicalReceiptPayload(receipt: M6ActionReceipt): String =
    buildString {
        append("{\"approval_id\":")
        appendJson(receipt.approvalId)
        append(",\"capability_id\":")
        appendJson(receipt.capabilityId)
        append(",\"confidence\":")
        append(canonicalPythonDouble(receipt.confidence))
        append(",\"error_type\":")
        appendJson(receipt.errorType)
        append(",\"evidence_record_ids\":[")
        receipt.evidenceRecordIds.forEachIndexed { index, id ->
            if (index != 0) append(',')
            append(id)
        }
        append("],\"lease_id\":")
        appendJson(receipt.leaseId)
        append(",\"plan_id\":")
        appendJson(receipt.planId)
        append(",\"previous_receipt_id\":")
        appendJson(receipt.previousReceiptId)
        append(",\"principal\":")
        appendJson(receipt.principal)
        append(",\"request_digest\":")
        appendJson(receipt.requestDigest)
        append(",\"result\":")
        appendJson(receipt.result)
        append(",\"retryable\":")
        append(if (receipt.retryable) "true" else "false")
        append(",\"scope_digest\":")
        appendJson(receipt.scopeDigest)
        append(",\"sequence\":")
        append(receipt.sequence)
        append(",\"status\":")
        append(receipt.status.code)
        append(",\"step_id\":")
        append(receipt.stepId)
        append(",\"timestamp_ns\":")
        append(receipt.timestampNs)
        append('}')
    }


internal fun canonicalReceiptRecord(receipt: M6ActionReceipt): String =
    buildString {
        append("{\"approval_id\":")
        appendJson(receipt.approvalId)
        append(",\"capability_id\":")
        appendJson(receipt.capabilityId)
        append(",\"confidence\":")
        append(canonicalPythonDouble(receipt.confidence))
        append(",\"error_type\":")
        appendJson(receipt.errorType)
        append(",\"evidence_record_ids\":[")
        receipt.evidenceRecordIds.forEachIndexed { index, id ->
            if (index != 0) append(',')
            append(id)
        }
        append("],\"lease_id\":")
        appendJson(receipt.leaseId)
        append(",\"plan_id\":")
        appendJson(receipt.planId)
        append(",\"previous_receipt_id\":")
        appendJson(receipt.previousReceiptId)
        append(",\"principal\":")
        appendJson(receipt.principal)
        append(",\"receipt_id\":")
        appendJson(receipt.receiptId)
        append(",\"request_digest\":")
        appendJson(receipt.requestDigest)
        append(",\"result\":")
        appendJson(receipt.result)
        append(",\"retryable\":")
        append(if (receipt.retryable) "true" else "false")
        append(",\"scope_digest\":")
        appendJson(receipt.scopeDigest)
        append(",\"sequence\":")
        append(receipt.sequence)
        append(",\"status\":")
        append(receipt.status.code)
        append(",\"step_id\":")
        append(receipt.stepId)
        append(",\"timestamp_ns\":")
        append(receipt.timestampNs)
        append('}')
    }

private fun canonicalPythonDouble(value: Double): String {
    require(value.isFinite()) { "JSON number must be finite" }
    if (value == 0.0) {
        return if (java.lang.Double.doubleToRawLongBits(value) < 0) {
            "-0.0"
        } else {
            "0.0"
        }
    }
    val decimal = BigDecimal.valueOf(value)
    val normalized = decimal.stripTrailingZeros()
    val exponent = normalized.precision() - normalized.scale() - 1
    if (exponent < -4 || exponent >= 16) {
        val negative = normalized.signum() < 0
        val digits = normalized.abs().unscaledValue().toString()
        val mantissa = if (digits.length == 1) {
            digits
        } else {
            digits.substring(0, 1) + "." + digits.substring(1)
        }
        val sign = if (exponent >= 0) "+" else "-"
        val exp = kotlin.math.abs(exponent).toString().padStart(2, '0')
        return (if (negative) "-" else "") + mantissa + "e" + sign + exp
    }
    val plain = normalized.toPlainString()
    return if (value == kotlin.math.floor(value) && '.' !in plain) {
        "$plain.0"
    } else {
        plain
    }
}

private fun StringBuilder.appendJson(value: String) {
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
                    if (
                        index + 1 >= value.length ||
                        !Character.isLowSurrogate(value[index + 1])
                    ) {
                        throw IllegalArgumentException(
                            "receipt JSON contains unpaired surrogate"
                        )
                    }
                    append(ch)
                    append(value[index + 1])
                    index += 1
                }
                Character.isLowSurrogate(ch) ->
                    throw IllegalArgumentException(
                        "receipt JSON contains unpaired surrogate"
                    )
                else -> append(ch)
            }
        }
        index += 1
    }
    append('"')
}
