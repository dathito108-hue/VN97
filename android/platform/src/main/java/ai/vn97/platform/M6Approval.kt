package ai.vn97.platform

import java.security.MessageDigest
import java.security.SecureRandom

internal interface ApprovalMac {
    fun hmacSha256(payload: ByteArray): ByteArray
}

data class M6ApprovalToken(
    val approvalId: String,
    val issuer: String,
    val principal: String,
    val requestDigest: String,
    val issuedNs: Long,
    val expiresNs: Long,
    val signature: String,
)

data class M6ApprovalPrompt(
    val promptId: String,
    val principal: String,
    val requestDigest: String,
    val presentationJson: String,
    val createdNs: Long,
    val expiresNs: Long,
)

class M6ApprovalException(message: String) : IllegalStateException(message)

internal class M6ApprovalAuthority(
    private val mac: ApprovalMac,
    val issuer: String = "local-user",
    private val maxTtlNs: Long = 60_000_000_000L,
    private val random: SecureRandom = SecureRandom(),
) {
    init {
        validateIdentifier(issuer, "issuer")
        require(maxTtlNs > 0L) { "maxTtlNs must be positive" }
    }

    fun approve(
        requestDigest: String,
        principal: String,
        ttlNs: Long,
        nowNs: Long,
    ): M6ApprovalToken {
        validateDigest(requestDigest)
        validateIdentifier(principal, "principal")
        require(nowNs >= 0L) { "nowNs must be non-negative" }
        require(ttlNs in 1..maxTtlNs) { "approval ttl exceeds configured bound" }
        val expires = try {
            Math.addExact(nowNs, ttlNs)
        } catch (exc: ArithmeticException) {
            throw IllegalArgumentException("approval expiry overflows", exc)
        }
        val approvalId = ByteArray(16).also(random::nextBytes).toHex()
        val signature = mac.hmacSha256(
            canonicalApprovalPayload(
                approvalId = approvalId,
                issuer = issuer,
                principal = principal,
                requestDigest = requestDigest,
                issuedNs = nowNs,
                expiresNs = expires,
            )
        ).also {
            require(it.size == 32) { "approval MAC must be HMAC-SHA256" }
        }.toHex()
        return M6ApprovalToken(
            approvalId = approvalId,
            issuer = issuer,
            principal = principal,
            requestDigest = requestDigest,
            issuedNs = nowNs,
            expiresNs = expires,
            signature = signature,
        )
    }

    fun verify(
        token: M6ApprovalToken,
        requestDigest: String,
        principal: String,
        nowNs: Long,
    ) {
        validateDigest(requestDigest)
        validateIdentifier(principal, "principal")
        if (token.issuer != issuer) throw M6ApprovalException("approval issuer mismatch")
        if (token.principal != principal) throw M6ApprovalException("approval principal mismatch")
        if (token.requestDigest != requestDigest) throw M6ApprovalException("approval request binding mismatch")
        if (token.issuedNs < 0L || token.expiresNs <= token.issuedNs) {
            throw M6ApprovalException("approval validity window is malformed")
        }
        if (nowNs < token.issuedNs || nowNs >= token.expiresNs) {
            throw M6ApprovalException("approval is outside its validity window")
        }
        try {
            validateApprovalId(token.approvalId)
            validateSignature(token.signature)
            validateIdentifier(token.issuer, "issuer")
            validateIdentifier(token.principal, "principal")
            validateDigest(token.requestDigest)
        } catch (exc: IllegalArgumentException) {
            throw M6ApprovalException("approval token structure is invalid")
        }
        val expected = mac.hmacSha256(
            canonicalApprovalPayload(
                approvalId = token.approvalId,
                issuer = token.issuer,
                principal = token.principal,
                requestDigest = token.requestDigest,
                issuedNs = token.issuedNs,
                expiresNs = token.expiresNs,
            )
        )
        val supplied = token.signature.hexToBytes()
        if (!MessageDigest.isEqual(expected, supplied)) {
            throw M6ApprovalException("approval signature mismatch")
        }
    }
}

internal class M6ApprovalCoordinator(
    private val authority: M6ApprovalAuthority,
    private val maxPendingPrompts: Int = 32,
    private val maxPromptTtlNs: Long = 300_000_000_000L,
    private val maxApprovalTtlNs: Long = 60_000_000_000L,
    private val random: SecureRandom = SecureRandom(),
) {
    private val pending = LinkedHashMap<String, M6ApprovalPrompt>()

    init {
        require(maxPendingPrompts > 0) { "maxPendingPrompts must be positive" }
        require(maxPromptTtlNs > 0L) { "maxPromptTtlNs must be positive" }
        require(maxApprovalTtlNs > 0L) { "maxApprovalTtlNs must be positive" }
    }

    @Synchronized
    fun createPrompt(
        requestDigest: String,
        principal: String,
        presentationJson: String,
        promptTtlNs: Long,
        nowNs: Long,
    ): M6ApprovalPrompt {
        validateDigest(requestDigest)
        validateIdentifier(principal, "principal")
        require(presentationJson.isNotEmpty()) { "presentationJson must not be empty" }
        require(presentationJson.toByteArray(Charsets.UTF_8).size <= 64 * 1024) {
            "presentationJson exceeds byte bound"
        }
        require(nowNs >= 0L) { "nowNs must be non-negative" }
        require(promptTtlNs in 1..maxPromptTtlNs) { "prompt ttl exceeds configured bound" }
        pruneExpired(nowNs)
        if (pending.size >= maxPendingPrompts) throw M6ApprovalException("too many pending approval prompts")
        val expires = try {
            Math.addExact(nowNs, promptTtlNs)
        } catch (exc: ArithmeticException) {
            throw IllegalArgumentException("prompt expiry overflows", exc)
        }
        var promptId: String
        do {
            promptId = ByteArray(16).also(random::nextBytes).toHex()
        } while (pending.containsKey(promptId))
        val prompt = M6ApprovalPrompt(
            promptId = promptId,
            principal = principal,
            requestDigest = requestDigest,
            presentationJson = presentationJson,
            createdNs = nowNs,
            expiresNs = expires,
        )
        pending[promptId] = prompt
        return prompt
    }

    @Synchronized
    fun resolve(
        prompt: M6ApprovalPrompt,
        approved: Boolean,
        approvalTtlNs: Long,
        nowNs: Long,
    ): M6ApprovalToken? {
        val stored = pending.remove(prompt.promptId)
            ?: throw M6ApprovalException("approval prompt is unknown, stale or already consumed")
        if (stored != prompt) throw M6ApprovalException("approval prompt was modified")
        if (nowNs < prompt.createdNs || nowNs >= prompt.expiresNs) {
            throw M6ApprovalException("approval prompt is outside its validity window")
        }
        if (!approved) return null
        require(approvalTtlNs in 1..maxApprovalTtlNs) { "approval ttl exceeds configured bound" }
        return authority.approve(
            requestDigest = prompt.requestDigest,
            principal = prompt.principal,
            ttlNs = approvalTtlNs,
            nowNs = nowNs,
        )
    }

    @Synchronized
    fun pendingCount(nowNs: Long): Int {
        pruneExpired(nowNs)
        return pending.size
    }

    private fun pruneExpired(nowNs: Long) {
        val iterator = pending.entries.iterator()
        while (iterator.hasNext()) {
            if (nowNs >= iterator.next().value.expiresNs) iterator.remove()
        }
    }
}

internal fun canonicalApprovalPayload(
    approvalId: String,
    issuer: String,
    principal: String,
    requestDigest: String,
    issuedNs: Long,
    expiresNs: Long,
): ByteArray {
    validateApprovalId(approvalId)
    validateIdentifier(issuer, "issuer")
    validateIdentifier(principal, "principal")
    validateDigest(requestDigest)
    require(issuedNs >= 0L) { "issuedNs must be non-negative" }
    require(expiresNs > issuedNs) { "expiresNs must be greater than issuedNs" }
    return buildString(256) {
        append("{\"approval_id\":\"").append(approvalId)
        append("\",\"expires_ns\":").append(expiresNs)
        append(",\"issued_ns\":").append(issuedNs)
        append(",\"issuer\":\"").append(issuer)
        append("\",\"principal\":\"").append(principal)
        append("\",\"request_digest\":\"").append(requestDigest)
        append("\"}")
    }.toByteArray(Charsets.UTF_8)
}

private val identifierChars =
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._:-/".toSet()
private val lowerHex = "0123456789abcdef".toSet()

internal fun validateIdentifier(value: String, label: String) {
    require(value.isNotEmpty()) { "$label must not be empty" }
    require(value.toByteArray(Charsets.UTF_8).size <= 256) { "$label exceeds byte bound" }
    require(value.all { it in identifierChars }) { "$label contains unsupported characters" }
}

internal fun validateDigest(value: String) {
    require(value.length == 64 && value.all { it in lowerHex }) {
        "requestDigest must be lowercase SHA-256 hex"
    }
}

private fun validateApprovalId(value: String) {
    require(value.length == 32 && value.all { it in lowerHex }) {
        "approvalId must be 16-byte lowercase hex"
    }
}

private fun validateSignature(value: String) {
    require(value.length == 64 && value.all { it in lowerHex }) {
        "signature must be lowercase HMAC-SHA256 hex"
    }
}

internal fun ByteArray.toHex(): String = joinToString(separator = "") { "%02x".format(it) }

internal fun String.hexToBytes(): ByteArray {
    require(length % 2 == 0) { "hex length must be even" }
    return ByteArray(length / 2) { index ->
        substring(index * 2, index * 2 + 2).toInt(16).toByte()
    }
}
