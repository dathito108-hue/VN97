package ai.vn97.platform

import android.security.keystore.KeyGenParameterSpec
import android.security.keystore.KeyProperties
import java.security.KeyStore
import javax.crypto.KeyGenerator
import javax.crypto.Mac
import javax.crypto.SecretKey

internal class AndroidKeystoreApprovalMac(
    private val alias: String,
) : ApprovalMac {
    init {
        validateKeyAlias(alias)
    }

    override fun hmacSha256(payload: ByteArray): ByteArray {
        val mac = Mac.getInstance("HmacSHA256")
        mac.init(loadOrCreateKey())
        return mac.doFinal(payload)
    }

    private fun loadOrCreateKey(): SecretKey = synchronized(keyLock) {
        val keyStore = KeyStore.getInstance(ANDROID_KEYSTORE).apply { load(null) }
        val existing = keyStore.getKey(alias, null)
        if (existing != null) {
            return@synchronized existing as? SecretKey
                ?: throw M6ApprovalException("approval key alias is not a secret key")
        }

        val generator = KeyGenerator.getInstance(
            KeyProperties.KEY_ALGORITHM_HMAC_SHA256,
            ANDROID_KEYSTORE,
        )
        generator.init(
            KeyGenParameterSpec.Builder(alias, KeyProperties.PURPOSE_SIGN)
                .setDigests(KeyProperties.DIGEST_SHA256)
                .build()
        )
        generator.generateKey()
    }

    companion object {
        private const val ANDROID_KEYSTORE = "AndroidKeyStore"
        private val keyLock = Any()
    }
}

class AndroidApprovalController(
    keyAlias: String = "vn97.approval.v1",
    issuer: String = "local-user",
    maxPendingPrompts: Int = 32,
    maxPromptTtlNs: Long = 300_000_000_000L,
    maxApprovalTtlNs: Long = 60_000_000_000L,
) {
    private val authority = M6ApprovalAuthority(
        mac = AndroidKeystoreApprovalMac(keyAlias),
        issuer = issuer,
        maxTtlNs = maxApprovalTtlNs,
    )
    private val coordinator = M6ApprovalCoordinator(
        authority = authority,
        maxPendingPrompts = maxPendingPrompts,
        maxPromptTtlNs = maxPromptTtlNs,
        maxApprovalTtlNs = maxApprovalTtlNs,
    )

    fun createPrompt(
        requestDigest: String,
        principal: String,
        presentationJson: String,
        promptTtlNs: Long,
        nowNs: Long = wallClockNs(),
    ): M6ApprovalPrompt = coordinator.createPrompt(
        requestDigest = requestDigest,
        principal = principal,
        presentationJson = presentationJson,
        promptTtlNs = promptTtlNs,
        nowNs = nowNs,
    )

    fun resolve(
        prompt: M6ApprovalPrompt,
        approved: Boolean,
        approvalTtlNs: Long,
        nowNs: Long = wallClockNs(),
    ): M6ApprovalToken? = coordinator.resolve(
        prompt = prompt,
        approved = approved,
        approvalTtlNs = approvalTtlNs,
        nowNs = nowNs,
    )

    fun verify(
        token: M6ApprovalToken,
        requestDigest: String,
        principal: String,
        nowNs: Long = wallClockNs(),
    ) = authority.verify(
        token = token,
        requestDigest = requestDigest,
        principal = principal,
        nowNs = nowNs,
    )

    fun pendingCount(nowNs: Long = wallClockNs()): Int = coordinator.pendingCount(nowNs)
}

class AndroidApprovalControllerPort(
    private val controller: AndroidApprovalController,
) : M6ApprovalControllerPort {
    override fun createPrompt(
        requestDigest: String,
        principal: String,
        presentationJson: String,
        promptTtlNs: Long,
        nowNs: Long,
    ): M6ApprovalPrompt = controller.createPrompt(
        requestDigest = requestDigest,
        principal = principal,
        presentationJson = presentationJson,
        promptTtlNs = promptTtlNs,
        nowNs = nowNs,
    )

    override fun resolve(
        prompt: M6ApprovalPrompt,
        approved: Boolean,
        approvalTtlNs: Long,
        nowNs: Long,
    ): M6ApprovalToken? = controller.resolve(
        prompt = prompt,
        approved = approved,
        approvalTtlNs = approvalTtlNs,
        nowNs = nowNs,
    )

    override fun verify(
        token: M6ApprovalToken,
        requestDigest: String,
        principal: String,
        nowNs: Long,
    ) = controller.verify(
        token = token,
        requestDigest = requestDigest,
        principal = principal,
        nowNs = nowNs,
    )
}

private fun validateKeyAlias(alias: String) {
    require(alias.isNotEmpty()) { "key alias must not be empty" }
    require(alias.toByteArray(Charsets.UTF_8).size <= 128) { "key alias exceeds byte bound" }
    require(alias.all { it.isLetterOrDigit() || it == '.' || it == '_' || it == '-' }) {
        "key alias contains unsupported characters"
    }
}

private fun wallClockNs(): Long =
    Math.multiplyExact(System.currentTimeMillis(), 1_000_000L)
