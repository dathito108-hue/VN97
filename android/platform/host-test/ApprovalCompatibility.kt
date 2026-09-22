import ai.vn97.platform.*
import javax.crypto.Mac
import javax.crypto.spec.SecretKeySpec

private class JvmMac(private val key: ByteArray) : ApprovalMac {
    override fun hmacSha256(payload: ByteArray): ByteArray {
        val mac = Mac.getInstance("HmacSHA256")
        mac.init(SecretKeySpec(key, "HmacSHA256"))
        return mac.doFinal(payload)
    }
}

fun main() {
    val key = "m7b2-test-secret-material-32bytes".toByteArray()
    val requestDigest = "a".repeat(64)
    val payload = canonicalApprovalPayload(
        approvalId = "00112233445566778899aabbccddeeff",
        issuer = "local-user",
        principal = "runtime.user",
        requestDigest = requestDigest,
        issuedNs = 100,
        expiresNs = 200,
    )
    check(payload.toString(Charsets.UTF_8) ==
        "{\"approval_id\":\"00112233445566778899aabbccddeeff\",\"expires_ns\":200,\"issued_ns\":100,\"issuer\":\"local-user\",\"principal\":\"runtime.user\",\"request_digest\":\"${"a".repeat(64)}\"}")
    check(JvmMac(key).hmacSha256(payload).toHex() ==
        "16006c3472602963c8d2a42cb03c3e331974d96ff6a76bc1ff60c3313f0923a3")

    val authority = M6ApprovalAuthority(JvmMac(key))
    val vector = M6ApprovalToken(
        approvalId = "00112233445566778899aabbccddeeff",
        issuer = "local-user",
        principal = "runtime.user",
        requestDigest = requestDigest,
        issuedNs = 100,
        expiresNs = 200,
        signature = "16006c3472602963c8d2a42cb03c3e331974d96ff6a76bc1ff60c3313f0923a3",
    )
    authority.verify(vector, requestDigest, "runtime.user", 150)

    val coordinator = M6ApprovalCoordinator(authority, maxPendingPrompts = 1)
    val prompt = coordinator.createPrompt(
        requestDigest,
        "runtime.user",
        "{\"request_digest\":\"$requestDigest\"}",
        1000,
        10,
    )
    check(coordinator.pendingCount(20) == 1)
    val token = coordinator.resolve(prompt, true, 100, 20)!!
    authority.verify(token, requestDigest, "runtime.user", 21)
    check(coordinator.pendingCount(21) == 0)
    try {
        coordinator.resolve(prompt, true, 100, 22)
        error("double resolve must fail")
    } catch (_: M6ApprovalException) {
    }


    val concurrent = coordinator.createPrompt(requestDigest, "runtime.user", "{}", 1000, 40)
    val results = java.util.Collections.synchronizedList(mutableListOf<M6ApprovalToken?>())
    val threads = List(2) {
        Thread {
            try {
                results += coordinator.resolve(concurrent, true, 100, 41)
            } catch (_: M6ApprovalException) {
                results += null
            }
        }
    }
    threads.forEach(Thread::start)
    threads.forEach(Thread::join)
    check(results.count { it != null } == 1)

    val denied = coordinator.createPrompt(requestDigest, "runtime.user", "{}", 100, 50)
    check(coordinator.resolve(denied, false, 1, 51) == null)
    println("M7B2_APPROVAL_COMPAT_PASS")
}
