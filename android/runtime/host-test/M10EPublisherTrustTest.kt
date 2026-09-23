package ai.vn97.runtime

private fun hex(value: String): ByteArray =
    ByteArray(value.length / 2) { index ->
        value.substring(index * 2, index * 2 + 2).toInt(16).toByte()
    }

private inline fun expectFailure(block: () -> Unit) {
    var failed = false
    try {
        block()
    } catch (_: VN97CapabilityTrustException) {
        failed = true
    }
    check(failed)
}

fun main() {
    val publicKey1 = hex(
        "d75a980182b10ab7d54bfed3c964073a" +
            "0ee172f3daa62325af021a68f707511a"
    )
    val signature1 = hex(
        "e5564300c360ac729086e2cc806e828a" +
            "84877f1eb8e5d974d873e06522490155" +
            "5fb8821590a33bacc61e39701cf9b46b" +
            "d25bf5f0595bbe24655141438e7a100b"
    )
    check(
        VN97Ed25519Verifier.verify(
            publicKey1,
            byteArrayOf(),
            signature1,
        )
    )

    val publicKey2 = hex(
        "3d4017c3e843895a92b70aa74d1b7ebc" +
            "9c982ccf2ec4968cc0cd55f12af4660c"
    )
    val signature2 = hex(
        "92a009a9f0d4cab8720e820b5f642540" +
            "a2b27b5416503f8fb3762223ebdb69da" +
            "085ac1e43e15996e458f3613d0f11d8" +
            "c387b2eaeb4302aeeb00d291612bb0c00"
    )
    check(
        VN97Ed25519Verifier.verify(
            publicKey2,
            byteArrayOf(0x72),
            signature2,
        )
    )

    val tampered = signature2.copyOf()
    tampered[0] = (tampered[0].toInt() xor 1).toByte()
    check(
        !VN97Ed25519Verifier.verify(
            publicKey2,
            byteArrayOf(0x72),
            tampered,
        )
    )

    val oversizedScalar = signature1.copyOf()
    for (index in 32 until 64) {
        oversizedScalar[index] = 0xff.toByte()
    }
    check(
        !VN97Ed25519Verifier.verify(
            publicKey1,
            byteArrayOf(),
            oversizedScalar,
        )
    )

    val identityKey = ByteArray(32).also { it[0] = 1 }
    check(
        !VN97Ed25519Verifier.verify(
            identityKey,
            byteArrayOf(),
            signature1,
        )
    )

    val manifest = VN97CapabilityManifest(
        capabilityId = "model.language",
        capabilityVersion = 7L,
        kind = "weights",
        source = VN97CapabilitySource(
            origin = "local",
            sourceSha256 = "11".repeat(32),
            license = "test",
        ),
        sections = emptyList(),
    )
    val key = VN97TrustedPublisherKey(
        keyId = "owner",
        publicKey = publicKey1,
        capabilityPrefixes = setOf("model"),
        allowedKinds = setOf("weights"),
        minVersion = 1L,
        maxVersion = 10L,
    )
    check(key.permits(manifest))
    check(
        !key.permits(
            manifest.copy(capabilityId = "modelevil.language")
        )
    )
    check(
        !key.permits(
            manifest.copy(kind = "tokenizer")
        )
    )
    check(
        !key.permits(
            manifest.copy(capabilityVersion = 11L)
        )
    )

    val revoked = VN97TrustedPublisherKey(
        keyId = "revoked",
        publicKey = publicKey1,
        capabilityPrefixes = setOf("model"),
        allowedKinds = setOf("weights"),
        revoked = true,
    )
    check(!revoked.permits(manifest))

    val store = VN97CapabilityTrustStore(listOf(key, revoked))
    check(store.require("owner") === key)
    expectFailure { store.require("missing") }
    expectFailure { VN97CapabilityTrustStore(listOf(key, key)) }

    val leaked = key.publicKey()
    leaked[0] = (leaked[0].toInt() xor 1).toByte()
    check(key.publicKey().contentEquals(publicKey1))

    println("M10E_PUBLISHER_TRUST_PASS")
}
