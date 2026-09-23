package ai.vn97.runtime

import java.io.ByteArrayInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.security.KeyPairGenerator
import java.security.MessageDigest
import java.security.Signature
import java.util.zip.CRC32
import kotlin.io.path.createTempDirectory

private fun sha(bytes: ByteArray): ByteArray =
    MessageDigest.getInstance("SHA-256").digest(bytes)

private fun hex(bytes: ByteArray): String =
    bytes.joinToString("") { "%02x".format(it.toInt() and 0xff) }

private fun modelPackage(payload: ByteArray): ByteArray {
    val payloadSha = hex(sha(payload))
    val manifest = VnStrictJson.canonical(
        VnStrictJson.objectOf(
            "capability_id" to VnStrictJson.string("model.language"),
            "capability_version" to VnJsonNumber("1"),
            "kind" to VnStrictJson.string("weights"),
            "schema" to VnStrictJson.string("VN97CAP1"),
            "sections" to VnStrictJson.array(
                listOf(
                    VnStrictJson.objectOf(
                        "format" to VnStrictJson.string("VN97MI1"),
                        "index" to VnJsonNumber("1"),
                        "role" to VnStrictJson.string("model_image"),
                        "sha256" to VnStrictJson.string(payloadSha),
                        "size" to VnJsonNumber(payload.size.toString()),
                    )
                )
            ),
            "source" to VnStrictJson.objectOf(
                "license" to VnStrictJson.string("test"),
                "origin" to VnStrictJson.string("local-test"),
                "source_sha256" to VnStrictJson.string("33".repeat(32)),
            ),
        )
    ).toByteArray()

    val sections = listOf(manifest, payload)
    val tableBytes = sections.size * 64
    val payloadOffset = 96 + tableBytes
    val totalSize = payloadOffset + sections.sumOf { it.size }
    val table = ByteBuffer.allocate(tableBytes).order(ByteOrder.LITTLE_ENDIAN)
    var cursor = payloadOffset.toLong()
    sections.forEachIndexed { index, section ->
        table.putInt(if (index == 0) 1 else 2)
        table.putInt(0)
        table.putLong(cursor)
        table.putLong(section.size.toLong())
        table.put(sha(section))
        table.putLong(0L)
        cursor += section.size
    }
    val content = table.array() +
        sections.fold(ByteArray(0)) { acc, section -> acc + section }
    val header = ByteBuffer.allocate(96).order(ByteOrder.LITTLE_ENDIAN)
    header.put("VN97CAP1".toByteArray())
    header.putShort(1.toShort())
    header.putShort(96.toShort())
    header.putInt(0)
    header.putInt(sections.size)
    header.putInt(64)
    header.putLong(96L)
    header.putLong(payloadOffset.toLong())
    header.putLong(totalSize.toLong())
    header.putInt(0)
    header.putInt(0)
    header.put(sha(content))
    val crc = CRC32().apply { update(header.array(), 0, 88) }.value
    header.putInt(crc.toInt())
    header.putInt(0)
    return header.array() + content
}

private fun signatureEnvelope(
    packageBytes: ByteArray,
    privateKey: java.security.PrivateKey,
): ByteArray {
    val packageSha = hex(sha(packageBytes))
    val claims = VnStrictJson.objectOf(
        "algorithm" to VnStrictJson.string("ed25519"),
        "capability_id" to VnStrictJson.string("model.language"),
        "capability_version" to VnJsonNumber("1"),
        "key_id" to VnStrictJson.string("owner"),
        "package_sha256" to VnStrictJson.string(packageSha),
        "schema" to VnStrictJson.string("VN97SIG1"),
    )
    val message = "VN97CAP1-SIGNATURE-V1\u0000".toByteArray() +
        VnStrictJson.canonical(claims).toByteArray()
    val signer = Signature.getInstance("Ed25519")
    signer.initSign(privateKey)
    signer.update(message)
    val signature = signer.sign()
    val values = claims.values.toMutableMap()
    values["signature"] = VnStrictJson.string(hex(signature))
    return VnStrictJson.canonical(VnJsonObject(values)).toByteArray()
}

fun main() {
    val root = createTempDirectory("m10h-provision-").toFile()
    val stageRoot = root.resolve("stage").apply { mkdirs() }
    val payload = "VN97MI1-host-candidate".toByteArray()
    val packageBytes = modelPackage(payload)

    val keyPair = KeyPairGenerator.getInstance("Ed25519").generateKeyPair()
    val encodedPublic = keyPair.public.encoded
    val rawPublic = encodedPublic.copyOfRange(
        encodedPublic.size - 32,
        encodedPublic.size,
    )
    val signatureBytes = signatureEnvelope(packageBytes, keyPair.private)

    var validatorCalls = 0
    val backend = VN97ModelImageActivationBackend(
        root = root,
        validator = VN97ModelImageCandidateValidator { candidate, artifactSha ->
            validatorCalls += 1
            check(candidate.readBytes().contentEquals(payload))
            check(sha(payload).contentEquals(artifactSha))
        },
    )
    val coordinator = VN97CapabilityActivationCoordinator(
        VN97CapabilityInventoryStore(root)
    )
    val session = VN97ModelImageProvisioningSession(
        stager = VN97CapabilityStager(stageRoot),
        stageRoot = stageRoot,
        profile = VN97ModelImageActivationBackend.productionProfile(),
        backend = backend,
        coordinator = coordinator,
    )

    check(session.recover().active.isEmpty())
    val review = session.review(
        packageInput = ByteArrayInputStream(packageBytes),
        signatureBytes = signatureBytes,
        publisherPublicKey = rawPublic,
    )
    check(review.capabilityId == "model.language")
    check(review.capabilityVersion == 1L)
    check(review.publisherKeyId == "owner")
    check(review.publisherKeySha256 == hex(sha(rawPublic)))
    check(coordinator.inventory().active.isEmpty())
    check(session.pendingReview() == review)

    val activated = session.activateReviewed()
    check(activated.capabilityId == "model.language")
    check(activated.artifactSha256 == hex(sha(payload)))
    check(coordinator.inventory().current("model.language") == activated)
    check(session.pendingReview() == null)
    check(validatorCalls >= 2)

    val inventoryBytes = root.resolve(VN97_INVENTORY_FILE).readBytes()
    val evidence = checkNotNull(
        VN97ActivatedModelInventoryEvidence.parse(inventoryBytes)
    )
    check(evidence.artifactSha256 == activated.artifactSha256)
    check(
        root.resolve("artifacts/${activated.artifactSha256}.vn97mi1")
            .readBytes()
            .contentEquals(payload)
    )

    println("M10H_TRUSTED_IMPORT_ACTIVATION_PASS")
}
