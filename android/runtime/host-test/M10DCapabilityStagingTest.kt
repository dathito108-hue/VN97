package ai.vn97.runtime

import java.io.ByteArrayInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.security.MessageDigest
import java.util.zip.CRC32
import kotlin.io.path.createTempDirectory

private fun sha(bytes: ByteArray): ByteArray =
    MessageDigest.getInstance("SHA-256").digest(bytes)

private fun hex(bytes: ByteArray): String =
    bytes.joinToString("") { "%02x".format(it.toInt() and 0xff) }

private fun buildPackage(
    role: String = "model_image",
    format: String = "VN97MI1",
): ByteArray {
    val payload = "VN97MI1-test-model".toByteArray()
    val payloadSha = hex(sha(payload))
    val sourceSha = "33".repeat(32)

    val manifest = VnStrictJson.canonical(
        VnStrictJson.objectOf(
            "capability_id" to VnStrictJson.string("model.language"),
            "capability_version" to VnJsonNumber("1"),
            "kind" to VnStrictJson.string("weights"),
            "schema" to VnStrictJson.string("VN97CAP1"),
            "sections" to VnStrictJson.array(
                listOf(
                    VnStrictJson.objectOf(
                        "format" to VnStrictJson.string(format),
                        "index" to VnJsonNumber("1"),
                        "role" to VnStrictJson.string(role),
                        "sha256" to VnStrictJson.string(payloadSha),
                        "size" to VnJsonNumber(payload.size.toString()),
                    )
                )
            ),
            "source" to VnStrictJson.objectOf(
                "license" to VnStrictJson.string("test"),
                "origin" to VnStrictJson.string("local"),
                "source_sha256" to VnStrictJson.string(sourceSha),
            ),
        )
    ).toByteArray()

    val sections = listOf(manifest, payload)
    val count = sections.size
    val tableBytes = count * 64
    val payloadOffset = 96 + tableBytes
    val total = payloadOffset + sections.sumOf { it.size }

    val table = ByteBuffer.allocate(tableBytes)
        .order(ByteOrder.LITTLE_ENDIAN)
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

    val content = table.array() + sections.fold(ByteArray(0)) { acc, value ->
        acc + value
    }
    val header = ByteBuffer.allocate(96)
        .order(ByteOrder.LITTLE_ENDIAN)
    header.put("VN97CAP1".toByteArray())
    header.putShort(1.toShort())
    header.putShort(96.toShort())
    header.putInt(0)
    header.putInt(count)
    header.putInt(64)
    header.putLong(96L)
    header.putLong(payloadOffset.toLong())
    header.putLong(total.toLong())
    header.putInt(0)
    header.putInt(0)
    header.put(sha(content))
    val crc = CRC32().apply {
        update(header.array(), 0, 88)
    }.value
    header.putInt(crc.toInt())
    header.putInt(0)
    return header.array() + content
}

private fun buildSignature(
    packageBytes: ByteArray,
    capabilityId: String = "model.language",
): ByteArray {
    val packageSha = hex(sha(packageBytes))
    return VnStrictJson.canonical(
        VnStrictJson.objectOf(
            "algorithm" to VnStrictJson.string("ed25519"),
            "capability_id" to VnStrictJson.string(capabilityId),
            "capability_version" to VnJsonNumber("1"),
            "key_id" to VnStrictJson.string("owner"),
            "package_sha256" to VnStrictJson.string(packageSha),
            "schema" to VnStrictJson.string("VN97SIG1"),
            "signature" to VnStrictJson.string("aa".repeat(64)),
        )
    ).toByteArray()
}

private inline fun expectFailure(block: () -> Unit) {
    var failed = false
    try {
        block()
    } catch (_: IllegalStateException) {
        failed = true
    }
    check(failed)
}

fun main() {
    val packageBytes = buildPackage()
    val signatureBytes = buildSignature(packageBytes)

    val parsed = VN97CapabilityPackageParser.parse(packageBytes)
    check(parsed.manifest.capabilityId == "model.language")
    check(parsed.manifest.capabilityVersion == 1L)
    check(parsed.manifest.kind == "weights")
    check(parsed.manifest.sections.single().role == "model_image")
    check(parsed.manifest.sections.single().format == "VN97MI1")

    val signature = VN97CapabilitySignatureParser.parse(signatureBytes)
    check(signature.packageSha256 == parsed.packageSha256)
    val signingPrefix = "VN97CAP1-SIGNATURE-V1\u0000".toByteArray()
    check(
        signature.signingMessage()
            .copyOfRange(0, signingPrefix.size)
            .contentEquals(signingPrefix)
    )

    val root = createTempDirectory("m10d-stage-").toFile()
    val stager = VN97CapabilityStager(root)
    val staged = stager.stage(
        ByteArrayInputStream(packageBytes),
        signatureBytes,
    )
    check(staged.packageFile.name == parsed.packageSha256 + ".vn97cap1")
    check(
        staged.signatureFile.name ==
            parsed.packageSha256 + ".owner.vn97sig1"
    )
    check(staged.packageFile.isFile)
    check(staged.signatureFile.isFile)
    check(!root.resolve("inventory.vn97inv1.json").exists())

    val stagedAgain = stager.stage(
        ByteArrayInputStream(packageBytes),
        signatureBytes,
    )
    check(stagedAgain.packageFile == staged.packageFile)
    check(stagedAgain.signatureFile == staged.signatureFile)

    val tampered = packageBytes.copyOf().also {
        it[it.lastIndex] = (it.last().toInt() xor 1).toByte()
    }
    expectFailure {
        VN97CapabilityPackageParser.parse(tampered)
    }

    expectFailure {
        VN97CapabilityStager(root).stage(
            ByteArrayInputStream(packageBytes),
            buildSignature(packageBytes, capabilityId = "model.other"),
        )
    }

    expectFailure {
        VN97CapabilitySignatureParser.parse(
            (" " + signatureBytes.toString(Charsets.UTF_8)).toByteArray()
        )
    }

    expectFailure {
        VN97CapabilityPackageParser.parse(
            buildPackage(role = "native-library")
        )
    }

    println("M10D_ANDROID_CAPABILITY_STAGING_PASS")
}
