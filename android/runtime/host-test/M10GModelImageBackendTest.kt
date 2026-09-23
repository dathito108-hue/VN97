package ai.vn97.runtime

import java.nio.file.Files
import java.security.MessageDigest

private fun digest(bytes: ByteArray): String =
    MessageDigest.getInstance("SHA-256").digest(bytes)
        .joinToString("") { "%02x".format(it.toInt() and 0xff) }

private inline fun expectFailure(block: () -> Unit) {
    var failed = false
    try { block() } catch (_: VN97CapabilityActivationException) { failed = true }
    check(failed)
}

fun main() {
    val root = Files.createTempDirectory("m10g-root-").toFile()
    val stage = Files.createTempDirectory("m10g-stage-").toFile()
    val prefix = ByteArray(128) { 7 }
    val image = "VN97MI1-production-candidate".toByteArray()
    val suffix = ByteArray(31) { 9 }
    val packageBytes = prefix + image + suffix
    val packageFile = stage.resolve("package.vn97cap1")
    packageFile.writeBytes(packageBytes)
    val imageSha = digest(image)
    val packageSha = digest(packageBytes)

    val section = VN97CapabilitySection(
        index = 1,
        role = "model_image",
        format = "VN97MI1",
        size = image.size.toLong(),
        sha256 = imageSha,
        packageOffset = prefix.size.toLong(),
    )
    val manifest = VN97CapabilityManifest(
        capabilityId = "model.language",
        capabilityVersion = 1,
        kind = "weights",
        source = VN97CapabilitySource("local", "11".repeat(32), "test"),
        sections = listOf(section),
    )
    val parsed = VN97ParsedCapabilityPackage(packageSha, manifest, packageBytes.size.toLong())
    val staged = VN97StagedCapability(parsed, packageFile = packageFile)
    val verified = VN97VerifiedCapability(staged, parsed, "owner", "22".repeat(32))
    val plan = VN97CompatibilityPlan(
        packageSha256 = packageSha,
        capabilityId = "model.language",
        capabilityVersion = 1,
        publisherKeyId = "owner",
        signatureSha256 = "22".repeat(32),
        profileId = "vn97.android.model-image.v1",
        profileSha256 = "33".repeat(32),
        runtimeApiVersion = 1,
        disposition = VN97CompatibilityDisposition.DIRECT,
        sections = listOf(
            VN97SectionPlan(1, "model_image", "VN97MI1", "VN97MI1", null, false)
        ),
    )

    var validations = 0
    val backend = VN97ModelImageActivationBackend(
        root,
        VN97ModelImageCandidateValidator { file, sha ->
            validations++
            check(file.readBytes().contentEquals(image))
            check(sha.joinToString("") { "%02x".format(it.toInt() and 0xff) } == imageSha)
        },
    )
    val profile = VN97ModelImageActivationBackend.productionProfile()
    check(profile.profileId == "vn97.android.model-image.v1")
    check(profile.formatRules.single().acceptedFormats == listOf("VN97MI1"))

    val prepared = backend.prepare(verified, plan)
    check(prepared.backendId == "vn97.model_image")
    check(prepared.artifactSha256 == imageSha)
    check(backend.inspect(prepared.token).state == VN97BackendTransactionState.PREPARED)
    check(validations == 1)

    val reopened = VN97ModelImageActivationBackend(root, VN97ModelImageCandidateValidator { file, _ ->
        check(file.readBytes().contentEquals(image))
    })
    check(reopened.inspect(prepared.token).state == VN97BackendTransactionState.PREPARED)
    val revision = reopened.commit(prepared.token)
    check(revision == "mi1-$imageSha")
    val status = reopened.inspect(prepared.token)
    check(status.state == VN97BackendTransactionState.COMMITTED)
    check(status.runtimeRevision == revision)
    val artifact = root.resolve("artifacts/$imageSha.vn97mi1")
    check(artifact.isFile)
    check(artifact.readBytes().contentEquals(image))

    val restarted = VN97ModelImageActivationBackend(root, VN97ModelImageCandidateValidator { _, _ -> })
    check(restarted.commit(prepared.token) == revision)

    restarted.rollback(prepared.token)
    check(restarted.inspect(prepared.token).state == VN97BackendTransactionState.ROLLED_BACK)
    check(artifact.isFile)
    restarted.rollback(prepared.token)

    expectFailure {
        restarted.prepare(
            verified,
            plan.copy(disposition = VN97CompatibilityDisposition.ADAPT_REQUIRED),
        )
    }
    expectFailure {
        restarted.prepare(
            verified.copy(parsed = parsed.copy(manifest = manifest.copy(capabilityId = "model.other"))),
            plan.copy(capabilityId = "model.other"),
        )
    }

    val tampered = packageBytes.copyOf()
    tampered[prefix.size] = (tampered[prefix.size].toInt() xor 1).toByte()
    packageFile.writeBytes(tampered)
    expectFailure { restarted.prepare(verified, plan) }

    check(!root.resolve("inventory.vn97inv1.json").exists())
    println("M10G_MODEL_IMAGE_BACKEND_PASS")
}
