import ai.vn97.runtime.*

private inline fun expectFailure(block: () -> Unit) {
    var failed = false
    try {
        block()
    } catch (_: NativeActivatedInventoryException) {
        failed = true
    }
    check(failed)
}

private val shaA = "11".repeat(32)
private val shaB = "22".repeat(32)
private val shaC = "33".repeat(32)
private val shaD = "44".repeat(32)
private val shaE = "55".repeat(32)
private val shaF = "66".repeat(32)

private fun inventory(
    backend: String = "vn97.model_image",
    pending: String = "null",
    artifactSha: String = shaB,
    capabilityVersion: Long = 1L,
): ByteArray {
    val text =
        """{"generation":1,"history":[{"action":"activate","activation_id":"$shaA","capability_id":"model.language","generation":1,"package_sha256":"$shaC"}],"pending":$pending,"schema":"VN97INV1","stacks":[{"capability_id":"model.language","records":[{"activation_id":"$shaA","artifact_sha256":"$artifactSha","backend_id":"$backend","backend_token":"tok-1","capability_id":"model.language","capability_version":$capabilityVersion,"package_sha256":"$shaC","plan_sha256":"$shaD","profile_id":"mobile","profile_sha256":"$shaE","publisher_key_id":"publisher","runtime_api_version":1,"runtime_revision":"rev-1","signature_sha256":"$shaF","source_license":"test","source_origin":"local","source_sha256":"$shaC"}]}]}"""
    return text.toByteArray(Charsets.UTF_8)
}

fun main() {
    val evidence = checkNotNull(
        VN97ActivatedModelInventoryEvidence.parse(inventory())
    )
    check(evidence.capabilityId == "model.language")
    check(evidence.backendId == "vn97.model_image")
    check(evidence.artifactSha256 == shaB)
    check(evidence.generation == 1L)
    checkNotNull(
        VN97ActivatedModelInventoryEvidence.parse(
            inventory(capabilityVersion = 0xffff_ffffL)
        )
    )

    check(
        VN97ActivatedModelInventoryEvidence.parse(
            """{"generation":0,"history":[],"pending":null,"schema":"VN97INV1","stacks":[]}"""
                .toByteArray()
        ) == null
    )

    expectFailure {
        VN97ActivatedModelInventoryEvidence.parse(
            inventory(backend = "other.backend")
        )
    }
    expectFailure {
        VN97ActivatedModelInventoryEvidence.parse(
            inventory(pending = "{}")
        )
    }
    expectFailure {
        VN97ActivatedModelInventoryEvidence.parse(
            inventory(artifactSha = "xyz")
        )
    }
    expectFailure {
        VN97ActivatedModelInventoryEvidence.parse(
            (" " + inventory().toString(Charsets.UTF_8)).toByteArray()
        )
    }

    println("M10B_INVENTORY_EVIDENCE_PASS")
}
