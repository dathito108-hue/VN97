import ai.vn97.app.*

fun main() {
    check(
        VN97BootstrapAssetContract.classify(emptySet()) ==
            VN97BootstrapAssetState.ABSENT
    )
    check(
        VN97BootstrapAssetContract.classify(setOf("README.txt")) ==
            VN97BootstrapAssetState.ABSENT
    )
    check(
        VN97BootstrapAssetContract.classify(
            setOf("model.vn97cap1")
        ) == VN97BootstrapAssetState.INCOMPLETE
    )
    check(
        VN97BootstrapAssetContract.classify(
            setOf(
                "model.vn97cap1",
                "model.vn97sig1",
                "publisher.ed25519",
            )
        ) == VN97BootstrapAssetState.COMPLETE
    )
    check(
        VN97BootstrapAssetContract.requiredEntries.size == 3
    )

    println("M10J_BOOTSTRAP_ASSET_CONTRACT_PASS")
}
