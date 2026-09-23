package ai.vn97.app

internal enum class VN97BootstrapAssetState {
    ABSENT,
    COMPLETE,
    INCOMPLETE,
}

internal object VN97BootstrapAssetContract {
    val requiredEntries: Set<String> = setOf(
        "model.vn97cap1",
        "model.vn97sig1",
        "publisher.ed25519",
    )

    fun classify(entries: Set<String>): VN97BootstrapAssetState {
        val present = entries.intersect(requiredEntries)
        return when {
            present.isEmpty() -> VN97BootstrapAssetState.ABSENT
            present == requiredEntries -> VN97BootstrapAssetState.COMPLETE
            else -> VN97BootstrapAssetState.INCOMPLETE
        }
    }
}
