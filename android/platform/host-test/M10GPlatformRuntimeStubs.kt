package ai.vn97.runtime

import java.io.File

fun interface VN97ModelImageCandidateValidator {
    fun validate(candidate: File, artifactSha256: ByteArray)
}

data class NativeActivatedModelInfo(
    val modelId: ByteArray,
    val imageBytes: Long,
)

object NativeModelImageCandidateValidator {
    fun validate(
        fd: Int,
        offset: Long,
        length: Long,
        artifactSha256: ByteArray,
    ): NativeActivatedModelInfo {
        @Suppress("UNUSED_VARIABLE") val ignored = listOf(fd, offset)
        return NativeActivatedModelInfo(artifactSha256.copyOf(), length)
    }
}

data class VN97FormatRule(
    val role: String,
    val acceptedFormats: List<String>,
)

data class VN97CompatibilityProfile(
    val profileId: String,
    val formatRules: List<VN97FormatRule>,
)

class VN97ModelImageActivationBackend(
    root: File,
    validator: VN97ModelImageCandidateValidator,
) {
    init {
        @Suppress("UNUSED_VARIABLE") val ignored = listOf(root, validator)
    }

    companion object {
        fun productionProfile() = VN97CompatibilityProfile(
            "vn97.android.model-image.v1",
            listOf(VN97FormatRule("model_image", listOf("VN97MI1"))),
        )
    }
}

class VN97CapabilityInventoryStore(root: File) {
    init {
        @Suppress("UNUSED_VARIABLE") val ignored = root
    }
}

class VN97CapabilityActivationCoordinator(
    store: VN97CapabilityInventoryStore,
) {
    init {
        @Suppress("UNUSED_VARIABLE") val ignored = store
    }
}

class VN97CapabilityStager(root: File) {
    init {
        @Suppress("UNUSED_VARIABLE") val ignored = root
    }
}
