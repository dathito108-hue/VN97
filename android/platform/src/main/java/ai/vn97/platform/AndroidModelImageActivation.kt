package ai.vn97.platform

import ai.vn97.runtime.NativeModelImageCandidateValidator
import ai.vn97.runtime.VN97ModelImageCandidateValidator
import android.os.ParcelFileDescriptor
import java.io.File
import java.nio.file.Files
import java.nio.file.LinkOption

internal object AndroidVN97ModelImageCandidateValidator :
    VN97ModelImageCandidateValidator {
    override fun validate(
        candidate: File,
        artifactSha256: ByteArray,
    ) {
        val path = candidate.toPath()
        require(
            Files.isRegularFile(path, LinkOption.NOFOLLOW_LINKS) &&
                !Files.isSymbolicLink(path)
        ) {
            "VN97MI1 candidate must be a non-symlink regular file"
        }
        ParcelFileDescriptor.open(
            candidate,
            ParcelFileDescriptor.MODE_READ_ONLY,
        ).use { descriptor ->
            val size = descriptor.statSize
            require(size > 0L && size == candidate.length()) {
                "VN97MI1 candidate descriptor size changed"
            }
            val info = NativeModelImageCandidateValidator.validate(
                fd = descriptor.fd,
                offset = 0L,
                length = size,
                artifactSha256 = artifactSha256.copyOf(),
            )
            require(info.modelId.contentEquals(artifactSha256)) {
                "native VN97MI1 candidate model identity mismatch"
            }
            require(info.imageBytes == size) {
                "native VN97MI1 candidate byte length mismatch"
            }
        }
    }
}

class AndroidVN97CapabilityProvisioner(
    context: android.content.Context,
) {
    private val appContext = context.applicationContext
    val capabilityRoot: File = File(
        appContext.noBackupFilesDir,
        "vn97-capabilities",
    )

    val stageRoot: File = File(capabilityRoot, "stage")
    val compatibilityProfile =
        ai.vn97.runtime.VN97ModelImageActivationBackend.productionProfile()

    val activationBackend: ai.vn97.runtime.VN97ModelImageActivationBackend
    val activationCoordinator: ai.vn97.runtime.VN97CapabilityActivationCoordinator

    init {
        requireDirectory(capabilityRoot, "VN97 capability root")
        requireDirectory(stageRoot, "VN97 capability stage root")
        activationBackend = ai.vn97.runtime.VN97ModelImageActivationBackend(
            root = capabilityRoot,
            validator = AndroidVN97ModelImageCandidateValidator,
        )
        activationCoordinator = ai.vn97.runtime.VN97CapabilityActivationCoordinator(
            ai.vn97.runtime.VN97CapabilityInventoryStore(capabilityRoot)
        )
    }

    fun createStager(): ai.vn97.runtime.VN97CapabilityStager =
        ai.vn97.runtime.VN97CapabilityStager(stageRoot)

    fun currentModelActivation():
        ai.vn97.runtime.VN97CapabilityInventoryItem? =
        ai.vn97.runtime.VN97CapabilityInventoryStore(
            capabilityRoot
        ).load().current(
            ai.vn97.runtime.VN97ModelImageActivationBackend
                .CAPABILITY_ID
        )

    private fun requireDirectory(file: File, label: String) {
        if (!file.exists()) {
            check(file.mkdirs()) { "failed to create $label" }
        }
        val path = file.toPath()
        require(
            Files.isDirectory(path, LinkOption.NOFOLLOW_LINKS) &&
                !Files.isSymbolicLink(path)
        ) {
            "$label must be a non-symlink directory"
        }
    }
}
