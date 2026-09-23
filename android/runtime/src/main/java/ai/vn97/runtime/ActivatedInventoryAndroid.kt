package ai.vn97.runtime

import android.os.ParcelFileDescriptor
import java.io.File
import java.io.FileInputStream
import java.nio.file.Files
import java.nio.file.LinkOption
import java.security.MessageDigest

private const val VN97_MODEL_IMAGE_MAX_BYTES = 512L * 1024L * 1024L

object NativeActivatedInventoryModelLoader {
    fun openOrNull(
        root: File,
        capabilityId: String =
            VN97ActivatedModelInventoryEvidence.DEFAULT_CAPABILITY_ID,
        backendId: String =
            VN97ActivatedModelInventoryEvidence.DEFAULT_BACKEND_ID,
    ): NativeActivatedModel? {
        val rootPath = root.toPath()
        if (!Files.exists(rootPath, LinkOption.NOFOLLOW_LINKS)) return null
        if (!Files.isDirectory(rootPath, LinkOption.NOFOLLOW_LINKS) ||
            Files.isSymbolicLink(rootPath)
        ) {
            throw NativeActivatedInventoryException(
                "activated model inventory root must be a real directory"
            )
        }

        val inventory = root.resolve("inventory.vn97inv1.json")
        if (!Files.exists(inventory.toPath(), LinkOption.NOFOLLOW_LINKS)) return null
        if (!Files.isRegularFile(inventory.toPath(), LinkOption.NOFOLLOW_LINKS) ||
            Files.isSymbolicLink(inventory.toPath())
        ) {
            throw NativeActivatedInventoryException(
                "VN97INV1 inventory must be a non-symlink regular file"
            )
        }
        val inventoryBytes = try {
            Files.readAllBytes(inventory.toPath())
        } catch (exc: Exception) {
            throw NativeActivatedInventoryException(
                "VN97INV1 inventory could not be read",
                exc,
            )
        }
        val evidence = VN97ActivatedModelInventoryEvidence.parse(
            inventoryBytes,
            capabilityId = capabilityId,
            backendId = backendId,
        ) ?: return null

        val artifacts = root.resolve("artifacts")
        if (!Files.isDirectory(artifacts.toPath(), LinkOption.NOFOLLOW_LINKS) ||
            Files.isSymbolicLink(artifacts.toPath())
        ) {
            throw NativeActivatedInventoryException(
                "activated model artifact directory is missing or unsafe"
            )
        }
        val image = artifacts.resolve("${evidence.artifactSha256}.vn97mi1")
        if (!Files.isRegularFile(image.toPath(), LinkOption.NOFOLLOW_LINKS) ||
            Files.isSymbolicLink(image.toPath())
        ) {
            throw NativeActivatedInventoryException(
                "active VN97 model artifact is missing or unsafe"
            )
        }

        ParcelFileDescriptor.open(
            image,
            ParcelFileDescriptor.MODE_READ_ONLY,
        ).use { pfd ->
            val length = pfd.statSize
            if (length <= 0L || length > VN97_MODEL_IMAGE_MAX_BYTES) {
                throw NativeActivatedInventoryException(
                    "active VN97 model artifact size is outside bounds"
                )
            }
            val digest = hashDescriptor(pfd)
            val digestHex = digest.joinToString("") {
                "%02x".format(it.toInt() and 0xff)
            }
            if (digestHex != evidence.artifactSha256) {
                throw NativeActivatedInventoryException(
                    "active VN97 model artifact SHA-256 does not match VN97INV1"
                )
            }
            val trusted = M9ActivatedModelBridge.committedArtifact(
                fd = pfd.fd,
                offset = 0L,
                length = length,
                artifactSha256 = digest,
            )
            return NativeActivatedModel.open(trusted).also { model ->
                if (!model.info.modelId.contentEquals(digest)) {
                    model.close()
                    throw NativeActivatedInventoryException(
                        "activated native model identity changed after inventory verification"
                    )
                }
            }
        }
    }

    private fun hashDescriptor(pfd: ParcelFileDescriptor): ByteArray {
        val digest = MessageDigest.getInstance("SHA-256")
        ParcelFileDescriptor.dup(pfd.fileDescriptor).use { duplicate ->
            ParcelFileDescriptor.AutoCloseInputStream(duplicate).use { input ->
                val buffer = ByteArray(64 * 1024)
                while (true) {
                    val read = input.read(buffer)
                    if (read < 0) break
                    if (read == 0) {
                        throw NativeActivatedInventoryException(
                            "active VN97 model artifact hash read made no progress"
                        )
                    }
                    digest.update(buffer, 0, read)
                }
            }
        }
        return digest.digest()
    }
}
