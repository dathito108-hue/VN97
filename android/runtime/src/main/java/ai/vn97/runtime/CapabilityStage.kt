package ai.vn97.runtime

import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.io.InputStream
import java.io.RandomAccessFile
import java.nio.channels.FileChannel
import java.nio.file.FileAlreadyExistsException
import java.nio.file.Files
import java.nio.file.LinkOption
import java.nio.file.StandardOpenOption
import java.security.MessageDigest

data class VN97StagedCapability(
    val parsed: VN97ParsedCapabilityPackage,
    val signature: VN97CapabilitySignatureEnvelope,
    val packageFile: File,
    val signatureFile: File,
)

class VN97CapabilityStager(
    private val root: File,
    private val maxPackageBytes: Long = VN97_CAP_MAX_PACKAGE_BYTES,
) {
    private val rootPath = root.toPath()

    init {
        require(maxPackageBytes >= 96L) {
            "maxPackageBytes is too small"
        }
        Files.createDirectories(rootPath)
        if (!Files.isDirectory(rootPath, LinkOption.NOFOLLOW_LINKS) ||
            Files.isSymbolicLink(rootPath)
        ) {
            throw VN97CapabilityPackageException(
                "capability stage root must be a real directory"
            )
        }
    }

    @Synchronized
    fun stage(
        packageInput: InputStream,
        signatureBytes: ByteArray,
    ): VN97StagedCapability {
        val signature = VN97CapabilitySignatureParser.parse(signatureBytes)
        val packageTemp = Files.createTempFile(
            rootPath,
            ".vn97cap-",
            ".tmp",
        )
        try {
            copyPackage(packageInput, packageTemp.toFile())
            val parsed = RandomAccessFile(packageTemp.toFile(), "r").use { file ->
                val size = file.length()
                if (size <= 0L || size > maxPackageBytes) {
                    throw VN97CapabilityPackageException(
                        "staged VN97CAP1 size is outside bounds"
                    )
                }
                val mapped = file.channel.map(
                    FileChannel.MapMode.READ_ONLY,
                    0L,
                    size,
                )
                VN97CapabilityPackageParser.parse(
                    mapped,
                    maxPackageBytes,
                )
            }

            requireSignatureBinding(parsed, signature)

            val packageTarget = root.resolve(
                parsed.packageSha256 + ".vn97cap1"
            )
            commitContentAddressedPackage(
                packageTemp.toFile(),
                packageTarget,
                parsed,
            )

            val signatureTarget = root.resolve(
                parsed.packageSha256 +
                    "." + signature.keyId +
                    ".vn97sig1"
            )
            commitContentAddressedSignature(
                signatureBytes,
                signatureTarget,
                signature,
            )

            fsyncDirectory()
            return VN97StagedCapability(
                parsed = parsed,
                signature = signature,
                packageFile = packageTarget,
                signatureFile = signatureTarget,
            )
        } finally {
            Files.deleteIfExists(packageTemp)
        }
    }

    fun stage(
        packageFile: File,
        signatureBytes: ByteArray,
    ): VN97StagedCapability =
        FileInputStream(packageFile).use { input ->
            stage(input, signatureBytes)
        }

    private fun copyPackage(input: InputStream, target: File) {
        FileOutputStream(target).use { output ->
            val buffer = ByteArray(64 * 1024)
            var total = 0L
            while (true) {
                val read = input.read(buffer)
                if (read < 0) break
                if (read == 0) {
                    throw VN97CapabilityPackageException(
                        "VN97CAP1 staging read made no progress"
                    )
                }
                total += read.toLong()
                if (total > maxPackageBytes) {
                    throw VN97CapabilityPackageException(
                        "VN97CAP1 exceeds stage byte limit"
                    )
                }
                output.write(buffer, 0, read)
            }
            if (total == 0L) {
                throw VN97CapabilityPackageException(
                    "VN97CAP1 package is empty"
                )
            }
            output.flush()
            output.fd.sync()
        }
    }

    private fun requireSignatureBinding(
        parsed: VN97ParsedCapabilityPackage,
        signature: VN97CapabilitySignatureEnvelope,
    ) {
        if (signature.packageSha256 != parsed.packageSha256 ||
            signature.capabilityId != parsed.manifest.capabilityId ||
            signature.capabilityVersion != parsed.manifest.capabilityVersion
        ) {
            throw VN97CapabilitySignatureException(
                "VN97SIG1 is not bound to staged VN97CAP1"
            )
        }
    }

    private fun commitContentAddressedPackage(
        temp: File,
        target: File,
        expected: VN97ParsedCapabilityPackage,
    ) {
        requireSafeTarget(target)
        if (target.exists()) {
            validateExistingPackage(target, expected)
            return
        }
        try {
            Files.createLink(target.toPath(), temp.toPath())
        } catch (_: FileAlreadyExistsException) {
            validateExistingPackage(target, expected)
        } catch (exc: Exception) {
            throw VN97CapabilityPackageException(
                "VN97CAP1 content-addressed commit failed",
                exc,
            )
        }
        validateExistingPackage(target, expected)
    }

    private fun validateExistingPackage(
        target: File,
        expected: VN97ParsedCapabilityPackage,
    ) {
        requireSafeTarget(target)
        if (!Files.isRegularFile(target.toPath(), LinkOption.NOFOLLOW_LINKS) ||
            Files.isSymbolicLink(target.toPath())
        ) {
            throw VN97CapabilityPackageException(
                "VN97CAP1 digest target is not a safe regular file"
            )
        }
        RandomAccessFile(target, "r").use { file ->
            if (file.length() != expected.size) {
                throw VN97CapabilityPackageException(
                    "VN97CAP1 digest target size mismatch"
                )
            }
            val mapped = file.channel.map(
                FileChannel.MapMode.READ_ONLY,
                0L,
                file.length(),
            )
            val actual = VN97CapabilityPackageParser.parse(
                mapped,
                maxPackageBytes,
            )
            if (actual.packageSha256 != expected.packageSha256 ||
                actual.manifest != expected.manifest
            ) {
                throw VN97CapabilityPackageException(
                    "VN97CAP1 digest target content mismatch"
                )
            }
        }
    }

    private fun commitContentAddressedSignature(
        bytes: ByteArray,
        target: File,
        expected: VN97CapabilitySignatureEnvelope,
    ) {
        requireSafeTarget(target)
        if (target.exists()) {
            validateExistingSignature(target, expected)
            return
        }
        val temp = Files.createTempFile(
            rootPath,
            ".vn97sig-",
            ".tmp",
        )
        try {
            FileOutputStream(temp.toFile()).use { output ->
                output.write(bytes)
                output.flush()
                output.fd.sync()
            }
            try {
                Files.createLink(target.toPath(), temp)
            } catch (_: FileAlreadyExistsException) {
                validateExistingSignature(target, expected)
            }
        } catch (exc: VN97CapabilitySignatureException) {
            throw exc
        } catch (exc: Exception) {
            throw VN97CapabilitySignatureException(
                "VN97SIG1 content-addressed commit failed",
                exc,
            )
        } finally {
            Files.deleteIfExists(temp)
        }
        validateExistingSignature(target, expected)
    }

    private fun validateExistingSignature(
        target: File,
        expected: VN97CapabilitySignatureEnvelope,
    ) {
        requireSafeTarget(target)
        if (!Files.isRegularFile(target.toPath(), LinkOption.NOFOLLOW_LINKS) ||
            Files.isSymbolicLink(target.toPath())
        ) {
            throw VN97CapabilitySignatureException(
                "VN97SIG1 target is not a safe regular file"
            )
        }
        val size = Files.size(target.toPath())
        if (size <= 0L || size > VN97_SIGNATURE_MAX_BYTES.toLong()) {
            throw VN97CapabilitySignatureException(
                "VN97SIG1 target size is outside bounds"
            )
        }
        val bytes = Files.readAllBytes(target.toPath())
        val actual = VN97CapabilitySignatureParser.parse(bytes)
        if (actual != expected) {
            throw VN97CapabilitySignatureException(
                "VN97SIG1 target content mismatch"
            )
        }
    }

    private fun requireSafeTarget(target: File) {
        if (target.parentFile != root ||
            Files.isSymbolicLink(target.toPath())
        ) {
            throw VN97CapabilityPackageException(
                "capability stage target is outside safe root"
            )
        }
    }

    private fun fsyncDirectory() {
        try {
            FileChannel.open(
                rootPath,
                StandardOpenOption.READ,
            ).use { it.force(true) }
        } catch (exc: Exception) {
            throw VN97CapabilityPackageException(
                "capability stage directory fsync failed",
                exc,
            )
        }
    }
}
