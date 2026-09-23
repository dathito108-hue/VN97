package ai.vn97.runtime

import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.channels.FileChannel
import java.nio.file.AtomicMoveNotSupportedException
import java.nio.file.FileAlreadyExistsException
import java.nio.file.Files
import java.nio.file.LinkOption
import java.nio.file.StandardCopyOption
import java.nio.file.StandardOpenOption
import java.security.MessageDigest
import java.security.SecureRandom

private const val COMMITTED_MAGIC = "VN97MI1COMMIT1"
private const val ROLLED_BACK_MAGIC = "VN97MI1ROLLBACK1"

fun interface VN97ModelImageCandidateValidator {
    fun validate(candidate: File, artifactSha256: ByteArray)
}

class VN97ModelImageActivationBackend(
    root: File,
    private val validator: VN97ModelImageCandidateValidator,
) : VN97CapabilityActivationBackend {
    private val rootPath = root.toPath().toAbsolutePath().normalize()
    private val artifactRoot = rootPath.resolve("artifacts")
    private val transactionRoot = rootPath.resolve(".model-image-transactions")
    private val random = SecureRandom()

    override val backendId: String = BACKEND_ID

    init {
        Files.createDirectories(rootPath)
        requireRealDirectory(rootPath.toFile(), "model-image activation root")
        Files.createDirectories(artifactRoot)
        requireRealDirectory(artifactRoot.toFile(), "model-image artifact root")
        Files.createDirectories(transactionRoot)
        requireRealDirectory(transactionRoot.toFile(), "model-image transaction root")
    }

    override fun prepare(
        verified: VN97VerifiedCapability,
        plan: VN97CompatibilityPlan,
    ): VN97PreparedActivation {
        val section = requireDirectModelImage(verified, plan)
        val packageFile = verified.staged.packageFile
        requireRegularFile(packageFile, "staged VN97CAP1")
        if (section.packageOffset < 0L ||
            section.size <= 0L ||
            section.packageOffset > packageFile.length() ||
            section.size > packageFile.length() - section.packageOffset
        ) {
            throw VN97CapabilityActivationException(
                "VN97MI1 section range is outside staged package"
            )
        }

        val token = newToken()
        val candidate = preparedPath(token).toFile()
        extractSection(
            packageFile = packageFile,
            offset = section.packageOffset,
            size = section.size,
            candidate = candidate,
        )
        val digest = try {
            hashRegularFile(candidate)
        } catch (exc: Throwable) {
            Files.deleteIfExists(candidate.toPath())
            throw exc
        }
        val digestHex = digest.toHex()
        if (digestHex != section.sha256) {
            Files.deleteIfExists(candidate.toPath())
            throw VN97CapabilityActivationException(
                "extracted VN97MI1 SHA-256 does not match package section"
            )
        }
        try {
            validator.validate(candidate, digest.copyOf())
        } catch (exc: Throwable) {
            Files.deleteIfExists(candidate.toPath())
            throw VN97CapabilityActivationException(
                "native VN97MI1 candidate validation failed",
                exc,
            )
        }
        forceDirectory(transactionRoot.toFile())
        return VN97PreparedActivation(
            backendId = backendId,
            token = token,
            artifactSha256 = digestHex,
        )
    }

    override fun commit(token: String): String {
        requireToken(token)
        if (rolledBackPath(token).toFile().exists()) {
            throw VN97CapabilityActivationException(
                "rolled-back VN97MI1 transaction cannot commit"
            )
        }
        readCommittedDigestOrNull(token)?.let { digest ->
            requireInstalledArtifact(digest)
            return runtimeRevision(digest)
        }

        val candidate = preparedPath(token).toFile()
        requireRegularFile(candidate, "prepared VN97MI1 candidate")
        val digestBytes = hashRegularFile(candidate)
        val digest = digestBytes.toHex()
        try {
            validator.validate(candidate, digestBytes.copyOf())
        } catch (exc: Throwable) {
            throw VN97CapabilityActivationException(
                "native VN97MI1 commit validation failed",
                exc,
            )
        }

        val target = artifactRoot.resolve("$digest.vn97mi1").toFile()
        commitContentAddressed(candidate, target, digest)
        writeMarker(committedPath(token).toFile(), COMMITTED_MAGIC, digest)
        Files.deleteIfExists(candidate.toPath())
        forceDirectory(transactionRoot.toFile())
        return runtimeRevision(digest)
    }

    override fun inspect(token: String): VN97BackendStatus {
        requireToken(token)
        if (rolledBackPath(token).toFile().exists()) {
            readMarker(rolledBackPath(token).toFile(), ROLLED_BACK_MAGIC)
            return VN97BackendStatus(VN97BackendTransactionState.ROLLED_BACK)
        }
        readCommittedDigestOrNull(token)?.let { digest ->
            requireInstalledArtifact(digest)
            return VN97BackendStatus(
                state = VN97BackendTransactionState.COMMITTED,
                runtimeRevision = runtimeRevision(digest),
            )
        }
        val candidate = preparedPath(token).toFile()
        if (candidate.exists()) {
            requireRegularFile(candidate, "prepared VN97MI1 candidate")
            return VN97BackendStatus(VN97BackendTransactionState.PREPARED)
        }
        throw VN97CapabilityActivationException(
            "VN97MI1 backend token has no durable transaction state"
        )
    }

    override fun rollback(token: String) {
        requireToken(token)
        if (rolledBackPath(token).toFile().exists()) {
            readMarker(rolledBackPath(token).toFile(), ROLLED_BACK_MAGIC)
            return
        }
        val prepared = preparedPath(token)
        val committedDigest = readCommittedDigestOrNull(token)
        if (!Files.exists(prepared, LinkOption.NOFOLLOW_LINKS) &&
            committedDigest == null
        ) {
            throw VN97CapabilityActivationException(
                "VN97MI1 backend token has no transaction to rollback"
            )
        }
        Files.deleteIfExists(prepared)
        writeMarker(
            rolledBackPath(token).toFile(),
            ROLLED_BACK_MAGIC,
            committedDigest ?: "-",
        )
        Files.deleteIfExists(committedPath(token))
        forceDirectory(transactionRoot.toFile())
    }

    private fun requireDirectModelImage(
        verified: VN97VerifiedCapability,
        plan: VN97CompatibilityPlan,
    ): VN97CapabilitySection {
        val manifest = verified.parsed.manifest
        if (manifest.capabilityId != CAPABILITY_ID || manifest.kind != "weights") {
            throw VN97CapabilityActivationException(
                "model-image backend only accepts model.language weights"
            )
        }
        if (manifest.sections.size != 1 || plan.sections.size != 1) {
            throw VN97CapabilityActivationException(
                "model-image backend requires exactly one package section"
            )
        }
        if (plan.disposition != VN97CompatibilityDisposition.DIRECT) {
            throw VN97CapabilityActivationException(
                "model-image backend accepts DIRECT compatibility only"
            )
        }
        if (plan.packageSha256 != verified.parsed.packageSha256 ||
            plan.capabilityId != manifest.capabilityId ||
            plan.capabilityVersion != manifest.capabilityVersion ||
            plan.publisherKeyId != verified.publisherKeyId ||
            plan.signatureSha256 != verified.signatureSha256
        ) {
            throw VN97CapabilityActivationException(
                "model-image plan identity does not match verified capability"
            )
        }
        val section = manifest.sections.single()
        val sectionPlan = plan.sections.single()
        if (section.role != MODEL_IMAGE_ROLE ||
            section.format != MODEL_IMAGE_FORMAT ||
            sectionPlan.sectionIndex != section.index ||
            sectionPlan.role != section.role ||
            sectionPlan.sourceFormat != MODEL_IMAGE_FORMAT ||
            sectionPlan.targetFormat != MODEL_IMAGE_FORMAT ||
            sectionPlan.adapterId != null ||
            sectionPlan.lossy
        ) {
            throw VN97CapabilityActivationException(
                "model-image section/plan is not direct VN97MI1"
            )
        }
        return section
    }

    private fun extractSection(
        packageFile: File,
        offset: Long,
        size: Long,
        candidate: File,
    ) {
        try {
            FileChannel.open(
                packageFile.toPath(),
                StandardOpenOption.READ,
                LinkOption.NOFOLLOW_LINKS,
            ).use { input ->
                FileChannel.open(
                    candidate.toPath(),
                    StandardOpenOption.CREATE_NEW,
                    StandardOpenOption.WRITE,
                    LinkOption.NOFOLLOW_LINKS,
                ).use { output ->
                    input.position(offset)
                    var remaining = size
                    val buffer = ByteBuffer.allocate(64 * 1024)
                    while (remaining > 0L) {
                        buffer.clear()
                        buffer.limit(minOf(buffer.capacity().toLong(), remaining).toInt())
                        val read = input.read(buffer)
                        if (read <= 0) {
                            throw VN97CapabilityActivationException(
                                "staged VN97MI1 section read was truncated"
                            )
                        }
                        remaining -= read.toLong()
                        buffer.flip()
                        while (buffer.hasRemaining()) {
                            if (output.write(buffer) <= 0) {
                                throw VN97CapabilityActivationException(
                                    "prepared VN97MI1 write made no progress"
                                )
                            }
                        }
                    }
                    output.force(true)
                }
            }
        } catch (exc: VN97CapabilityActivationException) {
            Files.deleteIfExists(candidate.toPath())
            throw exc
        } catch (exc: Exception) {
            Files.deleteIfExists(candidate.toPath())
            throw VN97CapabilityActivationException(
                "VN97MI1 section extraction failed",
                exc,
            )
        }
    }

    private fun commitContentAddressed(
        candidate: File,
        target: File,
        digest: String,
    ) {
        requireTargetInside(artifactRoot.toFile(), target)
        if (!target.exists()) {
            try {
                Files.createLink(target.toPath(), candidate.toPath())
            } catch (_: FileAlreadyExistsException) {
                Unit
            } catch (exc: Exception) {
                throw VN97CapabilityActivationException(
                    "VN97MI1 artifact content-addressed commit failed",
                    exc,
                )
            }
        }
        requireInstalledArtifact(digest)
        forceDirectory(artifactRoot.toFile())
    }

    private fun requireInstalledArtifact(digest: String) {
        requireSha(digest)
        val target = artifactRoot.resolve("$digest.vn97mi1").toFile()
        requireRegularFile(target, "committed VN97MI1 artifact")
        if (hashRegularFile(target).toHex() != digest) {
            throw VN97CapabilityActivationException(
                "committed VN97MI1 artifact digest mismatch"
            )
        }
    }

    private fun writeMarker(
        target: File,
        magic: String,
        value: String,
    ) {
        requireTargetInside(transactionRoot.toFile(), target)
        val payload = "$magic\n$value\n".toByteArray(Charsets.US_ASCII)
        val temp = Files.createTempFile(transactionRoot, ".marker-", ".tmp")
        try {
            FileOutputStream(temp.toFile()).use { output ->
                output.write(payload)
                output.flush()
                output.fd.sync()
            }
            try {
                Files.move(
                    temp,
                    target.toPath(),
                    StandardCopyOption.ATOMIC_MOVE,
                    StandardCopyOption.REPLACE_EXISTING,
                )
            } catch (exc: AtomicMoveNotSupportedException) {
                throw VN97CapabilityActivationException(
                    "VN97MI1 transaction filesystem lacks atomic replace",
                    exc,
                )
            }
            forceDirectory(transactionRoot.toFile())
        } finally {
            Files.deleteIfExists(temp)
        }
    }

    private fun readCommittedDigestOrNull(token: String): String? {
        val marker = committedPath(token).toFile()
        if (!marker.exists()) return null
        val value = readMarker(marker, COMMITTED_MAGIC)
        requireSha(value)
        return value
    }

    private fun readMarker(target: File, expectedMagic: String): String {
        requireRegularFile(target, "VN97MI1 transaction marker")
        val bytes = Files.readAllBytes(target.toPath())
        if (bytes.size > 256) {
            throw VN97CapabilityActivationException(
                "VN97MI1 transaction marker exceeds byte bound"
            )
        }
        val text = try {
            bytes.toString(Charsets.US_ASCII)
        } catch (exc: Exception) {
            throw VN97CapabilityActivationException(
                "VN97MI1 transaction marker is not ASCII",
                exc,
            )
        }
        val lines = text.split('\n')
        if (lines.size != 3 || lines[0] != expectedMagic || lines[2].isNotEmpty()) {
            throw VN97CapabilityActivationException(
                "VN97MI1 transaction marker is corrupt"
            )
        }
        return lines[1]
    }

    private fun hashRegularFile(file: File): ByteArray {
        requireRegularFile(file, "VN97MI1 file")
        val digest = MessageDigest.getInstance("SHA-256")
        FileChannel.open(
            file.toPath(),
            StandardOpenOption.READ,
            LinkOption.NOFOLLOW_LINKS,
        ).use { channel ->
            val buffer = ByteBuffer.allocate(64 * 1024)
            while (true) {
                buffer.clear()
                val read = channel.read(buffer)
                if (read < 0) break
                if (read == 0) {
                    throw VN97CapabilityActivationException(
                        "VN97MI1 hash read made no progress"
                    )
                }
                buffer.flip()
                digest.update(buffer)
            }
        }
        return digest.digest()
    }

    private fun requireRegularFile(file: File, label: String) {
        val path = file.toPath()
        if (!Files.isRegularFile(path, LinkOption.NOFOLLOW_LINKS) ||
            Files.isSymbolicLink(path)
        ) {
            throw VN97CapabilityActivationException(
                "$label must be a non-symlink regular file"
            )
        }
    }

    private fun requireRealDirectory(file: File, label: String) {
        val path = file.toPath()
        if (!Files.isDirectory(path, LinkOption.NOFOLLOW_LINKS) ||
            Files.isSymbolicLink(path)
        ) {
            throw VN97CapabilityActivationException(
                "$label must be a non-symlink directory"
            )
        }
    }

    private fun requireTargetInside(parent: File, target: File) {
        val expected = parent.toPath().toAbsolutePath().normalize()
        val actual = target.toPath().toAbsolutePath().normalize()
        if (actual.parent != expected || Files.isSymbolicLink(actual)) {
            throw VN97CapabilityActivationException(
                "VN97MI1 target is outside trusted activation root"
            )
        }
    }

    private fun newToken(): String {
        val bytes = ByteArray(16)
        random.nextBytes(bytes)
        return "mi1-" + bytes.toHex()
    }

    private fun requireToken(token: String) {
        if (!token.startsWith("mi1-") || token.length != 36 ||
            token.drop(4).any { it !in "0123456789abcdef" }
        ) {
            throw VN97CapabilityActivationException(
                "VN97MI1 backend token is invalid"
            )
        }
    }

    private fun requireSha(value: String) {
        if (value.length != 64 || value.any { it !in "0123456789abcdef" }) {
            throw VN97CapabilityActivationException(
                "VN97MI1 artifact SHA-256 is invalid"
            )
        }
    }

    private fun preparedPath(token: String) =
        transactionRoot.resolve("$token.prepared.vn97mi1")

    private fun committedPath(token: String) =
        transactionRoot.resolve("$token.committed")

    private fun rolledBackPath(token: String) =
        transactionRoot.resolve("$token.rolledback")

    private fun runtimeRevision(digest: String): String = "mi1-$digest"

    private fun forceDirectory(directory: File) {
        try {
            FileChannel.open(
                directory.toPath(),
                StandardOpenOption.READ,
                LinkOption.NOFOLLOW_LINKS,
            ).use { it.force(true) }
        } catch (exc: Exception) {
            throw VN97CapabilityActivationException(
                "VN97MI1 directory fsync failed",
                exc,
            )
        }
    }

    private fun ByteArray.toHex(): String =
        joinToString("") { "%02x".format(it.toInt() and 0xff) }

    companion object {
        const val BACKEND_ID = "vn97.model_image"
        const val CAPABILITY_ID = "model.language"
        const val MODEL_IMAGE_ROLE = "model_image"
        const val MODEL_IMAGE_FORMAT = "VN97MI1"

        fun productionProfile(): VN97CompatibilityProfile =
            VN97CompatibilityProfile(
                profileId = "vn97.android.model-image.v1",
                runtimeApiVersion = 1L,
                supportedKinds = setOf("weights"),
                minCapabilityVersion = 1L,
                maxCapabilityVersion = 0xffff_ffffL,
                formatRules = listOf(
                    VN97FormatRule(
                        role = MODEL_IMAGE_ROLE,
                        acceptedFormats = listOf(MODEL_IMAGE_FORMAT),
                    )
                ),
            )
    }
}
