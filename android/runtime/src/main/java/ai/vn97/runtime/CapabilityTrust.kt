package ai.vn97.runtime

import java.io.File
import java.io.RandomAccessFile
import java.math.BigInteger
import java.nio.channels.FileChannel
import java.nio.file.Files
import java.nio.file.LinkOption
import java.security.MessageDigest

private val VN97_TRUSTED_KINDS = setOf(
    "weights",
    "tokenizer",
    "memory",
    "avatar",
    "multimodal",
    "knowledge",
    "composite",
)

class VN97CapabilityTrustException(
    message: String,
    cause: Throwable? = null,
) : IllegalStateException(message, cause)

class VN97TrustedPublisherKey(
    val keyId: String,
    publicKey: ByteArray,
    capabilityPrefixes: Set<String>,
    allowedKinds: Set<String>,
    val minVersion: Long = 1L,
    val maxVersion: Long = 0xffff_ffffL,
    val revoked: Boolean = false,
) {
    private val publicKeyBytes = publicKey.copyOf()
    val capabilityPrefixes: Set<String> = capabilityPrefixes.toSet()
    val allowedKinds: Set<String> = allowedKinds.toSet()

    init {
        requireTrustId(keyId, "publisher key_id")
        if (publicKeyBytes.size != 32) {
            throw VN97CapabilityTrustException(
                "publisher Ed25519 public key must be exactly 32 bytes"
            )
        }
        if (this.capabilityPrefixes.isEmpty()) {
            throw VN97CapabilityTrustException(
                "publisher capability prefixes must not be empty"
            )
        }
        this.capabilityPrefixes.forEach {
            requireTrustId(it, "publisher capability prefix")
        }
        if (this.allowedKinds.isEmpty() ||
            !VN97_TRUSTED_KINDS.containsAll(this.allowedKinds)
        ) {
            throw VN97CapabilityTrustException(
                "publisher allowed kinds are invalid"
            )
        }
        if (minVersion !in 1L..0xffff_ffffL ||
            maxVersion !in minVersion..0xffff_ffffL
        ) {
            throw VN97CapabilityTrustException(
                "publisher version range is invalid"
            )
        }
    }

    fun publicKey(): ByteArray = publicKeyBytes.copyOf()

    fun permits(manifest: VN97CapabilityManifest): Boolean {
        val scoped = capabilityPrefixes.any { prefix ->
            manifest.capabilityId == prefix ||
                manifest.capabilityId.startsWith("$prefix.")
        }
        return !revoked &&
            scoped &&
            manifest.kind in allowedKinds &&
            manifest.capabilityVersion in minVersion..maxVersion
    }
}

class VN97CapabilityTrustStore(
    keys: List<VN97TrustedPublisherKey>,
) {
    private val byId: Map<String, VN97TrustedPublisherKey>

    init {
        if (keys.isEmpty()) {
            throw VN97CapabilityTrustException(
                "publisher trust store must not be empty"
            )
        }
        val map = LinkedHashMap<String, VN97TrustedPublisherKey>()
        for (key in keys) {
            if (map.put(key.keyId, key) != null) {
                throw VN97CapabilityTrustException(
                    "duplicate publisher key_id"
                )
            }
        }
        byId = map.toMap()
    }

    fun require(keyId: String): VN97TrustedPublisherKey =
        byId[keyId] ?: throw VN97CapabilityTrustException(
            "publisher key is not trusted"
        )
}

data class VN97VerifiedCapability(
    val staged: VN97StagedCapability,
    val parsed: VN97ParsedCapabilityPackage,
    val publisherKeyId: String,
    val signatureSha256: String,
)

object VN97CapabilityTrustVerifier {
    fun verify(
        staged: VN97StagedCapability,
        stageRoot: File,
        trustStore: VN97CapabilityTrustStore,
    ): VN97VerifiedCapability {
        val root = stageRoot.toPath().toAbsolutePath().normalize()
        if (!Files.isDirectory(root, LinkOption.NOFOLLOW_LINKS) ||
            Files.isSymbolicLink(root)
        ) {
            throw VN97CapabilityTrustException(
                "capability stage root is unsafe"
            )
        }

        val packagePath = staged.packageFile.toPath()
            .toAbsolutePath()
            .normalize()
        val signaturePath = staged.signatureFile.toPath()
            .toAbsolutePath()
            .normalize()
        if (packagePath.parent != root || signaturePath.parent != root) {
            throw VN97CapabilityTrustException(
                "staged capability paths are outside configured root"
            )
        }
        if (packagePath.fileName.toString() !=
            staged.parsed.packageSha256 + ".vn97cap1"
        ) {
            throw VN97CapabilityTrustException(
                "staged package filename is not content-addressed"
            )
        }
        val expectedSignatureName =
            staged.parsed.packageSha256 + "." +
                staged.signature.keyId + ".vn97sig1"
        if (signaturePath.fileName.toString() != expectedSignatureName) {
            throw VN97CapabilityTrustException(
                "staged signature filename is not content-addressed"
            )
        }
        requireRegularFile(packagePath.toFile(), "VN97CAP1")
        requireRegularFile(signaturePath.toFile(), "VN97SIG1")

        val parsed = try {
            RandomAccessFile(packagePath.toFile(), "r").use { file ->
                val size = file.length()
                if (size <= 0L || size > VN97_CAP_MAX_PACKAGE_BYTES) {
                    throw VN97CapabilityTrustException(
                        "staged VN97CAP1 size is outside bounds"
                    )
                }
                VN97CapabilityPackageParser.parse(
                    file.channel.map(
                        FileChannel.MapMode.READ_ONLY,
                        0L,
                        size,
                    ),
                )
            }
        } catch (exc: VN97CapabilityTrustException) {
            throw exc
        } catch (exc: Exception) {
            throw VN97CapabilityTrustException(
                "staged VN97CAP1 failed revalidation",
                exc,
            )
        }
        if (parsed != staged.parsed) {
            throw VN97CapabilityTrustException(
                "staged VN97CAP1 identity changed after staging"
            )
        }

        val signatureBytes = try {
            val size = Files.size(signaturePath)
            if (size <= 0L || size > VN97_SIGNATURE_MAX_BYTES.toLong()) {
                throw VN97CapabilityTrustException(
                    "staged VN97SIG1 size is outside bounds"
                )
            }
            Files.readAllBytes(signaturePath)
        } catch (exc: VN97CapabilityTrustException) {
            throw exc
        } catch (exc: Exception) {
            throw VN97CapabilityTrustException(
                "staged VN97SIG1 could not be re-read",
                exc,
            )
        }
        val envelope = try {
            VN97CapabilitySignatureParser.parse(signatureBytes)
        } catch (exc: Exception) {
            throw VN97CapabilityTrustException(
                "staged VN97SIG1 failed revalidation",
                exc,
            )
        }
        if (envelope != staged.signature) {
            throw VN97CapabilityTrustException(
                "staged VN97SIG1 identity changed after staging"
            )
        }
        if (envelope.packageSha256 != parsed.packageSha256 ||
            envelope.capabilityId != parsed.manifest.capabilityId ||
            envelope.capabilityVersion != parsed.manifest.capabilityVersion
        ) {
            throw VN97CapabilityTrustException(
                "VN97SIG1 is not bound to staged VN97CAP1"
            )
        }

        val key = trustStore.require(envelope.keyId)
        if (!key.permits(parsed.manifest)) {
            throw VN97CapabilityTrustException(
                "publisher scope does not permit staged capability"
            )
        }
        if (!VN97Ed25519Verifier.verify(
                publicKey = key.publicKey(),
                message = envelope.signingMessage(),
                signature = envelope.signature,
            )
        ) {
            throw VN97CapabilityTrustException(
                "publisher Ed25519 signature verification failed"
            )
        }

        return VN97VerifiedCapability(
            staged = staged,
            parsed = parsed,
            publisherKeyId = key.keyId,
            signatureSha256 = envelope.envelopeSha256,
        )
    }

    private fun requireRegularFile(file: File, label: String) {
        val path = file.toPath()
        if (!Files.isRegularFile(path, LinkOption.NOFOLLOW_LINKS) ||
            Files.isSymbolicLink(path)
        ) {
            throw VN97CapabilityTrustException(
                "$label must be a non-symlink regular file"
            )
        }
    }
}

/**
 * Strict Ed25519 verification for Android API 26+ without relying on the platform EdDSA provider.
 * This path is provisioning-only, not an inference hot path.
 */
object VN97Ed25519Verifier {
    private val ZERO = BigInteger.ZERO
    private val ONE = BigInteger.ONE
    private val TWO = BigInteger.valueOf(2L)
    private val P = TWO.pow(255).subtract(BigInteger.valueOf(19L))
    private val L = TWO.pow(252).add(
        BigInteger("27742317777372353535851937790883648493")
    )
    private val D = mod(
        BigInteger.valueOf(-121665L).multiply(
            BigInteger.valueOf(121666L).modInverse(P)
        )
    )
    private val SQRT_M1 = TWO.modPow(
        P.subtract(ONE).divide(BigInteger.valueOf(4L)),
        P,
    )
    private val SQRT_EXPONENT = P.add(BigInteger.valueOf(3L))
        .divide(BigInteger.valueOf(8L))

    private data class Point(
        val x: BigInteger,
        val y: BigInteger,
        val z: BigInteger,
        val t: BigInteger,
    )

    private val IDENTITY = Point(ZERO, ONE, ONE, ZERO)
    private val BASE = fromAffine(
        BigInteger(
            "15112221349535400772501151409588531511454012693041857206046113283949847762202"
        ),
        BigInteger(
            "46316835694926478169428394003475163141307993866256225615783033603165251855960"
        ),
    )

    fun verify(
        publicKey: ByteArray,
        message: ByteArray,
        signature: ByteArray,
    ): Boolean {
        if (publicKey.size != 32 || signature.size != 64) return false

        val encodedR = signature.copyOfRange(0, 32)
        val s = littleEndianUnsigned(signature.copyOfRange(32, 64))
        if (s >= L) return false

        val a = decodePoint(publicKey) ?: return false
        val r = decodePoint(encodedR) ?: return false
        if (isIdentity(a) || !isPrimeOrderPoint(a)) return false
        if (!isPrimeOrderPoint(r)) return false

        val hash = MessageDigest.getInstance("SHA-512")
        hash.update(encodedR)
        hash.update(publicKey)
        hash.update(message)
        val k = littleEndianUnsigned(hash.digest()).mod(L)

        val left = scalarMultiply(BASE, s)
        val right = add(r, scalarMultiply(a, k))
        return pointsEqual(left, right)
    }

    private fun decodePoint(encoded: ByteArray): Point? {
        if (encoded.size != 32) return null
        val yBytes = encoded.copyOf()
        val sign = (yBytes[31].toInt() ushr 7) and 1
        yBytes[31] = (yBytes[31].toInt() and 0x7f).toByte()
        val y = littleEndianUnsigned(yBytes)
        if (y >= P) return null

        val ySquared = mod(y.multiply(y))
        val numerator = mod(ySquared.subtract(ONE))
        val denominator = mod(D.multiply(ySquared).add(ONE))
        if (denominator == ZERO) return null

        val xSquared = try {
            mod(numerator.multiply(denominator.modInverse(P)))
        } catch (_: ArithmeticException) {
            return null
        }
        var x = xSquared.modPow(SQRT_EXPONENT, P)
        if (mod(x.multiply(x).subtract(xSquared)) != ZERO) {
            x = mod(x.multiply(SQRT_M1))
        }
        if (mod(x.multiply(x).subtract(xSquared)) != ZERO) return null
        if (x == ZERO && sign == 1) return null
        if ((if (x.testBit(0)) 1 else 0) != sign) {
            x = P.subtract(x)
        }
        return fromAffine(x, y)
    }

    private fun fromAffine(x: BigInteger, y: BigInteger): Point =
        Point(
            x = mod(x),
            y = mod(y),
            z = ONE,
            t = mod(x.multiply(y)),
        )

    private fun add(left: Point, right: Point): Point {
        val a = mod(
            left.y.subtract(left.x)
                .multiply(right.y.subtract(right.x))
        )
        val b = mod(
            left.y.add(left.x)
                .multiply(right.y.add(right.x))
        )
        val c = mod(
            TWO.multiply(D)
                .multiply(left.t)
                .multiply(right.t)
        )
        val d = mod(TWO.multiply(left.z).multiply(right.z))
        val e = mod(b.subtract(a))
        val f = mod(d.subtract(c))
        val g = mod(d.add(c))
        val h = mod(b.add(a))
        return Point(
            x = mod(e.multiply(f)),
            y = mod(g.multiply(h)),
            z = mod(f.multiply(g)),
            t = mod(e.multiply(h)),
        )
    }

    private fun double(point: Point): Point {
        val a = mod(point.x.multiply(point.x))
        val b = mod(point.y.multiply(point.y))
        val c = mod(TWO.multiply(point.z).multiply(point.z))
        val d = mod(a.negate())
        val e = mod(
            point.x.add(point.y)
                .multiply(point.x.add(point.y))
                .subtract(a)
                .subtract(b)
        )
        val g = mod(d.add(b))
        val f = mod(g.subtract(c))
        val h = mod(d.subtract(b))
        return Point(
            x = mod(e.multiply(f)),
            y = mod(g.multiply(h)),
            z = mod(f.multiply(g)),
            t = mod(e.multiply(h)),
        )
    }

    private fun scalarMultiply(point: Point, scalar: BigInteger): Point {
        var n = scalar
        var result = IDENTITY
        var addend = point
        while (n.signum() > 0) {
            if (n.testBit(0)) result = add(result, addend)
            addend = double(addend)
            n = n.shiftRight(1)
        }
        return result
    }

    private fun pointsEqual(left: Point, right: Point): Boolean =
        mod(left.x.multiply(right.z)) ==
            mod(right.x.multiply(left.z)) &&
            mod(left.y.multiply(right.z)) ==
            mod(right.y.multiply(left.z))

    private fun isIdentity(point: Point): Boolean =
        mod(point.x) == ZERO && mod(point.y.subtract(point.z)) == ZERO

    private fun isPrimeOrderPoint(point: Point): Boolean =
        isIdentity(scalarMultiply(point, L))

    private fun littleEndianUnsigned(bytes: ByteArray): BigInteger {
        val bigEndian = ByteArray(bytes.size + 1)
        for (index in bytes.indices) {
            bigEndian[bytes.size - index] = bytes[index]
        }
        return BigInteger(bigEndian)
    }

    private fun mod(value: BigInteger): BigInteger {
        val result = value.remainder(P)
        return if (result.signum() < 0) result.add(P) else result
    }
}

private fun requireTrustId(value: String, label: String) {
    if (value.isEmpty() ||
        value.length > 128 ||
        value[0] !in 'a'..'z' ||
        value.any {
            it !in 'a'..'z' &&
                it !in '0'..'9' &&
                it != '.' && it != '_' && it != '-'
        }
    ) {
        throw VN97CapabilityTrustException("$label is invalid")
    }
}
