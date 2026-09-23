package ai.vn97.runtime

import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.charset.StandardCharsets
import java.security.MessageDigest
import java.util.zip.CRC32

private const val VN97_CAP_MAGIC = "VN97CAP1"
private const val VN97_CAP_VERSION = 1
private const val VN97_CAP_HEADER_BYTES = 96
private const val VN97_CAP_ENTRY_BYTES = 64
private const val VN97_CAP_MAX_SECTIONS = 64
private const val VN97_CAP_MAX_MANIFEST_BYTES = 64 * 1024
const val VN97_CAP_MAX_PACKAGE_BYTES: Long = 512L * 1024L * 1024L

private const val VN97_CAP_TYPE_MANIFEST = 1
private const val VN97_CAP_TYPE_DATA = 2

private val VN97_CAP_KINDS = setOf(
    "weights",
    "tokenizer",
    "memory",
    "avatar",
    "multimodal",
    "knowledge",
    "composite",
)

private val VN97_CAP_FORBIDDEN_ROLES = setOf(
    "code",
    "executable",
    "plugin",
    "shared-library",
    "native-library",
    "script",
)

class VN97CapabilityPackageException(
    message: String,
    cause: Throwable? = null,
) : IllegalStateException(message, cause)

data class VN97CapabilitySource(
    val origin: String,
    val sourceSha256: String,
    val license: String,
)

data class VN97CapabilitySection(
    val index: Int,
    val role: String,
    val format: String,
    val size: Long,
    val sha256: String,
    val packageOffset: Long,
)

data class VN97CapabilityManifest(
    val capabilityId: String,
    val capabilityVersion: Long,
    val kind: String,
    val source: VN97CapabilitySource,
    val sections: List<VN97CapabilitySection>,
)

data class VN97ParsedCapabilityPackage(
    val packageSha256: String,
    val manifest: VN97CapabilityManifest,
    val size: Long,
)

object VN97CapabilityPackageParser {
    fun parse(
        bytes: ByteArray,
        maxPackageBytes: Long = VN97_CAP_MAX_PACKAGE_BYTES,
    ): VN97ParsedCapabilityPackage =
        parse(ByteBuffer.wrap(bytes), maxPackageBytes)

    fun parse(
        source: ByteBuffer,
        maxPackageBytes: Long = VN97_CAP_MAX_PACKAGE_BYTES,
    ): VN97ParsedCapabilityPackage {
        require(maxPackageBytes >= VN97_CAP_HEADER_BYTES) {
            "maxPackageBytes is too small"
        }
        val input = source.slice().asReadOnlyBuffer().order(ByteOrder.LITTLE_ENDIAN)
        val size = input.remaining().toLong()
        if (size < VN97_CAP_HEADER_BYTES || size > maxPackageBytes) {
            fail("VN97CAP1 package size is outside configured bounds")
        }

        if (ascii(input, 0, 8) != VN97_CAP_MAGIC) {
            fail("VN97CAP1 magic mismatch")
        }
        if (u16(input, 8) != VN97_CAP_VERSION ||
            u16(input, 10) != VN97_CAP_HEADER_BYTES
        ) {
            fail("VN97CAP1 version/header mismatch")
        }

        val flags = u32(input, 12)
        val sectionCount = u32(input, 16)
        val entryBytes = u32(input, 20)
        val tableOffset = input.getLong(24)
        val payloadOffset = input.getLong(32)
        val totalSize = input.getLong(40)
        val manifestIndex = u32(input, 48)
        val reserved = u32(input, 52)
        val trailingReserved = u32(input, 92)

        if (flags != 0L || reserved != 0L || trailingReserved != 0L) {
            fail("VN97CAP1 reserved header fields are nonzero")
        }
        if (sectionCount !in 2L..VN97_CAP_MAX_SECTIONS.toLong() ||
            entryBytes != VN97_CAP_ENTRY_BYTES.toLong() ||
            manifestIndex != 0L
        ) {
            fail("VN97CAP1 section table header is invalid")
        }

        val expectedPayload =
            VN97_CAP_HEADER_BYTES.toLong() + sectionCount * VN97_CAP_ENTRY_BYTES
        if (tableOffset != VN97_CAP_HEADER_BYTES.toLong() ||
            payloadOffset != expectedPayload ||
            totalSize != size
        ) {
            fail("VN97CAP1 offsets/length are invalid")
        }

        val expectedHeaderCrc = u32(input, 88)
        val actualHeaderCrc = CRC32().apply {
            update(bytes(input, 0, 88))
        }.value
        if (expectedHeaderCrc != actualHeaderCrc) {
            fail("VN97CAP1 header CRC mismatch")
        }

        val expectedContentDigest = bytes(input, 56, 32)
        val actualContentDigest = digestRange(
            input,
            VN97_CAP_HEADER_BYTES.toLong(),
            size - VN97_CAP_HEADER_BYTES,
        )
        if (!expectedContentDigest.contentEquals(actualContentDigest)) {
            fail("VN97CAP1 content SHA-256 mismatch")
        }

        var cursor = payloadOffset
        var manifestBytes: ByteArray? = null
        val dataEntries = ArrayList<DataEntry>()

        for (index in 0 until sectionCount.toInt()) {
            val entryOffset =
                VN97_CAP_HEADER_BYTES + index * VN97_CAP_ENTRY_BYTES
            val sectionType = u32(input, entryOffset)
            val sectionFlags = u32(input, entryOffset + 4)
            val sectionOffset = input.getLong(entryOffset + 8)
            val sectionSize = input.getLong(entryOffset + 16)
            val sectionDigest = bytes(input, entryOffset + 24, 32)
            val entryReserved = input.getLong(entryOffset + 56)
            val expectedType = if (index == 0) {
                VN97_CAP_TYPE_MANIFEST
            } else {
                VN97_CAP_TYPE_DATA
            }

            if (sectionType != expectedType.toLong() ||
                sectionFlags != 0L ||
                entryReserved != 0L
            ) {
                fail("VN97CAP1 section entry type/flags are invalid")
            }
            if (sectionOffset != cursor ||
                sectionSize <= 0L ||
                sectionSize > size - cursor
            ) {
                fail("VN97CAP1 section layout is invalid")
            }
            val actualDigest = digestRange(input, sectionOffset, sectionSize)
            if (!sectionDigest.contentEquals(actualDigest)) {
                fail("VN97CAP1 section SHA-256 mismatch")
            }

            if (index == 0) {
                if (sectionSize > VN97_CAP_MAX_MANIFEST_BYTES) {
                    fail("VN97CAP1 manifest exceeds byte limit")
                }
                manifestBytes = bytes(
                    input,
                    sectionOffset.toIntExact("manifest offset"),
                    sectionSize.toIntExact("manifest size"),
                )
            } else {
                dataEntries += DataEntry(
                    offset = sectionOffset,
                    size = sectionSize,
                    sha256 = sectionDigest.toHex(),
                )
            }
            cursor += sectionSize
        }

        if (cursor != size) {
            fail("VN97CAP1 package contains trailing/hidden bytes")
        }

        val manifest = parseManifest(
            checkNotNull(manifestBytes),
            dataEntries,
        )
        return VN97ParsedCapabilityPackage(
            packageSha256 = digestRange(input, 0L, size).toHex(),
            manifest = manifest,
            size = size,
        )
    }

    private data class DataEntry(
        val offset: Long,
        val size: Long,
        val sha256: String,
    )

    private fun parseManifest(
        data: ByteArray,
        dataEntries: List<DataEntry>,
    ): VN97CapabilityManifest {
        val text = strictUtf8(data, "VN97CAP1 manifest")
        val root = try {
            VnStrictJson.parseObject(
                text,
                VnJsonLimits(
                    maxInputUtf8Bytes = VN97_CAP_MAX_MANIFEST_BYTES,
                    maxDepth = 32,
                    maxNodes = 4096,
                    maxStringUtf8Bytes = 4096,
                ),
            )
        } catch (exc: RuntimeException) {
            throw VN97CapabilityPackageException(
                "VN97CAP1 manifest is not strict JSON",
                exc,
            )
        }
        if (VnStrictJson.canonical(root) != text) {
            fail("VN97CAP1 manifest must be canonical JSON")
        }
        requireKeys(
            root,
            setOf(
                "schema",
                "capability_id",
                "capability_version",
                "kind",
                "source",
                "sections",
            ),
            "manifest",
        )
        if (root.string("schema") != "VN97CAP1") {
            fail("VN97CAP1 manifest schema mismatch")
        }

        val capabilityId = requireId(
            root.string("capability_id"),
            "capability_id",
            128,
        )
        val version = root.unsigned32("capability_version")
        val kind = requireText(root.string("kind"), "kind", 32)
        if (kind !in VN97_CAP_KINDS) {
            fail("VN97CAP1 capability kind is unsupported")
        }

        val source = root.obj("source")
        requireKeys(
            source,
            setOf("origin", "source_sha256", "license"),
            "source",
        )
        val sourceValue = VN97CapabilitySource(
            origin = requireText(
                source.string("origin"),
                "source origin",
                1024,
            ),
            sourceSha256 = requireSha256(
                source.string("source_sha256"),
                "source_sha256",
            ),
            license = requireText(
                source.string("license"),
                "source license",
                128,
            ),
        )

        val rawSections = root.array("sections").values
        if (rawSections.size != dataEntries.size) {
            fail("VN97CAP1 manifest section list does not match package")
        }

        val roles = HashSet<String>()
        val sections = ArrayList<VN97CapabilitySection>()
        rawSections.forEachIndexed { position, raw ->
            val obj = raw as? VnJsonObject
                ?: fail("VN97CAP1 section metadata must be object")
            requireKeys(
                obj,
                setOf("index", "role", "format", "size", "sha256"),
                "section",
            )
            val expectedIndex = position + 1
            if (obj.positiveInt("index") != expectedIndex) {
                fail("VN97CAP1 section index mismatch")
            }
            val role = requireRole(obj.string("role"))
            if (!roles.add(role)) {
                fail("VN97CAP1 section roles must be unique")
            }
            val format = requireFormat(obj.string("format"))
            val expected = dataEntries[position]
            val sectionSize = obj.positiveLong("size")
            if (sectionSize != expected.size) {
                fail("VN97CAP1 section size mismatch")
            }
            val digest = requireSha256(
                obj.string("sha256"),
                "section sha256",
            )
            if (digest != expected.sha256) {
                fail("VN97CAP1 section manifest SHA-256 mismatch")
            }
            sections += VN97CapabilitySection(
                index = expectedIndex,
                role = role,
                format = format,
                size = sectionSize,
                sha256 = digest,
                packageOffset = expected.offset,
            )
        }

        return VN97CapabilityManifest(
            capabilityId = capabilityId,
            capabilityVersion = version,
            kind = kind,
            source = sourceValue,
            sections = sections,
        )
    }

    private fun requireRole(value: String): String {
        requireId(value, "section role", 64)
        if (value in VN97_CAP_FORBIDDEN_ROLES) {
            fail("VN97CAP1 executable/plugin section role is forbidden")
        }
        return value
    }

    private fun requireFormat(value: String): String {
        fun asciiAlphaNumeric(ch: Char): Boolean =
            ch in 'A'..'Z' || ch in 'a'..'z' || ch in '0'..'9'

        if (value.isEmpty() ||
            value.length > 64 ||
            !asciiAlphaNumeric(value[0]) ||
            value.any {
                !asciiAlphaNumeric(it) && it != '.' && it != '_' && it != '-'
            }
        ) {
            fail("VN97CAP1 section format is invalid")
        }
        return value
    }

    private fun requireId(
        value: String,
        label: String,
        maxChars: Int,
    ): String {
        if (value.isEmpty() ||
            value.length > maxChars ||
            value[0] !in 'a'..'z' ||
            value.any {
                it !in 'a'..'z' &&
                    it !in '0'..'9' &&
                    it != '.' && it != '_' && it != '-'
            }
        ) {
            fail("VN97CAP1 $label is invalid")
        }
        return value
    }

    private fun requireText(
        value: String,
        label: String,
        maxUtf8Bytes: Int,
    ): String {
        if (value.isEmpty() ||
            value.toByteArray(StandardCharsets.UTF_8).size > maxUtf8Bytes
        ) {
            fail("VN97CAP1 $label is invalid")
        }
        return value
    }

    private fun requireSha256(value: String, label: String): String {
        if (value.length != 64 ||
            value.any { it !in "0123456789abcdef" }
        ) {
            fail("VN97CAP1 $label is invalid")
        }
        return value
    }

    private fun requireKeys(
        value: VnJsonObject,
        expected: Set<String>,
        label: String,
    ) {
        if (value.values.keys != expected) {
            fail("VN97CAP1 $label keys are not exact")
        }
    }

    private fun VnJsonObject.string(key: String): String =
        (values[key] as? VnJsonString)?.value
            ?: fail("VN97CAP1 $key must be string")

    private fun VnJsonObject.obj(key: String): VnJsonObject =
        values[key] as? VnJsonObject
            ?: fail("VN97CAP1 $key must be object")

    private fun VnJsonObject.array(key: String): VnJsonArray =
        values[key] as? VnJsonArray
            ?: fail("VN97CAP1 $key must be array")

    private fun VnJsonObject.positiveLong(key: String): Long {
        val raw = (values[key] as? VnJsonNumber)?.canonical
            ?: fail("VN97CAP1 $key must be integer")
        if (raw.any { it == '.' || it == 'e' || it == 'E' }) {
            fail("VN97CAP1 $key must be integer")
        }
        val value = raw.toLongOrNull()
            ?: fail("VN97CAP1 $key is outside Long range")
        if (value <= 0L) {
            fail("VN97CAP1 $key must be positive")
        }
        return value
    }

    private fun VnJsonObject.positiveInt(key: String): Int {
        val value = positiveLong(key)
        if (value > Int.MAX_VALUE) {
            fail("VN97CAP1 $key exceeds Int range")
        }
        return value.toInt()
    }

    private fun VnJsonObject.unsigned32(key: String): Long {
        val value = positiveLong(key)
        if (value > 0xffff_ffffL) {
            fail("VN97CAP1 $key exceeds unsigned 32-bit range")
        }
        return value
    }

    private fun u16(input: ByteBuffer, offset: Int): Int =
        input.getShort(offset).toInt() and 0xffff

    private fun u32(input: ByteBuffer, offset: Int): Long =
        Integer.toUnsignedLong(input.getInt(offset))

    private fun ascii(
        input: ByteBuffer,
        offset: Int,
        size: Int,
    ): String = String(bytes(input, offset, size), StandardCharsets.US_ASCII)

    private fun bytes(
        input: ByteBuffer,
        offset: Int,
        size: Int,
    ): ByteArray {
        val out = ByteArray(size)
        val copy = input.duplicate()
        copy.position(offset)
        copy.get(out)
        return out
    }

    private fun digestRange(
        input: ByteBuffer,
        offset: Long,
        size: Long,
    ): ByteArray {
        if (offset < 0L || size < 0L ||
            offset > input.limit().toLong() ||
            size > input.limit().toLong() - offset
        ) {
            fail("VN97CAP1 digest range is invalid")
        }
        val digest = MessageDigest.getInstance("SHA-256")
        val copy = input.duplicate()
        copy.position(offset.toIntExact("digest offset"))
        var remaining = size
        val chunk = ByteArray(64 * 1024)
        while (remaining > 0L) {
            val count = minOf(chunk.size.toLong(), remaining).toInt()
            copy.get(chunk, 0, count)
            digest.update(chunk, 0, count)
            remaining -= count
        }
        return digest.digest()
    }

    private fun Long.toIntExact(label: String): Int {
        if (this !in 0L..Int.MAX_VALUE.toLong()) {
            fail("VN97CAP1 $label exceeds Int range")
        }
        return toInt()
    }

    private fun ByteArray.toHex(): String =
        joinToString("") { "%02x".format(it.toInt() and 0xff) }

    private fun strictUtf8(bytes: ByteArray, label: String): String = try {
        StandardCharsets.UTF_8.newDecoder()
            .onMalformedInput(java.nio.charset.CodingErrorAction.REPORT)
            .onUnmappableCharacter(java.nio.charset.CodingErrorAction.REPORT)
            .decode(ByteBuffer.wrap(bytes))
            .toString()
    } catch (exc: Exception) {
        throw VN97CapabilityPackageException(
            "$label is not strict UTF-8",
            exc,
        )
    }

    private fun fail(message: String): Nothing =
        throw VN97CapabilityPackageException(message)
}
