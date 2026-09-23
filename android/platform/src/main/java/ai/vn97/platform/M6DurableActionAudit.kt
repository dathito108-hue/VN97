package ai.vn97.platform

import java.io.File
import java.nio.ByteBuffer
import java.nio.charset.CodingErrorAction
import java.nio.charset.StandardCharsets
import java.nio.channels.FileChannel
import java.nio.file.Files
import java.nio.file.LinkOption
import java.nio.file.OpenOption
import java.nio.file.StandardOpenOption
import java.nio.file.attribute.BasicFileAttributes
import java.security.MessageDigest

class M6AuditIntegrityException(
    message: String,
    cause: Throwable? = null,
) : IllegalStateException(message, cause)

class M6AuditCapacityException(message: String) : IllegalStateException(message)

class M6DurableActionAudit(
    root: File,
    fileName: String = "m6-actions.jsonl",
    private val maxFileBytes: Long = 64L * 1024 * 1024,
    private val maxRecordBytes: Int = 256 * 1024,
    private val maxRecords: Int = 100_000,
) : M6ActionAudit {
    private val rootPath = root.toPath()
    private val target = root.resolve(fileName).toPath()
    private val records = ArrayList<M6ActionReceipt>()
    private var committedBytes = 0L
    private var committedFileKey: Any? = null

    init {
        require(fileName.isNotBlank()) { "audit file name must not be blank" }
        require('/' !in fileName && '\\' !in fileName && fileName != "." && fileName != "..") {
            "audit file name must be one path component"
        }
        require(maxFileBytes > 0L && maxFileBytes <= Int.MAX_VALUE.toLong()) {
            "maxFileBytes must fit bounded JVM recovery"
        }
        require(maxRecordBytes > 0 && maxRecordBytes.toLong() <= maxFileBytes) {
            "maxRecordBytes must be positive and within maxFileBytes"
        }
        require(maxRecords > 0) { "maxRecords must be positive" }

        Files.createDirectories(rootPath)
        if (!Files.isDirectory(rootPath, LinkOption.NOFOLLOW_LINKS) || Files.isSymbolicLink(rootPath)) {
            throw M6AuditIntegrityException("audit root must be a real directory")
        }
        loadExisting()
    }

    override val receipts: List<M6ActionReceipt>
        @Synchronized get() = records.toList()

    @Synchronized
    override fun append(
        status: M6ReceiptStatus,
        request: M6ExternalActionRequest,
        principal: String,
        leaseId: String,
        approvalId: String,
        outcome: M6ActionOutcome?,
        errorType: String,
        timestampNs: Long,
    ): M6ActionReceipt {
        require(timestampNs >= 0L) { "timestampNs must be non-negative" }
        validatePrincipal(principal)
        if (records.size >= maxRecords) {
            throw M6AuditCapacityException("audit record count limit reached")
        }

        val previous = records.lastOrNull()?.receiptId.orEmpty()
        val base = M6ActionReceipt(
            sequence = records.size + 1,
            timestampNs = timestampNs,
            previousReceiptId = previous,
            receiptId = "",
            status = status,
            requestDigest = request.requestDigest,
            planId = request.planId,
            stepId = request.stepId,
            capabilityId = request.capabilityId,
            scopeDigest = request.scope.digest,
            principal = principal,
            leaseId = leaseId,
            approvalId = approvalId,
            result = outcome?.result.orEmpty(),
            confidence = outcome?.confidence ?: 0.0,
            evidenceRecordIds = outcome?.evidenceRecordIds?.toList() ?: emptyList(),
            retryable = outcome?.retryable ?: false,
            errorType = errorType,
        )
        val receipt = base.copy(receiptId = receiptId(base))
        val record = (canonicalReceiptRecord(receipt) + "\n").toByteArray(StandardCharsets.UTF_8)
        if (record.size > maxRecordBytes) {
            throw M6AuditCapacityException("audit record exceeds byte limit")
        }
        val nextSize = try {
            Math.addExact(committedBytes, record.size.toLong())
        } catch (exc: ArithmeticException) {
            throw M6AuditCapacityException("audit file size overflows")
        }
        if (nextSize > maxFileBytes) {
            throw M6AuditCapacityException("audit file byte limit reached")
        }

        // Ensure the exact bytes we are about to persist are also acceptable to recovery.
        val parsed = parseCanonicalReceipt(
            String(record, 0, record.size - 1, StandardCharsets.UTF_8)
        )
        if (parsed != receipt) {
            throw M6AuditIntegrityException("audit receipt self-validation mismatch")
        }

        val existed = verifyTargetUnchanged()
        val options = arrayOf<OpenOption>(
            StandardOpenOption.CREATE,
            StandardOpenOption.WRITE,
            StandardOpenOption.APPEND,
            LinkOption.NOFOLLOW_LINKS,
        )
        try {
            FileChannel.open(target, *options).use { channel ->
                val buffer = ByteBuffer.wrap(record)
                while (buffer.hasRemaining()) {
                    if (channel.write(buffer) <= 0) {
                        throw M6AuditIntegrityException("audit append made no progress")
                    }
                }
                channel.force(true)
            }
            forceDirectory()
        } catch (exc: M6AuditIntegrityException) {
            throw exc
        } catch (exc: Exception) {
            throw M6AuditIntegrityException("durable audit append failed", exc)
        }

        val attrs = readTargetAttributes()
        if (attrs.size() != nextSize) {
            throw M6AuditIntegrityException("audit file size changed during append")
        }
        if (existed && committedFileKey != null && attrs.fileKey() != null && attrs.fileKey() != committedFileKey) {
            throw M6AuditIntegrityException("audit file identity changed during append")
        }
        committedBytes = nextSize
        committedFileKey = attrs.fileKey()
        records += receipt
        return receipt
    }

    @Synchronized
    override fun successfulReceipt(requestDigest: String): M6ActionReceipt? =
        records.asReversed().firstOrNull {
            it.requestDigest == requestDigest && it.status == M6ReceiptStatus.SUCCEEDED
        }

    private fun loadExisting() {
        if (!Files.exists(target, LinkOption.NOFOLLOW_LINKS)) return
        val before = readTargetAttributes()
        val size = before.size()
        if (size > maxFileBytes) {
            throw M6AuditCapacityException("audit file exceeds byte limit")
        }
        val bytes = try {
            val result = ByteArray(size.toInt())
            FileChannel.open(
                target,
                StandardOpenOption.READ,
                LinkOption.NOFOLLOW_LINKS,
            ).use { channel ->
                val buffer = ByteBuffer.wrap(result)
                while (buffer.hasRemaining()) {
                    val read = channel.read(buffer)
                    if (read < 0) {
                        throw M6AuditIntegrityException("audit ended during bounded read")
                    }
                    if (read == 0) {
                        throw M6AuditIntegrityException("audit read made no progress")
                    }
                }
            }
            result
        } catch (exc: M6AuditIntegrityException) {
            throw exc
        } catch (exc: Exception) {
            throw M6AuditIntegrityException("audit file could not be read", exc)
        }
        val after = readTargetAttributes()
        if (after.size() != size || bytes.size.toLong() != size) {
            throw M6AuditIntegrityException("audit changed while loading")
        }
        if (before.fileKey() != null && after.fileKey() != null && before.fileKey() != after.fileKey()) {
            throw M6AuditIntegrityException("audit file identity changed while loading")
        }
        if (bytes.isNotEmpty() && bytes.last() != '\n'.code.toByte()) {
            throw M6AuditIntegrityException("audit contains a torn final record")
        }

        var start = 0
        var expectedSequence = 1
        var previous = ""
        while (start < bytes.size) {
            var end = start
            while (end < bytes.size && bytes[end] != '\n'.code.toByte()) end += 1
            if (end == start) {
                throw M6AuditIntegrityException("audit contains an empty record")
            }
            val length = end - start
            if (length > maxRecordBytes) {
                throw M6AuditCapacityException("audit record exceeds byte limit")
            }
            if (records.size >= maxRecords) {
                throw M6AuditCapacityException("audit record count limit reached")
            }
            val line = strictUtf8(bytes, start, length)
            val receipt = parseCanonicalReceipt(line)
            if (receipt.sequence != expectedSequence) {
                throw M6AuditIntegrityException("audit sequence is not contiguous")
            }
            if (receipt.previousReceiptId != previous) {
                throw M6AuditIntegrityException("audit receipt hash chain is broken")
            }
            records += receipt
            previous = receipt.receiptId
            expectedSequence += 1
            start = end + 1
        }
        committedBytes = size
        committedFileKey = after.fileKey()
    }

    private fun verifyTargetUnchanged(): Boolean {
        val exists = Files.exists(target, LinkOption.NOFOLLOW_LINKS)
        if (committedBytes == 0L && committedFileKey == null) {
            if (exists) {
                val attrs = readTargetAttributes()
                if (attrs.size() != 0L) {
                    throw M6AuditIntegrityException("audit appeared or changed outside this instance")
                }
                committedFileKey = attrs.fileKey()
            }
            return exists
        }
        if (!exists) {
            throw M6AuditIntegrityException("audit file disappeared")
        }
        val attrs = readTargetAttributes()
        if (attrs.size() != committedBytes) {
            throw M6AuditIntegrityException("audit file changed outside this instance")
        }
        if (committedFileKey != null && attrs.fileKey() != null && attrs.fileKey() != committedFileKey) {
            throw M6AuditIntegrityException("audit file identity changed outside this instance")
        }
        return true
    }

    private fun readTargetAttributes(): BasicFileAttributes {
        if (Files.isSymbolicLink(target)) {
            throw M6AuditIntegrityException("audit target must not be a symlink")
        }
        val attrs = try {
            Files.readAttributes(target, BasicFileAttributes::class.java, LinkOption.NOFOLLOW_LINKS)
        } catch (exc: Exception) {
            throw M6AuditIntegrityException("audit target attributes unavailable", exc)
        }
        if (!attrs.isRegularFile) {
            throw M6AuditIntegrityException("audit target must be a regular file")
        }
        return attrs
    }

    private fun forceDirectory() {
        try {
            FileChannel.open(rootPath, StandardOpenOption.READ).use { it.force(true) }
        } catch (exc: Exception) {
            throw M6AuditIntegrityException("audit directory fsync failed", exc)
        }
    }

    private fun parseCanonicalReceipt(line: String): M6ActionReceipt {
        if (line.toByteArray(StandardCharsets.UTF_8).size > maxRecordBytes) {
            throw M6AuditCapacityException("audit record exceeds byte limit")
        }
        val parser = ReceiptParser(line)
        val receipt = parser.parse()
        val expectedId = receiptId(receipt)
        if (receipt.receiptId != expectedId) {
            throw M6AuditIntegrityException("audit receipt digest mismatch")
        }
        if (canonicalReceiptRecord(receipt) != line) {
            throw M6AuditIntegrityException("audit record is not canonical JSON")
        }
        return receipt
    }

    private fun receiptId(receipt: M6ActionReceipt): String = sha256Hex(
        canonicalReceiptPayload(receipt).toByteArray(StandardCharsets.UTF_8)
    )
}

private class ReceiptParser(private val text: String) {
    private var index = 0

    fun parse(): M6ActionReceipt {
        expect("{\"approval_id\":")
        val approvalId = string()
        expect(",\"capability_id\":")
        val capabilityId = string()
        expect(",\"confidence\":")
        val confidence = numberUntil(',').toDoubleOrNull()
            ?: corrupt("receipt confidence is invalid")
        if (!confidence.isFinite() || confidence !in 0.0..1.0) {
            corrupt("receipt confidence is outside [0,1]")
        }
        expect(",\"error_type\":")
        val errorType = string()
        expect(",\"evidence_record_ids\":[")
        val evidence = longArray()
        expect(",\"lease_id\":")
        val leaseId = string()
        expect(",\"plan_id\":")
        val planId = string()
        if (planId.isEmpty()) corrupt("receipt plan_id is empty")
        expect(",\"previous_receipt_id\":")
        val previous = string()
        if (previous.isNotEmpty()) requireSha(previous, "previous_receipt_id")
        expect(",\"principal\":")
        val principal = string()
        validatePrincipal(principal)
        expect(",\"receipt_id\":")
        val receiptId = string()
        requireSha(receiptId, "receipt_id")
        expect(",\"request_digest\":")
        val requestDigest = string()
        requireSha(requestDigest, "request_digest")
        expect(",\"result\":")
        val result = string()
        expect(",\"retryable\":")
        val retryable = bool()
        expect(",\"scope_digest\":")
        val scopeDigest = string()
        requireSha(scopeDigest, "scope_digest")
        expect(",\"sequence\":")
        val sequence = integerUntil(',').toIntOrNull()
            ?: corrupt("receipt sequence is outside Int range")
        if (sequence <= 0) corrupt("receipt sequence must be positive")
        expect(",\"status\":")
        val statusCode = integerUntil(',').toIntOrNull()
            ?: corrupt("receipt status is invalid")
        val status = M6ReceiptStatus.entries.firstOrNull { it.code == statusCode }
            ?: corrupt("receipt status is unknown")
        expect(",\"step_id\":")
        val stepId = integerUntil(',').toIntOrNull()
            ?: corrupt("receipt step_id is outside Int range")
        if (stepId <= 0) corrupt("receipt step_id must be positive")
        expect(",\"timestamp_ns\":")
        val timestampNs = integerUntil('}').toLongOrNull()
            ?: corrupt("receipt timestamp_ns is outside Long range")
        if (timestampNs < 0L) corrupt("receipt timestamp_ns must be non-negative")
        expect("}")
        if (index != text.length) corrupt("receipt contains trailing data")
        return M6ActionReceipt(
            sequence = sequence,
            timestampNs = timestampNs,
            previousReceiptId = previous,
            receiptId = receiptId,
            status = status,
            requestDigest = requestDigest,
            planId = planId,
            stepId = stepId,
            capabilityId = capabilityId,
            scopeDigest = scopeDigest,
            principal = principal,
            leaseId = leaseId,
            approvalId = approvalId,
            result = result,
            confidence = confidence,
            evidenceRecordIds = evidence,
            retryable = retryable,
            errorType = errorType,
        )
    }

    private fun longArray(): List<Long> {
        val out = ArrayList<Long>()
        val seen = HashSet<Long>()
        if (peek() == ']') {
            index += 1
            return out
        }
        while (true) {
            val token = integerUntil(',', ']')
            val value = token.toLongOrNull() ?: corrupt("evidence record ID is invalid")
            if (value <= 0L) corrupt("evidence record IDs must be positive")
            if (!seen.add(value)) corrupt("evidence record IDs must be unique")
            out += value
            when (val ch = peek()) {
                ',' -> index += 1
                ']' -> { index += 1; return out }
                else -> corrupt("invalid evidence list delimiter: $ch")
            }
        }
    }

    private fun string(): String {
        if (peek() != '"') corrupt("expected JSON string")
        index += 1
        val out = StringBuilder()
        while (index < text.length) {
            val ch = text[index++]
            when (ch) {
                '"' -> return out.toString()
                '\\' -> {
                    if (index >= text.length) corrupt("unterminated JSON escape")
                    when (val esc = text[index++]) {
                        '"', '\\', '/' -> out.append(esc)
                        'b' -> out.append('\b')
                        'f' -> out.append('\u000C')
                        'n' -> out.append('\n')
                        'r' -> out.append('\r')
                        't' -> out.append('\t')
                        'u' -> appendUnicodeEscape(out)
                        else -> corrupt("invalid JSON escape")
                    }
                }
                else -> {
                    if (ch.code < 0x20) corrupt("raw JSON control character")
                    when {
                        Character.isHighSurrogate(ch) -> {
                            if (index >= text.length || !Character.isLowSurrogate(text[index])) {
                                corrupt("JSON string contains unpaired surrogate")
                            }
                            out.append(ch)
                            out.append(text[index++])
                        }
                        Character.isLowSurrogate(ch) -> corrupt("JSON string contains unpaired surrogate")
                        else -> out.append(ch)
                    }
                }
            }
        }
        corrupt("unterminated JSON string")
    }

    private fun appendUnicodeEscape(out: StringBuilder) {
        val first = unicodeCodeUnit()
        val firstChar = first.toChar()
        when {
            Character.isHighSurrogate(firstChar) -> {
                if (index + 2 > text.length || text[index] != '\\' || text[index + 1] != 'u') {
                    corrupt("escaped high surrogate lacks low surrogate")
                }
                index += 2
                val second = unicodeCodeUnit().toChar()
                if (!Character.isLowSurrogate(second)) corrupt("escaped surrogate pair is invalid")
                out.append(firstChar).append(second)
            }
            Character.isLowSurrogate(firstChar) -> corrupt("escaped low surrogate is unpaired")
            else -> out.append(firstChar)
        }
    }

    private fun unicodeCodeUnit(): Int {
        if (index + 4 > text.length) corrupt("short unicode escape")
        var value = 0
        repeat(4) {
            val digit = text[index++].digitToIntOrNull(16) ?: corrupt("invalid unicode escape")
            value = (value shl 4) or digit
        }
        return value
    }

    private fun bool(): Boolean = when {
        text.startsWith("true", index) -> { index += 4; true }
        text.startsWith("false", index) -> { index += 5; false }
        else -> corrupt("receipt boolean is invalid")
    }

    private fun numberUntil(delimiter: Char): String {
        val start = index
        while (index < text.length && text[index] != delimiter) index += 1
        if (index == start || index >= text.length) corrupt("receipt number is malformed")
        return text.substring(start, index)
    }

    private fun integerUntil(vararg delimiters: Char): String {
        val start = index
        while (index < text.length && text[index] !in delimiters) index += 1
        if (index == start || index >= text.length) corrupt("receipt integer is malformed")
        val token = text.substring(start, index)
        if (token.any { it != '-' && !it.isDigit() }) corrupt("receipt integer contains non-digits")
        return token
    }

    private fun expect(literal: String) {
        if (!text.startsWith(literal, index)) corrupt("receipt schema mismatch")
        index += literal.length
    }

    private fun peek(): Char = if (index < text.length) text[index] else corrupt("unexpected end of receipt")
    private fun corrupt(message: String): Nothing = throw M6AuditIntegrityException(message)
}

private fun strictUtf8(bytes: ByteArray, offset: Int, length: Int): String = try {
    StandardCharsets.UTF_8.newDecoder()
        .onMalformedInput(CodingErrorAction.REPORT)
        .onUnmappableCharacter(CodingErrorAction.REPORT)
        .decode(ByteBuffer.wrap(bytes, offset, length))
        .toString()
} catch (exc: Exception) {
    throw M6AuditIntegrityException("audit contains invalid UTF-8", exc)
}

private fun validatePrincipal(value: String) {
    val allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._:-/".toSet()
    require(value.isNotEmpty()) { "principal must not be empty" }
    require(value.toByteArray(StandardCharsets.UTF_8).size <= 256) { "principal exceeds byte bound" }
    require(value.all { it in allowed }) { "principal contains unsupported characters" }
}

private fun requireSha(value: String, label: String) {
    if (value.length != 64 || value.any { it !in "0123456789abcdef" }) {
        throw M6AuditIntegrityException("$label must be lowercase SHA-256 hex")
    }
}

private fun sha256Hex(bytes: ByteArray): String =
    MessageDigest.getInstance("SHA-256").digest(bytes)
        .joinToString("") { "%02x".format(it.toInt() and 0xff) }
