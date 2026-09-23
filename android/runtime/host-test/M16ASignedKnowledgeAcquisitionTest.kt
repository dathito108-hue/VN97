package ai.vn97.runtime

import java.io.ByteArrayInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.security.KeyPair
import java.security.KeyPairGenerator
import java.security.MessageDigest
import java.security.Signature
import java.util.zip.CRC32
import kotlin.io.path.createTempDirectory

private fun sha(bytes: ByteArray): ByteArray =
    MessageDigest.getInstance("SHA-256").digest(bytes)

private fun hex(bytes: ByteArray): String =
    bytes.joinToString("") {
        "%02x".format(it.toInt() and 0xff)
    }

private fun knowledgePayload(): ByteArray =
    VnStrictJson.canonical(
        VnStrictJson.objectOf(
            "records" to VnStrictJson.array(
                listOf(
                    VnStrictJson.objectOf(
                        "content" to VnStrictJson.string(
                            "VN97 Mobile AGI uses one canonical cognition path."
                        ),
                        "title" to VnStrictJson.string(
                            "Canonical architecture"
                        ),
                    ),
                    VnStrictJson.objectOf(
                        "content" to VnStrictJson.string(
                            "Imported knowledge is evidence only and never execution authority."
                        ),
                        "title" to VnStrictJson.string(
                            "Authority boundary"
                        ),
                    ),
                )
            ),
            "schema" to VnStrictJson.string("VN97KN1"),
        )
    ).toByteArray()

private fun capabilityPackage(
    payload: ByteArray,
    capabilityId: String = "knowledge.vn97.architecture",
    kind: String = "knowledge",
    role: String = "knowledge_records",
    format: String = "VN97KN1",
): ByteArray {
    val payloadSha = hex(sha(payload))
    val manifest = VnStrictJson.canonical(
        VnStrictJson.objectOf(
            "capability_id" to
                VnStrictJson.string(capabilityId),
            "capability_version" to VnJsonNumber("1"),
            "kind" to VnStrictJson.string(kind),
            "schema" to VnStrictJson.string("VN97CAP1"),
            "sections" to VnStrictJson.array(
                listOf(
                    VnStrictJson.objectOf(
                        "format" to VnStrictJson.string(format),
                        "index" to VnJsonNumber("1"),
                        "role" to VnStrictJson.string(role),
                        "sha256" to
                            VnStrictJson.string(payloadSha),
                        "size" to
                            VnJsonNumber(
                                payload.size.toString()
                            ),
                    )
                )
            ),
            "source" to VnStrictJson.objectOf(
                "license" to VnStrictJson.string("test"),
                "origin" to
                    VnStrictJson.string("local-host-test"),
                "source_sha256" to
                    VnStrictJson.string("42".repeat(32)),
            ),
        )
    ).toByteArray()

    val sections = listOf(manifest, payload)
    val tableBytes = sections.size * 64
    val payloadOffset = 96 + tableBytes
    val totalSize =
        payloadOffset + sections.sumOf { it.size }
    val table =
        ByteBuffer.allocate(tableBytes)
            .order(ByteOrder.LITTLE_ENDIAN)
    var cursor = payloadOffset.toLong()
    sections.forEachIndexed { index, section ->
        table.putInt(if (index == 0) 1 else 2)
        table.putInt(0)
        table.putLong(cursor)
        table.putLong(section.size.toLong())
        table.put(sha(section))
        table.putLong(0L)
        cursor += section.size
    }
    val content =
        table.array() +
            sections.fold(ByteArray(0)) {
                    acc,
                    section,
                ->
                acc + section
            }
    val header =
        ByteBuffer.allocate(96)
            .order(ByteOrder.LITTLE_ENDIAN)
    header.put("VN97CAP1".toByteArray())
    header.putShort(1.toShort())
    header.putShort(96.toShort())
    header.putInt(0)
    header.putInt(sections.size)
    header.putInt(64)
    header.putLong(96L)
    header.putLong(payloadOffset.toLong())
    header.putLong(totalSize.toLong())
    header.putInt(0)
    header.putInt(0)
    header.put(sha(content))
    val crc = CRC32().apply {
        update(header.array(), 0, 88)
    }.value
    header.putInt(crc.toInt())
    header.putInt(0)
    return header.array() + content
}

private fun signatureEnvelope(
    packageBytes: ByteArray,
    keyPair: KeyPair,
    capabilityId: String =
        "knowledge.vn97.architecture",
): ByteArray {
    val packageSha = hex(sha(packageBytes))
    val claims = VnStrictJson.objectOf(
        "algorithm" to VnStrictJson.string("ed25519"),
        "capability_id" to
            VnStrictJson.string(capabilityId),
        "capability_version" to VnJsonNumber("1"),
        "key_id" to VnStrictJson.string("knowledge-owner"),
        "package_sha256" to
            VnStrictJson.string(packageSha),
        "schema" to VnStrictJson.string("VN97SIG1"),
    )
    val message =
        "VN97CAP1-SIGNATURE-V1\u0000".toByteArray() +
            VnStrictJson.canonical(claims).toByteArray()
    val signer = Signature.getInstance("Ed25519")
    signer.initSign(keyPair.private)
    signer.update(message)
    val signature = signer.sign()
    val values = claims.values.toMutableMap()
    values["signature"] =
        VnStrictJson.string(hex(signature))
    return VnStrictJson.canonical(
        VnJsonObject(values)
    ).toByteArray()
}

private class FakeMemory(
    override val vectorDim: Int = 8,
) : VN97KnowledgeMemoryPort {
    private val records =
        linkedMapOf<Long, VN97KnowledgeMemoryRecord>()
    private var lastId = 0L
    var crashAfterAppendOnce = false

    override fun lastRecordId(): Long = lastId

    override fun append(
        timestampNs: Long,
        source: String,
        content: String,
        vector: FloatArray,
    ): Long {
        check(timestampNs >= 0L)
        check(vector.size == vectorDim)
        val id = ++lastId
        records[id] = VN97KnowledgeMemoryRecord(
            recordId = id,
            source = source,
            content = content,
        )
        if (crashAfterAppendOnce) {
            crashAfterAppendOnce = false
            throw IllegalStateException(
                "simulated crash after durable memory append"
            )
        }
        return id
    }

    override fun recordOrNull(
        recordId: Long,
    ): VN97KnowledgeMemoryRecord? =
        records[recordId]

    fun size(): Int = records.size

    fun contents(): List<String> =
        records.values.map { it.content }
}

private fun newSession(
    root: java.io.File,
    stageRoot: java.io.File,
    memory: FakeMemory,
): VN97KnowledgeAcquisitionSession =
    VN97KnowledgeAcquisitionSession.forTest(
        stager = VN97CapabilityStager(stageRoot),
        stageRoot = stageRoot,
        trustRegistry =
            VN97KnowledgePublisherTrustRegistry(
                root.resolve("trust")
            ),
        ledger =
            VN97KnowledgeAcquisitionLedger(
                root.resolve("ledger")
            ),
        memory = memory,
        embedder = VN97KnowledgeEmbedder {
                text,
                dim,
            ->
            FloatArray(dim) { index ->
                (
                    (
                        text.hashCode().toLong() +
                            index +
                            1L
                    ) % 101L
                ).toFloat() + 1.0f
            }
        },
    )

private fun rawPublicKey(keyPair: KeyPair): ByteArray =
    keyPair.public.encoded.let {
        it.copyOfRange(it.size - 32, it.size)
    }

private inline fun expectFailure(
    label: String,
    block: () -> Unit,
) {
    check(runCatching(block).isFailure) {
        "expected M16A failure: $label"
    }
}

fun main() {
    val root =
        createTempDirectory("m16a-knowledge-").toFile()
    try {
        val stageRoot =
            root.resolve("stage").apply { mkdirs() }
        val payload = knowledgePayload()
        val packageBytes = capabilityPackage(payload)
        val keyPair =
            KeyPairGenerator.getInstance("Ed25519")
                .generateKeyPair()
        val publicKey = rawPublicKey(keyPair)
        val signature =
            signatureEnvelope(packageBytes, keyPair)
        val memory = FakeMemory()

        val firstSession =
            newSession(root, stageRoot, memory)
        val review = firstSession.review(
            packageInput =
                ByteArrayInputStream(packageBytes),
            signatureBytes = signature,
            publisherPublicKey = publicKey,
        )
        check(
            review.capabilityId ==
                "knowledge.vn97.architecture"
        )
        check(review.recordCount == 2)
        check(!review.publisherPreviouslyTrusted)
        check(memory.size() == 0)
        check(
            !root.resolve("trust")
                .resolve(
                    VN97_KNOWLEDGE_PUBLISHER_TRUST_FILE
                )
                .exists()
        )

        val acquired =
            firstSession.acquireReviewed(
                nowWallTimeMillis = 10_000L
            )
        check(!acquired.alreadyAcquired)
        check(acquired.recordIds == listOf(1L, 2L))
        check(memory.size() == 2)
        check(
            memory.contents().all {
                "\"authority\":\"evidence_only\"" in it
            }
        )
        check(
            memory.contents().all {
                "\"schema\":\"VN97KNMEM1\"" in it
            }
        )
        check(
            root.resolve("trust")
                .resolve(
                    VN97_KNOWLEDGE_PUBLISHER_TRUST_FILE
                )
                .isFile
        )

        val repeat =
            newSession(root, stageRoot, memory)
        val repeatReview = repeat.review(
            packageInput =
                ByteArrayInputStream(packageBytes),
            signatureBytes = signature,
            publisherPublicKey = publicKey,
        )
        check(repeatReview.publisherPreviouslyTrusted)
        val repeatResult =
            repeat.acquireReviewed(
                nowWallTimeMillis = 20_000L
            )
        check(repeatResult.alreadyAcquired)
        check(repeatResult.recordIds == listOf(1L, 2L))
        check(memory.size() == 2)

        val otherKey =
            KeyPairGenerator.getInstance("Ed25519")
                .generateKeyPair()
        expectFailure("publisher key rebinding") {
            newSession(root, stageRoot, memory)
                .review(
                    packageInput =
                        ByteArrayInputStream(packageBytes),
                    signatureBytes = signature,
                    publisherPublicKey =
                        rawPublicKey(otherKey),
                )
        }

        val badPackage =
            capabilityPackage(
                payload = payload,
                capabilityId = "avatar.bad",
                kind = "avatar",
            )
        val badSignature =
            signatureEnvelope(
                packageBytes = badPackage,
                keyPair = keyPair,
                capabilityId = "avatar.bad",
            )
        expectFailure("non-knowledge namespace") {
            newSession(root, stageRoot, memory)
                .review(
                    packageInput =
                        ByteArrayInputStream(badPackage),
                    signatureBytes = badSignature,
                    publisherPublicKey = publicKey,
                )
        }

        val crashRoot =
            createTempDirectory(
                "m16a-crash-"
            ).toFile()
        try {
            val crashStage =
                crashRoot.resolve("stage").apply {
                    mkdirs()
                }
            val crashMemory = FakeMemory().apply {
                crashAfterAppendOnce = true
            }
            val crashSession =
                newSession(
                    crashRoot,
                    crashStage,
                    crashMemory,
                )
            crashSession.review(
                ByteArrayInputStream(packageBytes),
                signature,
                publicKey,
            )
            expectFailure("simulated post-append crash") {
                crashSession.acquireReviewed(
                    nowWallTimeMillis = 30_000L
                )
            }
            check(crashMemory.size() == 1)

            val recovered =
                newSession(
                    crashRoot,
                    crashStage,
                    crashMemory,
                )
            recovered.review(
                ByteArrayInputStream(packageBytes),
                signature,
                publicKey,
            )
            val result =
                recovered.acquireReviewed(
                    nowWallTimeMillis = 40_000L
                )
            check(!result.alreadyAcquired)
            check(result.recordIds == listOf(1L, 2L))
            check(crashMemory.size() == 2)
        } finally {
            crashRoot.deleteRecursively()
        }

        println(
            "M16A signed knowledge acquisition contracts: PASS"
        )
    } finally {
        root.deleteRecursively()
    }
}
