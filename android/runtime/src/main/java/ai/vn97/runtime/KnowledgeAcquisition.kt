package ai.vn97.runtime

import java.io.File
import java.io.InputStream
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.charset.StandardCharsets
import java.nio.file.Files
import java.nio.file.LinkOption
import java.security.MessageDigest

data class VN97KnowledgeRecord(
    val title: String,
    val content: String,
) {
    init {
        requireUtf8Bound(
            title,
            VN97KnowledgePayload.MAX_TITLE_BYTES,
            "knowledge title",
            nonBlank = true,
        )
        requireUtf8Bound(
            content,
            VN97KnowledgePayload.MAX_CONTENT_BYTES,
            "knowledge content",
            nonBlank = true,
        )
    }
}

data class VN97KnowledgePayload(
    val records: List<VN97KnowledgeRecord>,
) {
    init {
        require(records.size in 1..MAX_RECORDS) {
            "VN97KN1 record count is outside bounds"
        }
    }

    companion object {
        const val MAX_RECORDS = 2048
        const val MAX_TITLE_BYTES = 256
        const val MAX_CONTENT_BYTES = 8 * 1024
        const val MAX_SECTION_BYTES = 8 * 1024 * 1024
    }
}

data class VN97KnowledgeAcquisitionReview(
    val capabilityId: String,
    val capabilityVersion: Long,
    val packageSha256: String,
    val publisherKeyId: String,
    val publisherKeySha256: String,
    val publisherPreviouslyTrusted: Boolean,
    val signatureSha256: String,
    val payloadSha256: String,
    val recordCount: Int,
    val sourceOrigin: String,
    val sourceLicense: String,
) {
    init {
        requireKnowledgeId(capabilityId, "capabilityId")
        require(capabilityId == "knowledge" || capabilityId.startsWith("knowledge."))
        require(capabilityVersion in 1L..0xffff_ffffL)
        requireKnowledgeSha(packageSha256, "packageSha256")
        requireKnowledgeId(publisherKeyId, "publisherKeyId")
        requireKnowledgeSha(publisherKeySha256, "publisherKeySha256")
        requireKnowledgeSha(signatureSha256, "signatureSha256")
        requireKnowledgeSha(payloadSha256, "payloadSha256")
        require(recordCount in 1..VN97KnowledgePayload.MAX_RECORDS)
        require(sourceOrigin.isNotBlank())
        require(sourceLicense.isNotBlank())
    }
}

data class VN97KnowledgeAcquisitionResult(
    val capabilityId: String,
    val capabilityVersion: Long,
    val packageSha256: String,
    val publisherKeyId: String,
    val recordIds: List<Long>,
    val alreadyAcquired: Boolean,
) {
    init {
        requireKnowledgeId(capabilityId, "capabilityId")
        require(capabilityVersion in 1L..0xffff_ffffL)
        requireKnowledgeSha(packageSha256, "packageSha256")
        requireKnowledgeId(publisherKeyId, "publisherKeyId")
        require(recordIds.isNotEmpty())
        require(recordIds.all { it > 0L })
        require(recordIds.zipWithNext().all { (a, b) -> b > a })
    }
}

internal data class VN97KnowledgeMemoryRecord(
    val recordId: Long,
    val source: String,
    val content: String,
)

internal interface VN97KnowledgeMemoryPort {
    val vectorDim: Int
    fun lastRecordId(): Long
    fun append(
        timestampNs: Long,
        source: String,
        content: String,
        vector: FloatArray,
    ): Long
    fun recordOrNull(recordId: Long): VN97KnowledgeMemoryRecord?
}

internal class NativeVN97KnowledgeMemoryPort(
    private val store: NativeMemoryStore,
) : VN97KnowledgeMemoryPort {
    override val vectorDim: Int
        get() = store.vectorDim

    override fun lastRecordId(): Long =
        store.stats().lastRecordId

    override fun append(
        timestampNs: Long,
        source: String,
        content: String,
        vector: FloatArray,
    ): Long =
        store.append(
            kind = NativeMemoryKind.SEMANTIC,
            timestampNs = timestampNs,
            importance = 0.85f,
            source = source,
            content = content,
            vector = vector,
            parentId = 0L,
            durable = true,
        )

    override fun recordOrNull(
        recordId: Long,
    ): VN97KnowledgeMemoryRecord? =
        runCatching {
            store.record(recordId)
        }.getOrNull()?.let {
            VN97KnowledgeMemoryRecord(
                recordId = it.recordId,
                source = it.source,
                content = it.content,
            )
        }
}

internal fun interface VN97KnowledgeEmbedder {
    fun embed(text: String, vectorDim: Int): FloatArray
}

class VN97KnowledgeAcquisitionSession private constructor(
    private val stager: VN97CapabilityStager,
    private val stageRoot: File,
    private val trustRegistry:
        VN97KnowledgePublisherTrustRegistry,
    private val ledger: VN97KnowledgeAcquisitionLedger,
    private val memory: VN97KnowledgeMemoryPort,
    private val embedder: VN97KnowledgeEmbedder,
) {
    private data class Pending(
        val verified: VN97VerifiedCapability,
        val publicKey: ByteArray,
        val payload: VN97KnowledgePayload,
        val payloadSha256: String,
        val review: VN97KnowledgeAcquisitionReview,
    )

    private var pending: Pending? = null

    init {
        require(memory.vectorDim > 0) {
            "knowledge acquisition memory vector dimension must be positive"
        }
    }

    @Synchronized
    fun review(
        packageInput: InputStream,
        signatureBytes: ByteArray,
        publisherPublicKey: ByteArray,
    ): VN97KnowledgeAcquisitionReview {
        check(pending == null) {
            "a knowledge acquisition review is already pending"
        }
        require(publisherPublicKey.size == 32) {
            "knowledge publisher Ed25519 public key must be exactly 32 bytes"
        }

        val staged =
            stager.stage(packageInput, signatureBytes)
        val key = VN97TrustedPublisherKey(
            keyId = staged.signature.keyId,
            publicKey = publisherPublicKey.copyOf(),
            capabilityPrefixes = setOf("knowledge"),
            allowedKinds = setOf("knowledge"),
            minVersion = 1L,
            maxVersion = 0xffff_ffffL,
            revoked = false,
        )
        val previouslyTrusted =
            trustRegistry.requireCompatibleOrUnenrolled(
                staged.signature.keyId,
                publisherPublicKey,
            )
        val verified =
            VN97CapabilityTrustVerifier.verify(
                staged = staged,
                stageRoot = stageRoot,
                trustStore =
                    VN97CapabilityTrustStore(listOf(key)),
            )
        val payload = requireKnowledgePayload(verified)
        val section =
            verified.parsed.manifest.sections.single()
        val review = VN97KnowledgeAcquisitionReview(
            capabilityId =
                verified.parsed.manifest.capabilityId,
            capabilityVersion =
                verified.parsed.manifest.capabilityVersion,
            packageSha256 =
                verified.parsed.packageSha256,
            publisherKeyId =
                verified.publisherKeyId,
            publisherKeySha256 =
                MessageDigest.getInstance("SHA-256")
                    .digest(publisherPublicKey)
                    .toKnowledgeHex(),
            publisherPreviouslyTrusted =
                previouslyTrusted,
            signatureSha256 =
                verified.signatureSha256,
            payloadSha256 =
                section.sha256,
            recordCount = payload.records.size,
            sourceOrigin =
                verified.parsed.manifest.source.origin,
            sourceLicense =
                verified.parsed.manifest.source.license,
        )
        pending = Pending(
            verified = verified,
            publicKey = publisherPublicKey.copyOf(),
            payload = payload,
            payloadSha256 = section.sha256,
            review = review,
        )
        return review
    }

    @Synchronized
    fun pendingReview():
        VN97KnowledgeAcquisitionReview? =
        pending?.review

    @Synchronized
    fun clearReview() {
        pending = null
    }

    /**
     * Explicit second phase. The exact publisher key is enrolled only here.
     * Verified knowledge is then imported into the existing VN97MEM1 store.
     */
    @Synchronized
    fun acquireReviewed(
        nowWallTimeMillis: Long =
            System.currentTimeMillis(),
    ): VN97KnowledgeAcquisitionResult {
        require(nowWallTimeMillis >= 0L) {
            "knowledge acquisition wall time must be non-negative"
        }
        val current = checkNotNull(pending) {
            "no knowledge acquisition review is pending"
        }

        val durableTrust =
            trustRegistry.enrollKnowledgePublisher(
                keyId = current.verified.publisherKeyId,
                publicKey = current.publicKey,
            )
        val fresh =
            VN97CapabilityTrustVerifier.verify(
                staged = current.verified.staged,
                stageRoot = stageRoot,
                trustStore = durableTrust,
            )
        check(fresh == current.verified) {
            "knowledge capability changed after review"
        }
        val freshPayload =
            requireKnowledgePayload(fresh)
        check(freshPayload == current.payload) {
            "knowledge payload changed after review"
        }

        val completed =
            ledger.loadOrNull(
                current.review.packageSha256
            )
        if (
            completed?.state ==
                VN97KnowledgeAcquisitionState.COMPLETED
        ) {
            requireLedgerMatches(current, completed)
            pending = null
            return resultFromLedger(
                completed,
                alreadyAcquired = true,
            )
        }

        var state =
            ledger.loadOrNull(
                current.review.packageSha256
            ) ?: VN97KnowledgeAcquisitionLedgerRecord(
                packageSha256 =
                    current.review.packageSha256,
                capabilityId =
                    current.review.capabilityId,
                capabilityVersion =
                    current.review.capabilityVersion,
                publisherKeyId =
                    current.review.publisherKeyId,
                signatureSha256 =
                    current.review.signatureSha256,
                payloadSha256 =
                    current.payloadSha256,
                totalRecords =
                    current.payload.records.size,
                state =
                    VN97KnowledgeAcquisitionState.IMPORTING,
                nextIndex = 0,
                recordIds = emptyList(),
                createdWallTimeMillis =
                    nowWallTimeMillis,
                updatedWallTimeMillis =
                    nowWallTimeMillis,
            ).also(ledger::save)
        requireLedgerMatches(current, state)

        state = reconcilePending(
            current = current,
            state = state,
            nowWallTimeMillis = nowWallTimeMillis,
        )

        while (state.nextIndex < state.totalRecords) {
            val index = state.nextIndex
            val knowledge = current.payload.records[index]
            val memoryContent =
                canonicalMemoryContent(
                    current,
                    index,
                    knowledge,
                )
            val contentSha =
                MessageDigest.getInstance("SHA-256")
                    .digest(
                        memoryContent.toByteArray(
                            StandardCharsets.UTF_8
                        )
                    )
                    .toKnowledgeHex()
            val preAppendLastId =
                memory.lastRecordId()
            val pendingState = state.copy(
                pendingIndex = index,
                pendingPreAppendLastRecordId =
                    preAppendLastId,
                pendingContentSha256 = contentSha,
                updatedWallTimeMillis =
                    maxOf(
                        state.updatedWallTimeMillis,
                        nowWallTimeMillis,
                    ),
            )
            ledger.save(pendingState)

            val vector = embedder.embed(
                memoryContent,
                memory.vectorDim,
            )
            require(
                vector.size == memory.vectorDim &&
                    vector.all { it.isFinite() }
            ) {
                "knowledge embedding is invalid"
            }
            val recordId = memory.append(
                timestampNs =
                    wallMillisToNs(nowWallTimeMillis),
                source = MEMORY_SOURCE,
                content = memoryContent,
                vector = vector,
            )
            check(recordId > preAppendLastId) {
                "knowledge memory append returned non-advancing record ID"
            }

            state = pendingState.copy(
                state =
                    if (index + 1 ==
                        pendingState.totalRecords
                    ) {
                        VN97KnowledgeAcquisitionState.COMPLETED
                    } else {
                        VN97KnowledgeAcquisitionState.IMPORTING
                    },
                nextIndex = index + 1,
                recordIds =
                    pendingState.recordIds + recordId,
                pendingIndex = -1,
                pendingPreAppendLastRecordId = 0L,
                pendingContentSha256 = "",
                updatedWallTimeMillis =
                    maxOf(
                        pendingState.updatedWallTimeMillis,
                        nowWallTimeMillis,
                    ),
            )
            ledger.save(state)
        }

        check(
            state.state ==
                VN97KnowledgeAcquisitionState.COMPLETED
        ) {
            "knowledge acquisition finished records without completion state"
        }
        pending = null
        return resultFromLedger(
            state,
            alreadyAcquired = false,
        )
    }

    private fun reconcilePending(
        current: Pending,
        state: VN97KnowledgeAcquisitionLedgerRecord,
        nowWallTimeMillis: Long,
    ): VN97KnowledgeAcquisitionLedgerRecord {
        if (state.pendingIndex < 0) {
            return state
        }
        val index = state.pendingIndex
        check(index == state.nextIndex)
        val expectedContent =
            canonicalMemoryContent(
                current,
                index,
                current.payload.records[index],
            )
        val expectedSha =
            MessageDigest.getInstance("SHA-256")
                .digest(
                    expectedContent.toByteArray(
                        StandardCharsets.UTF_8
                    )
                )
                .toKnowledgeHex()
        check(expectedSha == state.pendingContentSha256) {
            "knowledge pending content identity changed"
        }

        val currentLast = memory.lastRecordId()
        check(
            currentLast >=
                state.pendingPreAppendLastRecordId
        ) {
            "knowledge memory record ID moved backwards"
        }
        val delta =
            currentLast -
                state.pendingPreAppendLastRecordId
        check(delta <= MAX_RECONCILE_SCAN.toLong()) {
            "knowledge acquisition crash reconciliation scan exceeds bound"
        }
        val matches = ArrayList<Long>()
        var id =
            state.pendingPreAppendLastRecordId + 1L
        while (id <= currentLast) {
            val record = memory.recordOrNull(id)
            if (
                record != null &&
                record.source == MEMORY_SOURCE &&
                record.content == expectedContent
            ) {
                matches += id
            }
            id += 1L
        }
        check(matches.size <= 1) {
            "knowledge acquisition found duplicate pending memory records"
        }
        if (matches.size == 1) {
            val recordId = matches.single()
            val recovered = state.copy(
                state =
                    if (index + 1 ==
                        state.totalRecords
                    ) {
                        VN97KnowledgeAcquisitionState.COMPLETED
                    } else {
                        VN97KnowledgeAcquisitionState.IMPORTING
                    },
                nextIndex = index + 1,
                recordIds = state.recordIds + recordId,
                pendingIndex = -1,
                pendingPreAppendLastRecordId = 0L,
                pendingContentSha256 = "",
                updatedWallTimeMillis =
                    maxOf(
                        state.updatedWallTimeMillis,
                        nowWallTimeMillis,
                    ),
            )
            ledger.save(recovered)
            return recovered
        }

        val cleared = state.copy(
            pendingIndex = -1,
            pendingPreAppendLastRecordId = 0L,
            pendingContentSha256 = "",
            updatedWallTimeMillis =
                maxOf(
                    state.updatedWallTimeMillis,
                    nowWallTimeMillis,
                ),
        )
        ledger.save(cleared)
        return cleared
    }

    private fun requireLedgerMatches(
        current: Pending,
        state: VN97KnowledgeAcquisitionLedgerRecord,
    ) {
        check(
            state.packageSha256 ==
                current.review.packageSha256 &&
                state.capabilityId ==
                    current.review.capabilityId &&
                state.capabilityVersion ==
                    current.review.capabilityVersion &&
                state.publisherKeyId ==
                    current.review.publisherKeyId &&
                state.signatureSha256 ==
                    current.review.signatureSha256 &&
                state.payloadSha256 ==
                    current.payloadSha256 &&
                state.totalRecords ==
                    current.payload.records.size
        ) {
            "knowledge acquisition ledger identity does not match reviewed package"
        }
    }

    private fun resultFromLedger(
        state: VN97KnowledgeAcquisitionLedgerRecord,
        alreadyAcquired: Boolean,
    ): VN97KnowledgeAcquisitionResult =
        VN97KnowledgeAcquisitionResult(
            capabilityId = state.capabilityId,
            capabilityVersion = state.capabilityVersion,
            packageSha256 = state.packageSha256,
            publisherKeyId = state.publisherKeyId,
            recordIds = state.recordIds,
            alreadyAcquired = alreadyAcquired,
        )

    private fun canonicalMemoryContent(
        current: Pending,
        index: Int,
        knowledge: VN97KnowledgeRecord,
    ): String =
        VnStrictJson.canonical(
            VnStrictJson.objectOf(
                "authority" to
                    VnStrictJson.string("evidence_only"),
                "capability_id" to
                    VnStrictJson.string(
                        current.review.capabilityId
                    ),
                "capability_version" to
                    VnStrictJson.long(
                        current.review.capabilityVersion
                    ),
                "content" to
                    VnStrictJson.string(
                        knowledge.content
                    ),
                "package_sha256" to
                    VnStrictJson.string(
                        current.review.packageSha256
                    ),
                "publisher_key_id" to
                    VnStrictJson.string(
                        current.review.publisherKeyId
                    ),
                "record_index" to
                    VnStrictJson.int(index),
                "schema" to
                    VnStrictJson.string(
                        "VN97KNMEM1"
                    ),
                "source_license" to
                    VnStrictJson.string(
                        current.review.sourceLicense
                    ),
                "source_origin" to
                    VnStrictJson.string(
                        current.review.sourceOrigin
                    ),
                "title" to
                    VnStrictJson.string(knowledge.title),
            )
        )

    companion object {
        const val MEMORY_SOURCE =
            "vn97.capability.knowledge"
        private const val MAX_RECONCILE_SCAN = 4096

        fun production(
            stageRoot: File,
            trustRoot: File,
            ledgerRoot: File,
            memory: NativeMemoryStore,
            inference: NativeCognitionInference,
        ): VN97KnowledgeAcquisitionSession =
            VN97KnowledgeAcquisitionSession(
                stager =
                    VN97CapabilityStager(stageRoot),
                stageRoot = stageRoot,
                trustRegistry =
                    VN97KnowledgePublisherTrustRegistry(
                        trustRoot
                    ),
                ledger =
                    VN97KnowledgeAcquisitionLedger(
                        ledgerRoot
                    ),
                memory =
                    NativeVN97KnowledgeMemoryPort(memory),
                embedder =
                    VN97KnowledgeEmbedder {
                            text,
                            vectorDim,
                        ->
                        inference.embedText(
                            text,
                            vectorDim,
                        )
                    },
            )

        internal fun forTest(
            stager: VN97CapabilityStager,
            stageRoot: File,
            trustRegistry:
                VN97KnowledgePublisherTrustRegistry,
            ledger: VN97KnowledgeAcquisitionLedger,
            memory: VN97KnowledgeMemoryPort,
            embedder: VN97KnowledgeEmbedder,
        ): VN97KnowledgeAcquisitionSession =
            VN97KnowledgeAcquisitionSession(
                stager = stager,
                stageRoot = stageRoot,
                trustRegistry = trustRegistry,
                ledger = ledger,
                memory = memory,
                embedder = embedder,
            )
    }
}

private fun requireKnowledgePayload(
    verified: VN97VerifiedCapability,
): VN97KnowledgePayload {
    val manifest = verified.parsed.manifest
    require(
        manifest.kind == "knowledge" &&
            (
                manifest.capabilityId == "knowledge" ||
                    manifest.capabilityId.startsWith(
                        "knowledge."
                    )
            )
    ) {
        "knowledge acquisition accepts only knowledge namespace/kind"
    }
    require(manifest.sections.size == 1) {
        "knowledge acquisition requires exactly one data section"
    }
    val section = manifest.sections.single()
    require(
        section.role == "knowledge_records" &&
            section.format == "VN97KN1"
    ) {
        "knowledge acquisition requires knowledge_records/VN97KN1"
    }
    require(
        section.size in
            1L..VN97KnowledgePayload.MAX_SECTION_BYTES.toLong()
    ) {
        "VN97KN1 section size is outside bounds"
    }
    val path = verified.staged.packageFile.toPath()
    require(
        Files.isRegularFile(
            path,
            LinkOption.NOFOLLOW_LINKS,
        ) &&
            !Files.isSymbolicLink(path)
    ) {
        "knowledge package must remain a safe regular file"
    }

    val bytes = RandomAccessFile(
        verified.staged.packageFile,
        "r",
    ).use { file ->
        check(
            section.packageOffset >= 0L &&
                section.packageOffset <=
                    file.length() &&
                section.size <=
                    file.length() -
                        section.packageOffset
        ) {
            "knowledge section range is invalid"
        }
        val out = ByteArray(section.size.toInt())
        file.seek(section.packageOffset)
        file.readFully(out)
        out
    }
    val actualSha =
        MessageDigest.getInstance("SHA-256")
            .digest(bytes)
            .toKnowledgeHex()
    check(actualSha == section.sha256) {
        "knowledge section SHA-256 changed before import"
    }
    return parseKnowledgePayload(bytes)
}

private fun parseKnowledgePayload(
    bytes: ByteArray,
): VN97KnowledgePayload {
    require(
        bytes.isNotEmpty() &&
            bytes.size <=
                VN97KnowledgePayload.MAX_SECTION_BYTES
    ) {
        "VN97KN1 payload size is outside bounds"
    }
    val text = strictKnowledgeUtf8(bytes)
    require(0.toChar() !in text) {
        "VN97KN1 payload contains NUL"
    }
    val root = VnStrictJson.parseObject(
        text,
        VnJsonLimits(
            maxInputUtf8Bytes =
                VN97KnowledgePayload.MAX_SECTION_BYTES,
            maxDepth = 16,
            maxNodes =
                VN97KnowledgePayload.MAX_RECORDS * 4 +
                    32,
            maxStringUtf8Bytes =
                VN97KnowledgePayload.MAX_CONTENT_BYTES,
        ),
    )
    check(VnStrictJson.canonical(root) == text) {
        "VN97KN1 must be canonical JSON"
    }
    require(
        root.values.keys == setOf("records", "schema") &&
            root.stringValue("schema") == "VN97KN1"
    ) {
        "VN97KN1 schema/fields are invalid"
    }
    val rawRecords =
        root.arrayValue("records").values
    require(
        rawRecords.size in
            1..VN97KnowledgePayload.MAX_RECORDS
    ) {
        "VN97KN1 record count is outside bounds"
    }
    val records = rawRecords.mapIndexed { index, raw ->
        val obj = raw as? VnJsonObject
            ?: throw IllegalArgumentException(
                "VN97KN1 record $index must be object"
            )
        require(
            obj.values.keys ==
                setOf("content", "title")
        ) {
            "VN97KN1 record fields are invalid"
        }
        VN97KnowledgeRecord(
            title = obj.stringValue("title"),
            content = obj.stringValue("content"),
        )
    }
    return VN97KnowledgePayload(records)
}

private fun VnJsonObject.stringValue(
    key: String,
): String =
    (values[key] as? VnJsonString)?.value
        ?: throw IllegalArgumentException(
            "VN97KN1 $key must be string"
        )

private fun VnJsonObject.arrayValue(
    key: String,
): VnJsonArray =
    values[key] as? VnJsonArray
        ?: throw IllegalArgumentException(
            "VN97KN1 $key must be array"
        )

private fun requireUtf8Bound(
    value: String,
    maxBytes: Int,
    label: String,
    nonBlank: Boolean,
) {
    if (nonBlank) {
        require(value.isNotBlank()) {
            "$label must not be blank"
        }
    }
    require(
        value.toByteArray(StandardCharsets.UTF_8).size <=
            maxBytes
    ) {
        "$label exceeds UTF-8 byte bound"
    }
    require(0.toChar() !in value) {
        "$label contains NUL"
    }
}

private fun wallMillisToNs(value: Long): Long =
    Math.multiplyExact(value, 1_000_000L)
