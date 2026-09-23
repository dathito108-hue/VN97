package ai.vn97.runtime

import java.nio.charset.StandardCharsets
import java.security.MessageDigest

data class VN97KnowledgeProposalEvidence(
    val recordId: Long,
    val source: String,
    val content: String,
) {
    init {
        require(recordId > 0L) {
            "knowledge proposal evidence record ID must be positive"
        }
        requireBoundedProposalText(
            source,
            MAX_SOURCE_BYTES,
            "knowledge proposal evidence source",
            nonBlank = true,
        )
        requireBoundedProposalText(
            content,
            MAX_CONTENT_BYTES,
            "knowledge proposal evidence content",
            nonBlank = true,
        )
    }

    companion object {
        const val MAX_SOURCE_BYTES = 128
        const val MAX_CONTENT_BYTES = 1_024
    }
}

data class VN97KnowledgeGapProposal(
    val proposalId: String,
    val needed: Boolean,
    val capabilityId: String,
    val topic: String,
    val rationale: String,
    val sourceHint: String,
    val confidenceBasisPoints: Int,
    val evidenceRecordIds: List<Long>,
) {
    init {
        requireProposalSha(
            proposalId,
            "knowledge proposal ID",
        )
        require(confidenceBasisPoints in 0..10_000) {
            "knowledge proposal confidence must be 0..10000 basis points"
        }
        require(
            evidenceRecordIds.size <=
                VN97KnowledgeAcquisitionProposalEngine.MAX_EVIDENCE
        )
        require(
            evidenceRecordIds.all { it > 0L } &&
                evidenceRecordIds.distinct().size ==
                    evidenceRecordIds.size
        ) {
            "knowledge proposal evidence IDs are invalid"
        }
        requireBoundedProposalText(
            rationale,
            MAX_RATIONALE_BYTES,
            "knowledge proposal rationale",
            nonBlank = true,
        )
        if (needed) {
            requireKnowledgeProposalId(capabilityId)
            requireBoundedProposalText(
                topic,
                MAX_TOPIC_BYTES,
                "knowledge proposal topic",
                nonBlank = true,
            )
            requireBoundedProposalText(
                sourceHint,
                MAX_SOURCE_HINT_BYTES,
                "knowledge proposal source hint",
                nonBlank = true,
            )
            require(
                "://" !in sourceHint &&
                    !sourceHint.trimStart()
                        .startsWith(
                            "http",
                            ignoreCase = true,
                        )
            ) {
                "knowledge proposal source hint must not contain an actionable URL"
            }
        } else {
            require(
                capabilityId.isEmpty() &&
                    topic.isEmpty() &&
                    sourceHint.isEmpty()
            ) {
                "unneeded knowledge proposal must not contain acquisition target fields"
            }
        }
    }

    companion object {
        const val MAX_TOPIC_BYTES = 512
        const val MAX_RATIONALE_BYTES = 2 * 1_024
        const val MAX_SOURCE_HINT_BYTES = 512
    }
}

fun interface VN97KnowledgeProposalRecall {
    fun recall(goal: String): List<VN97KnowledgeProposalEvidence>
}

fun interface VN97KnowledgeProposalGenerator {
    fun generate(prompt: String): String
}

class VN97KnowledgeAcquisitionProposalEngine(
    private val recall: VN97KnowledgeProposalRecall,
    private val generator: VN97KnowledgeProposalGenerator,
) {
    fun propose(goal: String): VN97KnowledgeGapProposal {
        requireBoundedProposalText(
            goal,
            MAX_GOAL_BYTES,
            "knowledge acquisition goal",
            nonBlank = true,
        )
        val evidence = recall.recall(goal)
        require(evidence.size <= MAX_EVIDENCE) {
            "knowledge proposal recall exceeded evidence bound"
        }
        require(
            evidence.map { it.recordId }
                .distinct().size == evidence.size
        ) {
            "knowledge proposal recall returned duplicate evidence IDs"
        }

        val request = VnStrictJson.canonical(
            VnStrictJson.objectOf(
                "authority" to
                    VnStrictJson.string(
                        "proposal_only"
                    ),
                "evidence" to
                    VnStrictJson.array(
                        evidence.map { item ->
                            VnStrictJson.objectOf(
                                "content" to
                                    VnStrictJson.string(
                                        item.content
                                    ),
                                "record_id" to
                                    VnStrictJson.long(
                                        item.recordId
                                    ),
                                "source" to
                                    VnStrictJson.string(
                                        item.source
                                    ),
                            )
                        }
                    ),
                "goal" to VnStrictJson.string(goal),
                "schema" to
                    VnStrictJson.string(
                        "VN97CAPGAP1"
                    ),
            )
        )

        val prompt = buildString {
            append("VN97CAPGAP1")
            append('\n')
            append(
                "Assess only whether additional external KNOWLEDGE would materially help the user goal beyond the supplied memory evidence. "
            )
            append(
                "This is advisory only: do not execute tools, do not authorize anything, do not invent permissions, and do not propose code/plugins/scripts/native libraries. "
            )
            append(
                "Never output a URL. source_hint must be a non-actionable description of the kind of authoritative source a human could choose. "
            )
            append(
                "If no additional knowledge is needed, set needed=false and capability_id/topic/source_hint to empty strings. "
            )
            append(
                "If needed=true, capability_id must start with knowledge. and describe only a data knowledge namespace. "
            )
            append(
                "Return exactly one JSON object with keys: needed(bool), capability_id(string), topic(string), rationale(string), source_hint(string), confidence_bps(integer 0..10000). No markdown or extra prose."
            )
            append('\n')
            append("request=")
            append(request)
        }
        require(
            prompt.toByteArray(StandardCharsets.UTF_8).size <=
                MAX_PROMPT_BYTES
        ) {
            "knowledge proposal prompt exceeds byte bound"
        }

        val output = generator.generate(prompt)
        require(
            output.isNotBlank() &&
                output.toByteArray(
                    StandardCharsets.UTF_8
                ).size <= MAX_OUTPUT_BYTES
        ) {
            "knowledge proposal output is empty or too large"
        }
        val root = try {
            VnStrictJson.parseObject(
                output,
                VnJsonLimits(
                    maxInputUtf8Bytes =
                        MAX_OUTPUT_BYTES,
                    maxDepth = 8,
                    maxNodes = 64,
                    maxStringUtf8Bytes =
                        VN97KnowledgeGapProposal
                            .MAX_RATIONALE_BYTES,
                ),
            )
        } catch (exc: RuntimeException) {
            throw NativeCognitionContractException(
                "knowledge proposal output is not strict JSON",
                exc,
            )
        }
        require(
            root.values.keys == setOf(
                "needed",
                "capability_id",
                "topic",
                "rationale",
                "source_hint",
                "confidence_bps",
            )
        ) {
            "knowledge proposal output keys are invalid"
        }

        val needed =
            root.proposalBool("needed")
        val capabilityId =
            root.proposalString("capability_id")
        val topic =
            root.proposalString("topic")
        val rationale =
            root.proposalString("rationale")
        val sourceHint =
            root.proposalString("source_hint")
        val confidence =
            root.proposalInt(
                "confidence_bps",
                0,
                10_000,
            )

        val identity = VnStrictJson.canonical(
            VnStrictJson.objectOf(
                "evidence_record_ids" to
                    VnStrictJson.array(
                        evidence.map {
                            VnStrictJson.long(
                                it.recordId
                            )
                        }
                    ),
                "goal_sha256" to
                    VnStrictJson.string(
                        proposalSha(
                            goal.toByteArray(
                                StandardCharsets.UTF_8
                            )
                        )
                    ),
                "response" to root,
                "schema" to
                    VnStrictJson.string(
                        "VN97CAPGAPID1"
                    ),
            )
        )
        val proposal = VN97KnowledgeGapProposal(
            proposalId =
                proposalSha(
                    identity.toByteArray(
                        StandardCharsets.UTF_8
                    )
                ),
            needed = needed,
            capabilityId = capabilityId,
            topic = topic,
            rationale = rationale,
            sourceHint = sourceHint,
            confidenceBasisPoints = confidence,
            evidenceRecordIds =
                evidence.map { it.recordId },
        )
        return proposal
    }

    companion object {
        const val MAX_EVIDENCE = 4
        const val MAX_GOAL_BYTES = 4 * 1_024
        const val MAX_PROMPT_BYTES = 16 * 1_024
        const val MAX_OUTPUT_BYTES = 8 * 1_024
    }
}

private fun VnJsonObject.proposalString(
    key: String,
): String =
    (values[key] as? VnJsonString)?.value
        ?: throw NativeCognitionContractException(
            "knowledge proposal $key must be string"
        )

private fun VnJsonObject.proposalBool(
    key: String,
): Boolean =
    (values[key] as? VnJsonBoolean)?.value
        ?: throw NativeCognitionContractException(
            "knowledge proposal $key must be bool"
        )

private fun VnJsonObject.proposalInt(
    key: String,
    minimum: Int,
    maximum: Int,
): Int {
    val raw =
        (values[key] as? VnJsonNumber)?.canonical
            ?: throw NativeCognitionContractException(
                "knowledge proposal $key must be integer"
            )
    if (
        raw.any {
            it == '.' ||
                it == 'e' ||
                it == 'E'
        }
    ) {
        throw NativeCognitionContractException(
            "knowledge proposal $key must be integer"
        )
    }
    val value = raw.toIntOrNull()
        ?: throw NativeCognitionContractException(
            "knowledge proposal $key is outside integer range"
        )
    if (value !in minimum..maximum) {
        throw NativeCognitionContractException(
            "knowledge proposal $key is outside bounds"
        )
    }
    return value
}

private fun requireKnowledgeProposalId(
    value: String,
) {
    require(
        value.startsWith("knowledge.") &&
            value.length <= 128 &&
            value.all {
                it in 'a'..'z' ||
                    it in '0'..'9' ||
                    it == '.' ||
                    it == '_' ||
                    it == '-'
            }
    ) {
        "knowledge proposal capability ID is invalid"
    }
}

private fun requireBoundedProposalText(
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
    require(0.toChar() !in value) {
        "$label contains NUL"
    }
    require(
        value.toByteArray(
            StandardCharsets.UTF_8
        ).size <= maxBytes
    ) {
        "$label exceeds UTF-8 byte bound"
    }
}

private fun requireProposalSha(
    value: String,
    label: String,
) {
    require(
        value.length == 64 &&
            value.all {
                it in "0123456789abcdef"
            }
    ) {
        "$label must be lowercase SHA-256"
    }
}

private fun proposalSha(
    bytes: ByteArray,
): String =
    MessageDigest.getInstance("SHA-256")
        .digest(bytes)
        .joinToString("") {
            "%02x".format(
                it.toInt() and 0xff
            )
        }
