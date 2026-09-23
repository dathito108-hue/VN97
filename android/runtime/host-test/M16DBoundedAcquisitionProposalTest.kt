package ai.vn97.runtime

private inline fun expectM16DFailure(
    label: String,
    block: () -> Unit,
) {
    check(runCatching(block).isFailure) {
        "expected M16D failure: $label"
    }
}

private fun engine(
    output: String,
    evidence:
        List<VN97KnowledgeProposalEvidence> =
        listOf(
            VN97KnowledgeProposalEvidence(
                recordId = 11L,
                source =
                    "vn97.capability.knowledge",
                content =
                    "Existing memory covers the canonical VN97 architecture.",
            ),
            VN97KnowledgeProposalEvidence(
                recordId = 19L,
                source =
                    "vn97.trading.paper.performance",
                content =
                    "Existing memory has paper trading evidence only.",
            ),
        ),
): VN97KnowledgeAcquisitionProposalEngine =
    VN97KnowledgeAcquisitionProposalEngine(
        recall = { goal ->
            check(goal.isNotBlank())
            evidence
        },
        generator = { prompt ->
            check("VN97CAPGAP1" in prompt)
            check(
                "proposal_only" in prompt
            )
            check(
                "Never output a URL" in prompt
            )
            output
        },
    )

fun main() {
    val neededOutput =
        """{"needed":true,"capability_id":"knowledge.android.power","topic":"Android background power constraints","rationale":"The recalled evidence does not cover current platform power-management behavior.","source_hint":"official Android operating-system documentation for background execution limits","confidence_bps":8700}"""

    val first = engine(neededOutput)
        .propose(
            "Plan a battery-aware long-running Android capability."
        )
    check(first.needed)
    check(
        first.capabilityId ==
            "knowledge.android.power"
    )
    check(
        first.topic ==
            "Android background power constraints"
    )
    check(first.confidenceBasisPoints == 8700)
    check(
        first.evidenceRecordIds ==
            listOf(11L, 19L)
    )
    check(
        "://" !in first.sourceHint
    )

    val second = engine(
        "  $neededOutput  "
    ).propose(
        "Plan a battery-aware long-running Android capability."
    )
    check(second == first)

    val notNeeded =
        engine(
            """{"needed":false,"capability_id":"","topic":"","rationale":"Current bounded memory evidence is sufficient for this goal.","source_hint":"","confidence_bps":9200}"""
        ).propose(
            "Describe the existing canonical VN97 architecture."
        )
    check(!notNeeded.needed)
    check(notNeeded.capabilityId.isEmpty())
    check(notNeeded.topic.isEmpty())
    check(notNeeded.sourceHint.isEmpty())

    expectM16DFailure("actionable URL") {
        engine(
            """{"needed":true,"capability_id":"knowledge.android.power","topic":"Power","rationale":"Missing evidence.","source_hint":"https://developer.android.com/","confidence_bps":8000}"""
        ).propose("Need Android power rules")
    }

    expectM16DFailure("non-knowledge namespace") {
        engine(
            """{"needed":true,"capability_id":"plugin.android.power","topic":"Power","rationale":"Missing evidence.","source_hint":"official platform documentation","confidence_bps":8000}"""
        ).propose("Need Android power rules")
    }

    expectM16DFailure("unneeded target fields") {
        engine(
            """{"needed":false,"capability_id":"knowledge.bad","topic":"","rationale":"No need.","source_hint":"","confidence_bps":5000}"""
        ).propose("No extra knowledge")
    }

    expectM16DFailure("confidence overflow") {
        engine(
            """{"needed":false,"capability_id":"","topic":"","rationale":"No need.","source_hint":"","confidence_bps":10001}"""
        ).propose("No extra knowledge")
    }

    expectM16DFailure("extra output field") {
        engine(
            """{"needed":false,"capability_id":"","topic":"","rationale":"No need.","source_hint":"","confidence_bps":5000,"execute":true}"""
        ).propose("No extra knowledge")
    }

    expectM16DFailure("too much recall") {
        engine(
            output = neededOutput,
            evidence = (1L..5L).map { id ->
                VN97KnowledgeProposalEvidence(
                    recordId = id,
                    source = "test",
                    content = "evidence-$id",
                )
            },
        ).propose("Need bounded evidence")
    }

    expectM16DFailure("oversized goal") {
        engine(neededOutput)
            .propose("x".repeat(4 * 1024 + 1))
    }

    val unicode =
        VN97KnowledgeProposalEvidence.bounded(
            recordId = 77L,
            source = "nguồn",
            content = "ữ".repeat(2_000),
        )
    check(
        unicode.content.toByteArray(
            Charsets.UTF_8
        ).size <=
            VN97KnowledgeProposalEvidence
                .MAX_CONTENT_BYTES
    )
    check(unicode.content.isNotBlank())

    println(
        "M16D bounded acquisition proposal contracts: PASS"
    )
}
