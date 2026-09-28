package ai.vn97.runtime

/** VN97-owned scheduling policy over one identity-bound recurrent graph/state contract. */
enum class VN97NativeSchedulePriority {
    THROUGHPUT,
    RESPONSIVE_PREFILL,
}

data class VN97NativeExecutionSegment(
    val offset: Int,
    val validLength: Int,
) {
    init {
        require(offset >= 0)
        require(validLength > 0)
    }
}

data class VN97NativeExecutionPlan(
    val totalTokens: Int,
    val maxChunkSize: Int,
    val priority: VN97NativeSchedulePriority,
    val segments: List<VN97NativeExecutionSegment>,
) {
    init {
        require(totalTokens in 1..MAX_TOKENS)
        require(maxChunkSize in SUPPORTED_CHUNKS)
        require(segments.isNotEmpty())
        var expectedOffset = 0
        segments.forEach { segment ->
            require(segment.offset == expectedOffset) {
                "VN97 execution plan has a gap or overlap"
            }
            require(segment.validLength <= maxChunkSize) {
                "VN97 execution segment exceeds graph chunk capacity"
            }
            expectedOffset = Math.addExact(
                expectedOffset,
                segment.validLength,
            )
        }
        require(expectedOffset == totalTokens) {
            "VN97 execution plan does not cover every token exactly once"
        }
    }

    companion object {
        const val MAX_TOKENS = 1_000_000
        val SUPPORTED_CHUNKS = setOf(8, 16, 32)
    }
}

object VN97NativeExecutionPlanner {
    /**
     * Builds a deterministic schedule for the same graph and recurrent state.
     * It never selects another model or changes weights. THROUGHPUT preserves
     * the existing largest-chunk behavior. RESPONSIVE_PREFILL starts at up to
     * eight tokens and doubles toward the graph maximum; it must be selected
     * only when first-work latency matters more than invocation count.
     */
    fun plan(
        totalTokens: Int,
        maxChunkSize: Int,
        priority: VN97NativeSchedulePriority =
            VN97NativeSchedulePriority.THROUGHPUT,
    ): VN97NativeExecutionPlan {
        require(totalTokens in 1..VN97NativeExecutionPlan.MAX_TOKENS)
        require(maxChunkSize in VN97NativeExecutionPlan.SUPPORTED_CHUNKS)
        val segments = mutableListOf<VN97NativeExecutionSegment>()
        var offset = 0
        var target = if (
            priority == VN97NativeSchedulePriority.RESPONSIVE_PREFILL
        ) {
            minOf(8, maxChunkSize)
        } else {
            maxChunkSize
        }
        while (offset < totalTokens) {
            val validLength = minOf(target, totalTokens - offset)
            segments += VN97NativeExecutionSegment(offset, validLength)
            offset = Math.addExact(offset, validLength)
            if (priority == VN97NativeSchedulePriority.RESPONSIVE_PREFILL) {
                target = minOf(maxChunkSize, Math.multiplyExact(target, 2))
            }
        }
        return VN97NativeExecutionPlan(
            totalTokens = totalTokens,
            maxChunkSize = maxChunkSize,
            priority = priority,
            segments = segments,
        )
    }
}
