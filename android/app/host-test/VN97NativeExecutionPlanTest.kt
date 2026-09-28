package ai.vn97.runtime

fun main() {
    val throughput = VN97NativeExecutionPlanner.plan(
        totalTokens = 73,
        maxChunkSize = 32,
    )
    check(throughput.segments.map { it.validLength } == listOf(32, 32, 9))
    check(throughput.segments.map { it.offset } == listOf(0, 32, 64))

    val responsive = VN97NativeExecutionPlanner.plan(
        totalTokens = 73,
        maxChunkSize = 32,
        priority = VN97NativeSchedulePriority.RESPONSIVE_PREFILL,
    )
    check(responsive.segments.map { it.validLength } == listOf(8, 16, 32, 17))
    check(responsive.segments.sumOf { it.validLength } == 73)

    for (chunk in listOf(8, 16, 32)) {
        for (tokens in listOf(1, 7, 8, 9, 31, 32, 33, 257)) {
            val plan = VN97NativeExecutionPlanner.plan(tokens, chunk)
            check(plan.segments.sumOf { it.validLength } == tokens)
            check(plan.segments.all { it.validLength in 1..chunk })
            check(plan.segments.zipWithNext().all { (left, right) ->
                right.offset == left.offset + left.validLength
            })
        }
    }

    check(runCatching { VN97NativeExecutionPlanner.plan(0, 8) }.isFailure)
    check(runCatching { VN97NativeExecutionPlanner.plan(1, 7) }.isFailure)
    check(runCatching {
        VN97NativeExecutionPlan(
            totalTokens = 2,
            maxChunkSize = 8,
            priority = VN97NativeSchedulePriority.THROUGHPUT,
            segments = listOf(VN97NativeExecutionSegment(0, 1)),
        )
    }.isFailure)
    check(runCatching {
        VN97NativeExecutionPlan(
            totalTokens = 2,
            maxChunkSize = 8,
            priority = VN97NativeSchedulePriority.THROUGHPUT,
            segments = listOf(
                VN97NativeExecutionSegment(0, 1),
                VN97NativeExecutionSegment(0, 1),
            ),
        )
    }.isFailure)
    println("VN97 native execution plan: exact coverage/bounds/priorities PASS")
}
