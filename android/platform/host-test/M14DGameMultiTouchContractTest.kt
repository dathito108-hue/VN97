package ai.vn97.platform

private fun expectFailure(label: String, block: () -> Unit) {
    check(runCatching(block).isFailure) {
        "expected failure: $label"
    }
}

private fun payload(vararg strokes: String): String =
    "{\"strokes\":[" + strokes.joinToString(",") + "]}"

private fun stroke(
    duration: Long,
    endX: Int,
    endY: Int,
    startMs: Long,
    startX: Int,
    startY: Int,
): String =
    "{\"duration_ms\":$duration,\"end_x_bps\":$endX,\"end_y_bps\":$endY," +
        "\"start_ms\":$startMs,\"start_x_bps\":$startX,\"start_y_bps\":$startY}"

fun main() {
    val move = stroke(
        duration = 600,
        endX = 2600,
        endY = 7600,
        startMs = 0,
        startX = 1800,
        startY = 8200,
    )
    val skill = stroke(
        duration = 80,
        endX = 8600,
        endY = 7200,
        startMs = 120,
        startX = 8600,
        startY = 7200,
    )

    val parsed =
        VN97GameMultiTouchContract.parseCanonical(
            payload(move, skill)
        )
    check(parsed.size == 2)
    check(parsed[0].startMillis == 0L)
    check(parsed[0].durationMillis == 600L)
    check(parsed[1].startMillis == 120L)
    check(parsed[1].startXBasisPoints == 8600)
    check(parsed[1].startXBasisPoints == parsed[1].endXBasisPoints)
    check(parsed[1].startYBasisPoints == parsed[1].endYBasisPoints)

    val third = stroke(
        duration = 70,
        endX = 7400,
        endY = 6400,
        startMs = 180,
        startX = 7400,
        startY = 6400,
    )
    val fourth = stroke(
        duration = 60,
        endX = 6500,
        endY = 6100,
        startMs = 220,
        startX = 6500,
        startY = 6100,
    )
    check(
        VN97GameMultiTouchContract.parseCanonical(
            payload(move, skill, third, fourth)
        ).size == 4
    )

    expectFailure("one pointer is not multi-touch") {
        VN97GameMultiTouchContract.parseCanonical(payload(move))
    }

    expectFailure("more than four strokes") {
        VN97GameMultiTouchContract.parseCanonical(
            payload(move, skill, third, fourth, fourth)
        )
    }

    expectFailure("out-of-bounds coordinate") {
        VN97GameMultiTouchContract.parseCanonical(
            payload(
                move,
                stroke(
                    duration = 80,
                    endX = 10_001,
                    endY = 7200,
                    startMs = 120,
                    startX = 8600,
                    startY = 7200,
                ),
            )
        )
    }

    expectFailure("gesture exceeds three seconds") {
        VN97GameMultiTouchContract.parseCanonical(
            payload(
                stroke(
                    duration = 2900,
                    endX = 2600,
                    endY = 7600,
                    startMs = 200,
                    startX = 1800,
                    startY = 8200,
                ),
                skill,
            )
        )
    }

    expectFailure("unordered start times") {
        VN97GameMultiTouchContract.parseCanonical(
            payload(skill, move)
        )
    }

    expectFailure("strokes do not overlap") {
        VN97GameMultiTouchContract.parseCanonical(
            payload(
                stroke(
                    duration = 80,
                    endX = 2600,
                    endY = 7600,
                    startMs = 0,
                    startX = 1800,
                    startY = 8200,
                ),
                stroke(
                    duration = 80,
                    endX = 8600,
                    endY = 7200,
                    startMs = 100,
                    startX = 8600,
                    startY = 7200,
                ),
            )
        )
    }

    expectFailure("non-canonical whitespace") {
        VN97GameMultiTouchContract.parseCanonical(
            "{\"strokes\":[ $move,$skill]}"
        )
    }

    expectFailure("non-canonical field order") {
        VN97GameMultiTouchContract.parseCanonical(
            "{\"strokes\":[{\"start_ms\":0,\"duration_ms\":600," +
                "\"end_x_bps\":2600,\"end_y_bps\":7600," +
                "\"start_x_bps\":1800,\"start_y_bps\":8200},$skill]}"
        )
    }

    println("M14D multi-touch contract: PASS")
}
