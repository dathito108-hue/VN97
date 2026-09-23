package ai.vn97.platform

import java.nio.charset.StandardCharsets

internal data class VN97GameTouchStroke(
    val startXBasisPoints: Int,
    val startYBasisPoints: Int,
    val endXBasisPoints: Int,
    val endYBasisPoints: Int,
    val startMillis: Long,
    val durationMillis: Long,
)

internal object VN97GameMultiTouchContract {
    const val BASIS_POINTS = 10_000
    const val MIN_STROKES = 2
    const val MAX_STROKES = 4
    const val MIN_STROKE_MILLIS = 20L
    const val MAX_DURATION_MILLIS = 3_000L
    const val MAX_PAYLOAD_UTF8_BYTES = 1_024

    const val PAYLOAD_SCHEMA_JSON =
        "{\"strokes\":[{\"duration_ms\":600,\"end_x_bps\":2600,\"end_y_bps\":7600,\"start_ms\":0,\"start_x_bps\":1800,\"start_y_bps\":8200},{\"duration_ms\":80,\"end_x_bps\":8600,\"end_y_bps\":7200,\"start_ms\":120,\"start_x_bps\":8600,\"start_y_bps\":7200}]}"

    fun parseCanonical(payloadJson: String): List<VN97GameTouchStroke> {
        require(
            payloadJson.toByteArray(StandardCharsets.UTF_8).size <=
                MAX_PAYLOAD_UTF8_BYTES
        ) {
            "game multi-touch payload exceeds byte bound"
        }
        val wrapper = PAYLOAD.matchEntire(payloadJson)
            ?: throw IllegalArgumentException(
                "game multi-touch payload must match canonical schema"
            )
        val body = wrapper.groupValues[1]
        require(body.isNotEmpty()) {
            "game multi-touch strokes must not be empty"
        }
        val matches = STROKE.findAll(body).toList()
        require(matches.size in MIN_STROKES..MAX_STROKES) {
            "game multi-touch stroke count is outside bounds"
        }
        require(matches.joinToString(",") { it.value } == body) {
            "game multi-touch strokes must be canonical and contiguous"
        }

        val strokes = matches.map { match ->
            VN97GameTouchStroke(
                durationMillis = match.groupValues[1].toLong(),
                endXBasisPoints = match.groupValues[2].toInt(),
                endYBasisPoints = match.groupValues[3].toInt(),
                startMillis = match.groupValues[4].toLong(),
                startXBasisPoints = match.groupValues[5].toInt(),
                startYBasisPoints = match.groupValues[6].toInt(),
            )
        }
        requireValid(strokes)
        return strokes
    }

    fun requireValid(strokes: List<VN97GameTouchStroke>) {
        require(strokes.size in MIN_STROKES..MAX_STROKES) {
            "game multi-touch stroke count is outside bounds"
        }

        var previousStart = -1L
        strokes.forEach { stroke ->
            require(
                stroke.durationMillis in
                    MIN_STROKE_MILLIS..MAX_DURATION_MILLIS
            ) {
                "game multi-touch stroke duration is outside bounds"
            }
            require(stroke.startMillis in 0L..MAX_DURATION_MILLIS) {
                "game multi-touch stroke start is outside bounds"
            }
            require(stroke.startMillis >= previousStart) {
                "game multi-touch strokes must be ordered by start_ms"
            }
            previousStart = stroke.startMillis
            listOf(
                stroke.startXBasisPoints,
                stroke.startYBasisPoints,
                stroke.endXBasisPoints,
                stroke.endYBasisPoints,
            ).forEach { coordinate ->
                require(coordinate in 0..BASIS_POINTS) {
                    "game multi-touch coordinate is outside basis-point bounds"
                }
            }
            val endMillis =
                Math.addExact(stroke.startMillis, stroke.durationMillis)
            require(endMillis <= MAX_DURATION_MILLIS) {
                "game multi-touch gesture duration exceeds bound"
            }
        }

        require(hasTemporalOverlap(strokes)) {
            "game multi-touch requires overlapping strokes"
        }
    }

    private fun hasTemporalOverlap(
        strokes: List<VN97GameTouchStroke>,
    ): Boolean {
        for (leftIndex in strokes.indices) {
            val left = strokes[leftIndex]
            val leftEnd = left.startMillis + left.durationMillis
            for (rightIndex in leftIndex + 1 until strokes.size) {
                val right = strokes[rightIndex]
                val rightEnd = right.startMillis + right.durationMillis
                if (
                    left.startMillis < rightEnd &&
                    right.startMillis < leftEnd
                ) {
                    return true
                }
            }
        }
        return false
    }

    private val PAYLOAD =
        Regex("^\\{\\\"strokes\\\":\\[(.*)]}$")

    private val STROKE =
        Regex(
            "\\{\\\"duration_ms\\\":([0-9]+)," +
                "\\\"end_x_bps\\\":([0-9]+)," +
                "\\\"end_y_bps\\\":([0-9]+)," +
                "\\\"start_ms\\\":([0-9]+)," +
                "\\\"start_x_bps\\\":([0-9]+)," +
                "\\\"start_y_bps\\\":([0-9]+)\\}"
        )
}
