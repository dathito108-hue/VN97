package ai.vn97.app

data class VN97AutonomousSchedulePolicy(
    val notBeforeWallTimeMillis: Long = 0L,
    val deadlineWallTimeMillis: Long = 0L,
    val dependencyKey: String = "",
    val powerPolicy: VN97AutonomousPowerPolicy =
        VN97AutonomousPowerPolicy.ADAPTIVE,
) {
    init {
        require(notBeforeWallTimeMillis >= 0L) {
            "notBeforeWallTimeMillis must be non-negative"
        }
        require(deadlineWallTimeMillis >= 0L) {
            "deadlineWallTimeMillis must be non-negative"
        }
        if (
            deadlineWallTimeMillis > 0L &&
            notBeforeWallTimeMillis > 0L
        ) {
            require(deadlineWallTimeMillis >= notBeforeWallTimeMillis) {
                "deadlineWallTimeMillis precedes notBeforeWallTimeMillis"
            }
        }
        require(
            dependencyKey.toByteArray(Charsets.UTF_8).size <=
                VN97AutonomousGoalRecord.MAX_DEPENDENCY_KEY_BYTES
        ) {
            "dependencyKey exceeds UTF-8 byte bound"
        }
        require(dependencyKey.isEmpty() || dependencyKey.isNotBlank()) {
            "dependencyKey must not contain only whitespace"
        }
    }
}

internal data class VN97AutonomousScheduleWindow(
    val eligible: Boolean,
    val expired: Boolean,
    val minimumLatencyMillis: Long,
    val overrideDeadlineMillis: Long,
)

internal fun VN97AutonomousGoalRecord.scheduleWindow(
    nowWallTimeMillis: Long,
): VN97AutonomousScheduleWindow {
    require(nowWallTimeMillis >= 0L) {
        "nowWallTimeMillis must be non-negative"
    }
    val expired =
        deadlineWallTimeMillis > 0L &&
            nowWallTimeMillis >= deadlineWallTimeMillis
    if (expired) {
        return VN97AutonomousScheduleWindow(
            eligible = false,
            expired = true,
            minimumLatencyMillis = 0L,
            overrideDeadlineMillis = 0L,
        )
    }

    val dependencyReady =
        dependencyKey.isEmpty() || dependencySatisfied
    val minimumLatencyMillis =
        (notBeforeWallTimeMillis - nowWallTimeMillis)
            .coerceAtLeast(0L)
    val eligible =
        dependencyReady && minimumLatencyMillis == 0L

    val overrideDeadlineMillis =
        if (deadlineWallTimeMillis > 0L) {
            (deadlineWallTimeMillis - nowWallTimeMillis)
                .coerceAtLeast(1L)
        } else {
            0L
        }

    return VN97AutonomousScheduleWindow(
        eligible = eligible,
        expired = false,
        minimumLatencyMillis = minimumLatencyMillis,
        overrideDeadlineMillis = overrideDeadlineMillis,
    )
}
