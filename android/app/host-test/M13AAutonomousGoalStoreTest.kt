import ai.vn97.app.VN97AutonomousGoalRecord
import ai.vn97.app.VN97AutonomousGoalState
import ai.vn97.app.VN97AutonomousGoalStore
import ai.vn97.app.VN97AutonomousPowerPolicy
import java.nio.file.Files

private inline fun expectFailure(block: () -> Unit) {
    var failed = false
    try {
        block()
    } catch (_: Exception) {
        failed = true
    }
    check(failed)
}

fun main() {
    val root = Files.createTempDirectory("m13a-goals-").toFile()
    val store = VN97AutonomousGoalStore(root)
    val base = VN97AutonomousGoalRecord(
        jobId = 0x40000061,
        planId = "11".repeat(32),
        modelIdHex = "22".repeat(32),
        principal = "runtime.user",
        goal = "finish a durable background task",
        state = VN97AutonomousGoalState.SCHEDULED,
        createdNs = 100L,
        updatedNs = 100L,
    )

    store.save(base)
    check(store.loadOrNull(base.jobId) == base)
    check(store.list() == listOf(base))

    val running = base.copy(
        state = VN97AutonomousGoalState.RUNNING,
        updatedNs = 200L,
        wakeCount = 1,
    )
    store.save(running)
    check(store.loadOrNull(base.jobId) == running)

    val waiting = running.copy(
        state = VN97AutonomousGoalState.WAITING_APPROVAL,
        updatedNs = 300L,
        terminalReason = "foreground approval required",
    )
    store.save(waiting)
    check(store.loadOrNull(base.jobId) == waiting)

    expectFailure {
        store.save(
            waiting.copy(
                planId = "33".repeat(32),
                updatedNs = 400L,
            )
        )
    }
    expectFailure {
        store.save(
            waiting.copy(
                wakeCount = 0,
                updatedNs = 400L,
            )
        )
    }
    expectFailure {
        VN97AutonomousGoalRecord(
            jobId = 1,
            planId = "11".repeat(32),
            modelIdHex = "22".repeat(32),
            principal = "runtime.user",
            goal = "bad completed record",
            state = VN97AutonomousGoalState.COMPLETED,
            createdNs = 1L,
            updatedNs = 2L,
        )
    }

    val completed = waiting.copy(
        state = VN97AutonomousGoalState.COMPLETED,
        updatedNs = 500L,
        finalResponse = "done",
        terminalReason = "",
    )
    store.save(completed)
    check(store.loadOrNull(base.jobId) == completed)

    val successor = VN97AutonomousGoalRecord(
        jobId = 0x40000062,
        planId = "44".repeat(32),
        modelIdHex = base.modelIdHex,
        principal = base.principal,
        goal = base.goal,
        state = VN97AutonomousGoalState.SCHEDULED,
        rootJobId = base.jobId,
        generation = 1,
        previousJobId = base.jobId,
        createdNs = 600L,
        updatedNs = 600L,
    )
    store.save(successor)
    check(store.loadOrNull(successor.jobId) == successor)
    check(store.list().contains(successor))

    expectFailure {
        VN97AutonomousGoalRecord(
            jobId = 0x40000063,
            planId = "55".repeat(32),
            modelIdHex = base.modelIdHex,
            principal = base.principal,
            goal = base.goal,
            state = VN97AutonomousGoalState.SCHEDULED,
            rootJobId = base.jobId,
            generation = 1,
            previousJobId = 0,
            createdNs = 700L,
            updatedNs = 700L,
        )
    }

    val longHorizon = VN97AutonomousGoalRecord(
        jobId = 0x40000064,
        planId = "66".repeat(32),
        modelIdHex = base.modelIdHex,
        principal = base.principal,
        goal = "continue after a durable event",
        state = VN97AutonomousGoalState.WAITING_DEPENDENCY,
        createdNs = 800L,
        updatedNs = 800L,
        notBeforeWallTimeMillis = 10_000L,
        deadlineWallTimeMillis = 20_000L,
        dependencyKey = "network.ready",
        dependencySatisfied = false,
        powerPolicy = VN97AutonomousPowerPolicy.BATTERY_NOT_LOW,
    )
    store.save(longHorizon)
    check(store.loadOrNull(longHorizon.jobId) == longHorizon)

    val dependencySatisfied = longHorizon.copy(
        state = VN97AutonomousGoalState.SCHEDULED,
        updatedNs = 900L,
        dependencySatisfied = true,
        scheduleAttemptCount = 1,
        lastScheduledWallTimeMillis = 10_000L,
    )
    store.save(dependencySatisfied)
    check(
        store.loadOrNull(longHorizon.jobId) ==
            dependencySatisfied
    )

    expectFailure {
        store.save(
            dependencySatisfied.copy(
                updatedNs = 1_000L,
                dependencySatisfied = false,
            )
        )
    }
    expectFailure {
        store.save(
            dependencySatisfied.copy(
                updatedNs = 1_000L,
                scheduleAttemptCount = 0,
            )
        )
    }

    val file = root
        .resolve(base.jobId.toString())
        .resolve("goal.vn97goa1")
    val original = file.readBytes()
    val corrupt = original.copyOf()
    corrupt[corrupt.lastIndex] =
        (corrupt.last().toInt() xor 1).toByte()
    file.writeBytes(corrupt)
    expectFailure { store.loadOrNull(base.jobId) }
    file.writeBytes(original)
    check(store.loadOrNull(base.jobId) == completed)

    store.delete(base.jobId)
    check(store.loadOrNull(base.jobId) == null)

    println("M13A_AUTONOMOUS_GOAL_STORE_PASS")
}
