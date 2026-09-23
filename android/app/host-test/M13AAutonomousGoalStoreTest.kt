import ai.vn97.app.VN97AutonomousGoalRecord
import ai.vn97.app.VN97AutonomousGoalState
import ai.vn97.app.VN97AutonomousGoalStore
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
