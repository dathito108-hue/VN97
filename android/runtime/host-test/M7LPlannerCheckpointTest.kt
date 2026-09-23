import ai.vn97.runtime.*
import java.nio.file.Files
import java.security.MessageDigest

private inline fun expectCheckpoint(block: () -> Unit) {
    var failed = false
    try {
        block()
    } catch (_: NativePlannerCheckpointException) {
        failed = true
    }
    check(failed)
}

fun main() {
    val controller = NativePlanController.create(
        goal = "answer",
        specs = listOf(
            NativePlanStepSpec(
                kind = NativeStepKind.REASON,
                objective = "think",
                requiresVerification = true,
                minConfidence = 0.75,
            ),
            NativePlanStepSpec(
                kind = NativeStepKind.EXTERNAL,
                objective = "write",
                dependencies = listOf(1),
            ),
        ),
        createdNs = 1L,
    )
    controller.beginStep(1)
    controller.completeStep(1, "candidate", 0.8, listOf(7L))
    controller.verifyStep(1, true, "ok")
    controller.beginStep(2)

    val blob = NativePlannerCheckpoint.encode(controller.plan)
    check(blob.size == 861)
    val sha = MessageDigest.getInstance("SHA-256").digest(blob)
        .joinToString("") { "%02x".format(it.toInt() and 0xff) }
    check(sha == "eba8df81ddf32de6f7b2165bda26f8177a5de178ce57b040411ef7e7474c03b5")

    val restored = NativePlannerCheckpoint.decode(blob)
    check(restored.plan.planId == controller.plan.planId)
    check(restored.plan.status == NativePlanStatus.WAITING_EXTERNAL)
    check(restored.plan.transitionsUsed == 4)
    check(restored.plan.step(1).status == NativeStepStatus.SUCCEEDED)
    check(restored.plan.step(2).status == NativeStepStatus.WAITING_EXTERNAL)

    val badMagic = blob.copyOf()
    badMagic[0] = 'X'.code.toByte()
    expectCheckpoint { NativePlannerCheckpoint.decode(badMagic) }

    val badDigest = blob.copyOf()
    badDigest[20] = (badDigest[20].toInt() xor 1).toByte()
    expectCheckpoint { NativePlannerCheckpoint.decode(badDigest) }

    val badPayload = blob.copyOf()
    badPayload[badPayload.lastIndex] = (badPayload.last().toInt() xor 1).toByte()
    expectCheckpoint { NativePlannerCheckpoint.decode(badPayload) }

    val running = NativePlanController.create(
        goal = "running",
        specs = listOf(NativePlanStepSpec(NativeStepKind.REASON, "reason")),
        createdNs = 2L,
    )
    running.beginStep(1)
    expectCheckpoint { NativePlannerCheckpoint.encode(running.plan) }

    val root = Files.createTempDirectory("vn97-m7l-").toFile()
    try {
        val store = AtomicPlannerCheckpointStore(root)
        check(store.loadOrNull() == null)
        store.save(controller.plan)
        val loaded = checkNotNull(store.loadOrNull())
        check(loaded.plan.planId == controller.plan.planId)
        check(loaded.plan.status == NativePlanStatus.WAITING_EXTERNAL)
        store.delete()
        check(store.loadOrNull() == null)
    } finally {
        root.deleteRecursively()
    }

    println("M7L_PLANNER_CHECKPOINT_PASS")
}
