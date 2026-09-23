import ai.vn97.runtime.*
import java.nio.file.Files

private inline fun expectContinuity(block: () -> Unit) {
    var failed = false
    try { block() } catch (_: NativeContinuityException) { failed = true }
    check(failed)
}

fun main() {
    val config = NativeRuntimeConfig(1, 1, 2, 1)
    val session = NativeRuntimeSession.create(config)
    session.activate()
    session.advance(10)
    session.suspend()
    val info = session.info()
    val checkpoint = session.checkpoint()
    val unbound = session.modelBinding()
    check(!unbound.bound)
    session.close()

    val plan = NativePlanController.create(
        "continuity",
        listOf(NativePlanStepSpec(NativeStepKind.REASON, "done")),
        createdNs = 1L,
    )
    plan.beginStep(1)
    plan.completeStep(1, "done", 1.0)

    val root = Files.createTempDirectory("vn97-m7m-").toFile()
    try {
        val store = AtomicCompositeContinuityStore(root)
        val modelId = ByteArray(32) { it.toByte() }

        expectContinuity {
            store.save(
                NativeRuntimeCheckpointSnapshot(checkpoint, info, unbound),
                plan.plan,
            )
        }

        val fakeBound = NativeRuntimeCheckpointSnapshot(
            checkpoint,
            info,
            RuntimeModelBinding(true, modelId),
        )
        val first = store.save(fakeBound, plan.plan)
        check(first.epoch == 1L && first.slot == "a")
        val second = store.save(fakeBound, plan.plan)
        check(second.epoch == 2L && second.slot == "b")

        val bundle = checkNotNull(store.loadOrNull())
        check(bundle.manifest.epoch == 2L)
        check(bundle.planner.plan.planId == plan.plan.planId)

        // Runtime checkpoint itself is unbound, so manifest model identity cannot forge a binding.
        expectContinuity { bundle.restoreRuntime(config) }

        // Corrupt inactive A; committed B remains readable.
        root.resolve("runtime-a.vn97run").writeBytes(ByteArray(64) { 9 })
        check(checkNotNull(store.loadOrNull()).manifest.epoch == 2L)

        // Corrupt active B; committed generation fails closed.
        root.resolve("runtime-b.vn97run").writeBytes(ByteArray(64) { 8 })
        expectContinuity { store.loadOrNull() }

        store.delete()
        check(store.loadOrNull() == null)
    } finally {
        root.deleteRecursively()
    }

    println("M7M_COMPOSITE_CONTINUITY_PASS")
}
