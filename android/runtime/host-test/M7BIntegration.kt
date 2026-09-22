import ai.vn97.runtime.*
import java.nio.file.Files

fun main() {
    val config = NativeRuntimeConfig(2, 1, 3, 2)
    val session = NativeRuntimeSession.create(config)
    check(session.info().stateCount == 12L)
    val state = FloatArray(12) { it + 0.5f }
    session.writeState(state)
    session.activate()
    session.advance(9)
    session.suspend()
    val checkpoint = session.checkpoint()
    check(checkpoint.size == 64 + 12 * 4)
    session.close()

    val restored = NativeRuntimeSession.restore(checkpoint)
    check(restored.info().lifecycle == RuntimeLifecycle.SUSPENDED)
    check(restored.info().sequencePosition == 9L)
    check(restored.readState().contentEquals(state))
    restored.close()

    val root = Files.createTempDirectory("vn97-m7b").toFile()
    try {
        val store = AtomicCheckpointStore(root, maxCheckpointBytes = 4096)
        store.save(checkpoint)
        check(store.loadOrNull()!!.contentEquals(checkpoint))

        NativeRuntimeOwner(store).use { owner ->
            val info = owner.restoreOrCreate(config)
            check(info.lifecycle == RuntimeLifecycle.SUSPENDED)
            owner.resume()
            owner.advance(1)
            val persisted = owner.suspendAndPersist()
            check(persisted.sequencePosition == 10L)
        }
        NativeRuntimeOwner(store).use { owner ->
            val info = owner.restoreOrCreate(config)
            check(info.sequencePosition == 10L)
            check(info.lifecycle == RuntimeLifecycle.SUSPENDED)
        }
        store.delete()
        check(store.loadOrNull() == null)
    } finally {
        root.deleteRecursively()
    }

    println("M7B_JNI_KOTLIN_INTEGRATION_PASS")
}
