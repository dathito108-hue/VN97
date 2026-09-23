package ai.vn97.runtime

import java.io.File

class AtomicPlannerCheckpointStore(
    root: File,
    fileName: String = "planner.vn97pln1",
    maxCheckpointBytes: Int = (16 shl 20) + 48,
) {
    private val store = AtomicBinaryStore(
        root = root,
        fileName = fileName,
        minBytes = 48,
        maxBytes = maxCheckpointBytes,
        tempPrefix = ".vn97-planner-checkpoint-",
    )

    fun save(plan: NativePlan) = store.save(NativePlannerCheckpoint.encode(plan))
    fun loadOrNull(): NativePlanController? = store.loadOrNull()?.let(NativePlannerCheckpoint::decode)
    fun delete() = store.delete()
}
