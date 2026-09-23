package ai.vn97.runtime

import java.io.File

class AtomicCheckpointStore(
    root: File,
    fileName: String = "runtime.vn97run1",
    maxCheckpointBytes: Int = 512 * 1024 * 1024 + 100,
) {
    private val store = AtomicBinaryStore(
        root = root,
        fileName = fileName,
        minBytes = 64,
        maxBytes = maxCheckpointBytes,
        tempPrefix = ".vn97-runtime-checkpoint-",
    )

    fun save(checkpoint: ByteArray) = store.save(checkpoint)
    fun loadOrNull(): ByteArray? = store.loadOrNull()
    fun delete() = store.delete()
}
