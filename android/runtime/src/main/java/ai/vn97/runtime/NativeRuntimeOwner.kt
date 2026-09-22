package ai.vn97.runtime

class NativeRuntimeOwner(
    private val checkpointStore: AtomicCheckpointStore,
) : AutoCloseable {
    private var session: NativeRuntimeSession? = null

    @Synchronized
    fun restoreOrCreate(config: NativeRuntimeConfig): NativeRuntimeInfo {
        check(session == null) { "runtime owner already has a session" }
        val checkpoint = checkpointStore.loadOrNull()
        val opened = if (checkpoint == null) {
            NativeRuntimeSession.create(config)
        } else {
            NativeRuntimeSession.restore(checkpoint)
        }
        try {
            val info = opened.info()
            if (checkpoint != null && info.config != config) {
                throw IllegalStateException("VN97RUN1 checkpoint configuration does not match requested runtime")
            }
            session = opened
            return info
        } catch (exc: Throwable) {
            opened.close()
            throw exc
        }
    }

    @Synchronized
    fun info(): NativeRuntimeInfo = requireSession().info()

    @Synchronized
    fun activate(): NativeRuntimeInfo {
        val runtime = requireSession()
        runtime.activate()
        return runtime.info()
    }

    @Synchronized
    fun resume(): NativeRuntimeInfo {
        val runtime = requireSession()
        runtime.resume()
        return runtime.info()
    }

    @Synchronized
    fun advance(tokenCount: Long): NativeRuntimeInfo {
        val runtime = requireSession()
        runtime.advance(tokenCount)
        return runtime.info()
    }

    @Synchronized
    fun suspendAndPersist(): NativeRuntimeInfo {
        val runtime = requireSession()
        val before = runtime.info()
        if (before.lifecycle == RuntimeLifecycle.ACTIVE) {
            runtime.suspend()
        } else {
            check(before.lifecycle == RuntimeLifecycle.SUSPENDED) {
                "runtime must be ACTIVE or SUSPENDED before persistence"
            }
        }
        checkpointStore.save(runtime.checkpoint())
        return runtime.info()
    }

    @Synchronized
    fun deleteCheckpoint() = checkpointStore.delete()

    @Synchronized
    override fun close() {
        session?.close()
        session = null
    }

    private fun requireSession(): NativeRuntimeSession =
        checkNotNull(session) { "runtime owner has no open session" }
}
