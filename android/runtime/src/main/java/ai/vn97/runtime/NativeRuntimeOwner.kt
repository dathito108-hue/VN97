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
                throw IllegalStateException("VN97RUN checkpoint configuration does not match requested runtime")
            }
            session = opened
            return info
        } catch (exc: Throwable) {
            opened.close()
            throw exc
        }
    }

    @Synchronized
    fun restoreComposite(
        config: NativeRuntimeConfig,
        continuity: NativeCompositeContinuity,
    ): NativeRuntimeInfo {
        check(session == null) { "runtime owner already has a session" }
        val opened = continuity.restoreRuntime(config)
        return try {
            val info = opened.info()
            session = opened
            info
        } catch (exc: Throwable) {
            opened.close()
            throw exc
        }
    }

    @Synchronized
    fun info(): NativeRuntimeInfo = requireSession().info()

    @Synchronized
    fun modelBinding(): RuntimeModelBinding = requireSession().modelBinding()

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
    fun inferStep(model: NativeActivatedModel, inputIds: IntArray): FloatArray =
        requireSession().inferStep(model, inputIds)

    @Synchronized
    fun prefill(model: NativeActivatedModel, inputIds: IntArray): FloatArray =
        requireSession().prefill(model, inputIds)

    @Synchronized
    fun generateGreedy(
        model: NativeActivatedModel,
        promptIds: IntArray,
        maxNewTokens: Int,
        eosToken: Int = 2,
    ): IntArray = requireSession().generateGreedy(model, promptIds, maxNewTokens, eosToken)

    @Synchronized
    fun chatStreaming(
        model: NativeActivatedModel,
        userMessage: String,
        config: NativeChatConfig = NativeChatConfig(),
        cancellation: NativeChatCancellation = NativeChatCancellation(),
        onDelta: (NativeChatDelta) -> Boolean,
    ): NativeChatTurnResult = requireSession().chatStreaming(
        model = model,
        userMessage = userMessage,
        config = config,
        cancellation = cancellation,
        onDelta = onDelta,
    )

    @Synchronized
    fun suspendAndSnapshot(): NativeRuntimeCheckpointSnapshot {
        val runtime = requireSession()
        val before = runtime.info()
        if (before.lifecycle == RuntimeLifecycle.ACTIVE) {
            runtime.suspend()
        } else {
            check(before.lifecycle == RuntimeLifecycle.SUSPENDED) {
                "runtime must be ACTIVE or SUSPENDED before persistence"
            }
        }
        val info = runtime.info()
        val binding = runtime.modelBinding()
        return NativeRuntimeCheckpointSnapshot(
            checkpoint = runtime.checkpoint(),
            info = info,
            modelBinding = RuntimeModelBinding(binding.bound, binding.modelId.copyOf()),
        )
    }

    @Synchronized
    fun suspendAndPersist(): NativeRuntimeInfo {
        val snapshot = suspendAndSnapshot()
        checkpointStore.save(snapshot.checkpoint)
        return snapshot.info
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
