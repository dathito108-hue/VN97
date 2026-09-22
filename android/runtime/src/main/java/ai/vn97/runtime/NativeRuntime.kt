package ai.vn97.runtime

enum class NativeBackend(val code: Int) {
    AUTO(0),
    SCALAR(1),
    ARM64_NEON(2),
}

enum class RuntimeLifecycle(val code: Int) {
    CREATED(0),
    ACTIVE(1),
    SUSPENDED(2);

    companion object {
        internal fun fromCode(code: Int): RuntimeLifecycle =
            entries.firstOrNull { it.code == code }
                ?: throw IllegalStateException("unknown native runtime lifecycle: $code")
    }
}

enum class NativeRuntimeStatus(val code: Int) {
    OK(0),
    NULL_ARGUMENT(1),
    INVALID_CONFIG(2),
    SIZE_OVERFLOW(3),
    INVALID_HANDLE(4),
    INVALID_LIFECYCLE(5),
    OUTPUT_TOO_SMALL(6),
    CHECKPOINT_CORRUPT(7),
    CHECKPOINT_MISMATCH(8),
    COUNTER_OVERFLOW(9),
    BACKEND_UNAVAILABLE(10),
    MODEL_MISMATCH(11),
    INFERENCE_ERROR(12);

    companion object {
        internal fun fromCode(code: Int): NativeRuntimeStatus =
            entries.firstOrNull { it.code == code }
                ?: throw IllegalStateException("unknown native runtime status: $code")
    }
}

class NativeRuntimeException(
    val status: NativeRuntimeStatus,
    operation: String,
) : IllegalStateException("$operation failed: ${status.name} (${status.code})")

data class NativeRuntimeConfig(
    val layers: Int,
    val batch: Int,
    val dModel: Int,
    val dState: Int,
    val recurrentBackend: NativeBackend = NativeBackend.AUTO,
    val packedBackend: NativeBackend = NativeBackend.AUTO,
) {
    init {
        require(layers > 0) { "layers must be positive" }
        require(batch > 0) { "batch must be positive" }
        require(dModel > 0) { "dModel must be positive" }
        require(dState > 0) { "dState must be positive" }
    }

    internal fun toNative(): IntArray = intArrayOf(
        layers,
        batch,
        dModel,
        dState,
        recurrentBackend.code,
        packedBackend.code,
    )
}

data class NativeRuntimeInfo(
    val config: NativeRuntimeConfig,
    val resolvedRecurrentBackend: NativeBackend,
    val resolvedPackedBackend: NativeBackend,
    val lifecycle: RuntimeLifecycle,
    val stateCount: Long,
    val sequencePosition: Long,
)

internal object NativeRuntimeBindings {
    init {
        System.loadLibrary("vn97_jni")
    }

    external fun nativeCreate(config: IntArray, handleOut: LongArray): Int
    external fun nativeRestore(blob: ByteArray, handleOut: LongArray): Int
    external fun nativeDestroy(handle: Long): Int
    external fun nativeActivate(handle: Long): Int
    external fun nativeSuspend(handle: Long): Int
    external fun nativeResume(handle: Long): Int
    external fun nativeAdvance(handle: Long, tokenCount: Long): Int
    external fun nativeInfo(handle: Long, intsOut: IntArray, longsOut: LongArray): Int
    external fun nativeStateRead(handle: Long, out: FloatArray): Int
    external fun nativeStateWrite(handle: Long, state: FloatArray): Int
    external fun nativeCheckpointSize(handle: Long, out: LongArray): Int
    external fun nativeCheckpointWrite(handle: Long, out: ByteArray, writtenOut: LongArray): Int
    external fun nativeFsyncDirectory(path: String): Int
}

class NativeRuntimeSession private constructor(private var handle: Long) : AutoCloseable {
    private val lock = Any()

    companion object {
        fun create(config: NativeRuntimeConfig): NativeRuntimeSession {
            val out = LongArray(1)
            checkStatus(
                NativeRuntimeBindings.nativeCreate(config.toNative(), out),
                "runtime create",
            )
            check(out[0] != 0L) { "native runtime returned a zero handle" }
            return NativeRuntimeSession(out[0])
        }

        fun restore(checkpoint: ByteArray): NativeRuntimeSession {
            require(checkpoint.isNotEmpty()) { "checkpoint must not be empty" }
            val out = LongArray(1)
            checkStatus(
                NativeRuntimeBindings.nativeRestore(checkpoint, out),
                "runtime restore",
            )
            check(out[0] != 0L) { "native runtime returned a zero handle" }
            return NativeRuntimeSession(out[0])
        }

        private fun checkStatus(code: Int, operation: String) {
            val status = NativeRuntimeStatus.fromCode(code)
            if (status != NativeRuntimeStatus.OK) {
                throw NativeRuntimeException(status, operation)
            }
        }
    }

    private inline fun <T> withHandle(block: (Long) -> T): T = synchronized(lock) {
        check(handle != 0L) { "native runtime session is closed" }
        block(handle)
    }

    fun info(): NativeRuntimeInfo = withHandle { h ->
        val ints = IntArray(9)
        val longs = LongArray(2)
        checkStatus(NativeRuntimeBindings.nativeInfo(h, ints, longs), "runtime info")
        val requestedRecurrent = backendFromCode(ints[4])
        val requestedPacked = backendFromCode(ints[5])
        NativeRuntimeInfo(
            config = NativeRuntimeConfig(
                layers = ints[0],
                batch = ints[1],
                dModel = ints[2],
                dState = ints[3],
                recurrentBackend = requestedRecurrent,
                packedBackend = requestedPacked,
            ),
            resolvedRecurrentBackend = backendFromCode(ints[6]),
            resolvedPackedBackend = backendFromCode(ints[7]),
            lifecycle = RuntimeLifecycle.fromCode(ints[8]),
            stateCount = longs[0],
            sequencePosition = longs[1],
        )
    }

    fun activate() = withHandle<Unit> { h ->
        checkStatus(NativeRuntimeBindings.nativeActivate(h), "runtime activate")
    }

    fun suspend() = withHandle<Unit> { h ->
        checkStatus(NativeRuntimeBindings.nativeSuspend(h), "runtime suspend")
    }

    fun resume() = withHandle<Unit> { h ->
        checkStatus(NativeRuntimeBindings.nativeResume(h), "runtime resume")
    }

    fun advance(tokenCount: Long) {
        require(tokenCount >= 0L) { "tokenCount must be non-negative" }
        withHandle<Unit> { h ->
            checkStatus(NativeRuntimeBindings.nativeAdvance(h, tokenCount), "runtime advance")
        }
    }

    fun readState(): FloatArray = withHandle { h ->
        val count = checkedArrayCount(infoForHandle(h).stateCount, "stateCount")
        FloatArray(count).also { state ->
            checkStatus(NativeRuntimeBindings.nativeStateRead(h, state), "runtime state read")
        }
    }

    fun writeState(state: FloatArray) = withHandle<Unit> { h ->
        checkStatus(NativeRuntimeBindings.nativeStateWrite(h, state), "runtime state write")
    }

    fun checkpoint(): ByteArray = withHandle { h ->
        val sizeOut = LongArray(1)
        checkStatus(
            NativeRuntimeBindings.nativeCheckpointSize(h, sizeOut),
            "runtime checkpoint size",
        )
        val size = checkedArrayCount(sizeOut[0], "checkpoint size")
        val bytes = ByteArray(size)
        val written = LongArray(1)
        checkStatus(
            NativeRuntimeBindings.nativeCheckpointWrite(h, bytes, written),
            "runtime checkpoint write",
        )
        check(written[0] == sizeOut[0]) { "native runtime checkpoint length changed during write" }
        bytes
    }

    override fun close() {
        val old = synchronized(lock) {
            val value = handle
            handle = 0L
            value
        }
        if (old != 0L) {
            val status = NativeRuntimeStatus.fromCode(NativeRuntimeBindings.nativeDestroy(old))
            if (status != NativeRuntimeStatus.OK && status != NativeRuntimeStatus.INVALID_HANDLE) {
                throw NativeRuntimeException(status, "runtime destroy")
            }
        }
    }

    private fun infoForHandle(h: Long): NativeRuntimeInfo {
        val ints = IntArray(9)
        val longs = LongArray(2)
        checkStatus(NativeRuntimeBindings.nativeInfo(h, ints, longs), "runtime info")
        return NativeRuntimeInfo(
            config = NativeRuntimeConfig(
                ints[0], ints[1], ints[2], ints[3], backendFromCode(ints[4]), backendFromCode(ints[5])
            ),
            resolvedRecurrentBackend = backendFromCode(ints[6]),
            resolvedPackedBackend = backendFromCode(ints[7]),
            lifecycle = RuntimeLifecycle.fromCode(ints[8]),
            stateCount = longs[0],
            sequencePosition = longs[1],
        )
    }
}

private fun backendFromCode(code: Int): NativeBackend =
    NativeBackend.entries.firstOrNull { it.code == code }
        ?: throw IllegalStateException("unknown native backend: $code")

private fun checkedArrayCount(value: Long, label: String): Int {
    require(value in 0..Int.MAX_VALUE.toLong()) { "$label does not fit a JVM array" }
    return value.toInt()
}

private fun checkStatus(code: Int, operation: String) {
    val status = NativeRuntimeStatus.fromCode(code)
    if (status != NativeRuntimeStatus.OK) {
        throw NativeRuntimeException(status, operation)
    }
}
