package ai.vn97.app

import ai.vn97.runtime.NativePreparedVision

data class VN97CapturedVision(
    val capturedElapsedRealtimeNs: Long,
    val prepared: NativePreparedVision,
)

class VN97ScreenCaptureBroker {
    private val monitor = Object()
    private var active = false
    private var latest: VN97CapturedVision? = null
    private var failure: String? = null

    fun markActive() {
        synchronized(monitor) {
            active = true
            failure = null
            monitor.notifyAll()
        }
    }

    fun publish(frame: VN97CapturedVision) {
        require(frame.capturedElapsedRealtimeNs >= 0L)
        synchronized(monitor) {
            if (!active) return
            val previous = latest
            if (
                previous == null ||
                frame.capturedElapsedRealtimeNs >=
                    previous.capturedElapsedRealtimeNs
            ) {
                latest = frame
            }
            monitor.notifyAll()
        }
    }

    fun publishFailure(message: String) {
        require(message.isNotBlank())
        synchronized(monitor) {
            failure = message.take(1024)
            active = false
            monitor.notifyAll()
        }
    }

    fun markStopped() {
        synchronized(monitor) {
            active = false
            latest = null
            monitor.notifyAll()
        }
    }

    fun isActive(): Boolean = synchronized(monitor) { active }

    fun latestOrNull(): VN97CapturedVision? =
        synchronized(monitor) { latest }

    fun awaitFreshFrame(
        afterElapsedRealtimeNs: Long,
        timeoutMs: Long = 4_000L,
    ): VN97CapturedVision {
        require(afterElapsedRealtimeNs >= 0L)
        require(timeoutMs in 1L..30_000L)
        val deadline = System.nanoTime() +
            Math.multiplyExact(timeoutMs, 1_000_000L)
        synchronized(monitor) {
            while (true) {
                latest?.let { frame ->
                    if (
                        frame.capturedElapsedRealtimeNs >
                        afterElapsedRealtimeNs
                    ) {
                        return frame
                    }
                }
                failure?.let {
                    throw IllegalStateException(
                        "screen capture failed: $it"
                    )
                }
                check(active) {
                    "screen capture is not active"
                }
                val remainingNs = deadline - System.nanoTime()
                if (remainingNs <= 0L) {
                    throw IllegalStateException(
                        "timed out waiting for a fresh screen frame"
                    )
                }
                val millis = remainingNs / 1_000_000L
                val nanos = (remainingNs % 1_000_000L).toInt()
                monitor.wait(millis, nanos)
            }
        }
    }
}
