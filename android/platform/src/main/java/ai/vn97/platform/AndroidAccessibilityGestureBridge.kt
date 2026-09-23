package ai.vn97.platform

import android.accessibilityservice.AccessibilityService
import android.accessibilityservice.GestureDescription
import android.graphics.Path
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import java.lang.ref.WeakReference
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Process-local bridge from VN97's typed M6 action port to a user-enabled
 * AccessibilityService. The bridge never enables accessibility itself and
 * fails closed when the service is disconnected.
 */
object AndroidAccessibilityGestureBridge {
    private val lock = Any()
    private var serviceRef =
        WeakReference<AccessibilityService>(null)
    private var lastForegroundPackage = ""
    private var lastForegroundElapsedNs = 0L

    fun attach(service: AccessibilityService) {
        synchronized(lock) {
            serviceRef = WeakReference(service)
            lastForegroundPackage = ""
            lastForegroundElapsedNs = 0L
        }
    }

    fun detach(service: AccessibilityService) {
        synchronized(lock) {
            if (serviceRef.get() === service) {
                serviceRef.clear()
                serviceRef = WeakReference(null)
                lastForegroundPackage = ""
                lastForegroundElapsedNs = 0L
            }
        }
    }

    fun noteForegroundPackage(packageName: CharSequence?) {
        val value = packageName?.toString().orEmpty()
        if (value.isBlank()) return
        synchronized(lock) {
            if (serviceRef.get() == null) return
            lastForegroundPackage = value
            lastForegroundElapsedNs =
                SystemClock.elapsedRealtimeNanos()
        }
    }

    fun isConnected(): Boolean =
        synchronized(lock) { serviceRef.get() != null }

    fun foregroundPackageOrNull(): String? {
        val service = synchronized(lock) {
            serviceRef.get()
        } ?: return null

        val rootPackage = runCatching {
            service.rootInActiveWindow
                ?.packageName
                ?.toString()
                ?.takeIf { it.isNotBlank() }
        }.getOrNull()
        if (rootPackage != null) {
            noteForegroundPackage(rootPackage)
            return rootPackage
        }

        val now = SystemClock.elapsedRealtimeNanos()
        return synchronized(lock) {
            val age = now - lastForegroundElapsedNs
            if (
                lastForegroundPackage.isNotBlank() &&
                lastForegroundElapsedNs > 0L &&
                age in 0L..FOREGROUND_FRESHNESS_NS
            ) {
                lastForegroundPackage
            } else {
                null
            }
        }
    }

    internal fun performTap(
        xNormalized: Int,
        yNormalized: Int,
    ): String {
        requireNormalized(xNormalized, "x")
        requireNormalized(yNormalized, "y")
        val point = toPixels(xNormalized, yNormalized)
        val path = Path().apply {
            moveTo(point.first, point.second)
        }
        dispatch(
            GestureDescription.Builder()
                .addStroke(
                    GestureDescription.StrokeDescription(
                        path,
                        0L,
                        TAP_DURATION_MS,
                    )
                )
                .build(),
            timeoutMs = TAP_DURATION_MS + CALLBACK_GRACE_MS,
        )
        return "tap:${xNormalized},${yNormalized}"
    }

    internal fun performSwipe(
        fromXNormalized: Int,
        fromYNormalized: Int,
        toXNormalized: Int,
        toYNormalized: Int,
        durationMs: Long,
    ): String {
        requireNormalized(fromXNormalized, "from_x")
        requireNormalized(fromYNormalized, "from_y")
        requireNormalized(toXNormalized, "to_x")
        requireNormalized(toYNormalized, "to_y")
        require(durationMs in MIN_SWIPE_MS..MAX_SWIPE_MS) {
            "swipe duration is outside the bounded range"
        }
        val start = toPixels(
            fromXNormalized,
            fromYNormalized,
        )
        val end = toPixels(
            toXNormalized,
            toYNormalized,
        )
        val path = Path().apply {
            moveTo(start.first, start.second)
            lineTo(end.first, end.second)
        }
        dispatch(
            GestureDescription.Builder()
                .addStroke(
                    GestureDescription.StrokeDescription(
                        path,
                        0L,
                        durationMs,
                    )
                )
                .build(),
            timeoutMs = durationMs + CALLBACK_GRACE_MS,
        )
        return buildString {
            append("swipe:")
            append(fromXNormalized)
            append(',')
            append(fromYNormalized)
            append("->")
            append(toXNormalized)
            append(',')
            append(toYNormalized)
            append('@')
            append(durationMs)
        }
    }

    private fun toPixels(
        xNormalized: Int,
        yNormalized: Int,
    ): Pair<Float, Float> {
        val service = synchronized(lock) {
            serviceRef.get()
        } ?: throw IllegalStateException(
            "VN97 accessibility gesture service is not connected"
        )
        val metrics = service.resources.displayMetrics
        check(metrics.widthPixels > 0 && metrics.heightPixels > 0) {
            "display geometry is unavailable"
        }
        val maxX = (metrics.widthPixels - 1).coerceAtLeast(0)
        val maxY = (metrics.heightPixels - 1).coerceAtLeast(0)
        return Pair(
            maxX * (xNormalized / NORMALIZED_MAX.toFloat()),
            maxY * (yNormalized / NORMALIZED_MAX.toFloat()),
        )
    }

    private fun dispatch(
        gesture: GestureDescription,
        timeoutMs: Long,
    ) {
        check(Looper.myLooper() != Looper.getMainLooper()) {
            "blocking VN97 gesture execution must not run on the main thread"
        }
        val service = synchronized(lock) {
            serviceRef.get()
        } ?: throw IllegalStateException(
            "VN97 accessibility gesture service is not connected"
        )

        val completed = AtomicBoolean(false)
        val cancelled = AtomicBoolean(false)
        val latch = CountDownLatch(1)
        Handler(Looper.getMainLooper()).post {
            val accepted = runCatching {
                service.dispatchGesture(
                    gesture,
                    object :
                        AccessibilityService.GestureResultCallback() {
                        override fun onCompleted(
                            gestureDescription: GestureDescription,
                        ) {
                            completed.set(true)
                            latch.countDown()
                        }

                        override fun onCancelled(
                            gestureDescription: GestureDescription,
                        ) {
                            cancelled.set(true)
                            latch.countDown()
                        }
                    },
                    Handler(Looper.getMainLooper()),
                )
            }.getOrElse {
                latch.countDown()
                false
            }
            if (!accepted) {
                cancelled.set(true)
                latch.countDown()
            }
        }

        val signalled = latch.await(
            timeoutMs.coerceAtMost(MAX_CALLBACK_WAIT_MS),
            TimeUnit.MILLISECONDS,
        )
        check(signalled) {
            "VN97 accessibility gesture timed out"
        }
        check(completed.get() && !cancelled.get()) {
            "VN97 accessibility gesture was cancelled"
        }
    }

    private fun requireNormalized(value: Int, label: String) {
        require(value in 0..NORMALIZED_MAX) {
            "$label must be in 0..$NORMALIZED_MAX"
        }
    }

    private const val NORMALIZED_MAX = 1000
    private const val TAP_DURATION_MS = 55L
    private const val MIN_SWIPE_MS = 80L
    private const val MAX_SWIPE_MS = 2_000L
    private const val CALLBACK_GRACE_MS = 2_000L
    private const val MAX_CALLBACK_WAIT_MS = 5_000L
    private const val FOREGROUND_FRESHNESS_NS =
        5_000_000_000L
}
