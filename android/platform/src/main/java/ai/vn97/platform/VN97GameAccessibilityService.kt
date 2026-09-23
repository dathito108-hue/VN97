package ai.vn97.platform

import android.accessibilityservice.AccessibilityService
import android.accessibilityservice.GestureDescription
import android.graphics.Path
import android.os.Handler
import android.os.Looper
import android.view.accessibility.AccessibilityEvent
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean

object VN97GameAccessibilityController {
    @Volatile
    private var service: VN97GameAccessibilityService? = null

    @Volatile
    private var currentPackage: String? = null

    @Volatile
    private var lastExternalPackage: String? = null

    fun isConnected(): Boolean = service != null

    fun currentPackageName(): String? = currentPackage

    fun lastExternalPackageName(): String? = lastExternalPackage

    internal fun attach(value: VN97GameAccessibilityService) {
        service = value
    }

    internal fun detach(value: VN97GameAccessibilityService) {
        if (service === value) {
            service = null
        }
    }

    internal fun observePackage(
        packageName: String?,
        ownPackageName: String,
    ) {
        val normalized = packageName
            ?.trim()
            ?.takeIf { it.isNotEmpty() }
            ?: return
        currentPackage = normalized
        if (normalized != ownPackageName) {
            lastExternalPackage = normalized
        }
    }

    internal fun tap(
        packageName: String,
        xBasisPoints: Int,
        yBasisPoints: Int,
        durationMillis: Long,
    ): Boolean =
        checkNotNull(service) {
            "VN97 game accessibility service is not connected"
        }.dispatchTap(
            packageName,
            xBasisPoints,
            yBasisPoints,
            durationMillis,
        )

    internal fun swipe(
        packageName: String,
        startXBasisPoints: Int,
        startYBasisPoints: Int,
        endXBasisPoints: Int,
        endYBasisPoints: Int,
        durationMillis: Long,
    ): Boolean =
        checkNotNull(service) {
            "VN97 game accessibility service is not connected"
        }.dispatchSwipe(
            packageName,
            startXBasisPoints,
            startYBasisPoints,
            endXBasisPoints,
            endYBasisPoints,
            durationMillis,
        )

    internal fun multiTouch(
        packageName: String,
        strokes: List<VN97GameTouchStroke>,
    ): Boolean =
        checkNotNull(service) {
            "VN97 game accessibility service is not connected"
        }.dispatchMultiTouch(packageName, strokes)

    internal fun back(packageName: String): Boolean =
        checkNotNull(service) {
            "VN97 game accessibility service is not connected"
        }.dispatchBack(packageName)
}

class VN97GameAccessibilityService : AccessibilityService() {
    private val mainHandler by lazy {
        Handler(Looper.getMainLooper())
    }

    override fun onServiceConnected() {
        super.onServiceConnected()
        VN97GameAccessibilityController.attach(this)
    }

    override fun onAccessibilityEvent(event: AccessibilityEvent?) {
        VN97GameAccessibilityController.observePackage(
            event?.packageName?.toString(),
            packageName,
        )
    }

    override fun onInterrupt() = Unit

    override fun onDestroy() {
        VN97GameAccessibilityController.detach(this)
        super.onDestroy()
    }

    internal fun dispatchTap(
        targetPackage: String,
        xBasisPoints: Int,
        yBasisPoints: Int,
        durationMillis: Long,
    ): Boolean {
        requireForegroundPackage(targetPackage)
        val (x, y) = mapPoint(xBasisPoints, yBasisPoints)
        val path = Path().apply {
            moveTo(x, y)
        }
        val gesture = GestureDescription.Builder()
            .addStroke(
                GestureDescription.StrokeDescription(
                    path,
                    0L,
                    durationMillis,
                )
            )
            .build()
        return dispatchAndAwait(gesture)
    }

    internal fun dispatchSwipe(
        targetPackage: String,
        startXBasisPoints: Int,
        startYBasisPoints: Int,
        endXBasisPoints: Int,
        endYBasisPoints: Int,
        durationMillis: Long,
    ): Boolean {
        requireForegroundPackage(targetPackage)
        val start = mapPoint(
            startXBasisPoints,
            startYBasisPoints,
        )
        val end = mapPoint(
            endXBasisPoints,
            endYBasisPoints,
        )
        val path = Path().apply {
            moveTo(start.first, start.second)
            lineTo(end.first, end.second)
        }
        val gesture = GestureDescription.Builder()
            .addStroke(
                GestureDescription.StrokeDescription(
                    path,
                    0L,
                    durationMillis,
                )
            )
            .build()
        return dispatchAndAwait(gesture)
    }

    internal fun dispatchMultiTouch(
        targetPackage: String,
        strokes: List<VN97GameTouchStroke>,
    ): Boolean {
        requireForegroundPackage(targetPackage)
        VN97GameMultiTouchContract.requireValid(strokes)

        val builder = GestureDescription.Builder()
        strokes.forEach { stroke ->
            val start = mapPoint(
                stroke.startXBasisPoints,
                stroke.startYBasisPoints,
            )
            val end = mapPoint(
                stroke.endXBasisPoints,
                stroke.endYBasisPoints,
            )
            val path = Path().apply {
                moveTo(start.first, start.second)
                if (start != end) {
                    lineTo(end.first, end.second)
                }
            }
            builder.addStroke(
                GestureDescription.StrokeDescription(
                    path,
                    stroke.startMillis,
                    stroke.durationMillis,
                )
            )
        }
        return dispatchAndAwait(builder.build())
    }

    internal fun dispatchBack(targetPackage: String): Boolean {
        requireForegroundPackage(targetPackage)
        return runBooleanOnMain {
            performGlobalAction(GLOBAL_ACTION_BACK)
        }
    }

    private fun requireForegroundPackage(targetPackage: String) {
        val current =
            VN97GameAccessibilityController.currentPackageName()
        check(current == targetPackage) {
            "game action target is not the active foreground package"
        }
    }

    private fun mapPoint(
        xBasisPoints: Int,
        yBasisPoints: Int,
    ): Pair<Float, Float> {
        require(xBasisPoints in 0..BASIS_POINTS)
        require(yBasisPoints in 0..BASIS_POINTS)
        val metrics = resources.displayMetrics
        val maxX = (metrics.widthPixels - 1).coerceAtLeast(0)
        val maxY = (metrics.heightPixels - 1).coerceAtLeast(0)
        return Pair(
            maxX * (xBasisPoints / BASIS_POINTS.toFloat()),
            maxY * (yBasisPoints / BASIS_POINTS.toFloat()),
        )
    }

    private fun dispatchAndAwait(
        gesture: GestureDescription,
    ): Boolean {
        if (Looper.myLooper() == Looper.getMainLooper()) {
            return dispatchGesture(gesture, null, null)
        }

        val accepted = AtomicBoolean(false)
        val completed = AtomicBoolean(false)
        val latch = CountDownLatch(1)
        mainHandler.post {
            val dispatched = dispatchGesture(
                gesture,
                object : GestureResultCallback() {
                    override fun onCompleted(
                        gestureDescription: GestureDescription?,
                    ) {
                        completed.set(true)
                        latch.countDown()
                    }

                    override fun onCancelled(
                        gestureDescription: GestureDescription?,
                    ) {
                        latch.countDown()
                    }
                },
                null,
            )
            accepted.set(dispatched)
            if (!dispatched) {
                latch.countDown()
            }
        }
        check(
            latch.await(
                GESTURE_TIMEOUT_MILLIS,
                TimeUnit.MILLISECONDS,
            )
        ) {
            "game gesture timed out"
        }
        return accepted.get() && completed.get()
    }

    private fun runBooleanOnMain(
        block: () -> Boolean,
    ): Boolean {
        if (Looper.myLooper() == Looper.getMainLooper()) {
            return block()
        }
        val result = AtomicBoolean(false)
        val latch = CountDownLatch(1)
        mainHandler.post {
            try {
                result.set(block())
            } finally {
                latch.countDown()
            }
        }
        check(
            latch.await(
                GLOBAL_ACTION_TIMEOUT_MILLIS,
                TimeUnit.MILLISECONDS,
            )
        ) {
            "game global action timed out"
        }
        return result.get()
    }

    companion object {
        private const val BASIS_POINTS = 10_000
        private const val GESTURE_TIMEOUT_MILLIS = 5_000L
        private const val GLOBAL_ACTION_TIMEOUT_MILLIS = 2_000L
    }
}
