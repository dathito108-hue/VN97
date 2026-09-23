package ai.vn97.app

import ai.vn97.runtime.NativeVisionModality
import android.app.Activity
import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.graphics.PixelFormat
import android.hardware.display.DisplayManager
import android.hardware.display.VirtualDisplay
import android.media.Image
import android.media.ImageReader
import android.media.projection.MediaProjection
import android.media.projection.MediaProjectionManager
import android.os.Build
import android.os.Handler
import android.os.HandlerThread
import android.os.IBinder
import android.os.SystemClock

class VN97ScreenCaptureService : Service() {
    private lateinit var captureThread: HandlerThread
    private lateinit var captureHandler: Handler
    private var projection: MediaProjection? = null
    private var projectionCallback: MediaProjection.Callback? = null
    private var virtualDisplay: VirtualDisplay? = null
    private var imageReader: ImageReader? = null

    private val app: VN97Application
        get() = application as VN97Application

    override fun onCreate() {
        super.onCreate()
        captureThread = HandlerThread("vn97-screen-perception").apply {
            start()
        }
        captureHandler = Handler(captureThread.looper)
        ensureNotificationChannel()
    }

    override fun onStartCommand(
        intent: Intent?,
        flags: Int,
        startId: Int,
    ): Int {
        when (intent?.action) {
            ACTION_STOP -> {
                stopProjection()
                stopSelf()
                return START_NOT_STICKY
            }

            ACTION_START -> {
                startForegroundCompat()
                val resultCode = intent.getIntExtra(
                    EXTRA_RESULT_CODE,
                    Activity.RESULT_CANCELED,
                )
                val projectionData = projectionIntent(intent)
                if (
                    resultCode != Activity.RESULT_OK ||
                    projectionData == null
                ) {
                    failAndStop("screen-capture consent token is missing")
                    return START_NOT_STICKY
                }
                try {
                    startProjection(resultCode, projectionData)
                } catch (exc: Throwable) {
                    failAndStop(
                        "screen-capture startup failed: " +
                            exc::class.java.simpleName
                    )
                }
                return START_NOT_STICKY
            }

            else -> {
                stopSelf()
                return START_NOT_STICKY
            }
        }
    }

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onDestroy() {
        stopProjection()
        captureThread.quitSafely()
        super.onDestroy()
    }

    private fun startProjection(
        resultCode: Int,
        data: Intent,
    ) {
        stopProjection()
        val manager = checkNotNull(
            getSystemService(MediaProjectionManager::class.java)
        ) {
            "MediaProjectionManager unavailable"
        }
        val opened = checkNotNull(
            manager.getMediaProjection(resultCode, data)
        ) {
            "MediaProjection consent could not be opened"
        }

        val callback = object : MediaProjection.Callback() {
            override fun onStop() {
                captureHandler.post {
                    projection = null
                    projectionCallback = null
                    closeDisplayResources()
                    app.screenCaptureBroker.markStopped()
                    stopSelf()
                }
            }
        }
        opened.registerCallback(callback, captureHandler)
        projection = opened
        projectionCallback = callback

        val reader = ImageReader.newInstance(
            CAPTURE_WIDTH,
            CAPTURE_HEIGHT,
            PixelFormat.RGBA_8888,
            2,
        )
        reader.setOnImageAvailableListener(
            { source ->
                val image = source.acquireLatestImage() ?: return@setOnImageAvailableListener
                try {
                    publishImage(image)
                } catch (exc: Throwable) {
                    app.screenCaptureBroker.publishFailure(
                        "screen frame preprocessing failed: " +
                            exc::class.java.simpleName
                    )
                    stopSelf()
                } finally {
                    image.close()
                }
            },
            captureHandler,
        )
        imageReader = reader
        app.screenCaptureBroker.markActive()

        virtualDisplay = opened.createVirtualDisplay(
            "VN97ScreenPerception",
            CAPTURE_WIDTH,
            CAPTURE_HEIGHT,
            resources.displayMetrics.densityDpi,
            DisplayManager.VIRTUAL_DISPLAY_FLAG_AUTO_MIRROR,
            reader.surface,
            null,
            captureHandler,
        ) ?: throw IllegalStateException(
            "MediaProjection virtual display could not be created"
        )
    }

    private fun publishImage(image: Image) {
        require(
            image.width == CAPTURE_WIDTH &&
                image.height == CAPTURE_HEIGHT
        ) {
            "screen frame has unexpected geometry"
        }
        val planes = image.planes
        require(planes.size == 1) {
            "RGBA screen frame must contain one plane"
        }
        val plane = planes[0]
        val pixelStride = plane.pixelStride
        val rowStride = plane.rowStride
        require(pixelStride >= 4 && rowStride >= image.width * pixelStride) {
            "screen frame stride is invalid"
        }
        val buffer = plane.buffer
        val requiredLast =
            (image.height - 1) * rowStride +
                (image.width - 1) * pixelStride +
                2
        require(requiredLast < buffer.limit()) {
            "screen frame plane is truncated"
        }

        val rgb = ByteArray(image.width * image.height * 3)
        var target = 0
        for (y in 0 until image.height) {
            val row = y * rowStride
            for (x in 0 until image.width) {
                val source = row + x * pixelStride
                rgb[target++] = buffer.get(source)
                rgb[target++] = buffer.get(source + 1)
                rgb[target++] = buffer.get(source + 2)
            }
        }

        val prepared = NativeVisionModality.prepareRgb888(
            rgb,
            image.width,
            image.height,
        )
        app.screenCaptureBroker.publish(
            VN97CapturedVision(
                capturedElapsedRealtimeNs =
                    SystemClock.elapsedRealtimeNanos(),
                prepared = prepared,
            )
        )
    }

    private fun stopProjection() {
        val current = projection
        val callback = projectionCallback
        projection = null
        projectionCallback = null
        if (current != null && callback != null) {
            try {
                current.unregisterCallback(callback)
            } catch (_: RuntimeException) {
                // Already stopped by the system.
            }
        }
        try {
            current?.stop()
        } catch (_: RuntimeException) {
            // Token may already be revoked.
        }
        closeDisplayResources()
        app.screenCaptureBroker.markStopped()
    }

    private fun closeDisplayResources() {
        try {
            virtualDisplay?.release()
        } finally {
            virtualDisplay = null
            imageReader?.setOnImageAvailableListener(null, null)
            imageReader?.close()
            imageReader = null
        }
    }

    private fun failAndStop(message: String) {
        app.screenCaptureBroker.publishFailure(message)
        stopProjection()
        stopSelf()
    }

    private fun startForegroundCompat() {
        val notification = buildNotification()
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            startForeground(
                NOTIFICATION_ID,
                notification,
                ServiceInfo.FOREGROUND_SERVICE_TYPE_MEDIA_PROJECTION,
            )
        } else {
            startForeground(NOTIFICATION_ID, notification)
        }
    }

    private fun ensureNotificationChannel() {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.O) return
        val manager = checkNotNull(
            getSystemService(NotificationManager::class.java)
        )
        manager.createNotificationChannel(
            NotificationChannel(
                NOTIFICATION_CHANNEL,
                "VN97 screen perception",
                NotificationManager.IMPORTANCE_LOW,
            ).apply {
                description =
                    "Shows when VN97 is using user-approved screen capture for local perception."
                setShowBadge(false)
            }
        )
    }

    private fun buildNotification(): Notification {
        val openApp = PendingIntent.getActivity(
            this,
            0,
            Intent(this, VN97MainActivity::class.java),
            PendingIntent.FLAG_UPDATE_CURRENT or
                PendingIntent.FLAG_IMMUTABLE,
        )
        val stop = PendingIntent.getService(
            this,
            1,
            Intent(this, VN97ScreenCaptureService::class.java)
                .setAction(ACTION_STOP),
            PendingIntent.FLAG_UPDATE_CURRENT or
                PendingIntent.FLAG_IMMUTABLE,
        )
        return Notification.Builder(this, NOTIFICATION_CHANNEL)
            .setSmallIcon(R.drawable.ic_vn97_assistant)
            .setContentTitle("VN97 screen perception")
            .setContentText(
                "Screen sharing is active for local VN97 perception."
            )
            .setContentIntent(openApp)
            .setOngoing(true)
            .setCategory(Notification.CATEGORY_SERVICE)
            .addAction(
                Notification.Action.Builder(
                    R.drawable.ic_vn97_assistant,
                    "Stop",
                    stop,
                ).build()
            )
            .build()
    }

    @Suppress("DEPRECATION")
    private fun projectionIntent(source: Intent): Intent? =
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            source.getParcelableExtra(
                EXTRA_PROJECTION_DATA,
                Intent::class.java,
            )
        } else {
            source.getParcelableExtra(EXTRA_PROJECTION_DATA)
        }

    companion object {
        private const val ACTION_START =
            "ai.vn97.app.action.START_SCREEN_PERCEPTION"
        private const val ACTION_STOP =
            "ai.vn97.app.action.STOP_SCREEN_PERCEPTION"
        private const val EXTRA_RESULT_CODE = "result_code"
        private const val EXTRA_PROJECTION_DATA = "projection_data"
        private const val NOTIFICATION_CHANNEL =
            "vn97-screen-perception"
        private const val NOTIFICATION_ID = 9712
        private const val CAPTURE_WIDTH =
            NativeVisionModality.MAX_WIDTH
        private const val CAPTURE_HEIGHT =
            NativeVisionModality.MAX_HEIGHT

        fun start(
            context: Context,
            resultCode: Int,
            data: Intent,
        ) {
            require(resultCode == Activity.RESULT_OK) {
                "screen perception requires successful user consent"
            }
            val intent = Intent(
                context,
                VN97ScreenCaptureService::class.java,
            ).apply {
                action = ACTION_START
                putExtra(EXTRA_RESULT_CODE, resultCode)
                putExtra(EXTRA_PROJECTION_DATA, Intent(data))
            }
            context.startForegroundService(intent)
        }

        fun stop(context: Context) {
            context.stopService(
                Intent(
                    context,
                    VN97ScreenCaptureService::class.java,
                )
            )
        }
    }
}
