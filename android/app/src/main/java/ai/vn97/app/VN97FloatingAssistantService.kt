package ai.vn97.app

import ai.vn97.avatar.AssistantMode
import ai.vn97.avatar.AvatarCommand
import ai.vn97.avatar.CognitionPresentationState
import ai.vn97.avatar.SpeechAvatarInput
import ai.vn97.avatar.VN97AvatarView
import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.graphics.PixelFormat
import android.os.Build
import android.os.IBinder
import android.provider.Settings
import android.view.Gravity
import android.view.MotionEvent
import android.view.WindowManager
import kotlin.math.abs

class VN97FloatingAssistantService : Service() {
    private lateinit var windowManager: WindowManager
    private var avatar: VN97AvatarView? = null
    private var layoutParams: WindowManager.LayoutParams? = null
    private var interaction: VN97FloatingInteractionController? = null
    private var sequence = 1L

    private var downRawX = 0f
    private var downRawY = 0f
    private var downWindowX = 0
    private var downWindowY = 0
    private var moved = false

    override fun onCreate() {
        super.onCreate()
        windowManager = checkNotNull(
            getSystemService(WindowManager::class.java)
        )
        ensureNotificationChannel()
        startForegroundCompat(buildNotification())
    }

    override fun onStartCommand(
        intent: Intent?,
        flags: Int,
        startId: Int,
    ): Int {
        if (intent?.action == ACTION_DISABLE) {
            preferences(this)
                .edit()
                .putBoolean(PREF_ENABLED, false)
                .apply()
            removeOverlay()
            stopSelf()
            return START_NOT_STICKY
        }
        if (!isEnabled(this) || !Settings.canDrawOverlays(this)) {
            stopSelf()
            return START_NOT_STICKY
        }
        showOverlay()
        return START_STICKY
    }

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onConfigurationChanged(newConfig: android.content.res.Configuration) {
        super.onConfigurationChanged(newConfig)
        clampAndApplyPosition()
        interaction?.updatePosition()
    }

    override fun onDestroy() {
        removeOverlay()
        super.onDestroy()
    }

    private fun showOverlay() {
        if (avatar != null) return
        check(Settings.canDrawOverlays(this)) {
            "floating VN97 assistant requires overlay permission"
        }

        val size = overlaySizePx()
        val params = WindowManager.LayoutParams(
            size.first,
            size.second,
            WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY,
            WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE or
                WindowManager.LayoutParams.FLAG_LAYOUT_NO_LIMITS,
            PixelFormat.TRANSLUCENT,
        ).apply {
            gravity = Gravity.TOP or Gravity.START
            x = preferences(this@VN97FloatingAssistantService)
                .getInt(PREF_X, defaultX())
            y = preferences(this@VN97FloatingAssistantService)
                .getInt(PREF_Y, defaultY())
        }

        val view = VN97AvatarView(this).apply {
            enableTransparentOverlaySurface()
            contentDescription = "VN97 floating 3D assistant"
            publish(
                AvatarCommand(
                    sourceSequence = sequence++,
                    mode = AssistantMode.IDLE,
                    energy = 0.38f,
                )
            )
            setOnClickListener {
                interaction?.toggle()
            }
            setOnTouchListener { touchedView, event ->
                handleOverlayTouch(touchedView as VN97AvatarView, event)
            }
        }

        layoutParams = params
        avatar = view
        clampPosition(params)
        windowManager.addView(view, params)
        view.onAvatarResume()
        interaction = VN97FloatingInteractionController(
            context = this,
            windowManager = windowManager,
            assistant =
                (application as VN97Application)
                    .assistant,
            beforeAssistantOpen = {
                (application as VN97Application)
                    .mobileRecovery
                    .recover(
                        VN97MobileRecoveryTrigger
                            .EXECUTION_ENTRY
                    )
                    .requireActivationReady()
            },
            anchorProvider = { layoutParams },
            publishMode = { mode, energy ->
                view.publish(
                    AvatarCommand(
                        sourceSequence = sequence++,
                        mode = mode,
                        energy = energy,
                    )
                )
            },
            publishListeningLevel = { level, deltaMillis ->
                view.publishSpeech(
                    SpeechAvatarInput(
                        sourceSequence = sequence++,
                        cognitionState = CognitionPresentationState.LISTENING,
                        inputLevel = level,
                        energy = 0.58f,
                    ),
                    deltaMillis,
                )
            },
            requestMicrophonePermission = {
                openMicrophonePermissionActivity()
            },
            setMicrophoneForegroundActive = { active ->
                updateForegroundMicrophone(active)
            },
        ).also {
            it.initialize()
        }
    }

    private fun handleOverlayTouch(
        view: VN97AvatarView,
        event: MotionEvent,
    ): Boolean {
        val params = layoutParams ?: return false
        when (event.actionMasked) {
            MotionEvent.ACTION_DOWN -> {
                downRawX = event.rawX
                downRawY = event.rawY
                downWindowX = params.x
                downWindowY = params.y
                moved = false
                return true
            }

            MotionEvent.ACTION_MOVE -> {
                val dx = event.rawX - downRawX
                val dy = event.rawY - downRawY
                if (abs(dx) > DRAG_THRESHOLD_PX || abs(dy) > DRAG_THRESHOLD_PX) {
                    moved = true
                }
                if (moved) {
                    params.x = downWindowX + dx.toInt()
                    params.y = downWindowY + dy.toInt()
                    clampPosition(params)
                    windowManager.updateViewLayout(view, params)
                    interaction?.updatePosition()
                }
                return true
            }

            MotionEvent.ACTION_UP -> {
                if (moved) {
                    persistPosition(params)
                } else {
                    view.performClick()
                }
                return true
            }

            MotionEvent.ACTION_CANCEL -> return true
        }
        return false
    }

    private fun removeOverlay() {
        interaction?.close()
        interaction = null
        val view = avatar ?: return
        avatar = null
        layoutParams = null
        try {
            view.onAvatarPause()
        } finally {
            try {
                windowManager.removeView(view)
            } catch (_: IllegalArgumentException) {
                // Window was already detached by the system.
            }
        }
    }

    private fun clampAndApplyPosition() {
        val view = avatar ?: return
        val params = layoutParams ?: return
        clampPosition(params)
        windowManager.updateViewLayout(view, params)
        persistPosition(params)
    }

    private fun clampPosition(params: WindowManager.LayoutParams) {
        val bounds = displayBounds()
        val maxX = (bounds.first - params.width).coerceAtLeast(0)
        val maxY = (bounds.second - params.height).coerceAtLeast(0)
        params.x = params.x.coerceIn(0, maxX)
        params.y = params.y.coerceIn(0, maxY)
    }

    private fun persistPosition(params: WindowManager.LayoutParams) {
        preferences(this)
            .edit()
            .putInt(PREF_X, params.x)
            .putInt(PREF_Y, params.y)
            .apply()
    }

    private fun overlaySizePx(): Pair<Int, Int> {
        val density = resources.displayMetrics.density
        return Pair(
            (OVERLAY_WIDTH_DP * density).toInt().coerceAtLeast(1),
            (OVERLAY_HEIGHT_DP * density).toInt().coerceAtLeast(1),
        )
    }

    private fun defaultX(): Int {
        val bounds = displayBounds()
        return (bounds.first - overlaySizePx().first).coerceAtLeast(0)
    }

    private fun defaultY(): Int {
        val density = resources.displayMetrics.density
        return (96f * density).toInt()
    }

    @Suppress("DEPRECATION")
    private fun displayBounds(): Pair<Int, Int> =
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            val bounds = windowManager.currentWindowMetrics.bounds
            Pair(bounds.width(), bounds.height())
        } else {
            val metrics = resources.displayMetrics
            Pair(metrics.widthPixels, metrics.heightPixels)
        }

    private fun openMicrophonePermissionActivity(): Boolean =
        try {
            startActivity(
                Intent(this, VN97MainActivity::class.java)
                    .setAction(ACTION_REQUEST_MICROPHONE_PERMISSION)
                    .addFlags(
                        Intent.FLAG_ACTIVITY_NEW_TASK or
                            Intent.FLAG_ACTIVITY_SINGLE_TOP
                    )
            )
            true
        } catch (_: RuntimeException) {
            false
        }

    private fun updateForegroundMicrophone(active: Boolean): Boolean =
        try {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.UPSIDE_DOWN_CAKE) {
                val type =
                    ServiceInfo.FOREGROUND_SERVICE_TYPE_SPECIAL_USE or
                        if (active) {
                            ServiceInfo.FOREGROUND_SERVICE_TYPE_MICROPHONE
                        } else {
                            0
                        }
                startForeground(
                    NOTIFICATION_ID,
                    buildNotification(),
                    type,
                )
            } else {
                startForeground(NOTIFICATION_ID, buildNotification())
            }
            true
        } catch (_: SecurityException) {
            false
        } catch (_: IllegalStateException) {
            false
        }

    private fun startForegroundCompat(notification: Notification) {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.UPSIDE_DOWN_CAKE) {
            startForeground(
                NOTIFICATION_ID,
                notification,
                ServiceInfo.FOREGROUND_SERVICE_TYPE_SPECIAL_USE,
            )
        } else {
            startForeground(NOTIFICATION_ID, notification)
        }
    }

    private fun ensureNotificationChannel() {
        val manager = checkNotNull(
            getSystemService(NotificationManager::class.java)
        )
        manager.createNotificationChannel(
            NotificationChannel(
                NOTIFICATION_CHANNEL,
                "VN97 floating assistant",
                NotificationManager.IMPORTANCE_LOW,
            ).apply {
                description = "Keeps the user-enabled VN97 3D assistant visible over other apps."
                setShowBadge(false)
            }
        )
    }

    private fun buildNotification(): Notification {
        val openApp = PendingIntent.getActivity(
            this,
            0,
            Intent(this, VN97MainActivity::class.java),
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE,
        )
        val stop = PendingIntent.getService(
            this,
            1,
            Intent(this, VN97FloatingAssistantService::class.java)
                .setAction(ACTION_DISABLE),
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE,
        )

        return Notification.Builder(this, NOTIFICATION_CHANNEL)
            .setSmallIcon(R.drawable.ic_vn97_assistant)
            .setContentTitle("VN97")
            .setContentText("Floating assistant active — tap the avatar to interact")
            .setContentIntent(openApp)
            .setOngoing(true)
            .setCategory(Notification.CATEGORY_SERVICE)
            .addAction(
                Notification.Action.Builder(
                    R.drawable.ic_vn97_assistant,
                    "Hide",
                    stop,
                ).build()
            )
            .build()
    }

    companion object {
        const val ACTION_DISABLE = "ai.vn97.app.action.DISABLE_FLOATING_ASSISTANT"
        const val ACTION_REQUEST_MICROPHONE_PERMISSION =
            "ai.vn97.app.action.REQUEST_MICROPHONE_PERMISSION"

        private const val PREFS = "vn97-floating-assistant"
        private const val PREF_ENABLED = "enabled"
        private const val PREF_X = "x"
        private const val PREF_Y = "y"
        private const val NOTIFICATION_CHANNEL = "vn97-floating-assistant"
        private const val NOTIFICATION_ID = 9703
        private const val OVERLAY_WIDTH_DP = 180f
        private const val OVERLAY_HEIGHT_DP = 220f
        private const val DRAG_THRESHOLD_PX = 8f

        fun isEnabled(context: Context): Boolean =
            preferences(context).getBoolean(PREF_ENABLED, false)

        fun enable(context: Context) {
            check(Settings.canDrawOverlays(context)) {
                "floating VN97 assistant requires overlay permission"
            }
            preferences(context)
                .edit()
                .putBoolean(PREF_ENABLED, true)
                .apply()
            context.startForegroundService(
                Intent(context, VN97FloatingAssistantService::class.java)
            )
        }

        fun disable(context: Context) {
            preferences(context)
                .edit()
                .putBoolean(PREF_ENABLED, false)
                .apply()
            context.stopService(
                Intent(context, VN97FloatingAssistantService::class.java)
            )
        }

        fun startIfEnabled(context: Context) {
            if (!isEnabled(context) || !Settings.canDrawOverlays(context)) {
                return
            }
            try {
                context.startForegroundService(
                    Intent(context, VN97FloatingAssistantService::class.java)
                )
            } catch (_: RuntimeException) {
                // Android may temporarily reject a background FGS start.
                // The persisted user preference remains enabled for the next
                // permitted lifecycle entry point.
            }
        }

        private fun preferences(context: Context) =
            context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
    }
}
