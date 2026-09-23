package ai.vn97.app

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.os.Build
import android.os.IBinder
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

class VN97GameAgentService : Service() {
    private val app: VN97Application
        get() = application as VN97Application

    private val executor =
        Executors.newSingleThreadExecutor { runnable ->
            Thread(runnable, "vn97-game-agent")
        }
    private val running = AtomicBoolean(false)

    override fun onCreate() {
        super.onCreate()
        ensureChannel()
    }

    override fun onStartCommand(
        intent: Intent?,
        flags: Int,
        startId: Int,
    ): Int {
        when (intent?.action) {
            ACTION_STOP -> {
                app.gameAgent.requestStop()
                publishStatus(app.gameAgent.status())
                stopSelf()
                return START_NOT_STICKY
            }

            ACTION_START -> {
                val packageName =
                    intent.getStringExtra(EXTRA_PACKAGE)
                        .orEmpty()
                val objective =
                    intent.getStringExtra(EXTRA_OBJECTIVE)
                        .orEmpty()
                if (
                    packageName.isBlank() ||
                    objective.isBlank()
                ) {
                    stopSelf()
                    return START_NOT_STICKY
                }

                startForegroundCompat(
                    buildNotification(
                        "Preparing package-scoped game session",
                        ongoing = true,
                    )
                )
                if (!running.compareAndSet(false, true)) {
                    return START_NOT_STICKY
                }
                executor.execute {
                    try {
                        val started = app.gameAgent.start(
                            packageName = packageName,
                            objective = objective,
                        )
                        publishStatus(started)
                        app.gameAgent.runLoop(::publishStatus)
                    } catch (exc: Throwable) {
                        app.gameAgent.requestStop()
                        val message =
                            "Game agent stopped: " +
                                exc::class.java.simpleName
                        notificationManager().notify(
                            NOTIFICATION_ID,
                            buildNotification(
                                message,
                                ongoing = false,
                            )
                        )
                    } finally {
                        running.set(false)
                        stopSelf()
                    }
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
        app.gameAgent.requestStop()
        executor.shutdownNow()
        super.onDestroy()
    }

    private fun publishStatus(status: VN97GameAgentStatus) {
        val text = when (status.state) {
            VN97GameAgentState.IDLE ->
                "Game agent idle"

            VN97GameAgentState.STARTING ->
                "Preparing game session"

            VN97GameAgentState.RUNNING ->
                "Game agent running • round " +
                    status.round +
                    "/" +
                    status.maxRounds +
                    " • actions " +
                    status.actionsUsed +
                    "/" +
                    status.maxActions

            VN97GameAgentState.WAITING_APPROVAL ->
                "Game agent paused for explicit approval"

            VN97GameAgentState.COMPLETED ->
                "Game objective completed"

            VN97GameAgentState.BUDGET_EXHAUSTED ->
                "Game agent budget exhausted"

            VN97GameAgentState.STOPPED ->
                "Game agent stopped"

            VN97GameAgentState.FAILED ->
                "Game agent failed"
        }
        notificationManager().notify(
            NOTIFICATION_ID,
            buildNotification(
                text = text,
                ongoing =
                    status.state == VN97GameAgentState.STARTING ||
                        status.state == VN97GameAgentState.RUNNING,
            )
        )
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

    private fun ensureChannel() {
        notificationManager().createNotificationChannel(
            NotificationChannel(
                CHANNEL_ID,
                "VN97 game agent",
                NotificationManager.IMPORTANCE_LOW,
            ).apply {
                description =
                    "Shows while a user-authorized VN97 game session is active."
                setShowBadge(false)
            }
        )
    }

    private fun notificationManager(): NotificationManager =
        checkNotNull(
            getSystemService(NotificationManager::class.java)
        )

    private fun buildNotification(
        text: String,
        ongoing: Boolean,
    ): Notification {
        val open = PendingIntent.getActivity(
            this,
            0,
            Intent(this, VN97MainActivity::class.java),
            PendingIntent.FLAG_UPDATE_CURRENT or
                PendingIntent.FLAG_IMMUTABLE,
        )
        val stop = PendingIntent.getService(
            this,
            1,
            Intent(this, VN97GameAgentService::class.java)
                .setAction(ACTION_STOP),
            PendingIntent.FLAG_UPDATE_CURRENT or
                PendingIntent.FLAG_IMMUTABLE,
        )
        val builder =
            Notification.Builder(this, CHANNEL_ID)
                .setSmallIcon(R.drawable.ic_vn97_assistant)
                .setContentTitle("VN97 Game Agent")
                .setContentText(text)
                .setContentIntent(open)
                .setOngoing(ongoing)
                .setCategory(Notification.CATEGORY_SERVICE)
        if (ongoing) {
            builder.addAction(
                Notification.Action.Builder(
                    R.drawable.ic_vn97_assistant,
                    "Stop",
                    stop,
                ).build()
            )
        }
        return builder.build()
    }

    companion object {
        private const val ACTION_START =
            "ai.vn97.app.action.START_GAME_AGENT"
        private const val ACTION_STOP =
            "ai.vn97.app.action.STOP_GAME_AGENT"
        private const val EXTRA_PACKAGE = "package"
        private const val EXTRA_OBJECTIVE = "objective"
        private const val CHANNEL_ID = "vn97-game-agent"
        private const val NOTIFICATION_ID = 9714

        fun start(
            context: Context,
            packageName: String,
            objective: String,
        ) {
            require(packageName.isNotBlank())
            require(objective.isNotBlank())
            context.startForegroundService(
                Intent(
                    context,
                    VN97GameAgentService::class.java,
                ).apply {
                    action = ACTION_START
                    putExtra(EXTRA_PACKAGE, packageName)
                    putExtra(EXTRA_OBJECTIVE, objective)
                }
            )
        }

        fun stop(context: Context) {
            context.startService(
                Intent(
                    context,
                    VN97GameAgentService::class.java,
                ).setAction(ACTION_STOP)
            )
        }
    }
}
