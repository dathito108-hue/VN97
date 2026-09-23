package ai.vn97.app

import android.Manifest
import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Build

internal class VN97AutonomousNotifier(
    private val context: Context,
) {
    private val manager: NotificationManager =
        checkNotNull(
            context.getSystemService(NotificationManager::class.java)
        )

    fun postTerminal(record: VN97AutonomousGoalRecord): Boolean {
        if (
            record.state != VN97AutonomousGoalState.COMPLETED &&
            record.state != VN97AutonomousGoalState.FAILED &&
            record.state != VN97AutonomousGoalState.BUDGET_EXHAUSTED
        ) {
            return false
        }
        if (
            Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU &&
            context.checkSelfPermission(
                Manifest.permission.POST_NOTIFICATIONS
            ) != PackageManager.PERMISSION_GRANTED
        ) {
            return false
        }

        ensureChannel()
        val openApp = PendingIntent.getActivity(
            context,
            record.jobId,
            Intent(context, VN97MainActivity::class.java)
                .addFlags(
                    Intent.FLAG_ACTIVITY_NEW_TASK or
                        Intent.FLAG_ACTIVITY_SINGLE_TOP
                ),
            PendingIntent.FLAG_UPDATE_CURRENT or
                PendingIntent.FLAG_IMMUTABLE,
        )
        val completed =
            record.state == VN97AutonomousGoalState.COMPLETED
        val title =
            if (completed) {
                "VN97 autonomous goal completed"
            } else {
                "VN97 autonomous goal stopped"
            }
        val text =
            if (completed) {
                "Goal #${record.rootJobId}, generation ${record.generation} completed."
            } else {
                "Goal #${record.rootJobId}, generation ${record.generation}: ${record.state.name.lowercase()}."
            }

        val notification = Notification.Builder(
            context,
            CHANNEL_ID,
        )
            .setSmallIcon(R.drawable.ic_vn97_assistant)
            .setContentTitle(title)
            .setContentText(text)
            .setContentIntent(openApp)
            .setAutoCancel(true)
            .setCategory(Notification.CATEGORY_STATUS)
            .build()
        manager.notify(notificationId(record.jobId), notification)
        return true
    }

    private fun ensureChannel() {
        manager.createNotificationChannel(
            NotificationChannel(
                CHANNEL_ID,
                "VN97 autonomous operations",
                NotificationManager.IMPORTANCE_DEFAULT,
            ).apply {
                description =
                    "Completion and failure status for long-running VN97 autonomous goals."
            }
        )
    }

    private fun notificationId(jobId: Int): Int =
        NOTIFICATION_BASE or (jobId and 0x0000ffff)

    companion object {
        private const val CHANNEL_ID = "vn97-autonomous-operations"
        private const val NOTIFICATION_BASE = 0x00610000
    }
}
