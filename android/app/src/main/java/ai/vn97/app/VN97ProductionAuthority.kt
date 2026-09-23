package ai.vn97.app

import ai.vn97.platform.M6AndroidProductionCapabilities
import ai.vn97.platform.M6PolicyGrant
import android.content.Context
import android.content.Intent

internal object VN97ProductionAuthority {
    fun grants(
        context: Context,
        principal: String,
    ): List<M6PolicyGrant> = buildList {
        add(
            M6AndroidProductionCapabilities.userApprovedClipboardGrant(
                principal
            )
        )
        val app = context.applicationContext as? VN97Application
        val gameSession =
            app?.platformRuntime?.gameSession?.activeOrNull()
        if (gameSession != null) {
            add(
                M6AndroidProductionCapabilities
                    .gameSessionTapGrant(
                        principal,
                        gameSession.packageName,
                    )
            )
            add(
                M6AndroidProductionCapabilities
                    .gameSessionSwipeGrant(
                        principal,
                        gameSession.packageName,
                    )
            )
        }

        val launcher = Intent(Intent.ACTION_MAIN).apply {
            addCategory(Intent.CATEGORY_LAUNCHER)
        }
        context.packageManager
            .queryIntentActivities(launcher, 0)
            .asSequence()
            .mapNotNull { it.activityInfo?.packageName }
            .filter { it.isNotBlank() }
            .distinct()
            .sorted()
            .take(MAX_LAUNCHABLE_APP_GRANTS)
            .forEach { packageName ->
                runCatching {
                    M6AndroidProductionCapabilities
                        .userApprovedAppLaunchGrant(
                            principal,
                            packageName,
                        )
                }.getOrNull()?.let(::add)
            }
    }

    private const val MAX_LAUNCHABLE_APP_GRANTS = 512
}
