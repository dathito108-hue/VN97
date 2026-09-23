package ai.vn97.platform

import android.content.ClipData
import android.content.ClipDescription
import android.content.ClipboardManager
import android.content.Context
import android.content.Intent
import android.os.Build
import android.os.PersistableBundle

internal object PlatformCapabilityIds {
    const val APP_LAUNCH = M6AndroidProductionCapabilities.APP_LAUNCH_CAPABILITY
    const val CLIPBOARD_WRITE = M6AndroidProductionCapabilities.CLIPBOARD_WRITE_CAPABILITY
    const val GAME_TAP = M6AndroidProductionCapabilities.GAME_TAP_CAPABILITY
    const val GAME_SWIPE = M6AndroidProductionCapabilities.GAME_SWIPE_CAPABILITY
    const val GAME_BACK = M6AndroidProductionCapabilities.GAME_BACK_CAPABILITY
}

internal class AndroidAppDeviceAdapter(
    context: Context,
    private val permissionBroker: AndroidPermissionBroker,
    private val gameControlPolicy: VN97GameControlPolicy,
) : M6AndroidActionPort {
    private val appContext = context.applicationContext

    override fun launchPackage(packageName: String): String {
        validatePackageName(packageName)
        permissionBroker.requireGranted(PlatformCapabilityIds.APP_LAUNCH)

        if (Build.VERSION.SDK_INT >= 33) {
            val sender = appContext.packageManager.getLaunchIntentSenderForPackage(packageName)
            sender.sendIntent(appContext, 0, null, null, null)
        } else {
            val intent = appContext.packageManager.getLaunchIntentForPackage(packageName)
                ?: throw IllegalStateException("package has no visible launch activity")
            intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
            appContext.startActivity(intent)
        }
        return "launched:$packageName"
    }

    override fun gameTap(
        packageName: String,
        xBasisPoints: Int,
        yBasisPoints: Int,
        durationMillis: Long,
    ): String {
        validatePackageName(packageName)
        permissionBroker.requireGranted(PlatformCapabilityIds.GAME_TAP)
        gameControlPolicy.requireAuthorized(packageName)
        check(
            VN97GameAccessibilityController.tap(
                packageName,
                xBasisPoints,
                yBasisPoints,
                durationMillis,
            )
        ) {
            "game tap was not completed"
        }
        return "game:tap:$packageName"
    }

    override fun gameSwipe(
        packageName: String,
        startXBasisPoints: Int,
        startYBasisPoints: Int,
        endXBasisPoints: Int,
        endYBasisPoints: Int,
        durationMillis: Long,
    ): String {
        validatePackageName(packageName)
        permissionBroker.requireGranted(PlatformCapabilityIds.GAME_SWIPE)
        gameControlPolicy.requireAuthorized(packageName)
        check(
            VN97GameAccessibilityController.swipe(
                packageName,
                startXBasisPoints,
                startYBasisPoints,
                endXBasisPoints,
                endYBasisPoints,
                durationMillis,
            )
        ) {
            "game swipe was not completed"
        }
        return "game:swipe:$packageName"
    }

    override fun gameBack(packageName: String): String {
        validatePackageName(packageName)
        permissionBroker.requireGranted(PlatformCapabilityIds.GAME_BACK)
        gameControlPolicy.requireAuthorized(packageName)
        check(VN97GameAccessibilityController.back(packageName)) {
            "game back action was not completed"
        }
        return "game:back:$packageName"
    }

    override fun writeClipboard(text: String): String {
        val bytes = text.toByteArray(Charsets.UTF_8)
        require(bytes.size <= MAX_CLIPBOARD_UTF8_BYTES) { "clipboard text exceeds byte bound" }
        permissionBroker.requireGranted(PlatformCapabilityIds.CLIPBOARD_WRITE)
        val clipboard = appContext.getSystemService(ClipboardManager::class.java)
            ?: throw IllegalStateException("clipboard service unavailable")
        val clip = ClipData.newPlainText("VN97", text)
        clip.description.extras = PersistableBundle().apply {
            if (Build.VERSION.SDK_INT >= 33) {
                putBoolean(ClipDescription.EXTRA_IS_SENSITIVE, true)
            } else {
                putBoolean(SENSITIVE_CLIP_EXTRA, true)
            }
        }
        clipboard.setPrimaryClip(clip)
        return "clipboard:written"
    }

    companion object {
        private const val MAX_CLIPBOARD_UTF8_BYTES = 16 * 1024
        private const val SENSITIVE_CLIP_EXTRA = "android.content.extra.IS_SENSITIVE"
        private val PACKAGE_RE = Regex("^[A-Za-z][A-Za-z0-9_]*(?:\\.[A-Za-z][A-Za-z0-9_]*)+$")

        private fun validatePackageName(packageName: String) {
            require(packageName.isNotEmpty() && packageName.all { it.code <= 0x7f } && PACKAGE_RE.matches(packageName)) {
                "package must be an ASCII Android package name"
            }
        }
    }
}
