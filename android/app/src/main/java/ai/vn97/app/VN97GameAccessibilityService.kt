package ai.vn97.app

import ai.vn97.platform.AndroidAccessibilityGestureBridge
import android.accessibilityservice.AccessibilityService
import android.view.accessibility.AccessibilityEvent

/**
 * User-enabled Android accessibility endpoint used only as the physical
 * gesture transport for package-scoped VN97 game sessions.
 */
class VN97GameAccessibilityService : AccessibilityService() {
    override fun onServiceConnected() {
        super.onServiceConnected()
        AndroidAccessibilityGestureBridge.attach(this)
        AndroidAccessibilityGestureBridge.noteForegroundPackage(
            rootInActiveWindow?.packageName
        )
    }

    override fun onAccessibilityEvent(event: AccessibilityEvent?) {
        val type = event?.eventType ?: return
        if (
            type == AccessibilityEvent.TYPE_WINDOW_STATE_CHANGED ||
            type == AccessibilityEvent.TYPE_WINDOWS_CHANGED
        ) {
            AndroidAccessibilityGestureBridge.noteForegroundPackage(
                event.packageName
            )
        }
    }

    override fun onInterrupt() = Unit

    override fun onDestroy() {
        AndroidAccessibilityGestureBridge.detach(this)
        super.onDestroy()
    }
}
