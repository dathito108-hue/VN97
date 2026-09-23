package ai.vn97.platform

import android.os.SystemClock
import java.security.SecureRandom

data class VN97GameSessionSnapshot(
    val sessionId: String,
    val packageName: String,
    val startedElapsedNs: Long,
    val expiresElapsedNs: Long,
    val maxActions: Int,
    val actionsUsed: Int,
) {
    init {
        require(sessionId.matches(Regex("[0-9a-f]{32}")))
        require(packageName.isNotBlank())
        require(startedElapsedNs >= 0L)
        require(expiresElapsedNs > startedElapsedNs)
        require(maxActions > 0)
        require(actionsUsed in 0..maxActions)
    }

    val remainingActions: Int
        get() = maxActions - actionsUsed
}

class AndroidGameSessionController {
    private data class MutableSession(
        val sessionId: String,
        val packageName: String,
        val startedElapsedNs: Long,
        val expiresElapsedNs: Long,
        val maxActions: Int,
        var actionsUsed: Int,
    )

    private val lock = Any()
    private val random = SecureRandom()
    private var session: MutableSession? = null

    fun begin(
        packageName: String,
        durationMillis: Long = DEFAULT_DURATION_MS,
        maxActions: Int = DEFAULT_MAX_ACTIONS,
    ): VN97GameSessionSnapshot {
        validatePackageName(packageName)
        require(durationMillis in MIN_DURATION_MS..MAX_DURATION_MS) {
            "game session duration is outside the bounded range"
        }
        require(maxActions in 1..MAX_ACTIONS) {
            "game session action budget is outside the bounded range"
        }
        check(AndroidAccessibilityGestureBridge.isConnected()) {
            "VN97 game control requires the user-enabled accessibility service"
        }

        val now = SystemClock.elapsedRealtimeNanos()
        val durationNs = Math.multiplyExact(
            durationMillis,
            1_000_000L,
        )
        val expires = Math.addExact(now, durationNs)
        val created = MutableSession(
            sessionId = ByteArray(16)
                .also(random::nextBytes)
                .toLowerHex(),
            packageName = packageName,
            startedElapsedNs = now,
            expiresElapsedNs = expires,
            maxActions = maxActions,
            actionsUsed = 0,
        )
        synchronized(lock) {
            check(activeLocked(now) == null) {
                "a VN97 game session is already active"
            }
            session = created
        }
        return created.snapshot()
    }

    fun end(): VN97GameSessionSnapshot? =
        synchronized(lock) {
            val current = session ?: return null
            session = null
            current.snapshot()
        }

    fun activeOrNull(): VN97GameSessionSnapshot? {
        val now = SystemClock.elapsedRealtimeNanos()
        return synchronized(lock) {
            activeLocked(now)?.snapshot()
        }
    }

    fun isAccessibilityReady(): Boolean =
        AndroidAccessibilityGestureBridge.isConnected()

    fun foregroundPackageOrNull(): String? =
        AndroidAccessibilityGestureBridge.foregroundPackageOrNull()

    internal fun reserveAction(
        packageName: String,
    ): VN97GameSessionSnapshot {
        validatePackageName(packageName)
        val now = SystemClock.elapsedRealtimeNanos()
        return synchronized(lock) {
            val active = activeLocked(now)
                ?: throw IllegalStateException(
                    "no active VN97 game session"
                )
            check(active.packageName == packageName) {
                "gesture package is outside the active game session"
            }
            check(active.actionsUsed < active.maxActions) {
                "VN97 game action budget is exhausted"
            }
            val foreground =
                AndroidAccessibilityGestureBridge
                    .foregroundPackageOrNull()
            check(foreground == packageName) {
                "target game package is not foreground"
            }
            active.actionsUsed += 1
            active.snapshot()
        }
    }

    private fun activeLocked(
        nowElapsedNs: Long,
    ): MutableSession? {
        val current = session ?: return null
        if (
            nowElapsedNs >= current.expiresElapsedNs ||
            current.actionsUsed >= current.maxActions ||
            !AndroidAccessibilityGestureBridge.isConnected()
        ) {
            session = null
            return null
        }
        return current
    }

    private fun MutableSession.snapshot() =
        VN97GameSessionSnapshot(
            sessionId = sessionId,
            packageName = packageName,
            startedElapsedNs = startedElapsedNs,
            expiresElapsedNs = expiresElapsedNs,
            maxActions = maxActions,
            actionsUsed = actionsUsed,
        )

    companion object {
        const val DEFAULT_DURATION_MS = 30 * 60 * 1000L
        const val DEFAULT_MAX_ACTIONS = 512
        const val MAX_ACTIONS = 5_000
        const val MIN_DURATION_MS = 5_000L
        const val MAX_DURATION_MS = 2 * 60 * 60 * 1000L

        private val PACKAGE_RE =
            Regex(
                "^[A-Za-z][A-Za-z0-9_]*" +
                    "(?:\\.[A-Za-z][A-Za-z0-9_]*)+$"
            )

        private fun validatePackageName(packageName: String) {
            require(
                packageName.isNotEmpty() &&
                    packageName.all { it.code <= 0x7f } &&
                    PACKAGE_RE.matches(packageName)
            ) {
                "package must be a canonical Android package name"
            }
        }
    }
}

private fun ByteArray.toLowerHex(): String =
    joinToString("") { "%02x".format(it.toInt() and 0xff) }
