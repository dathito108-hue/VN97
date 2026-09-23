package ai.vn97.platform

import android.content.Context

data class VN97GameControlSession(
    val packageName: String,
    val authorizedAtWallTimeMillis: Long,
    val expiresAtWallTimeMillis: Long,
) {
    init {
        require(isGamePackageName(packageName)) {
            "game package is invalid"
        }
        require(authorizedAtWallTimeMillis >= 0L) {
            "game authorization time must be non-negative"
        }
        require(expiresAtWallTimeMillis > authorizedAtWallTimeMillis) {
            "game authorization expiry must follow authorization"
        }
    }

    fun activeAt(nowWallTimeMillis: Long): Boolean =
        nowWallTimeMillis in
            authorizedAtWallTimeMillis until expiresAtWallTimeMillis
}

class VN97GameControlPolicy(
    context: Context,
) {
    private val preferences =
        context.applicationContext.getSharedPreferences(
            PREFS,
            Context.MODE_PRIVATE,
        )

    @Synchronized
    fun authorize(
        packageName: String,
        durationMillis: Long = DEFAULT_SESSION_MILLIS,
        nowWallTimeMillis: Long = System.currentTimeMillis(),
    ): VN97GameControlSession {
        require(isGamePackageName(packageName)) {
            "game package is invalid"
        }
        require(
            durationMillis in MIN_SESSION_MILLIS..MAX_SESSION_MILLIS
        ) {
            "game session duration is outside bounds"
        }
        require(nowWallTimeMillis >= 0L) {
            "game authorization time must be non-negative"
        }
        val expires = Math.addExact(
            nowWallTimeMillis,
            durationMillis,
        )
        val session = VN97GameControlSession(
            packageName = packageName,
            authorizedAtWallTimeMillis = nowWallTimeMillis,
            expiresAtWallTimeMillis = expires,
        )
        check(
            preferences.edit()
                .putString(KEY_PACKAGE, packageName)
                .putLong(KEY_AUTHORIZED_AT, nowWallTimeMillis)
                .putLong(KEY_EXPIRES_AT, expires)
                .commit()
        ) {
            "failed to persist game control authorization"
        }
        return session
    }

    @Synchronized
    fun revoke() {
        check(
            preferences.edit()
                .remove(KEY_PACKAGE)
                .remove(KEY_AUTHORIZED_AT)
                .remove(KEY_EXPIRES_AT)
                .commit()
        ) {
            "failed to revoke game control authorization"
        }
    }

    @Synchronized
    fun activeSessionOrNull(
        nowWallTimeMillis: Long = System.currentTimeMillis(),
    ): VN97GameControlSession? {
        require(nowWallTimeMillis >= 0L) {
            "game session query time must be non-negative"
        }
        val packageName =
            preferences.getString(KEY_PACKAGE, null)
                ?: return null
        val authorizedAt =
            preferences.getLong(KEY_AUTHORIZED_AT, -1L)
        val expiresAt =
            preferences.getLong(KEY_EXPIRES_AT, -1L)
        val session = runCatching {
            VN97GameControlSession(
                packageName = packageName,
                authorizedAtWallTimeMillis = authorizedAt,
                expiresAtWallTimeMillis = expiresAt,
            )
        }.getOrNull() ?: run {
            revoke()
            return null
        }
        if (!session.activeAt(nowWallTimeMillis)) {
            revoke()
            return null
        }
        return session
    }

    fun requireAuthorized(
        packageName: String,
        nowWallTimeMillis: Long = System.currentTimeMillis(),
    ) {
        val session = activeSessionOrNull(nowWallTimeMillis)
            ?: throw SecurityException(
                "no active user-authorized game control session"
            )
        if (session.packageName != packageName) {
            throw SecurityException(
                "game action package does not match active authorization"
            )
        }
    }

    companion object {
        const val DEFAULT_SESSION_MILLIS = 2 * 60 * 60 * 1000L
        const val MIN_SESSION_MILLIS = 5 * 60 * 1000L
        const val MAX_SESSION_MILLIS = 4 * 60 * 60 * 1000L

        private const val PREFS = "vn97-game-control"
        private const val KEY_PACKAGE = "package"
        private const val KEY_AUTHORIZED_AT = "authorized_at"
        private const val KEY_EXPIRES_AT = "expires_at"
    }
}

private val GAME_PACKAGE_RE =
    Regex("^[A-Za-z][A-Za-z0-9_]*(?:\\.[A-Za-z][A-Za-z0-9_]*)+$")

private fun isGamePackageName(value: String): Boolean =
    value.isNotEmpty() &&
        value.all { it.code <= 0x7f } &&
        GAME_PACKAGE_RE.matches(value)
