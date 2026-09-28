package ai.vn97.platform

import java.net.URI

object VN97ExnessEndpointPolicy {
    fun normalizeBaseUrl(
        raw: String,
    ): String {
        val candidate =
            if ("://" in raw) {
                raw
            } else {
                "https://$raw"
            }
        val uri = URI(candidate)
        require(
            uri.scheme.equals(
                "https",
                ignoreCase = true,
            )
        ) {
            "Exness endpoint must use HTTPS"
        }
        val host =
            uri.host?.lowercase()
                ?: throw IllegalArgumentException(
                    "Exness endpoint host is missing"
                )
        require(isAllowedHost(host)) {
            "Exness endpoint host is outside approved domains"
        }
        require(
            uri.port == -1 || uri.port == 443
        ) {
            "Exness endpoint must use the default HTTPS port"
        }
        require(
            uri.rawUserInfo == null &&
                uri.rawQuery == null &&
                uri.rawFragment == null &&
                (uri.rawPath.isNullOrEmpty() ||
                    uri.rawPath == "/")
        ) {
            "Exness base URL must not contain credentials, path, query or fragment"
        }
        return "https://$host"
    }

    fun accessPointPath(
        accountId: String,
    ): String =
        "/v1/trading/access-point?account_id=" +
            validateAccountId(accountId)

    fun accountDetailsPath(
        accountId: String,
    ): String =
        "/v1/configuration/accounts/" +
            validateAccountId(accountId) +
            "/account"

    fun instrumentListPath(
        accountId: String,
    ): String =
        "/v1/configuration/accounts/" +
            validateAccountId(accountId) +
            "/instruments"

    private fun validateAccountId(
        accountId: String,
    ): String {
        require(
            accountId.isNotEmpty() &&
                accountId.length <= 128 &&
                accountId.all(Char::isDigit)
        ) {
            "Exness account ID must contain decimal digits only"
        }
        return accountId
    }

    private fun isAllowedHost(
        host: String,
    ): Boolean =
        host == "exness.com" ||
            host.endsWith(".exness.com") ||
            host == "exness-api.com" ||
            host.endsWith(".exness-api.com")
}
