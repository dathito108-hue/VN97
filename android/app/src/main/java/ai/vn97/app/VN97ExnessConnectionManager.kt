package ai.vn97.app

import ai.vn97.platform.VN97ExnessPrivateKeyCodec
import ai.vn97.platform.VN97ExnessReadOnlyClient
import java.util.concurrent.atomic.AtomicReference

data class VN97ExnessConnectionValidation(
    val evidence: VN97RevenueChannelEvidence,
    val accessPointBaseUrl: String?,
    val instrumentCount: Int?,
    val errorClass: String?,
) {
    init {
        require(
            (evidence.authenticated &&
                evidence.dryRunValidated) ==
                (accessPointBaseUrl != null &&
                    instrumentCount != null &&
                    errorClass == null)
        )
    }
}

class VN97ExnessConnectionManager(
    private val application: VN97Application,
) {
    private val validated =
        AtomicReference<VN97RevenueChannelEvidence?>(
            null
        )

    fun currentEvidence():
        VN97RevenueChannelEvidence {
        if (!application.exnessCredentialVault
                .hasCredentials()
        ) {
            validated.set(null)
            return unconfiguredEvidence()
        }
        return validated.get()
            ?: configuredEvidence()
    }

    fun invalidate() {
        validated.set(null)
    }

    fun validateReadOnly():
        VN97ExnessConnectionValidation {
        if (!application.exnessCredentialVault
                .hasCredentials()
        ) {
            validated.set(null)
            return VN97ExnessConnectionValidation(
                evidence = unconfiguredEvidence(),
                accessPointBaseUrl = null,
                instrumentCount = null,
                errorClass =
                    "credentials_not_configured",
            )
        }

        return try {
            val payload =
                application.exnessCredentialVault
                    .loadOrNull()
                    ?: error(
                        "Exness credential vault disappeared"
                    )
            payload.use {
                val result =
                    it.usePrivateKeySecret {
                            secret ->
                        val privateKey =
                            VN97ExnessPrivateKeyCodec
                                .decode(secret)
                        VN97ExnessReadOnlyClient(
                            apiKey = it.apiKey,
                            privateKey = privateKey,
                            discoveryBaseUrl =
                                it.baseUrl,
                            accountId = it.accountId,
                        ).validate()
                    }
                val evidence =
                    VN97RevenueChannelEvidence(
                        providerId = "exness",
                        configured = true,
                        credentialBackedByKeystore =
                            true,
                        authenticated = true,
                        dryRunValidated = true,
                        orderSubmissionAvailable =
                            false,
                    )
                validated.set(evidence)
                VN97ExnessConnectionValidation(
                    evidence = evidence,
                    accessPointBaseUrl =
                        result.accessPointBaseUrl,
                    instrumentCount =
                        result.instrumentCount,
                    errorClass = null,
                )
            }
        } catch (exc: Throwable) {
            val evidence =
                configuredEvidence()
            validated.set(evidence)
            VN97ExnessConnectionValidation(
                evidence = evidence,
                accessPointBaseUrl = null,
                instrumentCount = null,
                errorClass =
                    exc::class.java.simpleName
                        .take(96),
            )
        }
    }

    private fun configuredEvidence() =
        VN97RevenueChannelEvidence(
            providerId = "exness",
            configured = true,
            credentialBackedByKeystore = true,
            authenticated = false,
            dryRunValidated = false,
            orderSubmissionAvailable = false,
        )

    private fun unconfiguredEvidence() =
        VN97RevenueChannelEvidence(
            providerId = "exness",
            configured = false,
            credentialBackedByKeystore = false,
            authenticated = false,
            dryRunValidated = false,
            orderSubmissionAvailable = false,
        )
}
