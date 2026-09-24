package ai.vn97.app

import android.Manifest
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Build
import android.provider.Settings
import android.util.Base64

/**
 * M19M clean-device turnkey acceptance probe.
 *
 * This receiver is exported only behind the platform DUMP permission in the
 * manifest. On production Android builds that permission is held by shell /
 * system tooling, not ordinary third-party apps. The probe never accepts an
 * arbitrary autonomous goal; it creates only one fixed, no-tool persistence
 * goal carrying a bounded nonce supplied by the host acceptance orchestrator.
 */
class VN97TurnkeyAcceptanceReceiver : BroadcastReceiver() {
    override fun onReceive(
        context: Context,
        intent: Intent,
    ) {
        if (
            intent.action != ACTION_BEGIN &&
            intent.action != ACTION_VERIFY_AFTER_REBOOT
        ) {
            return
        }

        val pending = goAsync()
        Thread(
            {
                try {
                    val app =
                        context.applicationContext as
                            VN97Application
                    check(
                        BuildConfig
                            .VN97_TURNKEY_REQUIRED
                    ) {
                        "M19M acceptance requires a turnkey build"
                    }

                    val expectedPackageSha256 =
                        requireSha256(
                            intent.getStringExtra(
                                EXTRA_EXPECTED_PACKAGE_SHA256
                            ),
                            "expected package SHA-256",
                        )
                    val nonce =
                        requireNonce(
                            intent.getStringExtra(
                                EXTRA_NONCE
                            )
                        )
                    val phase =
                        if (
                            intent.action ==
                                ACTION_BEGIN
                        ) {
                            PHASE_PRE_REBOOT
                        } else {
                            PHASE_POST_REBOOT
                        }

                    app.mobileRecovery
                        .recover(
                            if (
                                phase ==
                                    PHASE_PRE_REBOOT
                            ) {
                                VN97MobileRecoveryTrigger
                                    .EXECUTION_ENTRY
                            } else {
                                VN97MobileRecoveryTrigger
                                    .SYSTEM_RESTART
                            }
                        )
                        .requireActivationReady()

                    var activation =
                        app.provisioner
                            .currentModelActivation()
                    if (activation == null) {
                        val outcome =
                            app.bundledBootstrap
                                .activateIfPresent(
                                    required = true
                                )
                        check(
                            outcome !=
                                VN97BundledBootstrapOutcome
                                    .ABSENT
                        ) {
                            "turnkey bootstrap remained absent"
                        }
                        activation =
                            app.provisioner
                                .currentModelActivation()
                    }
                    val active =
                        checkNotNull(
                            activation
                        ) {
                            "turnkey model activation is missing"
                        }
                    check(
                        active.packageSha256 ==
                            expectedPackageSha256
                    ) {
                        "active turnkey package identity mismatch"
                    }

                    val assistantOpen =
                        app.assistant
                            .openIfActivated() ||
                            app.assistant
                                .reloadActivatedModel()
                    check(assistantOpen) {
                        "turnkey VN97 model could not be opened"
                    }

                    val overlayPermission =
                        Settings.canDrawOverlays(
                            context
                        )
                    check(overlayPermission) {
                        "overlay permission is required for M19M acceptance"
                    }

                    val microphonePermission =
                        context.checkSelfPermission(
                            Manifest.permission
                                .RECORD_AUDIO
                        ) ==
                            PackageManager
                                .PERMISSION_GRANTED
                    check(
                        microphonePermission
                    ) {
                        "microphone permission is required for M19M acceptance"
                    }

                    val notificationPermission =
                        Build.VERSION.SDK_INT <
                            Build.VERSION_CODES
                                .TIRAMISU ||
                            context
                                .checkSelfPermission(
                                    Manifest.permission
                                        .POST_NOTIFICATIONS
                                ) ==
                            PackageManager
                                .PERMISSION_GRANTED
                    check(
                        notificationPermission
                    ) {
                        "notification permission is required for M19M acceptance"
                    }

                    if (
                        phase ==
                            PHASE_PRE_REBOOT
                    ) {
                        VN97FloatingAssistantService
                            .enable(context)
                    }
                    val floatingEnabled =
                        VN97FloatingAssistantService
                            .isEnabled(context)
                    check(floatingEnabled) {
                        "floating assistant preference is not enabled"
                    }

                    var autonomous =
                        app.autonomousWork
                            .listGoals()
                            .firstOrNull {
                                it.goal.contains(
                                    nonce
                                )
                            }
                    if (
                        phase ==
                            PHASE_PRE_REBOOT &&
                        autonomous == null
                    ) {
                        autonomous =
                            app.autonomousWork
                                .startGoal(
                                    acceptanceGoal(
                                        nonce
                                    )
                                )
                    }
                    val goal =
                        checkNotNull(
                            autonomous
                        ) {
                            "M19M autonomous acceptance goal is missing"
                        }
                    check(
                        goal.goal.contains(
                            nonce
                        )
                    ) {
                        "M19M autonomous acceptance marker mismatch"
                    }

                    val payload =
                        canonicalResult(
                            activePackageSha256 =
                                active.packageSha256,
                            assistantOpen =
                                assistantOpen,
                            autonomousGoalPresent =
                                true,
                            autonomousGoalState =
                                goal.state.name,
                            floatingEnabled =
                                floatingEnabled,
                            microphonePermission =
                                microphonePermission,
                            notificationPermission =
                                notificationPermission,
                            overlayPermission =
                                overlayPermission,
                            phase = phase,
                        )
                    pending.setResultCode(
                        RESULT_OK
                    )
                    pending.setResultData(
                        Base64.encodeToString(
                            payload
                                .toByteArray(
                                    Charsets.UTF_8
                                ),
                            Base64.NO_WRAP,
                        )
                    )
                } catch (exc: Throwable) {
                    pending.setResultCode(
                        RESULT_FAILED
                    )
                    val detail =
                        (
                            exc::class.java
                                .simpleName +
                                ":" +
                                (
                                    exc.message
                                        ?: "unknown"
                                )
                        )
                            .replace(
                                Regex(
                                    "[\\u0000-\\u001f\\u007f]"
                                ),
                                " ",
                            )
                            .take(768)
                    pending.setResultData(
                        Base64.encodeToString(
                            detail.toByteArray(
                                Charsets.UTF_8
                            ),
                            Base64.NO_WRAP,
                        )
                    )
                } finally {
                    pending.finish()
                }
            },
            "vn97-m19m-acceptance",
        ).start()
    }

    private fun acceptanceGoal(
        nonce: String,
    ): String =
        "M19M acceptance persistence marker " +
            nonce +
            ". Complete internally only. " +
            "Do not use tools, network, device actions, files, trading, " +
            "game controls, or any external capability."

    private fun canonicalResult(
        activePackageSha256: String,
        assistantOpen: Boolean,
        autonomousGoalPresent: Boolean,
        autonomousGoalState: String,
        floatingEnabled: Boolean,
        microphonePermission: Boolean,
        notificationPermission: Boolean,
        overlayPermission: Boolean,
        phase: String,
    ): String =
        buildString {
            append("{")
            append("\"active_package_sha256\":\"")
            append(activePackageSha256)
            append("\",")
            append("\"assistant_open\":")
            append(assistantOpen)
            append(",")
            append("\"autonomous_goal_present\":")
            append(autonomousGoalPresent)
            append(",")
            append("\"autonomous_goal_state\":\"")
            append(autonomousGoalState)
            append("\",")
            append("\"floating_enabled\":")
            append(floatingEnabled)
            append(",")
            append("\"microphone_permission\":")
            append(microphonePermission)
            append(",")
            append("\"notification_permission\":")
            append(notificationPermission)
            append(",")
            append("\"overlay_permission\":")
            append(overlayPermission)
            append(",")
            append("\"phase\":\"")
            append(phase)
            append("\",")
            append("\"schema\":\"VN97SELFTEST1\",")
            append("\"turnkey_required\":true")
            append("}")
        }

    private fun requireSha256(
        value: String?,
        label: String,
    ): String {
        val resolved =
            checkNotNull(value) {
                "$label is missing"
            }
        require(
            resolved.length == 64 &&
                resolved.all {
                    it in
                        "0123456789abcdef"
                }
        ) {
            "$label must be lowercase SHA-256"
        }
        return resolved
    }

    private fun requireNonce(
        value: String?,
    ): String {
        val resolved =
            checkNotNull(value) {
                "M19M nonce is missing"
            }
        require(
            resolved.length == 16 &&
                resolved.all {
                    it in
                        "0123456789abcdef"
                }
        ) {
            "M19M nonce must be 16 lowercase hex characters"
        }
        return resolved
    }

    companion object {
        const val ACTION_BEGIN =
            "ai.vn97.app.action.M19M_ACCEPTANCE_BEGIN"
        const val ACTION_VERIFY_AFTER_REBOOT =
            "ai.vn97.app.action.M19M_ACCEPTANCE_VERIFY_AFTER_REBOOT"
        const val EXTRA_EXPECTED_PACKAGE_SHA256 =
            "ai.vn97.app.extra.M19M_EXPECTED_PACKAGE_SHA256"
        const val EXTRA_NONCE =
            "ai.vn97.app.extra.M19M_NONCE"

        private const val PHASE_PRE_REBOOT =
            "PRE_REBOOT"
        private const val PHASE_POST_REBOOT =
            "POST_REBOOT"

        private const val RESULT_OK = 0
        private const val RESULT_FAILED = 1
    }
}
