package ai.vn97.app

private inline fun expectM18AFailure(
    label: String,
    block: () -> Unit,
) {
    check(runCatching(block).isFailure) {
        "expected M18A failure: $label"
    }
}

fun main() {
    var clock = 100L
    val processDomains =
        ArrayList<VN97MobileRecoveryDomain>()
    val process =
        executeVN97MobileRecovery(
            trigger =
                VN97MobileRecoveryTrigger
                    .PROCESS_START,
            nowWallTimeMillis = {
                clock++
            },
        ) { domain ->
            processDomains += domain
        }
    check(
        processDomains ==
            listOf(
                VN97MobileRecoveryDomain
                    .ACTIVATION
            )
    )
    check(process.activationReady)
    process.requireActivationReady()
    check(
        process.results.single().status ==
            VN97MobileRecoveryStatus
                .SUCCEEDED
    )

    val entryDomains =
        ArrayList<VN97MobileRecoveryDomain>()
    executeVN97MobileRecovery(
        trigger =
            VN97MobileRecoveryTrigger
                .EXECUTION_ENTRY,
        nowWallTimeMillis = {
            clock++
        },
    ) { domain ->
        entryDomains += domain
    }
    check(
        entryDomains ==
            listOf(
                VN97MobileRecoveryDomain
                    .ACTIVATION
            )
    )

    val rebootDomains =
        ArrayList<VN97MobileRecoveryDomain>()
    val reboot =
        executeVN97MobileRecovery(
            trigger =
                VN97MobileRecoveryTrigger
                    .SYSTEM_RESTART,
            nowWallTimeMillis = {
                clock++
            },
        ) { domain ->
            rebootDomains += domain
            if (
                domain ==
                    VN97MobileRecoveryDomain
                        .AUTONOMOUS
            ) {
                error(
                    "autonomous recovery failure"
                )
            }
        }
    check(
        rebootDomains ==
            listOf(
                VN97MobileRecoveryDomain
                    .ACTIVATION,
                VN97MobileRecoveryDomain
                    .AUTONOMOUS,
                VN97MobileRecoveryDomain
                    .PAPER_TRADING,
            )
    )
    check(reboot.activationReady)
    check(
        reboot.result(
            VN97MobileRecoveryDomain
                .AUTONOMOUS
        )?.status ==
            VN97MobileRecoveryStatus.FAILED
    )
    check(
        reboot.result(
            VN97MobileRecoveryDomain
                .PAPER_TRADING
        )?.status ==
            VN97MobileRecoveryStatus
                .SUCCEEDED
    )

    val failedActivationDomains =
        ArrayList<VN97MobileRecoveryDomain>()
    val failedActivation =
        executeVN97MobileRecovery(
            trigger =
                VN97MobileRecoveryTrigger
                    .PACKAGE_REPLACED,
            nowWallTimeMillis = {
                clock++
            },
        ) { domain ->
            failedActivationDomains += domain
            if (
                domain ==
                    VN97MobileRecoveryDomain
                        .ACTIVATION
            ) {
                throw IllegalStateException(
                    "inventory divergence"
                )
            }
        }
    check(
        failedActivationDomains ==
            listOf(
                VN97MobileRecoveryDomain
                    .ACTIVATION
            )
    )
    check(!failedActivation.activationReady)
    check(
        failedActivation.result(
            VN97MobileRecoveryDomain
                .AUTONOMOUS
        )?.status ==
            VN97MobileRecoveryStatus.SKIPPED
    )
    check(
        failedActivation.result(
            VN97MobileRecoveryDomain
                .PAPER_TRADING
        )?.status ==
            VN97MobileRecoveryStatus.SKIPPED
    )
    expectM18AFailure(
        "activation prerequisite"
    ) {
        failedActivation
            .requireActivationReady()
    }

    val huge =
        "x".repeat(10_000)
    val bounded =
        executeVN97MobileRecovery(
            trigger =
                VN97MobileRecoveryTrigger
                    .PROCESS_START,
            nowWallTimeMillis = {
                clock++
            },
        ) {
            throw IllegalArgumentException(
                huge
            )
        }
    val detail =
        checkNotNull(
            bounded.result(
                VN97MobileRecoveryDomain
                    .ACTIVATION
            )
        ).detail
    check(
        detail.toByteArray(
            Charsets.UTF_8
        ).size <=
            VN97MobileRecoveryDomainResult
                .MAX_DETAIL_BYTES
    )

    println(
        "M18A mobile recovery policy contracts: PASS"
    )
}
