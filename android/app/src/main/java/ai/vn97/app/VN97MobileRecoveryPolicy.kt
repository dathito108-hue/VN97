package ai.vn97.app

enum class VN97MobileRecoveryTrigger {
    PROCESS_START,
    SYSTEM_RESTART,
    PACKAGE_REPLACED,
    EXECUTION_ENTRY,
}

enum class VN97MobileRecoveryDomain {
    ACTIVATION,
    AUTONOMOUS,
    PAPER_TRADING,
}

enum class VN97MobileRecoveryStatus {
    SUCCEEDED,
    FAILED,
    SKIPPED,
}

data class VN97MobileRecoveryDomainResult(
    val domain: VN97MobileRecoveryDomain,
    val status: VN97MobileRecoveryStatus,
    val detail: String = "",
) {
    init {
        require(
            detail.toByteArray(
                Charsets.UTF_8
            ).size <= MAX_DETAIL_BYTES
        ) {
            "mobile recovery detail exceeds UTF-8 byte bound"
        }
        when (status) {
            VN97MobileRecoveryStatus.SUCCEEDED ->
                require(detail.isEmpty()) {
                    "successful recovery result must not carry detail"
                }

            VN97MobileRecoveryStatus.FAILED,
            VN97MobileRecoveryStatus.SKIPPED,
            ->
                require(detail.isNotBlank()) {
                    "non-success recovery result requires detail"
                }
        }
    }

    companion object {
        const val MAX_DETAIL_BYTES = 1024
    }
}

data class VN97MobileRecoveryReport(
    val trigger: VN97MobileRecoveryTrigger,
    val startedWallTimeMillis: Long,
    val completedWallTimeMillis: Long,
    val results: List<VN97MobileRecoveryDomainResult>,
) {
    init {
        require(startedWallTimeMillis >= 0L)
        require(
            completedWallTimeMillis >=
                startedWallTimeMillis
        )
        require(results.isNotEmpty())
        require(
            results.map { it.domain }
                .distinct()
                .size == results.size
        ) {
            "mobile recovery report contains duplicate domains"
        }
        require(
            results.first().domain ==
                VN97MobileRecoveryDomain.ACTIVATION
        ) {
            "activation recovery must be the first recovery domain"
        }
    }

    val activationReady: Boolean
        get() =
            result(
                VN97MobileRecoveryDomain.ACTIVATION
            )?.status ==
                VN97MobileRecoveryStatus.SUCCEEDED

    fun result(
        domain: VN97MobileRecoveryDomain,
    ): VN97MobileRecoveryDomainResult? =
        results.firstOrNull {
            it.domain == domain
        }

    fun requireActivationReady() {
        val activation =
            checkNotNull(
                result(
                    VN97MobileRecoveryDomain.ACTIVATION
                )
            )
        check(
            activation.status ==
                VN97MobileRecoveryStatus.SUCCEEDED
        ) {
            "VN97 activation recovery is not safe: " +
                activation.detail
        }
    }
}

internal fun executeVN97MobileRecovery(
    trigger: VN97MobileRecoveryTrigger,
    nowWallTimeMillis: () -> Long =
        System::currentTimeMillis,
    recover: (VN97MobileRecoveryDomain) -> Unit,
): VN97MobileRecoveryReport {
    val started =
        nowWallTimeMillis().also {
            require(it >= 0L)
        }
    val results =
        ArrayList<
            VN97MobileRecoveryDomainResult
        >(3)

    val activation =
        recoverDomain(
            domain =
                VN97MobileRecoveryDomain.ACTIVATION,
            recover = recover,
        )
    results += activation

    if (
        trigger ==
            VN97MobileRecoveryTrigger.SYSTEM_RESTART ||
        trigger ==
            VN97MobileRecoveryTrigger.PACKAGE_REPLACED
    ) {
        if (
            activation.status ==
                VN97MobileRecoveryStatus.SUCCEEDED
        ) {
            results +=
                recoverDomain(
                    domain =
                        VN97MobileRecoveryDomain
                            .AUTONOMOUS,
                    recover = recover,
                )
            results +=
                recoverDomain(
                    domain =
                        VN97MobileRecoveryDomain
                            .PAPER_TRADING,
                    recover = recover,
                )
        } else {
            val detail =
                "skipped because activation recovery failed"
            results +=
                VN97MobileRecoveryDomainResult(
                    domain =
                        VN97MobileRecoveryDomain
                            .AUTONOMOUS,
                    status =
                        VN97MobileRecoveryStatus
                            .SKIPPED,
                    detail = detail,
                )
            results +=
                VN97MobileRecoveryDomainResult(
                    domain =
                        VN97MobileRecoveryDomain
                            .PAPER_TRADING,
                    status =
                        VN97MobileRecoveryStatus
                            .SKIPPED,
                    detail = detail,
                )
        }
    }

    val completed =
        maxOf(
            started,
            nowWallTimeMillis(),
        )
    return VN97MobileRecoveryReport(
        trigger = trigger,
        startedWallTimeMillis = started,
        completedWallTimeMillis = completed,
        results = results,
    )
}

private fun recoverDomain(
    domain: VN97MobileRecoveryDomain,
    recover: (VN97MobileRecoveryDomain) -> Unit,
): VN97MobileRecoveryDomainResult =
    try {
        recover(domain)
        VN97MobileRecoveryDomainResult(
            domain = domain,
            status =
                VN97MobileRecoveryStatus.SUCCEEDED,
        )
    } catch (exc: Throwable) {
        when (exc) {
            is VirtualMachineError,
            is ThreadDeath,
            is LinkageError,
            -> throw exc

            else ->
                VN97MobileRecoveryDomainResult(
                    domain = domain,
                    status =
                        VN97MobileRecoveryStatus.FAILED,
                    detail =
                        boundedRecoveryFailure(exc),
                )
        }
    }

private fun boundedRecoveryFailure(
    exc: Throwable,
): String {
    val type =
        exc::class.java.simpleName
            .ifBlank { "Throwable" }
    val message =
        exc.message
            ?.replace('\n', ' ')
            ?.replace('\r', ' ')
            ?.trim()
            ?.takeIf { it.isNotEmpty() }
            ?: "unspecified"
    return "$type: $message"
        .toByteArray(Charsets.UTF_8)
        .let { bytes ->
            if (
                bytes.size <=
                    VN97MobileRecoveryDomainResult
                        .MAX_DETAIL_BYTES
            ) {
                "$type: $message"
            } else {
                truncateUtf8(
                    "$type: $message",
                    VN97MobileRecoveryDomainResult
                        .MAX_DETAIL_BYTES,
                )
            }
        }
}

private fun truncateUtf8(
    value: String,
    maxBytes: Int,
): String {
    require(maxBytes > 0)
    val out = StringBuilder()
    var index = 0
    var usedBytes = 0
    while (index < value.length) {
        val codePoint =
            Character.codePointAt(
                value,
                index,
            )
        val encoded =
            String(
                Character.toChars(
                    codePoint
                )
            ).toByteArray(Charsets.UTF_8)
        if (
            usedBytes + encoded.size >
                maxBytes
        ) {
            break
        }
        out.appendCodePoint(codePoint)
        usedBytes += encoded.size
        index += Character.charCount(codePoint)
    }
    return out.toString()
}
