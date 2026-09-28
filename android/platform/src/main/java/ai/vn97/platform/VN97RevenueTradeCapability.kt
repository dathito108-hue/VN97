package ai.vn97.platform

import java.math.BigDecimal
import org.json.JSONObject

enum class VN97RevenueTradeSide(
    val wireValue: String,
) {
    BUY("buy"),
    SELL("sell"),
}

data class VN97RevenueOpenPositionRequest(
    val accountId: String,
    val campaignId: String,
    val instrument: String,
    val side: VN97RevenueTradeSide,
    val volume: String,
    val idempotencyKey: String,
) {
    init {
        require(
            accountId.isNotEmpty() &&
                accountId.length <= 128 &&
                accountId.all(Char::isDigit)
        ) {
            "revenue account ID must contain decimal digits only"
        }
        require(
            campaignId.length == 64 &&
                campaignId.all {
                    it in "0123456789abcdef"
                }
        ) {
            "revenue campaign ID must be lowercase SHA-256 hex"
        }
        require(
            instrument.matches(
                Regex("^[A-Za-z0-9._-]{1,64}$")
            )
        ) {
            "revenue instrument identifier is invalid"
        }
        requireCanonicalPositiveDecimal(
            volume,
            "revenue volume",
        )
        require(
            idempotencyKey.matches(
                Regex("^[A-Za-z0-9._:-]{1,128}$")
            )
        ) {
            "revenue idempotency key is invalid"
        }
    }
}

enum class VN97RevenueTradeTerminalStatus {
    CONFIRMED,
    REJECTED,
    FAILED,
}

data class VN97RevenueTradeResult(
    val operationId: String,
    val clientRequestId: String?,
    val status: VN97RevenueTradeTerminalStatus,
    val errorCode: String?,
) {
    init {
        require(operationId.isNotBlank())
        require(
            clientRequestId == null ||
                clientRequestId.isNotBlank()
        )
        require(
            errorCode == null ||
                errorCode.length <= 128
        )
    }
}

fun interface VN97RevenueOrderPort {
    fun openPosition(
        request: VN97RevenueOpenPositionRequest,
    ): VN97RevenueTradeResult
}

class VN97RevenueProductionCapabilities(
    private val orders: VN97RevenueOrderPort,
) {
    val descriptor =
        M6CapabilityDescriptor(
            capabilityId = CAPABILITY_ID,
            requiredScopeKeys =
                setOf(
                    ACCOUNT_SCOPE,
                    CAMPAIGN_SCOPE,
                ),
            approvalRequired = true,
            maxPayloadUtf8Bytes = 256,
            payloadSchemaJson =
                "{\"instrument\":\"XAUUSD\",\"side\":\"buy|sell\",\"volume\":\"0.01\"}",
            maxLeaseNs = 60_000_000_000L,
            maxLeaseUses = 1,
        )

    val intentBinder =
        M6ExternalIntentBinder(
            listOf(descriptor)
        )

    fun createSealedRegistry():
        M6TypedCapabilityRegistry =
        M6TypedCapabilityRegistry().also {
                registry ->
            registry.register(
                descriptor,
                M6CapabilityHandler { action ->
                    val request =
                        validateRequest(
                            action.request
                        )
                    val result =
                        orders.openPosition(
                            request.copy(
                                idempotencyKey =
                                    idempotencyKeyFor(
                                        action.request
                                            .requestDigest
                                    )
                            )
                        )
                    val success =
                        result.status ==
                            VN97RevenueTradeTerminalStatus
                                .CONFIRMED
                    M6ActionOutcome(
                        success = success,
                        retryable = false,
                        result =
                            JSONObject()
                                .put(
                                    "operation_id",
                                    result.operationId,
                                )
                                .put(
                                    "client_request_id",
                                    result.clientRequestId,
                                )
                                .put(
                                    "status",
                                    result.status.name
                                        .lowercase(),
                                )
                                .put(
                                    "error_code",
                                    result.errorCode,
                                )
                                .toString(),
                    )
                },
                M6PayloadValidator(
                    ::validateRequest
                ),
            )
            registry.seal()
        }

    private fun validateRequest(
        request: M6ExternalActionRequest,
    ): VN97RevenueOpenPositionRequest {
        require(
            request.capabilityId ==
                CAPABILITY_ID
        ) {
            "revenue capability mismatch"
        }
        val scope = request.scope.asMap()
        val accountId =
            requireNotNull(
                scope[ACCOUNT_SCOPE]
            ) {
                "revenue account scope is missing"
            }
        val campaignId =
            requireNotNull(
                scope[CAMPAIGN_SCOPE]
            ) {
                "revenue campaign scope is missing"
            }
        val payload =
            JSONObject(request.payloadJson)
        require(
            payload.length() == 3 &&
                payload.has("instrument") &&
                payload.has("side") &&
                payload.has("volume")
        ) {
            "revenue payload has unsupported fields"
        }
        val side =
            when (
                payload.getString("side")
                    .lowercase()
            ) {
                "buy" ->
                    VN97RevenueTradeSide.BUY
                "sell" ->
                    VN97RevenueTradeSide.SELL
                else ->
                    throw IllegalArgumentException(
                        "unsupported revenue side"
                    )
            }
        return VN97RevenueOpenPositionRequest(
            accountId = accountId,
            campaignId = campaignId,
            instrument =
                payload.getString("instrument"),
            side = side,
            volume =
                canonicalPositiveDecimal(
                    payload.getString("volume"),
                    "revenue volume",
                ),
            idempotencyKey =
                idempotencyKeyFor(
                    request.requestDigest
                ),
        )
    }

    companion object {
        const val CAPABILITY_ID =
            "revenue.exness.open_position"
        const val ACCOUNT_SCOPE =
            "account_id"
        const val CAMPAIGN_SCOPE =
            "campaign_id"

        private fun idempotencyKeyFor(
            requestDigest: String,
        ): String {
            require(
                requestDigest.length == 64 &&
                    requestDigest.all {
                        it in "0123456789abcdef"
                    }
            )
            return "vn97-open-" +
                requestDigest.take(48)
        }
    }
}

internal fun canonicalPositiveDecimal(
    value: String,
    label: String,
): String {
    require(
        value.matches(
            Regex("^(?:0|[1-9][0-9]{0,11})(?:\\.[0-9]{1,8})?$")
        )
    ) {
        "$label must be a bounded plain decimal"
    }
    val decimal = BigDecimal(value)
    require(decimal > BigDecimal.ZERO) {
        "$label must be positive"
    }
    return decimal.stripTrailingZeros()
        .toPlainString()
}

internal fun requireCanonicalPositiveDecimal(
    value: String,
    label: String,
) {
    require(
        canonicalPositiveDecimal(
            value,
            label,
        ) == value
    ) {
        "$label must be canonical"
    }
}
