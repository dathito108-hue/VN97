package ai.vn97.app

import org.json.JSONArray
import org.json.JSONObject

internal object VN97ServiceLedgerCodec {
    const val MAX_BYTES = 2 * 1024 * 1024

    fun decode(encoded: String): VN97ServiceRevenueLedger {
        require(encoded.toByteArray(Charsets.UTF_8).size <= MAX_BYTES) { "stored ledger exceeds byte bound" }
        val root = JSONObject(encoded)
        require(root.getInt("schema") == 1)
        val orders = root.getJSONArray("orders")
        val payments = root.getJSONArray("payments")
        require(orders.length() <= VN97ServiceRevenueLedger.MAX_ORDERS)
        require(payments.length() <= VN97ServiceRevenueLedger.MAX_PAYMENTS)
        return VN97ServiceRevenueLedger(
            orders = (0 until orders.length()).map { index ->
                val value = orders.getJSONObject(index)
                VN97ServiceOrder(
                    orderId = value.getString("orderId"),
                    title = value.getString("title"),
                    service = VN97DigitalService.valueOf(value.getString("service")),
                    quotedPriceVnd = value.getLong("quotedPriceVnd"),
                    estimatedCostVnd = value.getLong("estimatedCostVnd"),
                    outputSha256 = value.getString("outputSha256"),
                    createdAtEpochMs = value.getLong("createdAtEpochMs"),
                    exportedAtEpochMs = if (value.isNull("exportedAtEpochMs")) null else {
                        value.getLong("exportedAtEpochMs")
                    },
                )
            },
            payments = (0 until payments.length()).map { index ->
                val value = payments.getJSONObject(index)
                VN97PaymentRecord(
                    orderId = value.getString("orderId"),
                    provider = value.getString("provider"),
                    providerAccountFingerprint = value.getString("providerAccountFingerprint"),
                    externalEventId = value.getString("externalEventId"),
                    grossVnd = value.getLong("grossVnd"),
                    providerFeeVnd = value.getLong("providerFeeVnd"),
                    refundVnd = value.getLong("refundVnd"),
                    realizedCostVnd = value.getLong("realizedCostVnd"),
                    verification = VN97PaymentVerification.valueOf(value.getString("verification")),
                    evidenceSha256 = if (value.isNull("evidenceSha256")) null else {
                        value.getString("evidenceSha256")
                    },
                    settledAtEpochMs = value.getLong("settledAtEpochMs"),
                )
            },
        )
    }

    fun encode(ledger: VN97ServiceRevenueLedger): String {
        val root = JSONObject().put("schema", 1)
            .put("orders", JSONArray().also { array ->
                ledger.orders().forEach { order ->
                    array.put(JSONObject().put("orderId", order.orderId).put("title", order.title)
                        .put("service", order.service.name).put("quotedPriceVnd", order.quotedPriceVnd)
                        .put("estimatedCostVnd", order.estimatedCostVnd)
                        .put("outputSha256", order.outputSha256)
                        .put("createdAtEpochMs", order.createdAtEpochMs)
                        .put("exportedAtEpochMs", order.exportedAtEpochMs ?: JSONObject.NULL))
                }
            }).put("payments", JSONArray().also { array ->
                ledger.payments().forEach { payment ->
                    array.put(JSONObject().put("orderId", payment.orderId)
                        .put("provider", payment.provider)
                        .put("providerAccountFingerprint", payment.providerAccountFingerprint)
                        .put("externalEventId", payment.externalEventId).put("grossVnd", payment.grossVnd)
                        .put("providerFeeVnd", payment.providerFeeVnd).put("refundVnd", payment.refundVnd)
                        .put("realizedCostVnd", payment.realizedCostVnd)
                        .put("verification", payment.verification.name)
                        .put("evidenceSha256", payment.evidenceSha256 ?: JSONObject.NULL)
                        .put("settledAtEpochMs", payment.settledAtEpochMs))
                }
            })
        val encoded = root.toString()
        require(encoded.toByteArray(Charsets.UTF_8).size <= MAX_BYTES) { "service ledger exceeds storage bound" }
        return encoded

    }
}
