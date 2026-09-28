package ai.vn97.runtime

import java.io.File
import java.nio.file.Files
import java.security.MessageDigest
import org.json.JSONArray
import org.json.JSONObject

data class Mamba2ProductionPromotion(
    val promotionId: String,
    val bridgeId: String,
    val runtimeId: String,
    val capsuleId: String,
    val sourceWeightSha256: String,
    val tokenizerId: String,
    val tuningId: String,
    val deviceProfileReceiptId: String,
    val modelParityReceiptId: String,
    val tokenParityReceiptId: String,
    val mobileQualificationReceiptId: String,
    val targetFamily: String,
) {
    init {
        listOf(
            promotionId,
            bridgeId,
            runtimeId,
            capsuleId,
            sourceWeightSha256,
            tokenizerId,
            tuningId,
            deviceProfileReceiptId,
            modelParityReceiptId,
            tokenParityReceiptId,
            mobileQualificationReceiptId,
        ).forEach {
            require(
                it.length == 64 &&
                    it.all { ch -> ch in '0'..'9' || ch in 'a'..'f' }
            )
        }
        require(targetFamily == "Samsung Galaxy S21 FE")
    }

    fun requireCompatible(
        binding: Mamba2CognitionBridgeBinding,
        runtime: Mamba2OrtRuntimePackage,
        tokenizer: Mamba2TokenizerPackage,
        tuning: Mamba2OrtTuningProfile,
    ) {
        require(binding.bridgeId == bridgeId) {
            "G0.10 promotion bridge identity mismatch"
        }
        require(runtime.runtimeId == runtimeId) {
            "G0.10 promotion runtime identity mismatch"
        }
        require(runtime.capsuleId == capsuleId) {
            "G0.10 promotion capsule identity mismatch"
        }
        require(runtime.sourceWeightSha256 == sourceWeightSha256) {
            "G0.10 promotion source-weight identity mismatch"
        }
        require(tokenizer.tokenizerId == tokenizerId) {
            "G0.10 promotion tokenizer identity mismatch"
        }
        require(tuning.tuningId == tuningId) {
            "G0.10 promotion tuning identity mismatch"
        }
        require(tuning.profileReceiptId == deviceProfileReceiptId) {
            "G0.10 promotion device-profile receipt mismatch"
        }
    }

    companion object {
        const val SCHEMA = "VN97M2G10PROMOTE1"
        const val FILENAME = "promotion.vn97m2g10.json"

        fun load(file: File): Mamba2ProductionPromotion {
            require(
                file.isFile &&
                    !Files.isSymbolicLink(file.toPath())
            ) {
                "G0.10 promotion must be regular non-symlink file"
            }
            val root = JSONObject(file.readText(Charsets.US_ASCII))
            val expectedFields = setOf(
                "schema",
                "bridge_id",
                "runtime_id",
                "capsule_id",
                "source_weight_sha256",
                "tokenizer_id",
                "tuning_id",
                "device_profile_receipt_id",
                "model_parity_receipt_id",
                "token_parity_receipt_id",
                "mobile_qualification_receipt_id",
                "model_max_abs_error",
                "state_max_abs_error",
                "token_parity_cases",
                "target_family",
                "same_weights_semantics",
                "same_token_ids_required",
                "real_evidence_required",
                "rollback_required",
                "production_activation_authorized",
                "promotion_id",
            )
            require(root.keys().asSequence().toSet() == expectedFields) {
                "G0.10 promotion fields mismatch"
            }
            require(root.getString("schema") == SCHEMA)
            require(root.getBoolean("same_weights_semantics"))
            require(root.getBoolean("same_token_ids_required"))
            require(root.getBoolean("real_evidence_required"))
            require(root.getBoolean("rollback_required"))
            require(root.getBoolean("production_activation_authorized"))
            require(root.getInt("token_parity_cases") >= 100)
            require(root.getDouble("model_max_abs_error") >= 0.0)
            require(root.getDouble("state_max_abs_error") >= 0.0)
            require(
                root.getString("target_family") ==
                    "Samsung Galaxy S21 FE"
            )

            val promotionId = root.getString("promotion_id")
            requireG10Sha(promotionId, "G0.10 promotion ID")
            val body = JSONObject(root.toString())
            body.remove("promotion_id")
            val prefix = "VN97M2G10PROMOTE1" +
                String(charArrayOf(0.toChar()))
            val expected = sha256G10(
                prefix.toByteArray(Charsets.US_ASCII) +
                    canonicalJsonG10(body)
                        .toByteArray(Charsets.US_ASCII)
            )
            require(promotionId == expected) {
                "G0.10 promotion identity mismatch"
            }

            return Mamba2ProductionPromotion(
                promotionId = promotionId,
                bridgeId = root.getString("bridge_id"),
                runtimeId = root.getString("runtime_id"),
                capsuleId = root.getString("capsule_id"),
                sourceWeightSha256 =
                    root.getString("source_weight_sha256"),
                tokenizerId = root.getString("tokenizer_id"),
                tuningId = root.getString("tuning_id"),
                deviceProfileReceiptId =
                    root.getString("device_profile_receipt_id"),
                modelParityReceiptId =
                    root.getString("model_parity_receipt_id"),
                tokenParityReceiptId =
                    root.getString("token_parity_receipt_id"),
                mobileQualificationReceiptId =
                    root.getString(
                        "mobile_qualification_receipt_id"
                    ),
                targetFamily = root.getString("target_family"),
            )
        }
    }
}

private fun requireG10Sha(value: String, label: String) {
    require(
        value.length == 64 &&
            value.all { it in '0'..'9' || it in 'a'..'f' }
    ) {
        label + " must be lowercase SHA-256"
    }
}

private fun sha256G10(bytes: ByteArray): String =
    MessageDigest.getInstance("SHA-256")
        .digest(bytes)
        .joinToString("") {
            "%02x".format(it.toInt() and 0xff)
        }

private fun canonicalJsonG10(value: Any?): String {
    return when (value) {
        JSONObject.NULL, null -> "null"
        is JSONObject -> {
            value.keys().asSequence().toList().sorted().joinToString(
                prefix = "{",
                postfix = "}",
                separator = ",",
            ) { key ->
                JSONObject.quote(key) + ":" +
                    canonicalJsonG10(value.get(key))
            }
        }
        is JSONArray -> {
            (0 until value.length()).joinToString(
                prefix = "[",
                postfix = "]",
                separator = ",",
            ) { index ->
                canonicalJsonG10(value.get(index))
            }
        }
        is String -> JSONObject.quote(value)
        is Boolean -> if (value) "true" else "false"
        is Int, is Long, is Short, is Byte -> value.toString()
        is Double -> {
            require(value.isFinite())
            value.toString()
        }
        is Float -> {
            require(value.isFinite())
            value.toString()
        }
        else -> error(
            "unsupported G0.10 canonical JSON type: " +
                value::class.java.name
        )
    }
}
