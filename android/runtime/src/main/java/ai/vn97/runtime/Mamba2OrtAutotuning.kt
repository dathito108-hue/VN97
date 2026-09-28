package ai.vn97.runtime

import java.io.File
import java.nio.file.Files
import java.security.MessageDigest
import org.json.JSONArray
import org.json.JSONObject

data class Mamba2OrtTuningDevice(
    val sdkInt: Int,
    val logicalCores: Int,
    val hardware: String,
    val socManufacturer: String,
    val socModel: String,
    val profiledAvailableMemoryBytes: Long,
) {
    init {
        require(sdkInt >= 26)
        require(logicalCores > 0)
        require(profiledAvailableMemoryBytes > 0L)
    }

    fun matches(device: OrtDeviceCapabilities): Boolean =
        sdkInt == device.sdkInt &&
            logicalCores == device.logicalCores &&
            hardware == device.hardware &&
            socManufacturer == device.socManufacturer &&
            socModel == device.socModel
}

data class Mamba2OrtWorkloadPolicy(
    val validLength: Int,
    val providerOrder: List<OrtProviderKind>,
    val preferredProvider: OrtProviderKind,
    val latencyP50Nanos: Long,
    val latencyP95Nanos: Long,
) {
    init {
        require(validLength > 0)
        require(providerOrder.isNotEmpty())
        require(providerOrder.distinct().size == providerOrder.size)
        require(providerOrder.last() == OrtProviderKind.CPU)
        require(preferredProvider == providerOrder.first())
        require(latencyP50Nanos > 0L)
        require(latencyP95Nanos >= latencyP50Nanos)
    }
}

data class Mamba2OrtControlPolicy(
    val thermalHotEnter: Int,
    val thermalCriticalEnter: Int,
    val xnnpackThreadsNormal: Int,
    val xnnpackThreadsHot: Int,
    val xnnpackThreadsCritical: Int,
) {
    init {
        require(thermalHotEnter in 0..6)
        require(thermalCriticalEnter in 0..6)
        require(thermalHotEnter < thermalCriticalEnter)
        require(xnnpackThreadsNormal > 0)
        require(xnnpackThreadsHot > 0)
        require(xnnpackThreadsCritical > 0)
    }

    fun xnnpackThreads(thermalStatus: Int): Int = when {
        thermalStatus >= thermalCriticalEnter ->
            xnnpackThreadsCritical
        thermalStatus >= thermalHotEnter ->
            xnnpackThreadsHot
        else -> xnnpackThreadsNormal
    }
}

data class Mamba2OrtProviderDecision(
    val profiledValidLength: Int,
    val providers: List<OrtProviderKind>,
    val xnnpackThreads: Int,
) {
    init {
        require(profiledValidLength > 0)
        require(providers.isNotEmpty())
        require(providers.last() == OrtProviderKind.CPU)
        require(xnnpackThreads > 0)
    }
}

data class Mamba2OrtTuningProfile(
    val tuningId: String,
    val runtimeId: String,
    val profileReceiptId: String,
    val graphFilename: String,
    val maxChunkSize: Int,
    val device: Mamba2OrtTuningDevice,
    val policies: Map<Int, Mamba2OrtWorkloadPolicy>,
    val control: Mamba2OrtControlPolicy,
) {
    init {
        requireG07Sha256(tuningId, "G0.7 tuning ID")
        requireG07Sha256(runtimeId, "G0.7 runtime ID")
        requireG07Sha256(profileReceiptId, "G0.7 profile receipt ID")
        require(maxChunkSize in setOf(8, 16, 32))
        require(graphFilename == "recurrent-" + maxChunkSize + ".onnx")
        require(policies.keys.sorted() == requiredG07Lengths(maxChunkSize))
    }

    fun requireCompatible(
        expectedRuntimeId: String,
        currentDevice: OrtDeviceCapabilities,
    ) {
        require(runtimeId == expectedRuntimeId) {
            "G0.7 tuning belongs to another runtime"
        }
        require(device.matches(currentDevice)) {
            "G0.7 tuning belongs to another device/OS identity"
        }
    }

    fun decide(
        validLength: Int,
        currentDevice: OrtDeviceCapabilities,
    ): Mamba2OrtProviderDecision {
        require(validLength in 1..maxChunkSize)
        require(device.matches(currentDevice)) {
            "G0.7 tuning device identity changed"
        }
        val profileLength = policies.keys.sorted().firstOrNull {
            it >= validLength
        } ?: maxChunkSize
        val policy = policies.getValue(profileLength)
        val filtered = policy.providerOrder.filter {
            providerAvailableG07(it, currentDevice)
        }.toMutableList()
        if (OrtProviderKind.CPU !in filtered) {
            filtered += OrtProviderKind.CPU
        }
        return Mamba2OrtProviderDecision(
            profiledValidLength = profileLength,
            providers = filtered,
            xnnpackThreads = control.xnnpackThreads(
                currentDevice.thermalStatus,
            ),
        )
    }

    companion object {
        const val SCHEMA = "VN97M2G07TUNE1"
        const val FILENAME = "tuning.vn97m2g07.json"

        fun load(
            file: File,
            expectedRuntimeId: String,
            currentDevice: OrtDeviceCapabilities,
        ): Mamba2OrtTuningProfile {
            require(
                file.isFile &&
                    !Files.isSymbolicLink(file.toPath())
            ) {
                "G0.7 tuning must be regular non-symlink file"
            }
            val root = JSONObject(file.readText(Charsets.US_ASCII))
            require(root.getString("schema") == SCHEMA)
            require(root.getBoolean("profile_is_device_measured"))
            require(root.getBoolean("same_weights_semantics"))
            require(!root.getBoolean("quantization_used"))
            require(!root.getBoolean("production_activation_authorized"))

            val tuningId = root.getString("tuning_id")
            requireG07Sha256(tuningId, "G0.7 tuning ID")
            val body = JSONObject(root.toString())
            body.remove("tuning_id")
            val prefix = "VN97M2G07TUNE1" +
                String(charArrayOf(0.toChar()))
            val expected = sha256G07(
                prefix.toByteArray(Charsets.US_ASCII) +
                    canonicalJsonG07(body).toByteArray(Charsets.US_ASCII)
            )
            require(tuningId == expected) {
                "G0.7 tuning identity mismatch"
            }

            val runtimeId = root.getString("runtime_id")
            requireG07Sha256(runtimeId, "G0.7 runtime ID")
            val receiptId = root.getString("profile_receipt_id")
            requireG07Sha256(receiptId, "G0.7 profile receipt ID")

            val maxChunk = root.getInt("max_chunk_size")
            require(maxChunk in setOf(8, 16, 32))
            val graph = root.getString("graph_filename")
            require(graph == "recurrent-" + maxChunk + ".onnx")

            val deviceJson = root.getJSONObject("device")
            require(
                deviceJson.keys().asSequence().toSet() == setOf(
                    "sdk_int",
                    "logical_cores",
                    "hardware",
                    "soc_manufacturer",
                    "soc_model",
                    "profiled_available_memory_bytes",
                )
            )
            val device = Mamba2OrtTuningDevice(
                sdkInt = deviceJson.getInt("sdk_int"),
                logicalCores = deviceJson.getInt("logical_cores"),
                hardware = deviceJson.getString("hardware"),
                socManufacturer = deviceJson.getString("soc_manufacturer"),
                socModel = deviceJson.getString("soc_model"),
                profiledAvailableMemoryBytes = deviceJson.getLong(
                    "profiled_available_memory_bytes"
                ),
            )

            val policies = linkedMapOf<Int, Mamba2OrtWorkloadPolicy>()
            val rawPolicies = root.getJSONArray("workload_policies")
            for (index in 0 until rawPolicies.length()) {
                val item = rawPolicies.getJSONObject(index)
                require(
                    item.keys().asSequence().toSet() == setOf(
                        "valid_length",
                        "provider_order",
                        "preferred_provider",
                        "latency_p50_ns",
                        "latency_p95_ns",
                    )
                )
                val validLength = item.getInt("valid_length")
                require(!policies.containsKey(validLength))
                val orderArray = item.getJSONArray("provider_order")
                val order = buildList {
                    for (providerIndex in 0 until orderArray.length()) {
                        add(
                            OrtProviderKind.valueOf(
                                orderArray.getString(providerIndex)
                            )
                        )
                    }
                }
                val policy = Mamba2OrtWorkloadPolicy(
                    validLength = validLength,
                    providerOrder = order,
                    preferredProvider = OrtProviderKind.valueOf(
                        item.getString("preferred_provider")
                    ),
                    latencyP50Nanos = item.getLong("latency_p50_ns"),
                    latencyP95Nanos = item.getLong("latency_p95_ns"),
                )
                policies[validLength] = policy
            }

            val controlJson = root.getJSONObject("control_policy")
            require(
                controlJson.keys().asSequence().toSet() == setOf(
                    "thermal_hot_enter",
                    "thermal_critical_enter",
                    "xnnpack_threads_normal",
                    "xnnpack_threads_hot",
                    "xnnpack_threads_critical",
                )
            )
            val profile = Mamba2OrtTuningProfile(
                tuningId = tuningId,
                runtimeId = runtimeId,
                profileReceiptId = receiptId,
                graphFilename = graph,
                maxChunkSize = maxChunk,
                device = device,
                policies = policies.toMap(),
                control = Mamba2OrtControlPolicy(
                    thermalHotEnter = controlJson.getInt(
                        "thermal_hot_enter"
                    ),
                    thermalCriticalEnter = controlJson.getInt(
                        "thermal_critical_enter"
                    ),
                    xnnpackThreadsNormal = controlJson.getInt(
                        "xnnpack_threads_normal"
                    ),
                    xnnpackThreadsHot = controlJson.getInt(
                        "xnnpack_threads_hot"
                    ),
                    xnnpackThreadsCritical = controlJson.getInt(
                        "xnnpack_threads_critical"
                    ),
                ),
            )
            profile.requireCompatible(
                expectedRuntimeId,
                currentDevice,
            )
            return profile
        }
    }
}

private fun requiredG07Lengths(maxChunk: Int): List<Int> =
    listOf(1, 8, 16, 32).filter {
        it == 1 || it <= maxChunk
    }

private fun providerAvailableG07(
    provider: OrtProviderKind,
    device: OrtDeviceCapabilities,
): Boolean = when (provider) {
    OrtProviderKind.QNN -> (
        device.qnnAvailable &&
            device.qnnBackendPath != null &&
            device.isQualcomm
        )
    OrtProviderKind.NNAPI -> (
        device.nnapiAvailable && device.sdkInt >= 27
        )
    OrtProviderKind.XNNPACK -> device.xnnpackAvailable
    OrtProviderKind.CPU -> true
}

private fun requireG07Sha256(value: String, label: String) {
    require(
        value.length == 64 &&
            value.all { it in '0'..'9' || it in 'a'..'f' }
    ) {
        label + " must be lowercase SHA-256"
    }
}

private fun sha256G07(bytes: ByteArray): String =
    MessageDigest.getInstance("SHA-256")
        .digest(bytes)
        .joinToString("") {
            "%02x".format(it.toInt() and 0xff)
        }

private fun canonicalJsonG07(value: Any?): String {
    return when (value) {
        JSONObject.NULL, null -> "null"
        is JSONObject -> {
            value.keys().asSequence().toList().sorted().joinToString(
                prefix = "{",
                postfix = "}",
                separator = ",",
            ) { key ->
                JSONObject.quote(key) + ":" +
                    canonicalJsonG07(value.get(key))
            }
        }
        is JSONArray -> {
            (0 until value.length()).joinToString(
                prefix = "[",
                postfix = "]",
                separator = ",",
            ) { index ->
                canonicalJsonG07(value.get(index))
            }
        }
        is String -> JSONObject.quote(value)
        is Boolean -> if (value) "true" else "false"
        is Int, is Long, is Short, is Byte -> value.toString()
        else -> error(
            "unsupported G0.7 canonical JSON type: " +
                value::class.java.name
        )
    }
}
