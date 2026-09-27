package ai.vn97.runtime

import java.io.File
import java.io.FileInputStream
import java.nio.file.Files
import java.security.MessageDigest
import org.json.JSONArray
import org.json.JSONObject

data class OrtProductionGraph(
    val filename: String,
    val kind: String,
    val sequenceLength: Int,
    val sha256: String,
    val bytes: Long,
    val file: File,
) {
    init {
        require(filename == "step.onnx" || filename.matches(
            Regex("^chunk-[1-9][0-9]*\\.onnx$")
        )) {
            "F1 graph filename is invalid"
        }
        require(kind == "step" || kind == "chunk")
        require(sequenceLength > 0)
        requireSha256F1(sha256, "graph sha256")
        require(bytes > 0L)
        if (kind == "step") {
            require(filename == "step.onnx")
            require(sequenceLength == 1)
        } else {
            require(filename == "chunk-" + sequenceLength + ".onnx")
        }
    }
}

data class OrtProductionPackage(
    val runtimeId: String,
    val bundleId: String,
    val architectureFingerprint: String,
    val modelProfile: String,
    val tuningId: String,
    val activeLayers: Int,
    val vocabSize: Int,
    val dInner: Int,
    val dState: Int,
    val dConvState: Int,
    val graphs: Map<String, OrtProductionGraph>,
    val supportedChunkSizes: List<Int>,
    val tuning: OrtAutotuneProfile,
) {
    init {
        requireSha256F1(runtimeId, "runtimeId")
        requireSha256F1(bundleId, "bundleId")
        requireSha256F1(
            architectureFingerprint,
            "architectureFingerprint",
        )
        requireSha256F1(tuningId, "tuningId")
        require(activeLayers > 0)
        require(vocabSize > 0)
        require(dInner > 0)
        require(dState > 0)
        require(dConvState > 0)
        require(graphs.containsKey("step.onnx"))
        require(supportedChunkSizes.isNotEmpty())
        require(supportedChunkSizes == supportedChunkSizes.sorted())
        require(supportedChunkSizes.distinct().size == supportedChunkSizes.size)
        require(
            supportedChunkSizes.all {
                graphs.containsKey("chunk-" + it + ".onnx")
            }
        )
        require(tuning.bundleId == bundleId)
        require(tuning.tuningId == tuningId)
        require(tuning.architectureFingerprint == architectureFingerprint)
        require(tuning.modelProfile == modelProfile)
        require(tuning.graphPolicies.keys == graphs.keys)
    }

    val convStateElements: Int
        get() = checkedArrayCountF1(
            "conv state",
            activeLayers.toLong(),
            dInner.toLong(),
            dConvState.toLong(),
        )

    val ssmStateElements: Int
        get() = checkedArrayCountF1(
            "ssm state",
            activeLayers.toLong(),
            dInner.toLong(),
            dState.toLong(),
        )

    companion object {
        const val SCHEMA = "VN97R2F1RUNTIME1"
        const val FILENAME = "runtime.vn97ort1.json"

        fun load(
            rootDir: File,
            tuningFile: File,
            currentDevice: OrtDeviceCapabilities,
        ): OrtProductionPackage {
            require(
                rootDir.isDirectory &&
                    !Files.isSymbolicLink(rootDir.toPath())
            ) {
                "F1 runtime root must be a real directory"
            }
            val root = rootDir.canonicalFile
            val descriptor = File(root, FILENAME)
            requireSafeRegularFileF1(
                descriptor,
                root,
                "F1 runtime descriptor",
            )
            val raw = descriptor.readText(Charsets.UTF_8)
            require(!raw.contains('\u0000')) {
                "F1 runtime descriptor contains NUL"
            }
            val json = JSONObject(raw)
            require(json.getString("schema") == SCHEMA) {
                "F1 runtime package schema mismatch"
            }

            val expectedFields = setOf(
                "schema",
                "bundle_id",
                "architecture_fingerprint",
                "profile",
                "tuning_id",
                "active_layers",
                "vocab_size",
                "d_inner",
                "d_state",
                "d_conv_state",
                "batch_size",
                "inputs",
                "outputs",
                "graphs",
                "supported_chunk_sizes",
                "state_dtype",
                "token_dtype",
                "same_weights_semantics",
                "quantization_used",
                "runtime_id",
            )
            require(json.keys().asSequence().toSet() == expectedFields) {
                "F1 runtime package fields mismatch"
            }

            val runtimeId = json.getString("runtime_id")
            requireSha256F1(runtimeId, "runtimeId")
            val body = JSONObject(json.toString())
            body.remove("runtime_id")
            val prefix = "VN97R2F1RUNTIME1" +
                String(charArrayOf(0.toChar()))
            val expectedRuntimeId = sha256F1(
                prefix.toByteArray(Charsets.UTF_8) +
                    canonicalJsonF1(body).toByteArray(Charsets.UTF_8)
            )
            require(runtimeId == expectedRuntimeId) {
                "F1 runtime package identity mismatch"
            }

            require(json.getInt("batch_size") == 1) {
                "F1 production runtime requires batch=1"
            }
            require(json.getString("state_dtype") == "float32")
            require(json.getString("token_dtype") == "int64")
            require(json.getBoolean("same_weights_semantics"))
            require(!json.getBoolean("quantization_used"))
            requireStringArrayF1(
                json.getJSONArray("inputs"),
                listOf("input_ids", "conv_state", "ssm_state"),
                "F1 input contract",
            )
            requireStringArrayF1(
                json.getJSONArray("outputs"),
                listOf(
                    "logits",
                    "next_conv_state",
                    "next_ssm_state",
                ),
                "F1 output contract",
            )

            val bundleId = json.getString("bundle_id")
            val architecture = json.getString(
                "architecture_fingerprint"
            )
            val tuningId = json.getString("tuning_id")
            requireSha256F1(bundleId, "bundleId")
            requireSha256F1(architecture, "architectureFingerprint")
            requireSha256F1(tuningId, "tuningId")

            val activeLayers = json.getInt("active_layers")
            val vocabSize = json.getInt("vocab_size")
            val dInner = json.getInt("d_inner")
            val dState = json.getInt("d_state")
            val dConvState = json.getInt("d_conv_state")
            require(activeLayers > 0)
            require(vocabSize > 0)
            require(dInner > 0)
            require(dState > 0)
            require(dConvState > 0)

            val graphMap = linkedMapOf<String, OrtProductionGraph>()
            val rawGraphs = json.getJSONArray("graphs")
            require(rawGraphs.length() >= 2) {
                "F1 requires step and chunk graphs"
            }
            for (index in 0 until rawGraphs.length()) {
                val item = rawGraphs.getJSONObject(index)
                require(
                    item.keys().asSequence().toSet() == setOf(
                        "filename",
                        "kind",
                        "sequence_length",
                        "sha256",
                        "bytes",
                    )
                ) {
                    "F1 graph fields mismatch"
                }
                val filename = item.getString("filename")
                require(!graphMap.containsKey(filename)) {
                    "F1 duplicate graph filename"
                }
                require(
                    !filename.contains('/') &&
                        !filename.contains('\\')
                ) {
                    "F1 graph filename must not contain path separators"
                }
                val graphFile = File(root, filename)
                requireSafeRegularFileF1(
                    graphFile,
                    root,
                    "F1 graph " + filename,
                )
                val expectedBytes = item.getLong("bytes")
                require(graphFile.length() == expectedBytes) {
                    "F1 graph byte size mismatch for " + filename
                }
                val expectedSha = item.getString("sha256")
                requireSha256F1(
                    expectedSha,
                    "F1 graph " + filename + " SHA",
                )
                require(sha256FileF1(graphFile) == expectedSha) {
                    "F1 graph SHA mismatch for " + filename
                }
                val graph = OrtProductionGraph(
                    filename = filename,
                    kind = item.getString("kind"),
                    sequenceLength = item.getInt("sequence_length"),
                    sha256 = expectedSha,
                    bytes = expectedBytes,
                    file = graphFile,
                )
                graphMap[filename] = graph
            }

            val chunksJson = json.getJSONArray("supported_chunk_sizes")
            val chunks = buildList {
                for (index in 0 until chunksJson.length()) {
                    add(chunksJson.getInt(index))
                }
            }
            require(chunks.isNotEmpty())
            require(chunks == chunks.sorted())
            require(chunks.distinct().size == chunks.size)
            require(
                graphMap["step.onnx"]?.sequenceLength == 1 &&
                    chunks.all {
                        graphMap.containsKey("chunk-" + it + ".onnx")
                    } &&
                    graphMap.values.count { it.kind == "chunk" } ==
                    chunks.size
            ) {
                "F1 graph/chunk coverage mismatch"
            }

            requireSafeRegularFileF1(
                tuningFile,
                tuningFile.parentFile?.canonicalFile
                    ?: root,
                "F1 E4 tuning profile",
                requireUnderRoot = false,
            )
            val tuning = OrtAutotuneProfile.load(
                tuningFile,
                bundleId,
                currentDevice,
            )
            require(tuning.tuningId == tuningId) {
                "F1 descriptor belongs to another E4 tuning profile"
            }
            require(tuning.architectureFingerprint == architecture) {
                "F1 descriptor architecture differs from E4"
            }
            val profile = json.getString("profile")
            require(tuning.modelProfile == profile) {
                "F1 descriptor fast/deep profile differs from E4"
            }
            require(tuning.graphPolicies.keys == graphMap.keys) {
                "F1 E4 graph coverage differs from descriptor"
            }

            return OrtProductionPackage(
                runtimeId = runtimeId,
                bundleId = bundleId,
                architectureFingerprint = architecture,
                modelProfile = profile,
                tuningId = tuningId,
                activeLayers = activeLayers,
                vocabSize = vocabSize,
                dInner = dInner,
                dState = dState,
                dConvState = dConvState,
                graphs = graphMap.toMap(),
                supportedChunkSizes = chunks,
                tuning = tuning,
            )
        }
    }
}

private fun requireSafeRegularFileF1(
    file: File,
    root: File,
    label: String,
    requireUnderRoot: Boolean = true,
) {
    require(
        file.isFile &&
            !Files.isSymbolicLink(file.toPath())
    ) {
        label + " must be a real regular file"
    }
    if (requireUnderRoot) {
        val canonical = file.canonicalFile
        val rootPath = root.canonicalFile.toPath()
        require(canonical.toPath().startsWith(rootPath)) {
            label + " escapes the runtime root"
        }
    }
}

private fun requireStringArrayF1(
    array: JSONArray,
    expected: List<String>,
    label: String,
) {
    val actual = buildList {
        for (index in 0 until array.length()) {
            add(array.getString(index))
        }
    }
    require(actual == expected) {
        label + " mismatch"
    }
}

private fun checkedArrayCountF1(
    label: String,
    vararg dimensions: Long,
): Int {
    var product = 1L
    for (dimension in dimensions) {
        require(dimension > 0L) {
            label + " dimension must be positive"
        }
        require(product <= Int.MAX_VALUE.toLong() / dimension) {
            label + " exceeds JVM array/buffer bounds"
        }
        product *= dimension
    }
    return product.toInt()
}

private fun sha256FileF1(file: File): String {
    val digest = MessageDigest.getInstance("SHA-256")
    FileInputStream(file).buffered(64 * 1024).use { input ->
        val buffer = ByteArray(64 * 1024)
        while (true) {
            val count = input.read(buffer)
            if (count < 0) break
            require(count > 0) {
                "F1 graph digest read made no progress"
            }
            digest.update(buffer, 0, count)
        }
    }
    return digest.digest().joinToString("") {
        "%02x".format(it.toInt() and 0xff)
    }
}

private fun requireSha256F1(value: String, label: String) {
    require(
        value.length == 64 &&
            value.all {
                it in '0'..'9' || it in 'a'..'f'
            }
    ) {
        label + " must be lowercase SHA-256"
    }
}

private fun sha256F1(bytes: ByteArray): String =
    MessageDigest.getInstance("SHA-256")
        .digest(bytes)
        .joinToString("") {
            "%02x".format(it.toInt() and 0xff)
        }

private fun canonicalJsonF1(value: Any?): String {
    return when (value) {
        JSONObject.NULL, null -> "null"
        is JSONObject -> {
            val keys = value.keys().asSequence().toList().sorted()
            keys.joinToString(
                prefix = "{",
                postfix = "}",
                separator = ",",
            ) { key ->
                JSONObject.quote(key) + ":" +
                    canonicalJsonF1(value.get(key))
            }
        }
        is JSONArray -> {
            (0 until value.length()).joinToString(
                prefix = "[",
                postfix = "]",
                separator = ",",
            ) { index ->
                canonicalJsonF1(value.get(index))
            }
        }
        is String -> JSONObject.quote(value)
        is Boolean -> if (value) "true" else "false"
        is Int, is Long, is Short, is Byte -> value.toString()
        else -> error(
            "unsupported F1 canonical JSON type: " +
                value::class.java.name
        )
    }
}
