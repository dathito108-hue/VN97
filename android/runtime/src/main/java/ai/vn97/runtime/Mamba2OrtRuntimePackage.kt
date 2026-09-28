package ai.vn97.runtime

import java.io.File
import java.io.FileInputStream
import java.nio.file.Files
import java.security.MessageDigest
import org.json.JSONArray
import org.json.JSONObject

data class Mamba2OrtGraphFile(
    val filename: String,
    val bytes: Long,
    val sha256: String,
    val file: File,
) {
    init {
        require(filename.isNotBlank())
        require(!filename.contains('/') && !filename.contains('\\'))
        require(bytes > 0L)
        requireMamba2Sha256(sha256, "G0.6 graph file SHA-256")
        require(file.isFile && !Files.isSymbolicLink(file.toPath()))
    }
}

data class Mamba2OrtRuntimePackage(
    val runtimeId: String,
    val g05ManifestId: String,
    val capsuleId: String,
    val capsuleManifestSha256: String,
    val sourceWeightSha256: String,
    val graphFilename: String,
    val graphFiles: Map<String, Mamba2OrtGraphFile>,
    val maxChunkSize: Int,
    val vocabSize: Int,
    val dModel: Int,
    val nLayers: Int,
    val dInner: Int,
    val nHeads: Int,
    val headDim: Int,
    val dState: Int,
    val dConv: Int,
    val convDim: Int,
    val stateDtype: String,
) {
    init {
        requireMamba2Sha256(runtimeId, "G0.6 runtime ID")
        requireMamba2Sha256(g05ManifestId, "G0.6 G0.5 manifest ID")
        requireMamba2Sha256(capsuleId, "G0.6 capsule ID")
        requireMamba2Sha256(
            capsuleManifestSha256,
            "G0.6 capsule manifest SHA-256",
        )
        requireMamba2Sha256(
            sourceWeightSha256,
            "G0.6 source weight SHA-256",
        )
        require(maxChunkSize in setOf(8, 16, 32))
        require(vocabSize == 50288)
        require(dModel == 2560)
        require(nLayers == 64)
        require(dInner == 5120)
        require(nHeads == 80)
        require(headDim == 64)
        require(dState == 128)
        require(dConv == 4)
        require(convDim == 5376)
        require(graphFilename == "recurrent-" + maxChunkSize + ".onnx")
        require(graphFiles.containsKey(graphFilename))
        require(stateDtype in setOf("float16", "float32")) {
            "G0.6 state dtype must be float16 or float32"
        }
    }

    val graphFile: File
        get() = graphFiles.getValue(graphFilename).file

    val convStateElements: Int
        get() = checkedMamba2Count(
            "G0.6 conv state",
            nLayers.toLong(),
            convDim.toLong(),
            dConv.toLong(),
        )

    val ssmStateElements: Int
        get() = checkedMamba2Count(
            "G0.6 SSD state",
            nLayers.toLong(),
            nHeads.toLong(),
            headDim.toLong(),
            dState.toLong(),
        )

    companion object {
        const val SCHEMA = "VN97M2G06RUNTIME1"
        const val FILENAME = "runtime.vn97m2g06.json"

        fun load(rootDir: File): Mamba2OrtRuntimePackage {
            require(
                rootDir.isDirectory &&
                    !Files.isSymbolicLink(rootDir.toPath())
            ) {
                "G0.6 runtime root must be a real directory"
            }
            val root = rootDir.canonicalFile
            val descriptor = File(root, FILENAME)
            requireMamba2RegularFile(
                descriptor,
                root,
                "G0.6 runtime descriptor",
            )
            val raw = descriptor.readText(Charsets.UTF_8)
            require(!raw.contains('\u0000'))
            val json = JSONObject(raw)
            require(json.getString("schema") == SCHEMA)

            val expectedFields = setOf(
                "schema",
                "g05_manifest_id",
                "capsule_id",
                "capsule_manifest_sha256",
                "source_weight_sha256",
                "graph_filename",
                "graph_files",
                "max_chunk_size",
                "valid_length_min",
                "valid_length_max",
                "inputs",
                "outputs",
                "batch_size",
                "vocab_size",
                "d_model",
                "n_layers",
                "d_inner",
                "n_heads",
                "head_dim",
                "d_state",
                "d_conv",
                "conv_dim",
                "conv_state_shape",
                "ssm_state_shape",
                "token_dtype",
                "valid_length_dtype",
                "state_dtype",
                "single_weight_graph",
                "decode_via_valid_length_one",
                "parallel_prefill_ready",
                "same_weights_semantics",
                "quantization_used",
                "source_runtime_required",
                "production_activation_authorized",
                "runtime_id",
            )
            require(json.keys().asSequence().toSet() == expectedFields) {
                "G0.6 runtime descriptor fields mismatch"
            }

            val runtimeId = json.getString("runtime_id")
            requireMamba2Sha256(runtimeId, "G0.6 runtime ID")
            val body = JSONObject(json.toString())
            body.remove("runtime_id")
            val prefix = "VN97M2G06RUNTIME1" +
                String(charArrayOf(0.toChar()))
            val expectedId = sha256Mamba2(
                prefix.toByteArray(Charsets.US_ASCII) +
                    canonicalJsonMamba2(body).toByteArray(Charsets.US_ASCII)
            )
            require(runtimeId == expectedId) {
                "G0.6 runtime descriptor identity mismatch"
            }

            require(json.getInt("batch_size") == 1)
            require(json.getInt("valid_length_min") == 1)
            val maxChunk = json.getInt("max_chunk_size")
            require(json.getInt("valid_length_max") == maxChunk)
            require(maxChunk in setOf(8, 16, 32))
            requireStringArrayMamba2(
                json.getJSONArray("inputs"),
                listOf(
                    "input_ids",
                    "valid_length",
                    "conv_state",
                    "ssm_state",
                ),
                "G0.6 input contract",
            )
            requireStringArrayMamba2(
                json.getJSONArray("outputs"),
                listOf(
                    "logits",
                    "next_conv_state",
                    "next_ssm_state",
                ),
                "G0.6 output contract",
            )
            require(json.getString("token_dtype") == "int64")
            require(json.getString("valid_length_dtype") == "int64")
            val stateDtype = json.getString("state_dtype")
            require(stateDtype in setOf("float16", "float32")) {
                "G0.6 state dtype must be float16 or float32"
            }
            require(json.getBoolean("single_weight_graph"))
            require(json.getBoolean("decode_via_valid_length_one"))
            require(json.getBoolean("parallel_prefill_ready"))
            require(json.getBoolean("same_weights_semantics"))
            require(!json.getBoolean("quantization_used"))
            require(!json.getBoolean("source_runtime_required"))
            require(!json.getBoolean("production_activation_authorized"))

            val nLayers = json.getInt("n_layers")
            val convDim = json.getInt("conv_dim")
            val dConv = json.getInt("d_conv")
            val nHeads = json.getInt("n_heads")
            val headDim = json.getInt("head_dim")
            val dState = json.getInt("d_state")
            requireIntArrayMamba2(
                json.getJSONArray("conv_state_shape"),
                listOf(nLayers, 1, convDim, dConv),
                "G0.6 conv state shape",
            )
            requireIntArrayMamba2(
                json.getJSONArray("ssm_state_shape"),
                listOf(nLayers, 1, nHeads, headDim, dState),
                "G0.6 SSD state shape",
            )

            val graphFiles = linkedMapOf<String, Mamba2OrtGraphFile>()
            val files = json.getJSONArray("graph_files")
            require(files.length() >= 1)
            for (index in 0 until files.length()) {
                val item = files.getJSONObject(index)
                require(
                    item.keys().asSequence().toSet() ==
                        setOf("filename", "bytes", "sha256")
                )
                val filename = item.getString("filename")
                require(!graphFiles.containsKey(filename))
                require(
                    filename.isNotBlank() &&
                        !filename.contains('/') &&
                        !filename.contains('\\')
                )
                val file = File(root, filename)
                requireMamba2RegularFile(
                    file,
                    root,
                    "G0.6 graph payload " + filename,
                )
                val bytes = item.getLong("bytes")
                require(file.length() == bytes) {
                    "G0.6 graph payload byte size mismatch: " + filename
                }
                val digest = item.getString("sha256")
                requireMamba2Sha256(digest, "G0.6 graph payload SHA-256")
                require(sha256FileMamba2(file) == digest) {
                    "G0.6 graph payload SHA-256 mismatch: " + filename
                }
                graphFiles[filename] = Mamba2OrtGraphFile(
                    filename = filename,
                    bytes = bytes,
                    sha256 = digest,
                    file = file,
                )
            }

            val graphFilename = json.getString("graph_filename")
            require(graphFiles.containsKey(graphFilename)) {
                "G0.6 primary graph is absent from graph inventory"
            }
            return Mamba2OrtRuntimePackage(
                runtimeId = runtimeId,
                g05ManifestId = json.getString("g05_manifest_id"),
                capsuleId = json.getString("capsule_id"),
                capsuleManifestSha256 =
                    json.getString("capsule_manifest_sha256"),
                sourceWeightSha256 =
                    json.getString("source_weight_sha256"),
                graphFilename = graphFilename,
                graphFiles = graphFiles.toMap(),
                maxChunkSize = maxChunk,
                vocabSize = json.getInt("vocab_size"),
                dModel = json.getInt("d_model"),
                nLayers = nLayers,
                dInner = json.getInt("d_inner"),
                nHeads = nHeads,
                headDim = headDim,
                dState = dState,
                dConv = dConv,
                convDim = convDim,
                stateDtype = stateDtype,
            )
        }
    }
}

private fun requireMamba2RegularFile(
    file: File,
    root: File,
    label: String,
) {
    require(file.isFile && !Files.isSymbolicLink(file.toPath())) {
        label + " must be a regular non-symlink file"
    }
    require(file.canonicalFile.toPath().startsWith(root.canonicalFile.toPath())) {
        label + " escapes runtime root"
    }
}

private fun requireStringArrayMamba2(
    array: JSONArray,
    expected: List<String>,
    label: String,
) {
    val actual = buildList {
        for (index in 0 until array.length()) {
            add(array.getString(index))
        }
    }
    require(actual == expected) { label + " mismatch" }
}

private fun requireIntArrayMamba2(
    array: JSONArray,
    expected: List<Int>,
    label: String,
) {
    val actual = buildList {
        for (index in 0 until array.length()) {
            add(array.getInt(index))
        }
    }
    require(actual == expected) { label + " mismatch" }
}

private fun checkedMamba2Count(
    label: String,
    vararg dimensions: Long,
): Int {
    var product = 1L
    for (dimension in dimensions) {
        require(dimension > 0L)
        require(product <= Int.MAX_VALUE.toLong() / dimension) {
            label + " exceeds JVM direct-buffer bounds"
        }
        product *= dimension
    }
    return product.toInt()
}

private fun requireMamba2Sha256(value: String, label: String) {
    require(
        value.length == 64 &&
            value.all { it in '0'..'9' || it in 'a'..'f' }
    ) {
        label + " must be lowercase SHA-256"
    }
}

private fun sha256FileMamba2(file: File): String {
    val digest = MessageDigest.getInstance("SHA-256")
    FileInputStream(file).buffered(1024 * 1024).use { input ->
        val buffer = ByteArray(1024 * 1024)
        while (true) {
            val count = input.read(buffer)
            if (count < 0) break
            require(count > 0)
            digest.update(buffer, 0, count)
        }
    }
    return digest.digest().joinToString("") {
        "%02x".format(it.toInt() and 0xff)
    }
}

private fun sha256Mamba2(bytes: ByteArray): String =
    MessageDigest.getInstance("SHA-256")
        .digest(bytes)
        .joinToString("") {
            "%02x".format(it.toInt() and 0xff)
        }

private fun canonicalJsonMamba2(value: Any?): String {
    return when (value) {
        JSONObject.NULL, null -> "null"
        is JSONObject -> {
            value.keys().asSequence().toList().sorted().joinToString(
                prefix = "{",
                postfix = "}",
                separator = ",",
            ) { key ->
                JSONObject.quote(key) + ":" +
                    canonicalJsonMamba2(value.get(key))
            }
        }
        is JSONArray -> {
            (0 until value.length()).joinToString(
                prefix = "[",
                postfix = "]",
                separator = ",",
            ) { index ->
                canonicalJsonMamba2(value.get(index))
            }
        }
        is String -> JSONObject.quote(value)
        is Boolean -> if (value) "true" else "false"
        is Int, is Long, is Short, is Byte -> value.toString()
        else -> error(
            "unsupported G0.6 canonical JSON type: " +
                value::class.java.name
        )
    }
}
