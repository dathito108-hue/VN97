package ai.vn97.runtime

import java.io.ByteArrayOutputStream
import java.io.File
import java.io.FileInputStream
import java.nio.file.Files
import java.text.Normalizer
import java.security.MessageDigest
import java.util.regex.Pattern
import org.json.JSONArray
import org.json.JSONObject

data class Mamba2TokenizerAsset(
    val filename: String,
    val bytes: Long,
    val sha256: String,
    val file: File,
) {
    init {
        require(filename.isNotBlank())
        require(!filename.contains('/') && !filename.contains('\\'))
        require(bytes > 0L)
        requireG08Sha256(sha256, "G0.8 tokenizer asset SHA-256")
        require(file.isFile && !Files.isSymbolicLink(file.toPath()))
    }
}

data class Mamba2TokenizerPackage(
    val tokenizerId: String,
    val capsuleId: String,
    val sourceTokenizerModelId: String,
    val sourceTokenizerRevision: String,
    val bosToken: String,
    val eosToken: String,
    val unkToken: String,
    val bosTokenId: Int,
    val eosTokenId: Int,
    val unkTokenId: Int,
    val tokenIdSpace: Int,
    val runtimeLogitsSize: Int,
    val invalidPaddedTokenStart: Int,
    val invalidPaddedTokenEndExclusive: Int,
    val mergeCount: Int,
    val assets: Map<String, Mamba2TokenizerAsset>,
) {
    init {
        requireG08Sha256(tokenizerId, "G0.8 tokenizer ID")
        requireG08Sha256(capsuleId, "G0.8 capsule ID")
        require(sourceTokenizerModelId == "EleutherAI/gpt-neox-20b")
        require(
            sourceTokenizerRevision ==
                "364ae95407723fadd1d47b023c1efb92a4d891c3"
        )
        require(tokenIdSpace == 50_277)
        require(runtimeLogitsSize == 50_288)
        require(invalidPaddedTokenStart == tokenIdSpace)
        require(invalidPaddedTokenEndExclusive == runtimeLogitsSize)
        require(bosTokenId in 0 until tokenIdSpace)
        require(eosTokenId in 0 until tokenIdSpace)
        require(unkTokenId in 0 until tokenIdSpace)
        require(mergeCount > 0)
        require(assets.keys == REQUIRED_ASSETS)
    }

    val vocabFile: File
        get() = assets.getValue("vocab.json").file

    val mergesFile: File
        get() = assets.getValue("merges.txt").file

    fun maskInvalidPaddedLogits(logits: FloatArray) {
        require(logits.size == runtimeLogitsSize) {
            "G0.8 logits width differs from Mamba-2 runtime"
        }
        for (index in invalidPaddedTokenStart until invalidPaddedTokenEndExclusive) {
            logits[index] = Float.NEGATIVE_INFINITY
        }
    }

    companion object {
        const val SCHEMA = "VN97M2G08TOK1"
        const val FILENAME = "tokenizer.vn97m2g08.json"
        val REQUIRED_ASSETS = setOf(
            "merges.txt",
            "special_tokens_map.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.json",
        )

        fun load(rootDir: File): Mamba2TokenizerPackage {
            require(
                rootDir.isDirectory &&
                    !Files.isSymbolicLink(rootDir.toPath())
            ) {
                "G0.8 tokenizer root must be real directory"
            }
            val root = rootDir.canonicalFile
            val descriptor = File(root, FILENAME)
            requireG08RegularFile(
                descriptor,
                root,
                "G0.8 tokenizer descriptor",
            )
            val json = JSONObject(
                descriptor.readText(Charsets.US_ASCII)
            )
            require(json.getString("schema") == SCHEMA)
            require(
                json.getString("tokenizer_class") ==
                    "GPTNeoXTokenizer"
            )
            require(
                json.getString("algorithm") ==
                    "gpt2_byte_level_bpe_ranked_merges"
            )
            require(
                json.getString("pretokenizer_pattern") ==
                    GPT_NEOX_PATTERN_G08
            )
            require(!json.getBoolean("add_prefix_space"))
            require(json.getBoolean("same_token_ids_required"))
            require(json.getBoolean("sampling_must_mask_padded_ids"))
            require(!json.getBoolean("source_runtime_required"))
            require(!json.getBoolean("production_activation_authorized"))

            val tokenizerId = json.getString("tokenizer_id")
            requireG08Sha256(tokenizerId, "G0.8 tokenizer ID")
            val body = JSONObject(json.toString())
            body.remove("tokenizer_id")
            val prefix = "VN97M2G08TOK1" +
                String(charArrayOf(0.toChar()))
            val expectedId = sha256G08(
                prefix.toByteArray(Charsets.US_ASCII) +
                    canonicalJsonG08(body)
                        .toByteArray(Charsets.US_ASCII)
            )
            require(tokenizerId == expectedId) {
                "G0.8 tokenizer descriptor identity mismatch"
            }

            val assetJson = json.getJSONObject("assets")
            require(
                assetJson.keys().asSequence().toSet() ==
                    REQUIRED_ASSETS
            ) {
                "G0.8 tokenizer asset inventory mismatch"
            }
            val assets = linkedMapOf<String, Mamba2TokenizerAsset>()
            for (filename in REQUIRED_ASSETS.sorted()) {
                val item = assetJson.getJSONObject(filename)
                require(
                    item.keys().asSequence().toSet() ==
                        setOf("bytes", "sha256")
                )
                val file = File(root, filename)
                requireG08RegularFile(
                    file,
                    root,
                    "G0.8 tokenizer asset " + filename,
                )
                val bytes = item.getLong("bytes")
                require(file.length() == bytes) {
                    "G0.8 tokenizer asset size mismatch: " + filename
                }
                val digest = item.getString("sha256")
                requireG08Sha256(
                    digest,
                    "G0.8 tokenizer asset SHA-256",
                )
                require(sha256FileG08(file) == digest) {
                    "G0.8 tokenizer asset hash mismatch: " + filename
                }
                assets[filename] = Mamba2TokenizerAsset(
                    filename = filename,
                    bytes = bytes,
                    sha256 = digest,
                    file = file,
                )
            }

            return Mamba2TokenizerPackage(
                tokenizerId = tokenizerId,
                capsuleId = json.getString("capsule_id"),
                sourceTokenizerModelId =
                    json.getString("source_tokenizer_model_id"),
                sourceTokenizerRevision =
                    json.getString("source_tokenizer_revision"),
                bosToken = json.getString("bos_token"),
                eosToken = json.getString("eos_token"),
                unkToken = json.getString("unk_token"),
                bosTokenId = json.getInt("bos_token_id"),
                eosTokenId = json.getInt("eos_token_id"),
                unkTokenId = json.getInt("unk_token_id"),
                tokenIdSpace = json.getInt("token_id_space"),
                runtimeLogitsSize = json.getInt("runtime_logits_size"),
                invalidPaddedTokenStart =
                    json.getInt("invalid_padded_token_start"),
                invalidPaddedTokenEndExclusive =
                    json.getInt(
                        "invalid_padded_token_end_exclusive"
                    ),
                mergeCount = json.getInt("merge_count"),
                assets = assets.toMap(),
            )
        }
    }
}

class VN97GptNeoXTokenizer(
    val packageInfo: Mamba2TokenizerPackage,
) {
    private val vocab: Map<String, Int>
    private val inverse: Array<String?>
    private val mergeRanks: Map<Pair<String, String>, Int>
    private val byteEncoder: Array<String>
    private val byteDecoder: Map<Char, Int>
    private val addedTokens: Map<String, Int>
    private val addedIds: Set<Int>
    private val addedPattern: Pattern
    private val cache = LinkedHashMap<String, List<String>>()
    private val pattern = Pattern.compile(
        GPT_NEOX_PATTERN_G08,
        Pattern.UNICODE_CHARACTER_CLASS,
    )

    init {
        val parsed = JSONObject(
            packageInfo.vocabFile.readText(Charsets.UTF_8)
        )
        val mutable = linkedMapOf<String, Int>()
        val ids = mutableSetOf<Int>()
        var maximum = -1
        for (token in parsed.keys().asSequence().toList().sorted()) {
            val id = parsed.getInt(token)
            require(id >= 0 && ids.add(id)) {
                "G0.8 vocabulary ID invalid or duplicate"
            }
            mutable[token] = id
            maximum = maxOf(maximum, id)
        }
        require(maximum + 1 == packageInfo.tokenIdSpace) {
            "G0.8 vocabulary token ID space mismatch"
        }
        require(ids == (0..maximum).toSet()) {
            "G0.8 vocabulary IDs must be contiguous"
        }
        vocab = mutable.toMap()
        inverse = arrayOfNulls(packageInfo.tokenIdSpace)
        for ((token, id) in vocab) {
            inverse[id] = token
        }
        require(vocab[packageInfo.bosToken] == packageInfo.bosTokenId)
        require(vocab[packageInfo.eosToken] == packageInfo.eosTokenId)
        require(vocab[packageInfo.unkToken] == packageInfo.unkTokenId)

        val metadata = JSONObject(packageInfo.assets.getValue("tokenizer.json").file.readText(Charsets.UTF_8))
        require(metadata.getJSONObject("normalizer").toString() == JSONObject().put("type", "NFC").toString()) {
            "G0.8 requires the pinned NFC normalizer"
        }
        val added = linkedMapOf<String, Int>()
        val entries = metadata.getJSONArray("added_tokens")
        for (i in 0 until entries.length()) {
            val entry = entries.getJSONObject(i)
            require(listOf("single_word", "lstrip", "rstrip").all { !entry.getBoolean(it) })
            val token = entry.getString("content")
            val id = entry.getInt("id")
            require(token.isNotEmpty() && vocab[token] == id && !added.containsKey(token))
            added[token] = id
        }
        require(added[packageInfo.eosToken] == packageInfo.eosTokenId)
        addedTokens = added.toMap()
        addedIds = added.values.toSet()
        addedPattern = Pattern.compile(added.keys.sortedByDescending { it.length }.joinToString("|") { Pattern.quote(it) })

        val ranks = linkedMapOf<Pair<String, String>, Int>()
        var rank = 0
        packageInfo.mergesFile.forEachLine(Charsets.UTF_8) { raw ->
            val line = raw.trim()
            if (line.isEmpty() || line.startsWith("#version:")) {
                return@forEachLine
            }
            val parts = line.split(Regex("\\s+"))
            require(parts.size == 2) {
                "G0.8 merges line must contain two symbols"
            }
            val pair = parts[0] to parts[1]
            require(!ranks.containsKey(pair)) {
                "G0.8 duplicate BPE merge"
            }
            ranks[pair] = rank++
        }
        require(rank == packageInfo.mergeCount) {
            "G0.8 merge count differs from descriptor"
        }
        mergeRanks = ranks.toMap()

        val mapping = buildByteUnicodeG08()
        byteEncoder = mapping.first
        byteDecoder = mapping.second
    }

    fun encode(text: String): IntArray {
        if (text.isEmpty()) return IntArray(0)
        val output = ArrayList<Int>()
        val normalized = Normalizer.normalize(text, Normalizer.Form.NFC)
        val matcher = addedPattern.matcher(normalized)
        var cursor = 0
        while (matcher.find()) {
            if (matcher.start() > cursor) encodeOrdinary(normalized.substring(cursor, matcher.start()), output)
            output += addedTokens.getValue(matcher.group())
            cursor = matcher.end()
        }
        if (cursor < normalized.length) encodeOrdinary(normalized.substring(cursor), output)
        return output.toIntArray()
    }

    private fun encodeOrdinary(
        text: String,
        output: MutableList<Int>,
    ) {
        val matcher = pattern.matcher(text)
        var cursor = 0
        while (matcher.find()) {
            require(matcher.start() == cursor) {
                "G0.8 pretokenizer left unmatched text"
            }
            val piece = matcher.group()
            val encoded = StringBuilder()
            for (byte in piece.toByteArray(Charsets.UTF_8)) {
                encoded.append(
                    byteEncoder[byte.toInt() and 0xff]
                )
            }
            for (token in bpe(encoded.toString())) {
                output += requireNotNull(vocab[token]) {
                    "G0.8 BPE produced token outside vocabulary"
                }
            }
            cursor = matcher.end()
        }
        require(cursor == text.length) {
            "G0.8 pretokenizer did not consume full text"
        }
    }

    fun decodeBytes(
        tokenIds: IntArray,
        skipEos: Boolean = false,
    ): ByteArray {
        val output = ByteArrayOutputStream()
        for (tokenId in tokenIds) {
            require(tokenId in 0 until packageInfo.tokenIdSpace) {
                "G0.8 token ID outside tokenizer vocabulary"
            }
            val token = requireNotNull(inverse[tokenId])
            if (tokenId in addedIds) {
                if (tokenId != packageInfo.eosTokenId || !skipEos) {
                    output.write(token.toByteArray(Charsets.UTF_8))
                }
                continue
            }
            for (character in token) {
                val value = byteDecoder[character]
                require(value != null) {
                    "G0.8 vocabulary token outside byte decoder"
                }
                output.write(value)
            }
        }
        return output.toByteArray()
    }

    fun decode(
        tokenIds: IntArray,
        skipEos: Boolean = false,
    ): String = decodeBytes(
        tokenIds,
        skipEos=skipEos,
    ).toString(Charsets.UTF_8)

    fun maskInvalidPaddedLogits(logits: FloatArray) {
        packageInfo.maskInvalidPaddedLogits(logits)
    }

    private fun bpe(source: String): List<String> {
        cache[source]?.let { return it }
        var word = source.map { it.toString() }
        if (word.size <= 1) {
            cache[source] = word
            return word
        }

        while (word.size > 1) {
            var selected: Pair<String, String>? = null
            var selectedRank = Int.MAX_VALUE
            for (index in 0 until word.size - 1) {
                val pair = word[index] to word[index + 1]
                val rank = mergeRanks[pair] ?: continue
                if (rank < selectedRank) {
                    selected = pair
                    selectedRank = rank
                }
            }
            val pair = selected ?: break
            val merged = ArrayList<String>()
            var index = 0
            while (index < word.size) {
                if (
                    index + 1 < word.size &&
                    word[index] == pair.first &&
                    word[index + 1] == pair.second
                ) {
                    merged += pair.first + pair.second
                    index += 2
                } else {
                    merged += word[index]
                    index += 1
                }
            }
            word = merged
        }
        val result = word.toList()
        if (cache.size >= 16_384) {
            val iterator = cache.entries.iterator()
            repeat(4_096) {
                if (iterator.hasNext()) {
                    iterator.next()
                    iterator.remove()
                }
            }
        }
        cache[source] = result
        return result
    }
}

private const val GPT_NEOX_PATTERN_G08 =
    "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| " +
        "?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+"

private fun buildByteUnicodeG08(): Pair<Array<String>, Map<Char, Int>> {
    val base = mutableListOf<Int>()
    for (value in '!'.code..'~'.code) base += value
    for (value in '¡'.code..'¬'.code) base += value
    for (value in '®'.code..'ÿ'.code) base += value
    val codepoints = base.toMutableList()
    var extra = 0
    for (value in 0..255) {
        if (value !in base) {
            base += value
            codepoints += 256 + extra
            extra += 1
        }
    }
    val encoder = Array(256) { "" }
    val decoder = linkedMapOf<Char, Int>()
    for (index in base.indices) {
        val character = codepoints[index].toChar()
        encoder[base[index]] = character.toString()
        require(decoder.put(character, base[index]) == null)
    }
    require(encoder.all { it.isNotEmpty() })
    require(decoder.size == 256)
    return encoder to decoder.toMap()
}

private fun requireG08RegularFile(
    file: File,
    root: File,
    label: String,
) {
    require(file.isFile && !Files.isSymbolicLink(file.toPath())) {
        label + " must be regular non-symlink file"
    }
    require(file.canonicalFile.toPath().startsWith(root.canonicalFile.toPath())) {
        label + " escapes tokenizer root"
    }
}

private fun sha256FileG08(file: File): String {
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

private fun requireG08Sha256(value: String, label: String) {
    require(
        value.length == 64 &&
            value.all { it in '0'..'9' || it in 'a'..'f' }
    ) {
        label + " must be lowercase SHA-256"
    }
}

private fun sha256G08(bytes: ByteArray): String =
    MessageDigest.getInstance("SHA-256")
        .digest(bytes)
        .joinToString("") {
            "%02x".format(it.toInt() and 0xff)
        }

private fun canonicalJsonG08(value: Any?): String {
    return when (value) {
        JSONObject.NULL, null -> "null"
        is JSONObject -> {
            value.keys().asSequence().toList().sorted().joinToString(
                prefix = "{",
                postfix = "}",
                separator = ",",
            ) { key ->
                JSONObject.quote(key) + ":" +
                    canonicalJsonG08(value.get(key))
            }
        }
        is JSONArray -> {
            (0 until value.length()).joinToString(
                prefix = "[",
                postfix = "]",
                separator = ",",
            ) { index ->
                canonicalJsonG08(value.get(index))
            }
        }
        is String -> JSONObject.quote(value)
        is Boolean -> if (value) "true" else "false"
        is Int, is Long, is Short, is Byte -> value.toString()
        else -> error(
            "unsupported G0.8 canonical JSON type: " +
                value::class.java.name
        )
    }
}
