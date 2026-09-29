package ai.vn97.runtime

import java.io.File
import org.json.JSONArray
import org.json.JSONObject

/** Executes the actual Android tokenizer source on the JVM, not a replacement implementation. */
fun main(args: Array<String>) {
    require(args.size == 3)
    val packageInfo = Mamba2TokenizerPackage.load(File(args[0]))
    val tokenizer = VN97GptNeoXTokenizer(packageInfo)
    val reference = JSONObject(File(args[1]).readText())
    require(reference.getString("schema") == "VN97G08REFERENCECASES1")
    require(reference.getString("tokenizer_id") == packageInfo.tokenizerId)
    val cases = reference.getJSONArray("cases")
    require(cases.length() >= 100)
    val mismatches = JSONArray()
    for (i in 0 until cases.length()) {
        val case = cases.getJSONObject(i)
        val ids = case.getJSONArray("ids")
        val expected = IntArray(ids.length()) { ids.getInt(it) }
        try {
            val actual = tokenizer.encode(case.getString("text"))
            val decoded = tokenizer.decodeBytes(expected, skipEos = false)
                .joinToString("") { "%02x".format(it.toInt() and 0xff) }
            if (!actual.contentEquals(expected) || decoded != case.getString("decode_hex")) {
                mismatches.put(JSONObject().put("case", i)
                    .put("token_ids_exact", actual.contentEquals(expected))
                    .put("decode_bytes_exact", decoded == case.getString("decode_hex")))
            }
        } catch (error: Exception) {
            mismatches.put(JSONObject().put("case", i).put("error", error.toString()))
        }
    }
    val logits = FloatArray(packageInfo.runtimeLogitsSize)
    tokenizer.maskInvalidPaddedLogits(logits)
    check((0 until packageInfo.tokenIdSpace).all { logits[it] == 0.0f })
    check((packageInfo.tokenIdSpace until logits.size).all { logits[it] == Float.NEGATIVE_INFINITY })
    val report = JSONObject().put("schema", "VN97G08KOTLINREFERENCEAUDIT1")
        .put("tokenizer_id", packageInfo.tokenizerId)
        .put("corpus_sha256", reference.getString("corpus_sha256"))
        .put("cases", cases.length()).put("mismatches", mismatches)
        .put("passed", mismatches.length() == 0).put("padded_logits_mask_passed", true)
        .put("execution_environment", "host JVM with Android tokenizer source")
        .put("device_measured", false).put("production_activation_authorized", false)
    File(args[2]).apply { parentFile.mkdirs(); writeText(report.toString(2) + "\n") }
    println("G08 Android tokenizer source: cases=${cases.length()} mismatches=${mismatches.length()}")
    check(mismatches.length() == 0) { "G08 Kotlin/reference parity failed; see audit report" }
}
