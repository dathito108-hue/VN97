package ai.vn97.runtime

import java.io.File
import java.security.MessageDigest
import org.json.JSONArray
import org.json.JSONObject

fun main(args: Array<String>) {
    val descriptor = JSONObject(File(args[0]).readText())
    val expected = descriptor.getString("tokenizer_id")
    descriptor.remove("tokenizer_id")
    fun identity(body: JSONObject): String = MessageDigest.getInstance("SHA-256")
        .digest(("VN97M2G08TOK1" + 0.toChar() + canonicalJsonG08(body)).toByteArray(Charsets.US_ASCII))
        .joinToString("") { "%02x".format(it.toInt() and 255) }
    check(identity(descriptor) == expected) { "Pinned G08 identity mismatch: ${identity(descriptor)}" }
    check(expected == "27ce0a2f005befaa98ec4ee05d5d83a71aac1bcc5e4dd00032e33a17615ca575")
    descriptor.put("source_tokenizer_model_id", "modified/source")
    check(identity(descriptor) != expected) { "Tampering must not pass" }
    val vectors = JSONArray(File(args[1]).readText())
    for (i in 0 until vectors.length()) {
        val entry = vectors.getJSONObject(i)
        check(canonicalJsonG08(entry.get("value")) == entry.getString("canonical")) { "Canonical vector $i failed" }
    }
    println("G08 pinned descriptor + ${vectors.length()} Python canonical vectors + tamper rejection PASS")
    println("Platform quote of slash: ${JSONObject.quote("a/b")}")
}
