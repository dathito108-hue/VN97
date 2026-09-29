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
    checkG08PortableRegex()
    println("Platform quote of slash: ${JSONObject.quote("a/b")}")
}


private fun checkG08PortableRegex() {
    val portable = compileG08Pretokenizer()
    check(portable.flags() == 0) { "Android tokenizer must not request unsupported flags" }
    val reference = java.util.regex.Pattern.compile(
        "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+",
        java.util.regex.Pattern.UNICODE_CHARACTER_CLASS,
    )
    fun pieces(pattern: java.util.regex.Pattern, text: String): List<String> {
        val out = mutableListOf<String>()
        val matcher = pattern.matcher(text)
        while (matcher.find()) out.add(matcher.group())
        check(out.joinToString("") == text)
        return out
    }
    val points = (0x09..0x0d).toList() + listOf(0x20, 0x85, 0xa0, 0x1680) +
        (0x2000..0x200a).toList() + listOf(0x2028, 0x2029, 0x202f, 0x205f, 0x3000, 0x200b, 0xfeff)
    val cases = points.flatMap { point ->
        val space = point.toChar().toString()
        listOf(space, "a${space}b", "Việt${space}${space}Nam 123", "😀${space}!", "end${space}${space}")
    } + listOf("Xin chào Việt Nam!", "I'm here, they're fine.", "中文 日本語 العربية", "e\u0301", "𐐀 𝟘")
    for (text in cases) check(pieces(portable, text) == pieces(reference, text)) { "Regex split mismatch: $text" }
    println("G08 flag-free pretokenizer: ${cases.size} Unicode/whitespace cases PASS")
}
