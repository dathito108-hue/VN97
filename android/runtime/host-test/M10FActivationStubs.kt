package ai.vn97.runtime

import java.io.File

internal sealed interface VnJsonValue
internal data object VnJsonNull : VnJsonValue
internal data class VnJsonBoolean(val value: Boolean) : VnJsonValue
internal data class VnJsonString(val value: String) : VnJsonValue
internal data class VnJsonNumber(val canonical: String) : VnJsonValue
internal data class VnJsonArray(val values: List<VnJsonValue>) : VnJsonValue
internal data class VnJsonObject(val values: Map<String, VnJsonValue>) : VnJsonValue
internal data class VnJsonLimits(
    val maxInputUtf8Bytes: Int = 128 * 1024,
    val maxDepth: Int = 64,
    val maxNodes: Int = 16 * 1024,
    val maxStringUtf8Bytes: Int = 128 * 1024,
)

internal object VnStrictJson {
    fun canonical(value: VnJsonValue): String = buildString { appendValue(value) }
    fun objectOf(vararg fields: Pair<String, VnJsonValue>): VnJsonObject =
        VnJsonObject(linkedMapOf(*fields))
    fun string(value: String) = VnJsonString(value)
    fun bool(value: Boolean) = VnJsonBoolean(value)
    fun array(values: Iterable<VnJsonValue>) = VnJsonArray(values.toList())

    fun parseObject(text: String, limits: VnJsonLimits = VnJsonLimits()): VnJsonObject {
        @Suppress("UNUSED_VARIABLE") val ignored = limits
        val parser = Parser(text)
        val value = parser.value()
        parser.ws()
        check(parser.end()) { "trailing JSON" }
        return value as? VnJsonObject ?: error("root not object")
    }

    private fun StringBuilder.appendValue(value: VnJsonValue) {
        when (value) {
            VnJsonNull -> append("null")
            is VnJsonBoolean -> append(if (value.value) "true" else "false")
            is VnJsonString -> appendString(value.value)
            is VnJsonNumber -> append(value.canonical)
            is VnJsonArray -> {
                append('[')
                value.values.forEachIndexed { i, v ->
                    if (i > 0) append(',')
                    appendValue(v)
                }
                append(']')
            }
            is VnJsonObject -> {
                append('{')
                value.values.keys.sorted().forEachIndexed { i, k ->
                    if (i > 0) append(',')
                    appendString(k)
                    append(':')
                    appendValue(checkNotNull(value.values[k]))
                }
                append('}')
            }
        }
    }

    private fun StringBuilder.appendString(value: String) {
        append('"')
        for (ch in value) when (ch) {
            '"' -> append("\"")
            '\\' -> append("\\")
            '\b' -> append("\b")
            '\u000c' -> append("\f")
            '\n' -> append("\n")
            '\r' -> append("\r")
            '\t' -> append("\t")
            else -> if (ch.code < 0x20) {
                append("\u%04x".format(ch.code))
            } else {
                append(ch)
            }
        }
        append('"')
    }

    private class Parser(private val s: String) {
        var i = 0
        fun end() = i == s.length
        fun ws() { while (i < s.length && s[i].isWhitespace()) i++ }
        fun value(): VnJsonValue {
            ws()
            check(i < s.length)
            return when (s[i]) {
                '{' -> obj()
                '[' -> arr()
                '"' -> VnJsonString(str())
                'n' -> { lit("null"); VnJsonNull }
                't' -> { lit("true"); VnJsonBoolean(true) }
                'f' -> { lit("false"); VnJsonBoolean(false) }
                else -> num()
            }
        }
        fun obj(): VnJsonObject {
            i++
            ws()
            val m = linkedMapOf<String, VnJsonValue>()
            if (s[i] == '}') {
                i++
                return VnJsonObject(m)
            }
            while (true) {
                val k = str()
                ws()
                check(s[i++] == ':')
                val v = value()
                check(!m.containsKey(k))
                m[k] = v
                ws()
                if (s[i] == '}') {
                    i++
                    return VnJsonObject(m)
                }
                check(s[i++] == ',')
                ws()
            }
        }
        fun arr(): VnJsonArray {
            i++
            ws()
            val values = mutableListOf<VnJsonValue>()
            if (s[i] == ']') {
                i++
                return VnJsonArray(values)
            }
            while (true) {
                values += value()
                ws()
                if (s[i] == ']') {
                    i++
                    return VnJsonArray(values)
                }
                check(s[i++] == ',')
            }
        }
        fun str(): String {
            check(s[i++] == '"')
            val out = StringBuilder()
            while (true) {
                val ch = s[i++]
                if (ch == '"') return out.toString()
                if (ch != '\\') {
                    out.append(ch)
                    continue
                }
                when (val e = s[i++]) {
                    '"', '\\', '/' -> out.append(e)
                    'b' -> out.append('\b')
                    'f' -> out.append('\u000c')
                    'n' -> out.append('\n')
                    'r' -> out.append('\r')
                    't' -> out.append('\t')
                    'u' -> {
                        val hex = s.substring(i, i + 4)
                        i += 4
                        out.append(hex.toInt(16).toChar())
                    }
                    else -> error("escape")
                }
            }
        }
        fun lit(value: String) {
            check(s.startsWith(value, i))
            i += value.length
        }
        fun num(): VnJsonNumber {
            val start = i
            if (s[i] == '-') i++
            while (i < s.length && s[i].isDigit()) i++
            return VnJsonNumber(s.substring(start, i))
        }
    }
}

data class VN97CapabilitySource(
    val origin: String,
    val sourceSha256: String,
    val license: String,
)
data class VN97CapabilitySection(
    val index: Int,
    val role: String,
    val format: String,
    val size: Long,
    val sha256: String,
    val packageOffset: Long,
)
data class VN97CapabilityManifest(
    val capabilityId: String,
    val capabilityVersion: Long,
    val kind: String,
    val source: VN97CapabilitySource,
    val sections: List<VN97CapabilitySection>,
)
data class VN97ParsedCapabilityPackage(
    val packageSha256: String,
    val manifest: VN97CapabilityManifest,
    val size: Long,
)
data class VN97StagedCapability(val marker: String = "stage")
data class VN97VerifiedCapability(
    val staged: VN97StagedCapability,
    val parsed: VN97ParsedCapabilityPackage,
    val publisherKeyId: String,
    val signatureSha256: String,
)
class VN97CapabilityTrustStore
object VN97CapabilityTrustVerifier {
    var next: VN97VerifiedCapability? = null

    @Suppress("UNUSED_PARAMETER")
    fun verify(
        staged: VN97StagedCapability,
        stageRoot: File,
        trustStore: VN97CapabilityTrustStore,
    ): VN97VerifiedCapability = checkNotNull(next)
}
