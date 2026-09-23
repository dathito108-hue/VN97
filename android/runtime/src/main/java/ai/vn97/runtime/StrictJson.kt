package ai.vn97.runtime

import java.math.BigInteger

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
) {
    init {
        require(maxInputUtf8Bytes > 0) { "maxInputUtf8Bytes must be positive" }
        require(maxDepth > 0) { "maxDepth must be positive" }
        require(maxNodes > 0) { "maxNodes must be positive" }
        require(maxStringUtf8Bytes > 0) { "maxStringUtf8Bytes must be positive" }
    }
}

internal object VnStrictJson {
    fun parseObject(
        text: String,
        limits: VnJsonLimits = VnJsonLimits(),
    ): VnJsonObject {
        if (text.toByteArray(Charsets.UTF_8).size > limits.maxInputUtf8Bytes) {
            throw NativeCognitionContractException("JSON exceeds UTF-8 byte budget")
        }
        val value = Parser(text, limits).parseDocument()
        return value as? VnJsonObject
            ?: throw NativeCognitionContractException("JSON root must be an object")
    }

    fun canonical(value: VnJsonValue): String = buildString {
        appendCanonical(value)
    }

    fun objectOf(vararg fields: Pair<String, VnJsonValue>): VnJsonObject =
        VnJsonObject(linkedMapOf(*fields))

    fun string(value: String): VnJsonString = VnJsonString(value)
    fun bool(value: Boolean): VnJsonBoolean = VnJsonBoolean(value)
    fun long(value: Long): VnJsonNumber = VnJsonNumber(value.toString())
    fun int(value: Int): VnJsonNumber = VnJsonNumber(value.toString())
    fun double(value: Double): VnJsonNumber {
        if (!value.isFinite()) {
            throw NativeCognitionContractException("JSON number must be finite")
        }
        return VnJsonNumber(canonicalDouble(value))
    }
    fun array(values: Iterable<VnJsonValue>): VnJsonArray = VnJsonArray(values.toList())

    private fun StringBuilder.appendCanonical(value: VnJsonValue) {
        when (value) {
            VnJsonNull -> append("null")
            is VnJsonBoolean -> append(if (value.value) "true" else "false")
            is VnJsonString -> appendEscapedString(value.value)
            is VnJsonNumber -> append(value.canonical)
            is VnJsonArray -> {
                append('[')
                value.values.forEachIndexed { index, item ->
                    if (index != 0) append(',')
                    appendCanonical(item)
                }
                append(']')
            }
            is VnJsonObject -> {
                append('{')
                value.values.keys.sorted().forEachIndexed { index, key ->
                    if (index != 0) append(',')
                    appendEscapedString(key)
                    append(':')
                    appendCanonical(checkNotNull(value.values[key]))
                }
                append('}')
            }
        }
    }

    private fun StringBuilder.appendEscapedString(value: String) {
        append('"')
        var index = 0
        while (index < value.length) {
            val ch = value[index]
            when (ch) {
                '"' -> append("\\\"")
                '\\' -> append("\\\\")
                '\b' -> append("\\b")
                '\u000C' -> append("\\f")
                '\n' -> append("\\n")
                '\r' -> append("\\r")
                '\t' -> append("\\t")
                else -> {
                    when {
                        ch.code < 0x20 -> appendUnicodeEscape(ch.code)
                        Character.isHighSurrogate(ch) -> {
                            if (index + 1 >= value.length ||
                                !Character.isLowSurrogate(value[index + 1])) {
                                throw NativeCognitionContractException("JSON string contains unpaired surrogate")
                            }
                            append(ch)
                            append(value[index + 1])
                            index += 1
                        }
                        Character.isLowSurrogate(ch) ->
                            throw NativeCognitionContractException("JSON string contains unpaired surrogate")
                        else -> append(ch)
                    }
                }
            }
            index += 1
        }
        append('"')
    }

    private fun StringBuilder.appendUnicodeEscape(code: Int) {
        val hex = "0123456789abcdef"
        append("\\u")
        append(hex[(code ushr 12) and 0xf])
        append(hex[(code ushr 8) and 0xf])
        append(hex[(code ushr 4) and 0xf])
        append(hex[code and 0xf])
    }

    private fun canonicalDouble(value: Double): String {
        val raw = java.lang.Double.toString(value)
        val e = raw.indexOf('E')
        if (e < 0) return raw
        val mantissa = raw.substring(0, e)
        val exponent = raw.substring(e + 1).toInt()
        val sign = if (exponent >= 0) "+" else "-"
        val digits = kotlin.math.abs(exponent).toString().padStart(2, '0')
        return "$mantissa" + "e" + sign + digits
    }

    private class Parser(
        private val text: String,
        private val limits: VnJsonLimits,
    ) {
        private var index = 0
        private var nodes = 0

        fun parseDocument(): VnJsonValue {
            skipWhitespace()
            val value = parseValue(0)
            skipWhitespace()
            if (index != text.length) fail("trailing content after JSON value")
            return value
        }

        private fun parseValue(depth: Int): VnJsonValue {
            if (depth > limits.maxDepth) fail("JSON nesting exceeds maxDepth")
            nodes += 1
            if (nodes > limits.maxNodes) fail("JSON node count exceeds maxNodes")
            if (index >= text.length) fail("unexpected end of JSON")
            return when (text[index]) {
                '{' -> parseObject(depth)
                '[' -> parseArray(depth)
                '"' -> VnJsonString(parseString())
                't' -> { consumeLiteral("true"); VnJsonBoolean(true) }
                'f' -> { consumeLiteral("false"); VnJsonBoolean(false) }
                'n' -> { consumeLiteral("null"); VnJsonNull }
                '-', in '0'..'9' -> parseNumber()
                else -> fail("invalid JSON value")
            }
        }

        private fun parseObject(depth: Int): VnJsonObject {
            expect('{')
            skipWhitespace()
            val values = LinkedHashMap<String, VnJsonValue>()
            if (peek('}')) {
                index += 1
                return VnJsonObject(values)
            }
            while (true) {
                if (!peek('"')) fail("JSON object key must be a string")
                val key = parseString()
                if (values.containsKey(key)) fail("JSON object contains duplicate key: $key")
                skipWhitespace()
                expect(':')
                skipWhitespace()
                values[key] = parseValue(depth + 1)
                skipWhitespace()
                when {
                    peek(',') -> { index += 1; skipWhitespace() }
                    peek('}') -> { index += 1; return VnJsonObject(values) }
                    else -> fail("JSON object requires ',' or '}'")
                }
            }
        }

        private fun parseArray(depth: Int): VnJsonArray {
            expect('[')
            skipWhitespace()
            val values = ArrayList<VnJsonValue>()
            if (peek(']')) {
                index += 1
                return VnJsonArray(values)
            }
            while (true) {
                values += parseValue(depth + 1)
                skipWhitespace()
                when {
                    peek(',') -> { index += 1; skipWhitespace() }
                    peek(']') -> { index += 1; return VnJsonArray(values) }
                    else -> fail("JSON array requires ',' or ']' ")
                }
            }
        }

        private fun parseString(): String {
            expect('"')
            val out = StringBuilder()
            while (index < text.length) {
                val ch = text[index++]
                when {
                    ch == '"' -> {
                        val result = out.toString()
                        if (result.toByteArray(Charsets.UTF_8).size > limits.maxStringUtf8Bytes) {
                            fail("JSON string exceeds UTF-8 byte budget")
                        }
                        return result
                    }
                    ch == '\\' -> parseEscape(out)
                    ch.code < 0x20 -> fail("unescaped control character in JSON string")
                    Character.isHighSurrogate(ch) -> {
                        if (index >= text.length || !Character.isLowSurrogate(text[index])) {
                            fail("JSON string contains unpaired surrogate")
                        }
                        out.append(ch)
                        out.append(text[index++])
                    }
                    Character.isLowSurrogate(ch) -> fail("JSON string contains unpaired surrogate")
                    else -> out.append(ch)
                }
            }
            fail("unterminated JSON string")
        }

        private fun parseEscape(out: StringBuilder) {
            if (index >= text.length) fail("unterminated JSON escape")
            when (val escaped = text[index++]) {
                '"', '\\', '/' -> out.append(escaped)
                'b' -> out.append('\b')
                'f' -> out.append('\u000C')
                'n' -> out.append('\n')
                'r' -> out.append('\r')
                't' -> out.append('\t')
                'u' -> {
                    val first = parseHex4()
                    when {
                        first in 0xD800..0xDBFF -> {
                            if (index + 2 > text.length || text[index] != '\\' || text[index + 1] != 'u') {
                                fail("high surrogate must be followed by low surrogate escape")
                            }
                            index += 2
                            val second = parseHex4()
                            if (second !in 0xDC00..0xDFFF) fail("invalid low surrogate escape")
                            out.appendCodePoint(Character.toCodePoint(first.toChar(), second.toChar()))
                        }
                        first in 0xDC00..0xDFFF -> fail("unpaired low surrogate escape")
                        else -> out.append(first.toChar())
                    }
                }
                else -> fail("invalid JSON escape")
            }
        }

        private fun parseHex4(): Int {
            if (index + 4 > text.length) fail("short unicode escape")
            var value = 0
            repeat(4) {
                val digit = Character.digit(text[index++], 16)
                if (digit < 0) fail("invalid unicode escape")
                value = (value shl 4) or digit
            }
            return value
        }

        private fun parseNumber(): VnJsonNumber {
            val start = index
            if (peek('-')) index += 1
            if (index >= text.length) fail("incomplete JSON number")
            if (peek('0')) {
                index += 1
                if (index < text.length && text[index] in '0'..'9') {
                    fail("JSON number has leading zero")
                }
            } else {
                if (text[index] !in '1'..'9') fail("invalid JSON number")
                while (index < text.length && text[index] in '0'..'9') index += 1
            }
            var floating = false
            if (peek('.')) {
                floating = true
                index += 1
                val fractionStart = index
                while (index < text.length && text[index] in '0'..'9') index += 1
                if (index == fractionStart) fail("JSON fraction requires digits")
            }
            if (index < text.length && (text[index] == 'e' || text[index] == 'E')) {
                floating = true
                index += 1
                if (index < text.length && (text[index] == '+' || text[index] == '-')) index += 1
                val exponentStart = index
                while (index < text.length && text[index] in '0'..'9') index += 1
                if (index == exponentStart) fail("JSON exponent requires digits")
            }
            val raw = text.substring(start, index)
            val canonical = if (!floating) {
                try {
                    BigInteger(raw).toString()
                } catch (exc: NumberFormatException) {
                    throw NativeCognitionContractException("invalid integer JSON number", exc)
                }
            } else {
                val value = raw.toDoubleOrNull()
                    ?: fail("invalid floating JSON number")
                if (!value.isFinite()) fail("JSON number must be finite")
                canonicalDouble(value)
            }
            return VnJsonNumber(canonical)
        }

        private fun consumeLiteral(value: String) {
            if (!text.startsWith(value, index)) fail("invalid JSON literal")
            index += value.length
        }

        private fun skipWhitespace() {
            while (index < text.length && when (text[index]) {
                    ' ', '\t', '\n', '\r' -> true
                    else -> false
                }) {
                index += 1
            }
        }

        private fun expect(ch: Char) {
            if (!peek(ch)) fail("expected '$ch'")
            index += 1
        }

        private fun peek(ch: Char): Boolean = index < text.length && text[index] == ch

        private fun fail(message: String): Nothing =
            throw NativeCognitionContractException("strict JSON error at $index: $message")
    }
}
