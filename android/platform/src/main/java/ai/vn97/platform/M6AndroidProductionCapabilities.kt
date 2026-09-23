package ai.vn97.platform

import java.nio.charset.StandardCharsets

internal interface M6AndroidActionPort {
    fun launchPackage(packageName: String): String
    fun writeClipboard(text: String): String
}

class M6AndroidProductionCapabilities internal constructor(
    private val actions: M6AndroidActionPort,
) {
    val descriptors: List<M6CapabilityDescriptor> = listOf(
        M6CapabilityDescriptor(
            capabilityId = APP_LAUNCH_CAPABILITY,
            requiredScopeKeys = setOf(APP_PACKAGE_SCOPE),
            approvalRequired = true,
            maxPayloadUtf8Bytes = 2,
            maxLeaseNs = 30_000_000_000L,
            maxLeaseUses = 1,
        ),
        M6CapabilityDescriptor(
            capabilityId = CLIPBOARD_WRITE_CAPABILITY,
            requiredScopeKeys = setOf(CLIPBOARD_CHANNEL_SCOPE),
            approvalRequired = true,
            maxPayloadUtf8Bytes = MAX_CLIPBOARD_PAYLOAD_UTF8_BYTES,
            maxLeaseNs = 30_000_000_000L,
            maxLeaseUses = 1,
        ),
    )

    val intentBinder: M6ExternalIntentBinder = M6ExternalIntentBinder(descriptors)

    fun createSealedRegistry(): M6TypedCapabilityRegistry =
        M6TypedCapabilityRegistry().also { registry ->
            registry.register(
                descriptors[0],
                M6CapabilityHandler { action ->
                    val packageName = validateAppLaunchRequest(action.request)
                    M6ActionOutcome(
                        success = true,
                        result = actions.launchPackage(packageName),
                    )
                },
                M6PayloadValidator(::validateAppLaunchRequest),
            )
            registry.register(
                descriptors[1],
                M6CapabilityHandler { action ->
                    val text = validateClipboardWriteRequest(action.request)
                    M6ActionOutcome(
                        success = true,
                        result = actions.writeClipboard(text),
                    )
                },
                M6PayloadValidator(::validateClipboardWriteRequest),
            )
            registry.seal()
        }

    private fun validateAppLaunchRequest(request: M6ExternalActionRequest): String {
        require(request.capabilityId == APP_LAUNCH_CAPABILITY) {
            "app launch handler received wrong capability"
        }
        val scope = request.scope.asMap()
        require(scope.keys == setOf(APP_PACKAGE_SCOPE)) {
            "app launch scope must contain only package"
        }
        val packageName = checkNotNull(scope[APP_PACKAGE_SCOPE])
        require(isCanonicalAndroidPackage(packageName)) {
            "app launch package is invalid"
        }
        require(request.payloadJson == "{}") {
            "app launch payload must be an empty canonical object"
        }
        return packageName
    }

    private fun validateClipboardWriteRequest(request: M6ExternalActionRequest): String {
        require(request.capabilityId == CLIPBOARD_WRITE_CAPABILITY) {
            "clipboard handler received wrong capability"
        }
        val scope = request.scope.asMap()
        require(scope == mapOf(CLIPBOARD_CHANNEL_SCOPE to CLIPBOARD_CHANNEL_VALUE)) {
            "clipboard scope must target the system clipboard"
        }
        val text = parseCanonicalTextPayload(request.payloadJson)
        require(text.toByteArray(StandardCharsets.UTF_8).size <= MAX_CLIPBOARD_TEXT_UTF8_BYTES) {
            "clipboard text exceeds byte bound"
        }
        return text
    }

    companion object {
        fun userApprovedClipboardGrant(
            principal: String,
        ): M6PolicyGrant {
            val scope = M6CapabilityScope.fromMap(
                mapOf(CLIPBOARD_CHANNEL_SCOPE to CLIPBOARD_CHANNEL_VALUE)
            )
            return M6PolicyGrant(
                principal = principal,
                capabilityId = CLIPBOARD_WRITE_CAPABILITY,
                scopeDigest = scope.digest,
                approvalRequired = true,
                maxLeaseNs = 30_000_000_000L,
                maxLeaseUses = 1,
            )
        }

        const val APP_LAUNCH_CAPABILITY = "app.launch"
        const val CLIPBOARD_WRITE_CAPABILITY = "device.clipboard.write"
        const val APP_PACKAGE_SCOPE = "package"
        const val CLIPBOARD_CHANNEL_SCOPE = "channel"
        const val CLIPBOARD_CHANNEL_VALUE = "system-clipboard"

        private const val MAX_CLIPBOARD_TEXT_UTF8_BYTES = 16 * 1024
        private const val MAX_CLIPBOARD_PAYLOAD_UTF8_BYTES = 64 * 1024
        private val PACKAGE_RE =
            Regex("^[A-Za-z][A-Za-z0-9_]*(?:\\.[A-Za-z][A-Za-z0-9_]*)+$")

        private fun isCanonicalAndroidPackage(value: String): Boolean =
            value.isNotEmpty() && value.all { it.code <= 0x7f } && PACKAGE_RE.matches(value)

        private fun parseCanonicalTextPayload(payload: String): String {
            val prefix = "{\"text\":"
            require(payload.startsWith(prefix) && payload.endsWith('}')) {
                "clipboard payload must contain exactly text"
            }
            val parser = CanonicalJsonStringParser(payload, prefix.length)
            val text = parser.parse()
            require(parser.position == payload.length - 1) {
                "clipboard payload must contain exactly text"
            }
            val canonical = buildString {
                append(prefix)
                appendCanonicalJsonString(text)
                append('}')
            }
            require(canonical == payload) {
                "clipboard payload must be canonical JSON"
            }
            return text
        }
    }
}

private class CanonicalJsonStringParser(
    private val source: String,
    start: Int,
) {
    var position: Int = start
        private set

    fun parse(): String {
        require(position < source.length && source[position] == '"') {
            "clipboard text must be a JSON string"
        }
        position += 1
        val out = StringBuilder()
        while (position < source.length) {
            val ch = source[position++]
            when (ch) {
                '"' -> return out.toString()
                '\\' -> parseEscape(out)
                else -> {
                    require(ch.code >= 0x20) { "clipboard text contains JSON control character" }
                    if (Character.isHighSurrogate(ch)) {
                        require(position < source.length && Character.isLowSurrogate(source[position])) {
                            "clipboard text contains unpaired surrogate"
                        }
                        out.append(ch)
                        out.append(source[position++])
                    } else {
                        require(!Character.isLowSurrogate(ch)) {
                            "clipboard text contains unpaired surrogate"
                        }
                        out.append(ch)
                    }
                }
            }
        }
        throw IllegalArgumentException("clipboard text JSON string is unterminated")
    }

    private fun parseEscape(out: StringBuilder) {
        require(position < source.length) { "clipboard text JSON escape is truncated" }
        when (val escaped = source[position++]) {
            '"' -> out.append('"')
            '\\' -> out.append('\\')
            '/' -> throw IllegalArgumentException("clipboard payload must use canonical JSON escaping")
            'b' -> out.append('\b')
            'f' -> out.append('\u000c')
            'n' -> out.append('\n')
            'r' -> out.append('\r')
            't' -> out.append('\t')
            'u' -> {
                val value = readHexCodeUnit().toChar()
                if (Character.isHighSurrogate(value)) {
                    require(position + 2 <= source.length && source[position] == '\\' && source[position + 1] == 'u') {
                        "clipboard text unicode escape has unpaired surrogate"
                    }
                    position += 2
                    val low = readHexCodeUnit().toChar()
                    require(Character.isLowSurrogate(low)) {
                        "clipboard text unicode escape has invalid surrogate pair"
                    }
                    out.append(value)
                    out.append(low)
                } else {
                    require(!Character.isLowSurrogate(value)) {
                        "clipboard text unicode escape has unpaired surrogate"
                    }
                    out.append(value)
                }
            }
            else -> throw IllegalArgumentException("clipboard text JSON escape is invalid: $escaped")
        }
    }

    private fun readHexCodeUnit(): Int {
        require(position + 4 <= source.length) { "clipboard text unicode escape is truncated" }
        var value = 0
        repeat(4) {
            val digit = source[position++].digitToIntOrNull(16)
                ?: throw IllegalArgumentException("clipboard text unicode escape is invalid")
            value = (value shl 4) or digit
        }
        return value
    }
}

private fun StringBuilder.appendCanonicalJsonString(value: String) {
    append('"')
    var index = 0
    while (index < value.length) {
        val ch = value[index]
        when (ch) {
            '"' -> append("\\\"")
            '\\' -> append("\\\\")
            '\b' -> append("\\b")
            '\u000c' -> append("\\f")
            '\n' -> append("\\n")
            '\r' -> append("\\r")
            '\t' -> append("\\t")
            else -> when {
                ch.code < 0x20 -> append("\\u%04x".format(ch.code))
                Character.isHighSurrogate(ch) -> {
                    require(index + 1 < value.length && Character.isLowSurrogate(value[index + 1])) {
                        "clipboard text contains unpaired surrogate"
                    }
                    append(ch)
                    append(value[++index])
                }
                Character.isLowSurrogate(ch) ->
                    throw IllegalArgumentException("clipboard text contains unpaired surrogate")
                else -> append(ch)
            }
        }
        index += 1
    }
    append('"')
}
