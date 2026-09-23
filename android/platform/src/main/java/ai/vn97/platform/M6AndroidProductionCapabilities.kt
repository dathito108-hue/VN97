package ai.vn97.platform

import java.nio.charset.StandardCharsets

internal interface M6AndroidActionPort {
    fun launchPackage(packageName: String): String
    fun writeClipboard(text: String): String
    fun gameTap(
        packageName: String,
        xBasisPoints: Int,
        yBasisPoints: Int,
        durationMillis: Long,
    ): String
    fun gameSwipe(
        packageName: String,
        startXBasisPoints: Int,
        startYBasisPoints: Int,
        endXBasisPoints: Int,
        endYBasisPoints: Int,
        durationMillis: Long,
    ): String
    fun gameBack(packageName: String): String
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
            payloadSchemaJson = "{}",
            maxLeaseNs = 30_000_000_000L,
            maxLeaseUses = 1,
        ),
        M6CapabilityDescriptor(
            capabilityId = CLIPBOARD_WRITE_CAPABILITY,
            requiredScopeKeys = setOf(CLIPBOARD_CHANNEL_SCOPE),
            approvalRequired = true,
            maxPayloadUtf8Bytes = MAX_CLIPBOARD_PAYLOAD_UTF8_BYTES,
            payloadSchemaJson = "{\"text\":\"string\"}",
            maxLeaseNs = 30_000_000_000L,
            maxLeaseUses = 1,
        ),
        M6CapabilityDescriptor(
            capabilityId = GAME_TAP_CAPABILITY,
            requiredScopeKeys = setOf(APP_PACKAGE_SCOPE),
            approvalRequired = false,
            maxPayloadUtf8Bytes = 96,
            payloadSchemaJson =
                "{\"duration_ms\":80,\"x_bps\":5000,\"y_bps\":5000}",
            maxLeaseNs = GAME_ACTION_LEASE_NS,
            maxLeaseUses = 1,
        ),
        M6CapabilityDescriptor(
            capabilityId = GAME_SWIPE_CAPABILITY,
            requiredScopeKeys = setOf(APP_PACKAGE_SCOPE),
            approvalRequired = false,
            maxPayloadUtf8Bytes = 192,
            payloadSchemaJson =
                "{\"duration_ms\":300,\"end_x_bps\":8000,\"end_y_bps\":5000,\"start_x_bps\":2000,\"start_y_bps\":5000}",
            maxLeaseNs = GAME_ACTION_LEASE_NS,
            maxLeaseUses = 1,
        ),
        M6CapabilityDescriptor(
            capabilityId = GAME_BACK_CAPABILITY,
            requiredScopeKeys = setOf(APP_PACKAGE_SCOPE),
            approvalRequired = false,
            maxPayloadUtf8Bytes = 2,
            payloadSchemaJson = "{}",
            maxLeaseNs = GAME_ACTION_LEASE_NS,
            maxLeaseUses = 1,
        ),
    )

    val intentBinder: M6ExternalIntentBinder =
        M6ExternalIntentBinder(descriptors)

    val gameDescriptors: List<M6CapabilityDescriptor> =
        descriptors.filter {
            it.capabilityId in GAME_CAPABILITY_IDS
        }

    val gameIntentBinder: M6ExternalIntentBinder =
        M6ExternalIntentBinder(gameDescriptors)

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
            registry.register(
                descriptors[2],
                M6CapabilityHandler { action ->
                    val request = validateGameTapRequest(action.request)
                    M6ActionOutcome(
                        success = true,
                        result = actions.gameTap(
                            request.packageName,
                            request.xBasisPoints,
                            request.yBasisPoints,
                            request.durationMillis,
                        ),
                    )
                },
                M6PayloadValidator(::validateGameTapRequest),
            )
            registry.register(
                descriptors[3],
                M6CapabilityHandler { action ->
                    val request = validateGameSwipeRequest(action.request)
                    M6ActionOutcome(
                        success = true,
                        result = actions.gameSwipe(
                            request.packageName,
                            request.startXBasisPoints,
                            request.startYBasisPoints,
                            request.endXBasisPoints,
                            request.endYBasisPoints,
                            request.durationMillis,
                        ),
                    )
                },
                M6PayloadValidator(::validateGameSwipeRequest),
            )
            registry.register(
                descriptors[4],
                M6CapabilityHandler { action ->
                    val packageName =
                        validateGameBackRequest(action.request)
                    M6ActionOutcome(
                        success = true,
                        result = actions.gameBack(packageName),
                    )
                },
                M6PayloadValidator(::validateGameBackRequest),
            )
            registry.seal()
        }

    fun createSealedGameRegistry(): M6TypedCapabilityRegistry =
        M6TypedCapabilityRegistry().also { registry ->
            registry.register(
                gameDescriptors[0],
                M6CapabilityHandler { action ->
                    val request =
                        validateGameTapRequest(action.request)
                    M6ActionOutcome(
                        success = true,
                        result = actions.gameTap(
                            request.packageName,
                            request.xBasisPoints,
                            request.yBasisPoints,
                            request.durationMillis,
                        ),
                    )
                },
                M6PayloadValidator(::validateGameTapRequest),
            )
            registry.register(
                gameDescriptors[1],
                M6CapabilityHandler { action ->
                    val request =
                        validateGameSwipeRequest(action.request)
                    M6ActionOutcome(
                        success = true,
                        result = actions.gameSwipe(
                            request.packageName,
                            request.startXBasisPoints,
                            request.startYBasisPoints,
                            request.endXBasisPoints,
                            request.endYBasisPoints,
                            request.durationMillis,
                        ),
                    )
                },
                M6PayloadValidator(::validateGameSwipeRequest),
            )
            registry.register(
                gameDescriptors[2],
                M6CapabilityHandler { action ->
                    val packageName =
                        validateGameBackRequest(action.request)
                    M6ActionOutcome(
                        success = true,
                        result = actions.gameBack(packageName),
                    )
                },
                M6PayloadValidator(::validateGameBackRequest),
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

    private data class GameTapRequest(
        val packageName: String,
        val xBasisPoints: Int,
        val yBasisPoints: Int,
        val durationMillis: Long,
    )

    private data class GameSwipeRequest(
        val packageName: String,
        val startXBasisPoints: Int,
        val startYBasisPoints: Int,
        val endXBasisPoints: Int,
        val endYBasisPoints: Int,
        val durationMillis: Long,
    )

    private fun validateGameScope(
        request: M6ExternalActionRequest,
        capabilityId: String,
    ): String {
        require(request.capabilityId == capabilityId) {
            "game handler received wrong capability"
        }
        val scope = request.scope.asMap()
        require(scope.keys == setOf(APP_PACKAGE_SCOPE)) {
            "game scope must contain only package"
        }
        val packageName = checkNotNull(scope[APP_PACKAGE_SCOPE])
        require(isCanonicalAndroidPackage(packageName)) {
            "game package is invalid"
        }
        return packageName
    }

    private fun validateGameTapRequest(
        request: M6ExternalActionRequest,
    ): GameTapRequest {
        val packageName = validateGameScope(
            request,
            GAME_TAP_CAPABILITY,
        )
        val match = GAME_TAP_PAYLOAD.matchEntire(request.payloadJson)
            ?: throw IllegalArgumentException(
                "game tap payload must match canonical schema"
            )
        val duration = match.groupValues[1].toLong()
        val x = match.groupValues[2].toInt()
        val y = match.groupValues[3].toInt()
        require(duration in MIN_TAP_MILLIS..MAX_TAP_MILLIS) {
            "game tap duration is outside bounds"
        }
        require(x in 0..BASIS_POINTS && y in 0..BASIS_POINTS) {
            "game tap coordinates are outside basis-point bounds"
        }
        return GameTapRequest(packageName, x, y, duration)
    }

    private fun validateGameSwipeRequest(
        request: M6ExternalActionRequest,
    ): GameSwipeRequest {
        val packageName = validateGameScope(
            request,
            GAME_SWIPE_CAPABILITY,
        )
        val match = GAME_SWIPE_PAYLOAD.matchEntire(request.payloadJson)
            ?: throw IllegalArgumentException(
                "game swipe payload must match canonical schema"
            )
        val duration = match.groupValues[1].toLong()
        val endX = match.groupValues[2].toInt()
        val endY = match.groupValues[3].toInt()
        val startX = match.groupValues[4].toInt()
        val startY = match.groupValues[5].toInt()
        require(duration in MIN_SWIPE_MILLIS..MAX_SWIPE_MILLIS) {
            "game swipe duration is outside bounds"
        }
        require(
            listOf(startX, startY, endX, endY).all {
                it in 0..BASIS_POINTS
            }
        ) {
            "game swipe coordinates are outside basis-point bounds"
        }
        require(startX != endX || startY != endY) {
            "game swipe requires distinct endpoints"
        }
        return GameSwipeRequest(
            packageName,
            startX,
            startY,
            endX,
            endY,
            duration,
        )
    }

    private fun validateGameBackRequest(
        request: M6ExternalActionRequest,
    ): String {
        val packageName = validateGameScope(
            request,
            GAME_BACK_CAPABILITY,
        )
        require(request.payloadJson == "{}") {
            "game back payload must be empty"
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
        fun userApprovedAppLaunchGrant(
            principal: String,
            packageName: String,
        ): M6PolicyGrant {
            require(isCanonicalAndroidPackage(packageName)) {
                "app launch package is invalid"
            }
            val scope = M6CapabilityScope.fromMap(
                mapOf(APP_PACKAGE_SCOPE to packageName)
            )
            return M6PolicyGrant(
                principal = principal,
                capabilityId = APP_LAUNCH_CAPABILITY,
                scopeDigest = scope.digest,
                approvalRequired = true,
                maxLeaseNs = 30_000_000_000L,
                maxLeaseUses = 1,
            )
        }

        fun userApprovedGameControlGrants(
            principal: String,
            packageName: String,
        ): List<M6PolicyGrant> {
            require(isCanonicalAndroidPackage(packageName)) {
                "game package is invalid"
            }
            val scope = M6CapabilityScope.fromMap(
                mapOf(APP_PACKAGE_SCOPE to packageName)
            )
            return listOf(
                GAME_TAP_CAPABILITY,
                GAME_SWIPE_CAPABILITY,
                GAME_BACK_CAPABILITY,
            ).map { capabilityId ->
                M6PolicyGrant(
                    principal = principal,
                    capabilityId = capabilityId,
                    scopeDigest = scope.digest,
                    approvalRequired = false,
                    maxLeaseNs = GAME_ACTION_LEASE_NS,
                    maxLeaseUses = 1,
                )
            }
        }

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
        const val GAME_TAP_CAPABILITY = "device.game.tap"
        const val GAME_SWIPE_CAPABILITY = "device.game.swipe"
        const val GAME_BACK_CAPABILITY = "device.game.back"
        const val APP_PACKAGE_SCOPE = "package"
        const val CLIPBOARD_CHANNEL_SCOPE = "channel"
        const val CLIPBOARD_CHANNEL_VALUE = "system-clipboard"

        private const val MAX_CLIPBOARD_TEXT_UTF8_BYTES = 16 * 1024
        private const val MAX_CLIPBOARD_PAYLOAD_UTF8_BYTES = 64 * 1024
        private const val BASIS_POINTS = 10_000
        private const val MIN_TAP_MILLIS = 20L
        private const val MAX_TAP_MILLIS = 1_500L
        private const val MIN_SWIPE_MILLIS = 50L
        private const val MAX_SWIPE_MILLIS = 3_000L
        private const val GAME_ACTION_LEASE_NS = 10_000_000_000L
        private val GAME_CAPABILITY_IDS = setOf(
            GAME_TAP_CAPABILITY,
            GAME_SWIPE_CAPABILITY,
            GAME_BACK_CAPABILITY,
        )
        private val GAME_TAP_PAYLOAD =
            Regex("^\\{\\\"duration_ms\\\":([0-9]+),\\\"x_bps\\\":([0-9]+),\\\"y_bps\\\":([0-9]+)\\}$")
        private val GAME_SWIPE_PAYLOAD =
            Regex("^\\{\\\"duration_ms\\\":([0-9]+),\\\"end_x_bps\\\":([0-9]+),\\\"end_y_bps\\\":([0-9]+),\\\"start_x_bps\\\":([0-9]+),\\\"start_y_bps\\\":([0-9]+)\\}$")
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
