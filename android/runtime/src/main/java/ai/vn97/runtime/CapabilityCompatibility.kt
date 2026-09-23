package ai.vn97.runtime

import java.nio.charset.StandardCharsets
import java.security.MessageDigest

enum class VN97CompatibilityDisposition {
    DIRECT,
    ADAPT_REQUIRED,
}

data class VN97FormatRule(
    val role: String,
    val acceptedFormats: List<String>,
) {
    init {
        requireCompatRole(role, "format role")
        if (acceptedFormats.isEmpty() || acceptedFormats.toSet().size != acceptedFormats.size) {
            throw VN97CompatibilityException("accepted formats must be non-empty and unique")
        }
        acceptedFormats.forEach { requireCompatFormat(it, "accepted format") }
    }
}

data class VN97CompatibilityProfile(
    val profileId: String,
    val runtimeApiVersion: Long,
    val supportedKinds: Set<String>,
    val minCapabilityVersion: Long,
    val maxCapabilityVersion: Long,
    val formatRules: List<VN97FormatRule>,
) {
    init {
        requireCompatId(profileId, "profile_id")
        if (runtimeApiVersion <= 0L) {
            throw VN97CompatibilityException("runtime_api_version must be positive")
        }
        if (supportedKinds.isEmpty() || !VN97_COMPAT_KINDS.containsAll(supportedKinds)) {
            throw VN97CompatibilityException("supported capability kinds are invalid")
        }
        if (minCapabilityVersion !in 1L..0xffff_ffffL ||
            maxCapabilityVersion !in minCapabilityVersion..0xffff_ffffL
        ) {
            throw VN97CompatibilityException("profile capability version range is invalid")
        }
        if (formatRules.isEmpty() || formatRules.map { it.role }.toSet().size != formatRules.size) {
            throw VN97CompatibilityException("format rules must be non-empty and unique by role")
        }
    }

    fun rule(role: String): VN97FormatRule? = formatRules.firstOrNull { it.role == role }

    fun fingerprint(): String {
        val rules = VnStrictJson.array(
            formatRules.map { rule ->
                VnStrictJson.objectOf(
                    "accepted_formats" to VnStrictJson.array(
                        rule.acceptedFormats.map(VnStrictJson::string)
                    ),
                    "role" to VnStrictJson.string(rule.role),
                )
            }
        )
        val root = VnStrictJson.objectOf(
            "format_rules" to rules,
            "max_capability_version" to VnJsonNumber(maxCapabilityVersion.toString()),
            "min_capability_version" to VnJsonNumber(minCapabilityVersion.toString()),
            "profile_id" to VnStrictJson.string(profileId),
            "runtime_api_version" to VnJsonNumber(runtimeApiVersion.toString()),
            "supported_kinds" to VnStrictJson.array(
                supportedKinds.sorted().map(VnStrictJson::string)
            ),
        )
        return sha256Hex(VnStrictJson.canonical(root).toByteArray(StandardCharsets.UTF_8))
    }
}

data class VN97AdapterSpec(
    val adapterId: String,
    val role: String,
    val inputFormat: String,
    val outputFormat: String,
    val lossy: Boolean = false,
) {
    init {
        requireCompatId(adapterId, "adapter_id")
        requireCompatRole(role, "adapter role")
        requireCompatFormat(inputFormat, "adapter input format")
        requireCompatFormat(outputFormat, "adapter output format")
        if (inputFormat == outputFormat) {
            throw VN97CompatibilityException("adapter must change format")
        }
    }
}

data class VN97SectionPlan(
    val sectionIndex: Int,
    val role: String,
    val sourceFormat: String,
    val targetFormat: String,
    val adapterId: String?,
    val lossy: Boolean,
)

data class VN97CompatibilityPlan(
    val packageSha256: String,
    val capabilityId: String,
    val capabilityVersion: Long,
    val publisherKeyId: String,
    val signatureSha256: String,
    val profileId: String,
    val profileSha256: String,
    val runtimeApiVersion: Long,
    val disposition: VN97CompatibilityDisposition,
    val sections: List<VN97SectionPlan>,
) {
    init {
        requireCompatSha(packageSha256, "package_sha256")
        requireCompatId(capabilityId, "capability_id")
        if (capabilityVersion !in 1L..0xffff_ffffL) {
            throw VN97CompatibilityException("capability_version is invalid")
        }
        requireCompatId(publisherKeyId, "publisher_key_id")
        requireCompatSha(signatureSha256, "signature_sha256")
        requireCompatId(profileId, "profile_id")
        requireCompatSha(profileSha256, "profile_sha256")
        if (runtimeApiVersion <= 0L) {
            throw VN97CompatibilityException("runtime_api_version is invalid")
        }
        if (sections.isEmpty()) {
            throw VN97CompatibilityException("compatibility plan requires sections")
        }
    }

    fun sha256(): String = sha256Hex(
        VnStrictJson.canonical(toJson()).toByteArray(StandardCharsets.UTF_8)
    )

    internal fun toJson(): VnJsonObject = VnStrictJson.objectOf(
        "capability_id" to VnStrictJson.string(capabilityId),
        "capability_version" to VnJsonNumber(capabilityVersion.toString()),
        "disposition" to VnStrictJson.string(
            when (disposition) {
                VN97CompatibilityDisposition.DIRECT -> "direct"
                VN97CompatibilityDisposition.ADAPT_REQUIRED -> "adapt_required"
            }
        ),
        "package_sha256" to VnStrictJson.string(packageSha256),
        "profile_id" to VnStrictJson.string(profileId),
        "profile_sha256" to VnStrictJson.string(profileSha256),
        "publisher_key_id" to VnStrictJson.string(publisherKeyId),
        "runtime_api_version" to VnJsonNumber(runtimeApiVersion.toString()),
        "sections" to VnStrictJson.array(
            sections.map { section ->
                VnStrictJson.objectOf(
                    "adapter_id" to (section.adapterId?.let(VnStrictJson::string) ?: VnJsonNull),
                    "lossy" to VnStrictJson.bool(section.lossy),
                    "role" to VnStrictJson.string(section.role),
                    "section_index" to VnJsonNumber(section.sectionIndex.toString()),
                    "source_format" to VnStrictJson.string(section.sourceFormat),
                    "target_format" to VnStrictJson.string(section.targetFormat),
                )
            }
        ),
        "signature_sha256" to VnStrictJson.string(signatureSha256),
    )
}

class VN97CompatibilityException(
    message: String,
    cause: Throwable? = null,
) : IllegalStateException(message, cause)

object VN97CapabilityCompatibility {
    fun plan(
        verified: VN97VerifiedCapability,
        profile: VN97CompatibilityProfile,
        adapters: List<VN97AdapterSpec> = emptyList(),
        allowLossy: Boolean = false,
    ): VN97CompatibilityPlan {
        val manifest = verified.parsed.manifest
        if (manifest.kind !in profile.supportedKinds) {
            throw VN97CompatibilityException("capability kind is unsupported by profile")
        }
        if (manifest.capabilityVersion !in profile.minCapabilityVersion..profile.maxCapabilityVersion) {
            throw VN97CompatibilityException("capability version is outside profile range")
        }
        if (adapters.map { it.adapterId }.toSet().size != adapters.size) {
            throw VN97CompatibilityException("adapter IDs must be unique")
        }

        var adapted = false
        val sectionPlans = manifest.sections.map { section ->
            val rule = profile.rule(section.role)
                ?: throw VN97CompatibilityException(
                    "profile has no format rule for role=${section.role}"
                )
            if (section.format in rule.acceptedFormats) {
                VN97SectionPlan(
                    sectionIndex = section.index,
                    role = section.role,
                    sourceFormat = section.format,
                    targetFormat = section.format,
                    adapterId = null,
                    lossy = false,
                )
            } else {
                val matches = adapters.filter {
                    it.role == section.role &&
                        it.inputFormat == section.format &&
                        it.outputFormat in rule.acceptedFormats
                }
                if (matches.isEmpty()) {
                    throw VN97CompatibilityException(
                        "no adaptation path for role=${section.role} format=${section.format}"
                    )
                }
                if (matches.size != 1) {
                    throw VN97CompatibilityException(
                        "ambiguous adaptation path for role=${section.role} format=${section.format}"
                    )
                }
                val adapter = matches.single()
                if (adapter.lossy && !allowLossy) {
                    throw VN97CompatibilityException(
                        "lossy adaptation denied for role=${section.role}"
                    )
                }
                adapted = true
                VN97SectionPlan(
                    sectionIndex = section.index,
                    role = section.role,
                    sourceFormat = section.format,
                    targetFormat = adapter.outputFormat,
                    adapterId = adapter.adapterId,
                    lossy = adapter.lossy,
                )
            }
        }

        return VN97CompatibilityPlan(
            packageSha256 = verified.parsed.packageSha256,
            capabilityId = manifest.capabilityId,
            capabilityVersion = manifest.capabilityVersion,
            publisherKeyId = verified.publisherKeyId,
            signatureSha256 = verified.signatureSha256,
            profileId = profile.profileId,
            profileSha256 = profile.fingerprint(),
            runtimeApiVersion = profile.runtimeApiVersion,
            disposition = if (adapted) {
                VN97CompatibilityDisposition.ADAPT_REQUIRED
            } else {
                VN97CompatibilityDisposition.DIRECT
            },
            sections = sectionPlans,
        )
    }
}

private val VN97_COMPAT_KINDS = setOf(
    "weights",
    "tokenizer",
    "memory",
    "avatar",
    "multimodal",
    "knowledge",
    "composite",
)

internal fun requireCompatId(value: String, label: String) {
    if (value.isEmpty() ||
        value.length > 128 ||
        value[0] !in 'a'..'z' ||
        value.any {
            it !in 'a'..'z' &&
                it !in '0'..'9' &&
                it != '.' && it != '_' && it != '-'
        }
    ) {
        throw VN97CompatibilityException("$label is invalid")
    }
}

private fun requireCompatRole(value: String, label: String) {
    if (value.isEmpty() ||
        value.length > 64 ||
        value[0] !in 'a'..'z' ||
        value.any {
            it !in 'a'..'z' &&
                it !in '0'..'9' &&
                it != '.' && it != '_' && it != '-'
        }
    ) {
        throw VN97CompatibilityException("$label is invalid")
    }
}

private fun requireCompatFormat(value: String, label: String) {
    fun alphaNum(ch: Char): Boolean =
        ch in 'A'..'Z' || ch in 'a'..'z' || ch in '0'..'9'
    if (value.isEmpty() ||
        value.length > 64 ||
        !alphaNum(value[0]) ||
        value.any { !alphaNum(it) && it != '.' && it != '_' && it != '-' }
    ) {
        throw VN97CompatibilityException("$label is invalid")
    }
}

internal fun requireCompatSha(value: String, label: String) {
    if (value.length != 64 || value.any { it !in "0123456789abcdef" }) {
        throw VN97CompatibilityException("$label is invalid")
    }
}

internal fun sha256Hex(bytes: ByteArray): String =
    MessageDigest.getInstance("SHA-256")
        .digest(bytes)
        .joinToString("") { "%02x".format(it.toInt() and 0xff) }
