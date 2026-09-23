package ai.vn97.runtime

enum class NativeStepKind { REASON, RETRIEVE, VERIFY, RESPOND, EXTERNAL }
enum class NativeStepStatus {
    PENDING,
    RUNNING,
    WAITING_VERIFICATION,
    WAITING_EXTERNAL,
    SUCCEEDED,
    FAILED,
    CANCELLED,
}
enum class NativePlanStatus {
    READY,
    RUNNING,
    WAITING_EXTERNAL,
    PAUSED,
    COMPLETED,
    FAILED,
    CANCELLED,
    BUDGET_EXHAUSTED,
}

data class NativePlanStepSpec(
    val kind: NativeStepKind,
    val objective: String,
    val dependencies: List<Int> = emptyList(),
)

class NativePlannerStep(
    val stepId: Int,
    val spec: NativePlanStepSpec,
    var status: NativeStepStatus,
)

class NativePlan(
    val planId: String,
    val goal: String,
    val steps: List<NativePlannerStep>,
    var status: NativePlanStatus,
)

class NativePlanController(val plan: NativePlan)

data class NativeExternalCapabilityView(
    val capabilityId: String,
    val requiredScopeKeys: List<String>,
    val optionalScopeKeys: List<String>,
    val approvalRequired: Boolean,
    val maxPayloadUtf8Bytes: Int,
    val payloadSchemaJson: String = "{}",
)

data class NativeExternalIntentRequest(
    val planId: String,
    val goal: String,
    val stepId: Int,
    val objective: String,
    val capabilities: List<NativeExternalCapabilityView>,
)

data class NativeExternalIntent(
    val capabilityId: String,
    val scope: Map<String, String>,
    val payloadJson: String,
)

open class NativeTypedCognitionAdapter(
    private val result: NativeExternalIntent,
    private val beforeReturn: (() -> Unit)? = null,
) {
    open fun proposeExternalIntent(
        request: NativeExternalIntentRequest,
    ): NativeExternalIntent {
        beforeReturn?.invoke()
        return result
    }
}
