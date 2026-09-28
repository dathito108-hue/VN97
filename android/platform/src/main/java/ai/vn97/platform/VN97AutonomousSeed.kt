package ai.vn97.platform

import ai.vn97.runtime.VN97G06Model
import ai.vn97.runtime.NativeCognitionLimits
import ai.vn97.runtime.NativeCognitionLoop
import ai.vn97.runtime.NativeCognitionRuntimeConfig
import ai.vn97.runtime.NativePlan
import ai.vn97.runtime.NativePlanController
import ai.vn97.runtime.NativeReasoningBudget
import ai.vn97.runtime.NativeRuntimeCheckpointSnapshot
import ai.vn97.runtime.NativeRuntimeConfig
import ai.vn97.runtime.NativeTypedCognitionAdapter

data class VN97AutonomousContinuationSeed(
    val runtimeConfig: NativeRuntimeConfig,
    val snapshot: NativeRuntimeCheckpointSnapshot,
    val plan: NativePlan,
) {
    init {
        require(!plan.isTerminal()) {
            "autonomous continuation seed plan must be non-terminal"
        }
        require(snapshot.modelBinding.bound) {
            "autonomous continuation seed must be model-bound"
        }
        require(snapshot.info.config == runtimeConfig) {
            "autonomous continuation seed runtime config mismatch"
        }
    }
}


/** Planner continuity must not serialize old SSM state as if it were G06 state. */
@Suppress("UNUSED_PARAMETER")
fun createVN97AutonomousContinuationSeed(
    model: VN97G06Model,
    goal: String,
    budget: NativeReasoningBudget = NativeReasoningBudget(),
    cognitionRuntimeConfig: NativeCognitionRuntimeConfig = NativeCognitionRuntimeConfig(),
    cognitionLimits: NativeCognitionLimits = NativeCognitionLimits(),
    createdNs: Long = System.currentTimeMillis() * 1_000_000L,
): VN97AutonomousContinuationSeed = error(
    "G06: tác vụ qua khởi động lại đang chờ chuyển checkpoint; chat và tác vụ trong phiên vẫn dùng được. Không chạy lại lõi cũ."
)

@Suppress("UNUSED_PARAMETER")
fun createVN97AutonomousReplanSeed(
    model: VN97G06Model,
    previousPlan: NativePlan,
    feedback: String,
    cognitionRuntimeConfig: NativeCognitionRuntimeConfig = NativeCognitionRuntimeConfig(),
    cognitionLimits: NativeCognitionLimits = NativeCognitionLimits(),
    createdNs: Long = System.currentTimeMillis() * 1_000_000L,
): VN97AutonomousContinuationSeed = error(
    "G06: checkpoint tác vụ cũ không tương thích; không tự tiếp tục bằng lõi cũ."
)
