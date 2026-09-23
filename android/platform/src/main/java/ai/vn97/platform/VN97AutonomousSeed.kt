package ai.vn97.platform

import ai.vn97.runtime.NativeActivatedModel
import ai.vn97.runtime.NativeCognitionInferenceEngine
import ai.vn97.runtime.NativeCognitionLimits
import ai.vn97.runtime.NativeCognitionLoop
import ai.vn97.runtime.NativeCognitionRuntimeConfig
import ai.vn97.runtime.NativePlan
import ai.vn97.runtime.NativeReasoningBudget
import ai.vn97.runtime.NativeRuntimeCheckpointSnapshot
import ai.vn97.runtime.NativeRuntimeConfig
import ai.vn97.runtime.NativeRuntimeOwner
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

private fun autonomousRuntimeConfig(
    model: NativeActivatedModel,
    cognitionRuntimeConfig: NativeCognitionRuntimeConfig,
): NativeRuntimeConfig = NativeRuntimeConfig(
    layers = model.info.layers,
    batch = 1,
    dModel = model.info.dModel,
    dState = model.info.dState,
    recurrentBackend = cognitionRuntimeConfig.recurrentBackend,
    packedBackend = cognitionRuntimeConfig.packedBackend,
)

private fun autonomousSeed(
    model: NativeActivatedModel,
    controller: NativePlanController,
    cognitionRuntimeConfig: NativeCognitionRuntimeConfig,
): VN97AutonomousContinuationSeed {
    val runtimeConfig =
        autonomousRuntimeConfig(model, cognitionRuntimeConfig)
    val snapshot =
        NativeRuntimeOwner.createModelBoundSeedSnapshot(
            model = model,
            config = runtimeConfig,
        )
    if (!snapshot.modelBinding.modelId.contentEquals(model.info.modelId)) {
        throw IllegalStateException(
            "autonomous continuation seed model identity mismatch"
        )
    }
    return VN97AutonomousContinuationSeed(
        runtimeConfig = runtimeConfig,
        snapshot = snapshot,
        plan = controller.plan,
    )
}

fun createVN97AutonomousContinuationSeed(
    model: NativeActivatedModel,
    goal: String,
    budget: NativeReasoningBudget = NativeReasoningBudget(
        maxTransitions = 128,
        maxRetriesPerStep = 3,
        maxMemoryQueries = 16,
        maxMemoryHits = 8,
    ),
    cognitionRuntimeConfig: NativeCognitionRuntimeConfig =
        NativeCognitionRuntimeConfig(),
    cognitionLimits: NativeCognitionLimits = NativeCognitionLimits(
        maxPlanSteps = 24,
        maxExternalSteps = 8,
        maxCyclesPerRun = 32,
    ),
    createdNs: Long = System.currentTimeMillis() * 1_000_000L,
): VN97AutonomousContinuationSeed {
    require(goal.isNotBlank()) {
        "autonomous goal must not be blank"
    }
    require(createdNs >= 0L) {
        "autonomous goal createdNs must be non-negative"
    }
    require(model.info.hasTokenizer) {
        "autonomous goal planning requires VN97TK1 tokenizer"
    }

    val cognition = NativeTypedCognitionAdapter(
        NativeCognitionInferenceEngine(
            model = model,
            config = cognitionRuntimeConfig,
        )
    )
    val controller = NativeCognitionLoop(
        cognition,
        cognitionLimits,
    ).buildPlan(
        goal = goal,
        budget = budget,
        createdNs = createdNs,
    )

    return autonomousSeed(
        model = model,
        controller = controller,
        cognitionRuntimeConfig = cognitionRuntimeConfig,
    )
}


fun createVN97AutonomousReplanSeed(
    model: NativeActivatedModel,
    previousPlan: NativePlan,
    feedback: String,
    cognitionRuntimeConfig: NativeCognitionRuntimeConfig =
        NativeCognitionRuntimeConfig(),
    cognitionLimits: NativeCognitionLimits = NativeCognitionLimits(
        maxPlanSteps = 24,
        maxExternalSteps = 8,
        maxCyclesPerRun = 32,
    ),
    createdNs: Long = System.currentTimeMillis() * 1_000_000L,
): VN97AutonomousContinuationSeed {
    require(previousPlan.isTerminal()) {
        "autonomous replan requires terminal previous plan"
    }
    require(previousPlan.goal.isNotBlank()) {
        "autonomous replan goal must not be blank"
    }
    require(feedback.isNotBlank()) {
        "autonomous replan feedback must not be blank"
    }
    require(model.info.hasTokenizer) {
        "autonomous replanning requires VN97TK1 tokenizer"
    }

    val cognition = NativeTypedCognitionAdapter(
        NativeCognitionInferenceEngine(
            model = model,
            config = cognitionRuntimeConfig,
        )
    )
    val revision = NativeCognitionLoop(
        cognition,
        cognitionLimits,
    ).replanTerminalPlan(
        previousPlan = previousPlan,
        feedback = feedback,
        createdNs = createdNs,
    )
    return autonomousSeed(
        model = model,
        controller = revision.controller,
        cognitionRuntimeConfig = cognitionRuntimeConfig,
    )
}
