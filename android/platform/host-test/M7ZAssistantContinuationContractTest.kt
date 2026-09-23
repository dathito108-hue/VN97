import ai.vn97.platform.*
import ai.vn97.runtime.*
import java.nio.file.Files

private inline fun expectFailure(block: () -> Unit) {
    var failed = false
    try {
        block()
    } catch (_: Exception) {
        failed = true
    }
    check(failed)
}

private fun modelId(seed: Int): ByteArray =
    ByteArray(32) { index -> ((seed + index) and 0xff).toByte() }

private fun hex(bytes: ByteArray): String =
    bytes.joinToString("") { "%02x".format(it.toInt() and 0xff) }

private fun continuity(
    controller: NativePlanController,
    modelId: ByteArray,
    epoch: Long = 1L,
): NativeCompositeContinuity {
    val manifest = NativeContinuityManifest(
        epoch = epoch,
        slot = "a",
        runtimeSha256 = "11".repeat(32),
        plannerSha256 = "22".repeat(32),
        modelId = modelId.copyOf(),
        planId = controller.plan.planId,
        sequencePosition = 7L,
    )
    return NativeCompositeContinuity(
        manifest = manifest,
        runtimeCheckpoint = byteArrayOf(1, 2, 3),
        planner = controller,
    )
}

fun main() {
    val model = modelId(1)
    val controller = NativePlanController.create(
        goal = "continue after reboot",
        specs = listOf(
            NativePlanStepSpec(
                kind = NativeStepKind.REASON,
                objective = "reason",
            ),
            NativePlanStepSpec(
                kind = NativeStepKind.RESPOND,
                objective = "answer",
                dependencies = listOf(1),
            ),
        ),
        createdNs = 10L,
    )
    val binding = VN97AssistantContinuationBinding(
        principal = "runtime.user",
        planId = controller.plan.planId,
        modelIdHex = hex(model),
    )
    val restored = restoreAssistantContinuation(
        expected = binding,
        continuity = continuity(controller, model),
    )
    check(restored.controller === controller)
    check(restored.binding == binding)
    check(restored.epoch == 1L)
    binding.requireModelId(model)

    expectFailure {
        restoreAssistantContinuation(
            expected = binding.copy(planId = "00".repeat(32)),
            continuity = continuity(controller, model),
        )
    }
    expectFailure {
        restoreAssistantContinuation(
            expected = binding.copy(modelIdHex = "ff".repeat(32)),
            continuity = continuity(controller, model),
        )
    }
    expectFailure {
        binding.requireModelId(modelId(99))
    }
    expectFailure {
        VN97AssistantContinuationBinding(
            principal = "bad principal",
            planId = controller.plan.planId,
            modelIdHex = hex(model),
        )
    }

    val root = Files.createTempDirectory("m7z-binding-").toFile()
    val store = VN97AssistantContinuationBindingStore(root)
    check(store.bind(binding) == binding)
    check(store.loadOrNull() == binding)
    check(store.require(binding) == binding)
    check(store.bind(binding) == binding)
    expectFailure {
        store.bind(binding.copy(principal = "runtime.other"))
    }
    expectFailure {
        store.require(binding.copy(modelIdHex = "aa".repeat(32)))
    }

    val bindingFile = root.resolve("assistant.vn97acb1")
    val original = bindingFile.readBytes()
    val corrupt = original.copyOf()
    corrupt[corrupt.lastIndex] =
        (corrupt.last().toInt() xor 1).toByte()
    bindingFile.writeBytes(corrupt)
    expectFailure { store.loadOrNull() }
    bindingFile.writeBytes(original)
    check(store.loadOrNull() == binding)
    store.delete()
    check(store.loadOrNull() == null)

    val terminal = NativePlanController.create(
        goal = "already done",
        specs = listOf(
            NativePlanStepSpec(
                kind = NativeStepKind.RESPOND,
                objective = "answer",
            )
        ),
        createdNs = 20L,
    )
    terminal.beginStep(1)
    terminal.completeStep(1, "done", 1.0)
    check(terminal.plan.status == NativePlanStatus.COMPLETED)
    val terminalBinding = VN97AssistantContinuationBinding(
        principal = "runtime.user",
        planId = terminal.plan.planId,
        modelIdHex = hex(model),
    )
    expectFailure {
        restoreAssistantContinuation(
            expected = terminalBinding,
            continuity = continuity(terminal, model),
        )
    }

    println("M7Z_ASSISTANT_CONTINUATION_CONTRACT_PASS")
}
