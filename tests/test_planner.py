from pathlib import Path

import pytest

from vn97.memory import MemoryJournal, MemoryKind
from vn97.planner import (
    CheckpointCorruptionError,
    PlanController,
    PlanStatus,
    PlanStepSpec,
    PlannerError,
    ReasoningBudget,
    StepKind,
    StepStatus,
    compute_plan_id,
    decode_checkpoint,
    encode_checkpoint,
    load_checkpoint,
    save_checkpoint,
)


def _specs():
    return (
        PlanStepSpec(StepKind.REASON, "analyze the task"),
        PlanStepSpec(
            StepKind.RETRIEVE,
            "retrieve supporting memory",
            dependencies=(1,),
        ),
        PlanStepSpec(
            StepKind.RESPOND,
            "produce verified answer",
            dependencies=(2,),
            requires_verification=True,
            min_confidence=0.8,
        ),
    )


def test_plan_id_is_stable_and_created_time_is_not_identity():
    budget = ReasoningBudget(max_transitions=20)
    first = PlanController.create(
        "solve task",
        _specs(),
        budget=budget,
        created_ns=1,
    ).plan
    second = PlanController.create(
        "solve task",
        _specs(),
        budget=budget,
        created_ns=999,
    ).plan
    assert first.plan_id == second.plan_id
    assert first.plan_id == compute_plan_id(
        "solve task", _specs(), budget
    )


def test_dependencies_execute_in_deterministic_order():
    controller = PlanController.create("goal", _specs())
    assert controller.next_directive().step_id == 1
    controller.begin_step(1)
    controller.complete_step(1, result="analysis", confidence=1.0)
    assert controller.next_directive().step_id == 2
    controller.begin_step(2)
    controller.complete_step(2, result="memory", confidence=1.0)
    assert controller.next_directive().step_id == 3


def test_low_confidence_requires_verification_and_can_retry():
    controller = PlanController.create(
        "goal",
        (
            PlanStepSpec(
                StepKind.REASON,
                "reason carefully",
                min_confidence=0.9,
            ),
        ),
        budget=ReasoningBudget(max_retries_per_step=1),
    )
    controller.begin_step(1)
    controller.complete_step(1, result="draft", confidence=0.5)
    assert controller.plan.step(1).status == StepStatus.WAITING_VERIFICATION
    controller.verify_step(1, passed=False, note="weak evidence")
    assert controller.plan.step(1).status == StepStatus.PENDING
    assert controller.plan.step(1).attempts == 1
    controller.begin_step(1)
    controller.complete_step(1, result="better", confidence=0.95)
    assert controller.plan.status == PlanStatus.COMPLETED


def test_failed_verification_exhausts_retry_budget():
    controller = PlanController.create(
        "goal",
        (
            PlanStepSpec(
                StepKind.VERIFY,
                "verify",
                requires_verification=True,
            ),
        ),
        budget=ReasoningBudget(max_retries_per_step=0),
    )
    controller.begin_step(1)
    controller.complete_step(1, result="candidate", confidence=1.0)
    controller.verify_step(1, passed=False, note="contradiction")
    assert controller.plan.status == PlanStatus.FAILED
    assert controller.plan.step(1).status == StepStatus.FAILED


def test_external_step_never_executes_inside_m5_controller():
    controller = PlanController.create(
        "goal",
        (PlanStepSpec(StepKind.EXTERNAL, "write a file"),),
    )
    directive = controller.begin_step(1)
    assert directive.external_required
    assert controller.plan.status == PlanStatus.WAITING_EXTERNAL
    assert controller.plan.step(1).status == StepStatus.WAITING_EXTERNAL
    with pytest.raises(PlannerError):
        controller.complete_step(1, result="not allowed", confidence=1.0)
    controller.record_external_result(
        1,
        result="authorized result supplied by future tool fabric",
        confidence=1.0,
    )
    assert controller.plan.status == PlanStatus.COMPLETED


def test_transition_budget_exhaustion_fails_closed():
    controller = PlanController.create(
        "goal",
        (PlanStepSpec(StepKind.REASON, "one"),),
        budget=ReasoningBudget(max_transitions=1),
    )
    controller.begin_step(1)
    with pytest.raises(PlannerError, match="budget exhausted"):
        controller.complete_step(1, result="answer", confidence=1.0)
    assert controller.plan.status == PlanStatus.BUDGET_EXHAUSTED
    assert controller.plan.step(1).status == StepStatus.CANCELLED
    assert decode_checkpoint(
        controller.checkpoint_bytes()
    ).status == PlanStatus.BUDGET_EXHAUSTED


def test_pause_requeues_inflight_step_and_checkpoint_roundtrips(tmp_path: Path):
    controller = PlanController.create(
        "goal", _specs(), created_ns=123
    )
    controller.begin_step(1)
    controller.pause("app backgrounded")
    assert controller.plan.status == PlanStatus.PAUSED
    assert controller.plan.step(1).status == StepStatus.PENDING
    blob = controller.checkpoint_bytes()
    restored = decode_checkpoint(blob)
    assert encode_checkpoint(restored) == blob
    path = tmp_path / "plan.vn97plan"
    save_checkpoint(restored, path)
    loaded = load_checkpoint(path)
    assert loaded.plan_id == restored.plan_id
    resumed = PlanController(loaded)
    resumed.resume()
    assert resumed.next_directive().step_id == 1


def test_running_step_checkpoint_is_rejected():
    controller = PlanController.create(
        "goal", (PlanStepSpec(StepKind.REASON, "reason"),)
    )
    controller.begin_step(1)
    with pytest.raises(PlannerError, match="safe point"):
        controller.checkpoint_bytes()


def test_checkpoint_corruption_fails_closed():
    controller = PlanController.create(
        "goal", (PlanStepSpec(StepKind.REASON, "reason"),)
    )
    blob = bytearray(controller.checkpoint_bytes())
    blob[-1] ^= 1
    with pytest.raises(CheckpointCorruptionError, match="SHA-256"):
        decode_checkpoint(bytes(blob))


def test_retrieval_context_is_bounded_and_uses_memory_ids(tmp_path: Path):
    journal = MemoryJournal.create(
        tmp_path / "memory.vn97mem", vector_dim=2
    )
    first = journal.append(
        MemoryKind.EPISODIC,
        "exact",
        source="chat",
        vector=[1.0, 0.0],
        timestamp_ns=1,
    )
    second = journal.append(
        MemoryKind.SEMANTIC,
        "near",
        source="summary",
        vector=[0.9, 0.1],
        timestamp_ns=2,
    )
    journal.append(
        MemoryKind.SEMANTIC,
        "other",
        source="summary",
        vector=[0.0, 1.0],
        timestamp_ns=3,
    )
    controller = PlanController.create(
        "goal",
        (PlanStepSpec(StepKind.RETRIEVE, "retrieve"),),
        budget=ReasoningBudget(
            max_memory_queries=1,
            max_memory_hits=2,
        ),
    )
    context = controller.retrieve_context(
        journal,
        [1.0, 0.0],
        top_k=99,
    )
    assert context.record_ids == (first.record_id, second.record_id)
    assert [item.content for item in context.items] == ["exact", "near"]
    with pytest.raises(PlannerError, match="memory query budget exhausted"):
        controller.retrieve_context(journal, [1.0, 0.0])


def test_invalid_forward_dependency_is_rejected():
    with pytest.raises(ValueError, match="earlier steps"):
        PlanController.create(
            "goal",
            (
                PlanStepSpec(
                    StepKind.REASON,
                    "bad dependency",
                    dependencies=(2,),
                ),
                PlanStepSpec(StepKind.REASON, "second"),
            ),
        )
