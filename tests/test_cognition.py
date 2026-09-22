from pathlib import Path

import pytest

from vn97.cognition import (
    CognitionContractError,
    CognitionLimits,
    CognitionLoop,
    CognitionLoopError,
    LoopBoundary,
    MemoryQuery,
    PlanDraft,
    StepProposal,
    VerificationDecision,
)
from vn97.memory import MemoryJournal, MemoryKind
from vn97.planner import (
    PlanController,
    PlanStatus,
    PlanStepSpec,
    ReasoningBudget,
    StepKind,
    StepStatus,
)


class ScriptedBackend:
    def __init__(
        self,
        *,
        drafts=(),
        queries=(),
        proposals=(),
        verifications=(),
    ):
        self.drafts = list(drafts)
        self.queries = list(queries)
        self.proposals = list(proposals)
        self.verifications = list(verifications)
        self.plan_requests = []
        self.query_requests = []
        self.step_requests = []
        self.verification_requests = []

    @staticmethod
    def _next(queue):
        if not queue:
            raise RuntimeError("script exhausted")
        value = queue.pop(0)
        if isinstance(value, Exception):
            raise value
        return value

    def propose_plan(self, request):
        self.plan_requests.append(request)
        return self._next(self.drafts)

    def memory_query(self, request):
        self.query_requests.append(request)
        return self._next(self.queries)

    def propose_step(self, request):
        self.step_requests.append(request)
        return self._next(self.proposals)

    def verify_step(self, request):
        self.verification_requests.append(request)
        return self._next(self.verifications)


def test_build_plan_validates_backend_draft_and_final_response():
    backend = ScriptedBackend(
        drafts=[
            PlanDraft(
                (
                    PlanStepSpec(StepKind.REASON, "analyze"),
                    PlanStepSpec(
                        StepKind.RESPOND,
                        "answer",
                        dependencies=(1,),
                    ),
                )
            )
        ]
    )
    loop = CognitionLoop(backend)
    controller = loop.build_plan(
        "solve",
        budget=ReasoningBudget(max_transitions=20),
        created_ns=1,
    )
    assert controller.plan.steps[-1].spec.kind == StepKind.RESPOND
    assert backend.plan_requests[0].goal == "solve"
    assert backend.plan_requests[0].max_steps == 16


def test_build_plan_rejects_missing_final_response():
    backend = ScriptedBackend(
        drafts=[
            PlanDraft(
                (PlanStepSpec(StepKind.REASON, "only reasoning"),)
            )
        ]
    )
    with pytest.raises(CognitionContractError, match="end with RESPOND"):
        CognitionLoop(backend).build_plan("goal")


def test_refine_unstarted_plan_creates_new_identity_and_preserves_old():
    first_draft = PlanDraft(
        (PlanStepSpec(StepKind.RESPOND, "answer directly"),)
    )
    second_draft = PlanDraft(
        (
            PlanStepSpec(StepKind.REASON, "analyze first"),
            PlanStepSpec(
                StepKind.RESPOND,
                "answer after analysis",
                dependencies=(1,),
            ),
        )
    )
    backend = ScriptedBackend(drafts=[first_draft, second_draft])
    loop = CognitionLoop(backend)
    original = loop.build_plan("goal", created_ns=1)
    old_id = original.plan.plan_id
    revision = loop.refine_unstarted_plan(
        original,
        feedback="add analysis",
        created_ns=2,
    )
    assert revision.previous_plan_id == old_id
    assert original.plan.plan_id == old_id
    assert revision.controller.plan.plan_id != old_id
    assert len(revision.controller.plan.steps) == 2
    assert backend.plan_requests[1].previous_plan_id == old_id


def test_refine_started_plan_is_rejected():
    backend = ScriptedBackend()
    controller = PlanController.create(
        "goal",
        (PlanStepSpec(StepKind.RESPOND, "answer"),),
    )
    controller.begin_step(1)
    with pytest.raises(CognitionLoopError, match="unstarted READY"):
        CognitionLoop(backend).refine_unstarted_plan(
            controller,
            feedback="change it",
        )


def test_reasoning_loop_completes_and_passes_dependency_context():
    backend = ScriptedBackend(
        proposals=[
            StepProposal("analysis result", 0.95),
            StepProposal("final answer", 0.99),
        ]
    )
    controller = PlanController.create(
        "goal",
        (
            PlanStepSpec(StepKind.REASON, "analyze"),
            PlanStepSpec(
                StepKind.RESPOND,
                "answer",
                dependencies=(1,),
            ),
        ),
        budget=ReasoningBudget(max_transitions=20),
    )
    result = CognitionLoop(backend).run_until_boundary(controller)
    assert result.boundary == LoopBoundary.COMPLETED
    assert result.final_response == "final answer"
    assert backend.step_requests[1].dependencies[0].result == "analysis result"
    assert backend.step_requests[1].dependencies[0].confidence == 0.95


def test_retrieval_context_propagates_real_evidence_ids(tmp_path: Path):
    journal = MemoryJournal.create(
        tmp_path / "memory.vn97mem",
        vector_dim=2,
    )
    exact = journal.append(
        MemoryKind.EPISODIC,
        "exact fact",
        source="chat",
        vector=[1.0, 0.0],
        timestamp_ns=1,
    )
    near = journal.append(
        MemoryKind.SEMANTIC,
        "near fact",
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

    backend = ScriptedBackend(
        queries=[
            MemoryQuery(
                (1.0, 0.0),
                top_k=2,
            )
        ],
        proposals=[
            StepProposal("retrieved synthesis", 1.0),
            StepProposal("answer from memory", 1.0),
        ],
    )
    controller = PlanController.create(
        "goal",
        (
            PlanStepSpec(StepKind.RETRIEVE, "find memory"),
            PlanStepSpec(
                StepKind.RESPOND,
                "answer",
                dependencies=(1,),
            ),
        ),
        budget=ReasoningBudget(
            max_transitions=20,
            max_memory_queries=2,
            max_memory_hits=2,
        ),
    )
    result = CognitionLoop(backend).run_until_boundary(
        controller,
        journal=journal,
    )
    assert result.boundary == LoopBoundary.COMPLETED
    assert controller.plan.step(1).evidence_record_ids == (
        exact.record_id,
        near.record_id,
    )
    assert controller.plan.step(2).evidence_record_ids == (
        exact.record_id,
        near.record_id,
    )
    assert backend.step_requests[0].memory_context.record_ids == (
        exact.record_id,
        near.record_id,
    )
    assert backend.step_requests[1].dependencies[0].evidence_record_ids == (
        exact.record_id,
        near.record_id,
    )


def test_verification_failure_retries_with_previous_failure_feedback():
    backend = ScriptedBackend(
        proposals=[
            StepProposal("weak answer", 0.5),
            StepProposal("revised answer", 0.95),
        ],
        verifications=[
            VerificationDecision(False, "missing support"),
            VerificationDecision(True, "supported"),
        ],
    )
    controller = PlanController.create(
        "goal",
        (
            PlanStepSpec(
                StepKind.RESPOND,
                "answer",
                requires_verification=True,
            ),
        ),
        budget=ReasoningBudget(
            max_transitions=20,
            max_retries_per_step=1,
        ),
    )
    result = CognitionLoop(backend).run_until_boundary(controller)
    assert result.boundary == LoopBoundary.COMPLETED
    assert result.final_response == "revised answer"
    assert backend.step_requests[1].previous_failure == "missing support"
    assert controller.plan.step(1).attempts == 2


def test_external_step_stops_before_backend_execution():
    backend = ScriptedBackend()
    controller = PlanController.create(
        "goal",
        (
            PlanStepSpec(StepKind.EXTERNAL, "write file"),
            PlanStepSpec(
                StepKind.RESPOND,
                "report result",
                dependencies=(1,),
            ),
        ),
    )
    result = CognitionLoop(backend).run_until_boundary(controller)
    assert result.boundary == LoopBoundary.WAITING_EXTERNAL
    assert controller.plan.step(1).status == StepStatus.WAITING_EXTERNAL
    assert backend.step_requests == []


def test_backend_exception_consumes_retry_budget_then_recovers():
    backend = ScriptedBackend(
        proposals=[
            RuntimeError("transient"),
            StepProposal("recovered", 1.0),
        ]
    )
    controller = PlanController.create(
        "goal",
        (PlanStepSpec(StepKind.RESPOND, "answer"),),
        budget=ReasoningBudget(
            max_transitions=20,
            max_retries_per_step=1,
        ),
    )
    result = CognitionLoop(backend).run_until_boundary(controller)
    assert result.boundary == LoopBoundary.COMPLETED
    assert controller.plan.step(1).attempts == 2
    assert result.final_response == "recovered"


def test_invalid_backend_proposal_fails_plan_nonretryably():
    backend = ScriptedBackend(proposals=["not a proposal"])
    controller = PlanController.create(
        "goal",
        (PlanStepSpec(StepKind.RESPOND, "answer"),),
        budget=ReasoningBudget(max_retries_per_step=10),
    )
    result = CognitionLoop(backend).run_until_boundary(controller)
    assert result.boundary == LoopBoundary.FAILED
    assert controller.plan.status == PlanStatus.FAILED
    assert controller.plan.step(1).attempts == 1
    assert "must return StepProposal" in controller.plan.step(1).failure_reason


def test_dependency_context_is_utf8_bounded_and_marked_truncated():
    backend = ScriptedBackend(
        proposals=[
            StepProposal("x" * 200, 1.0),
            StepProposal("done", 1.0),
        ]
    )
    loop = CognitionLoop(
        backend,
        limits=CognitionLimits(
            max_dependency_context_utf8_bytes=16,
        ),
    )
    controller = PlanController.create(
        "goal",
        (
            PlanStepSpec(StepKind.REASON, "a"),
            PlanStepSpec(
                StepKind.RESPOND,
                "b",
                dependencies=(1,),
            ),
        ),
    )
    result = loop.run_until_boundary(controller)
    assert result.boundary == LoopBoundary.COMPLETED
    request = backend.step_requests[1]
    assert request.context_truncated
    total = sum(
        len(dep.objective.encode("utf-8"))
        + len(dep.result.encode("utf-8"))
        for dep in request.dependencies
    )
    assert total <= 16


def test_run_yields_only_at_safe_step_boundary():
    backend = ScriptedBackend(
        proposals=[
            StepProposal("first", 1.0),
            StepProposal("second", 1.0),
        ]
    )
    controller = PlanController.create(
        "goal",
        (
            PlanStepSpec(StepKind.REASON, "first"),
            PlanStepSpec(
                StepKind.RESPOND,
                "second",
                dependencies=(1,),
            ),
        ),
    )
    result = CognitionLoop(backend).run_until_boundary(
        controller,
        max_cycles=1,
    )
    assert result.boundary == LoopBoundary.YIELDED
    assert controller.plan.step(1).status == StepStatus.SUCCEEDED
    assert controller.plan.step(2).status == StepStatus.PENDING
    assert all(
        step.status != StepStatus.RUNNING
        for step in controller.plan.steps
    )


def test_verification_backend_failure_is_bounded_by_retry_policy():
    backend = ScriptedBackend(
        proposals=[StepProposal("candidate", 1.0)],
        verifications=[RuntimeError("verifier unavailable")],
    )
    controller = PlanController.create(
        "goal",
        (
            PlanStepSpec(
                StepKind.RESPOND,
                "answer",
                requires_verification=True,
            ),
        ),
        budget=ReasoningBudget(
            max_transitions=20,
            max_retries_per_step=0,
        ),
    )
    result = CognitionLoop(backend).run_until_boundary(controller)
    assert result.boundary == LoopBoundary.FAILED
    assert "verification backend failure" in controller.plan.step(1).failure_reason


def test_retrieval_without_journal_fails_step_closed():
    backend = ScriptedBackend()
    controller = PlanController.create(
        "goal",
        (
            PlanStepSpec(StepKind.RETRIEVE, "retrieve"),
            PlanStepSpec(
                StepKind.RESPOND,
                "answer",
                dependencies=(1,),
            ),
        ),
    )
    result = CognitionLoop(backend).run_until_boundary(controller)
    assert result.boundary == LoopBoundary.FAILED
    assert "requires a memory journal" in controller.plan.step(1).failure_reason
