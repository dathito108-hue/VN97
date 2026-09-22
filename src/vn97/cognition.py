from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import math
from typing import Protocol, Sequence, runtime_checkable

from .memory import MemoryJournal, MemoryKind
from .planner import (
    MemoryContext,
    MemoryContextItem,
    PlanController,
    PlanStatus,
    PlanStep,
    PlanStepSpec,
    PlannerError,
    ReasoningBudget,
    StepKind,
    StepStatus,
)


class CognitionLoopError(RuntimeError):
    pass


class CognitionContractError(CognitionLoopError):
    pass


@dataclass(frozen=True)
class CognitionLimits:
    max_plan_steps: int = 16
    max_external_steps: int = 4
    max_objective_utf8_bytes: int = 4096
    max_result_utf8_bytes: int = 64 * 1024
    max_note_utf8_bytes: int = 8192
    max_dependency_context_utf8_bytes: int = 32 * 1024
    max_memory_context_utf8_bytes: int = 64 * 1024
    max_cycles_per_run: int = 64
    require_final_response: bool = True

    def __post_init__(self) -> None:
        numeric = (
            self.max_plan_steps,
            self.max_external_steps,
            self.max_objective_utf8_bytes,
            self.max_result_utf8_bytes,
            self.max_note_utf8_bytes,
            self.max_dependency_context_utf8_bytes,
            self.max_memory_context_utf8_bytes,
            self.max_cycles_per_run,
        )
        if any(value <= 0 for value in numeric):
            raise ValueError("all cognition limits must be positive")
        if self.max_external_steps > self.max_plan_steps:
            raise ValueError("max_external_steps cannot exceed max_plan_steps")


@dataclass(frozen=True)
class DependencyResult:
    step_id: int
    kind: StepKind
    objective: str
    result: str
    confidence: float
    evidence_record_ids: tuple[int, ...]


@dataclass(frozen=True)
class PlanDraftRequest:
    goal: str
    max_steps: int
    previous_plan_id: str = ""
    previous_steps: tuple[PlanStepSpec, ...] = ()
    feedback: str = ""


@dataclass(frozen=True)
class PlanDraft:
    steps: tuple[PlanStepSpec, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "steps", tuple(self.steps))
        if not self.steps:
            raise ValueError("plan draft requires at least one step")


@dataclass(frozen=True)
class MemoryQueryRequest:
    plan_id: str
    goal: str
    step_id: int
    objective: str
    attempt: int
    previous_failure: str
    dependencies: tuple[DependencyResult, ...]
    vector_dim: int


@dataclass(frozen=True)
class MemoryQuery:
    vector: tuple[float, ...]
    top_k: int = 5
    kinds: tuple[MemoryKind, ...] | None = None
    semantic_weight: float = 1.0
    recency_weight: float = 0.0
    importance_weight: float = 0.0
    now_ns: int | None = None
    recency_half_life_ns: int = 86_400_000_000_000

    def __post_init__(self) -> None:
        vector = tuple(float(value) for value in self.vector)
        object.__setattr__(self, "vector", vector)
        if not vector or not all(math.isfinite(value) for value in vector):
            raise ValueError("memory query vector must be finite and non-empty")
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.kinds is not None:
            object.__setattr__(
                self,
                "kinds",
                tuple(MemoryKind(kind) for kind in self.kinds),
            )
        weights = (
            self.semantic_weight,
            self.recency_weight,
            self.importance_weight,
        )
        if not all(math.isfinite(weight) and weight >= 0.0 for weight in weights):
            raise ValueError("memory query weights must be finite and non-negative")
        if sum(weights) <= 0.0:
            raise ValueError("at least one memory query weight must be positive")
        if self.now_ns is not None and self.now_ns < 0:
            raise ValueError("now_ns must be non-negative")
        if self.recency_half_life_ns <= 0:
            raise ValueError("recency_half_life_ns must be positive")


@dataclass(frozen=True)
class StepReasoningRequest:
    plan_id: str
    goal: str
    step_id: int
    kind: StepKind
    objective: str
    attempt: int
    previous_failure: str
    dependencies: tuple[DependencyResult, ...]
    memory_context: MemoryContext
    context_truncated: bool


@dataclass(frozen=True)
class StepProposal:
    result: str
    confidence: float

    def __post_init__(self) -> None:
        if not self.result:
            raise ValueError("proposal result must not be empty")
        confidence = float(self.confidence)
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("proposal confidence must be finite and in [0, 1]")
        object.__setattr__(self, "confidence", confidence)


@dataclass(frozen=True)
class VerificationRequest:
    plan_id: str
    goal: str
    step_id: int
    kind: StepKind
    objective: str
    attempt: int
    candidate: str
    confidence: float
    dependencies: tuple[DependencyResult, ...]
    evidence_record_ids: tuple[int, ...]


@dataclass(frozen=True)
class VerificationDecision:
    passed: bool
    note: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.passed, bool):
            raise TypeError("passed must be bool")


@dataclass(frozen=True)
class PlanRevision:
    previous_plan_id: str
    controller: PlanController


class LoopBoundary(IntEnum):
    COMPLETED = 1
    WAITING_EXTERNAL = 2
    PAUSED = 3
    FAILED = 4
    CANCELLED = 5
    BUDGET_EXHAUSTED = 6
    YIELDED = 7
    STALLED = 8


@dataclass(frozen=True)
class LoopRunResult:
    boundary: LoopBoundary
    plan_status: PlanStatus
    cycles: int
    final_response: str = ""


@runtime_checkable
class CognitionBackend(Protocol):
    def propose_plan(self, request: PlanDraftRequest) -> PlanDraft:
        ...

    def memory_query(self, request: MemoryQueryRequest) -> MemoryQuery:
        ...

    def propose_step(self, request: StepReasoningRequest) -> StepProposal:
        ...

    def verify_step(self, request: VerificationRequest) -> VerificationDecision:
        ...


def _utf8_size(text: str) -> int:
    return len(text.encode("utf-8"))


def _truncate_utf8(text: str, max_bytes: int) -> tuple[str, bool]:
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text, False
    return encoded[:max_bytes].decode("utf-8", errors="ignore"), True


class CognitionLoop:
    """Bounded adapter between a cognition backend and the M5A plan state machine."""

    def __init__(
        self,
        backend: CognitionBackend,
        *,
        limits: CognitionLimits | None = None,
    ) -> None:
        self.backend = backend
        self.limits = limits or CognitionLimits()

    def _validate_plan_draft(self, draft: object) -> PlanDraft:
        if not isinstance(draft, PlanDraft):
            raise CognitionContractError("backend propose_plan must return PlanDraft")
        if len(draft.steps) > self.limits.max_plan_steps:
            raise CognitionContractError("backend plan exceeds max_plan_steps")
        external_count = sum(step.kind == StepKind.EXTERNAL for step in draft.steps)
        if external_count > self.limits.max_external_steps:
            raise CognitionContractError("backend plan exceeds max_external_steps")
        for step in draft.steps:
            if _utf8_size(step.objective) > self.limits.max_objective_utf8_bytes:
                raise CognitionContractError("backend step objective exceeds byte limit")
        if (
            self.limits.require_final_response
            and draft.steps[-1].kind != StepKind.RESPOND
        ):
            raise CognitionContractError("backend plan must end with RESPOND")
        return draft

    def build_plan(
        self,
        goal: str,
        *,
        budget: ReasoningBudget | None = None,
        created_ns: int | None = None,
    ) -> PlanController:
        if not goal.strip():
            raise ValueError("goal must not be empty")
        request = PlanDraftRequest(goal=goal, max_steps=self.limits.max_plan_steps)
        try:
            raw = self.backend.propose_plan(request)
        except Exception as exc:
            raise CognitionLoopError(
                f"backend propose_plan raised {type(exc).__name__}"
            ) from exc
        draft = self._validate_plan_draft(raw)
        try:
            return PlanController.create(
                goal,
                draft.steps,
                budget=budget,
                created_ns=created_ns,
            )
        except (ValueError, PlannerError) as exc:
            raise CognitionContractError("backend plan draft is structurally invalid") from exc

    def refine_unstarted_plan(
        self,
        controller: PlanController,
        *,
        feedback: str,
        created_ns: int | None = None,
    ) -> PlanRevision:
        plan = controller.plan
        if (
            plan.status != PlanStatus.READY
            or plan.transitions_used != 0
            or plan.memory_queries_used != 0
            or any(step.status != StepStatus.PENDING for step in plan.steps)
        ):
            raise CognitionLoopError("only an unstarted READY plan can be refined")
        bounded_feedback, _ = _truncate_utf8(
            feedback,
            self.limits.max_note_utf8_bytes,
        )
        request = PlanDraftRequest(
            goal=plan.goal,
            max_steps=self.limits.max_plan_steps,
            previous_plan_id=plan.plan_id,
            previous_steps=tuple(step.spec for step in plan.steps),
            feedback=bounded_feedback,
        )
        try:
            raw = self.backend.propose_plan(request)
        except Exception as exc:
            raise CognitionLoopError(
                f"backend propose_plan raised {type(exc).__name__}"
            ) from exc
        draft = self._validate_plan_draft(raw)
        try:
            replacement = PlanController.create(
                plan.goal,
                draft.steps,
                budget=plan.budget,
                created_ns=created_ns,
            )
        except (ValueError, PlannerError) as exc:
            raise CognitionContractError("refined plan draft is structurally invalid") from exc
        return PlanRevision(plan.plan_id, replacement)

    def _dependency_results(
        self,
        controller: PlanController,
        step: PlanStep,
    ) -> tuple[tuple[DependencyResult, ...], bool]:
        remaining = self.limits.max_dependency_context_utf8_bytes
        truncated = False
        results: list[DependencyResult] = []
        for dependency_id in step.spec.dependencies:
            dependency = controller.plan.step(dependency_id)
            if dependency.status != StepStatus.SUCCEEDED:
                raise CognitionLoopError("dependency is not succeeded")
            objective, objective_cut = _truncate_utf8(
                dependency.spec.objective,
                min(remaining, self.limits.max_objective_utf8_bytes),
            )
            remaining = max(0, remaining - _utf8_size(objective))
            result, result_cut = _truncate_utf8(dependency.result, remaining)
            remaining = max(0, remaining - _utf8_size(result))
            truncated = truncated or objective_cut or result_cut
            confidence = dependency.confidence
            if confidence is None:
                raise CognitionLoopError("succeeded dependency lacks confidence")
            results.append(
                DependencyResult(
                    step_id=dependency.step_id,
                    kind=dependency.spec.kind,
                    objective=objective,
                    result=result,
                    confidence=confidence,
                    evidence_record_ids=dependency.evidence_record_ids,
                )
            )
            if remaining == 0 and dependency_id != step.spec.dependencies[-1]:
                truncated = True
        return tuple(results), truncated

    def _bounded_memory_context(
        self,
        context: MemoryContext,
    ) -> tuple[MemoryContext, bool]:
        remaining = self.limits.max_memory_context_utf8_bytes
        truncated = False
        items: list[MemoryContextItem] = []
        for item in context.items:
            source, source_cut = _truncate_utf8(item.source, remaining)
            remaining = max(0, remaining - _utf8_size(source))
            content, content_cut = _truncate_utf8(item.content, remaining)
            remaining = max(0, remaining - _utf8_size(content))
            truncated = truncated or source_cut or content_cut
            items.append(
                MemoryContextItem(
                    record_id=item.record_id,
                    content=content,
                    source=source,
                    score=item.score,
                    semantic_score=item.semantic_score,
                    recency_score=item.recency_score,
                    importance_score=item.importance_score,
                )
            )
            if remaining == 0 and item is not context.items[-1]:
                truncated = True
        return MemoryContext(tuple(items)), truncated

    def _inherited_evidence(
        self,
        dependencies: Sequence[DependencyResult],
        memory_context: MemoryContext,
    ) -> tuple[int, ...]:
        seen: set[int] = set()
        ordered: list[int] = []
        for dependency in dependencies:
            for record_id in dependency.evidence_record_ids:
                if record_id not in seen:
                    seen.add(record_id)
                    ordered.append(record_id)
        for record_id in memory_context.record_ids:
            if record_id not in seen:
                seen.add(record_id)
                ordered.append(record_id)
        return tuple(ordered)

    def _validate_memory_query(
        self,
        raw: object,
        *,
        vector_dim: int,
    ) -> MemoryQuery:
        if not isinstance(raw, MemoryQuery):
            raise CognitionContractError("backend memory_query must return MemoryQuery")
        if len(raw.vector) != vector_dim:
            raise CognitionContractError("backend memory query vector has wrong dimension")
        return raw

    def _validate_proposal(self, raw: object) -> StepProposal:
        if not isinstance(raw, StepProposal):
            raise CognitionContractError("backend propose_step must return StepProposal")
        if _utf8_size(raw.result) > self.limits.max_result_utf8_bytes:
            raise CognitionContractError("backend proposal result exceeds byte limit")
        return raw

    def _validate_verification(self, raw: object) -> VerificationDecision:
        if not isinstance(raw, VerificationDecision):
            raise CognitionContractError(
                "backend verify_step must return VerificationDecision"
            )
        if _utf8_size(raw.note) > self.limits.max_note_utf8_bytes:
            raise CognitionContractError("backend verification note exceeds byte limit")
        return raw

    def _safe_fail_active(
        self,
        controller: PlanController,
        step_id: int,
        *,
        reason: str,
        retryable: bool,
    ) -> None:
        try:
            controller.fail_step(
                step_id,
                reason=reason,
                retryable=retryable,
            )
        except PlannerError:
            if controller.plan.status != PlanStatus.BUDGET_EXHAUSTED:
                raise

    def _drive_verification(
        self,
        controller: PlanController,
        step: PlanStep,
    ) -> None:
        dependencies, _ = self._dependency_results(controller, step)
        confidence = step.confidence
        if confidence is None:
            raise CognitionLoopError("verification candidate lacks confidence")
        request = VerificationRequest(
            plan_id=controller.plan.plan_id,
            goal=controller.plan.goal,
            step_id=step.step_id,
            kind=step.spec.kind,
            objective=step.spec.objective,
            attempt=step.attempts,
            candidate=step.result,
            confidence=confidence,
            dependencies=dependencies,
            evidence_record_ids=step.evidence_record_ids,
        )
        try:
            raw = self.backend.verify_step(request)
        except Exception as exc:
            note = f"verification backend failure: {type(exc).__name__}"
            controller.verify_step(step.step_id, passed=False, note=note)
            return
        try:
            decision = self._validate_verification(raw)
        except CognitionContractError as exc:
            self._safe_fail_active(
                controller,
                step.step_id,
                reason=str(exc),
                retryable=False,
            )
            return
        controller.verify_step(
            step.step_id,
            passed=decision.passed,
            note=decision.note,
        )

    def _drive_step(
        self,
        controller: PlanController,
        step: PlanStep,
        *,
        journal: MemoryJournal | None,
    ) -> None:
        previous_failure = step.failure_reason
        controller.begin_step(step.step_id)
        if step.spec.kind == StepKind.EXTERNAL:
            return

        dependencies, dependency_truncated = self._dependency_results(
            controller,
            step,
        )
        memory_context = MemoryContext(())
        memory_truncated = False

        if step.spec.kind == StepKind.RETRIEVE:
            if journal is None:
                self._safe_fail_active(
                    controller,
                    step.step_id,
                    reason="retrieval step requires a memory journal",
                    retryable=False,
                )
                return
            query_request = MemoryQueryRequest(
                plan_id=controller.plan.plan_id,
                goal=controller.plan.goal,
                step_id=step.step_id,
                objective=step.spec.objective,
                attempt=step.attempts,
                previous_failure=previous_failure,
                dependencies=dependencies,
                vector_dim=journal.vector_dim,
            )
            try:
                raw_query = self.backend.memory_query(query_request)
                query = self._validate_memory_query(
                    raw_query,
                    vector_dim=journal.vector_dim,
                )
                retrieved = controller.retrieve_context(
                    journal,
                    query.vector,
                    top_k=query.top_k,
                    kinds=query.kinds,
                    semantic_weight=query.semantic_weight,
                    recency_weight=query.recency_weight,
                    importance_weight=query.importance_weight,
                    now_ns=query.now_ns,
                    recency_half_life_ns=query.recency_half_life_ns,
                )
                memory_context, memory_truncated = self._bounded_memory_context(
                    retrieved
                )
            except CognitionContractError as exc:
                self._safe_fail_active(
                    controller,
                    step.step_id,
                    reason=str(exc),
                    retryable=False,
                )
                return
            except Exception as exc:
                self._safe_fail_active(
                    controller,
                    step.step_id,
                    reason=f"memory cognition failure: {type(exc).__name__}",
                    retryable=True,
                )
                return

        request = StepReasoningRequest(
            plan_id=controller.plan.plan_id,
            goal=controller.plan.goal,
            step_id=step.step_id,
            kind=step.spec.kind,
            objective=step.spec.objective,
            attempt=step.attempts,
            previous_failure=previous_failure,
            dependencies=dependencies,
            memory_context=memory_context,
            context_truncated=dependency_truncated or memory_truncated,
        )
        try:
            raw_proposal = self.backend.propose_step(request)
            proposal = self._validate_proposal(raw_proposal)
        except CognitionContractError as exc:
            self._safe_fail_active(
                controller,
                step.step_id,
                reason=str(exc),
                retryable=False,
            )
            return
        except Exception as exc:
            self._safe_fail_active(
                controller,
                step.step_id,
                reason=f"cognition backend failure: {type(exc).__name__}",
                retryable=True,
            )
            return

        evidence = self._inherited_evidence(dependencies, memory_context)
        controller.complete_step(
            step.step_id,
            result=proposal.result,
            confidence=proposal.confidence,
            evidence_record_ids=evidence,
        )

    def final_response(self, controller: PlanController) -> str:
        for step in reversed(controller.plan.steps):
            if (
                step.spec.kind == StepKind.RESPOND
                and step.status == StepStatus.SUCCEEDED
            ):
                return step.result
        return ""

    def _boundary(self, controller: PlanController) -> LoopBoundary:
        status = controller.plan.status
        if status == PlanStatus.COMPLETED:
            return LoopBoundary.COMPLETED
        if status == PlanStatus.WAITING_EXTERNAL:
            return LoopBoundary.WAITING_EXTERNAL
        if status == PlanStatus.PAUSED:
            return LoopBoundary.PAUSED
        if status == PlanStatus.FAILED:
            return LoopBoundary.FAILED
        if status == PlanStatus.CANCELLED:
            return LoopBoundary.CANCELLED
        if status == PlanStatus.BUDGET_EXHAUSTED:
            return LoopBoundary.BUDGET_EXHAUSTED
        return LoopBoundary.STALLED

    def run_until_boundary(
        self,
        controller: PlanController,
        *,
        journal: MemoryJournal | None = None,
        max_cycles: int | None = None,
    ) -> LoopRunResult:
        cycle_limit = (
            self.limits.max_cycles_per_run
            if max_cycles is None
            else max_cycles
        )
        if cycle_limit <= 0:
            raise ValueError("max_cycles must be positive")

        cycles = 0
        while cycles < cycle_limit:
            if controller.plan.is_terminal():
                break
            if controller.plan.status in {
                PlanStatus.PAUSED,
                PlanStatus.WAITING_EXTERNAL,
            }:
                break

            verification_step = next(
                (
                    step
                    for step in controller.plan.steps
                    if step.status == StepStatus.WAITING_VERIFICATION
                ),
                None,
            )
            if verification_step is not None:
                try:
                    self._drive_verification(controller, verification_step)
                except PlannerError:
                    if controller.plan.status != PlanStatus.BUDGET_EXHAUSTED:
                        raise
                cycles += 1
                continue

            directive = controller.next_directive()
            if directive is None:
                break
            step = controller.plan.step(directive.step_id)
            try:
                self._drive_step(
                    controller,
                    step,
                    journal=journal,
                )
            except PlannerError:
                if controller.plan.status != PlanStatus.BUDGET_EXHAUSTED:
                    raise
            cycles += 1

        if cycles >= cycle_limit and not (
            controller.plan.is_terminal()
            or controller.plan.status
            in {PlanStatus.PAUSED, PlanStatus.WAITING_EXTERNAL}
        ):
            boundary = LoopBoundary.YIELDED
        else:
            boundary = self._boundary(controller)
        return LoopRunResult(
            boundary=boundary,
            plan_status=controller.plan.status,
            cycles=cycles,
            final_response=self.final_response(controller),
        )
