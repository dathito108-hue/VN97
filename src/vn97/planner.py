from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import tempfile
import time
from typing import Iterable, Sequence

from .memory import MemoryJournal, MemoryKind


CHECKPOINT_MAGIC = b"VN97PLN1"
CHECKPOINT_VERSION = 1
_CHECKPOINT_HEADER = struct.Struct("<8sII32s")
MAX_CHECKPOINT_PAYLOAD = 16 << 20


class PlannerError(ValueError):
    pass


class CheckpointCorruptionError(PlannerError):
    pass


class StepKind(IntEnum):
    REASON = 1
    RETRIEVE = 2
    VERIFY = 3
    RESPOND = 4
    EXTERNAL = 5


class StepStatus(IntEnum):
    PENDING = 1
    RUNNING = 2
    WAITING_VERIFICATION = 3
    WAITING_EXTERNAL = 4
    SUCCEEDED = 5
    FAILED = 6
    CANCELLED = 7


class PlanStatus(IntEnum):
    READY = 1
    RUNNING = 2
    WAITING_EXTERNAL = 3
    PAUSED = 4
    COMPLETED = 5
    FAILED = 6
    CANCELLED = 7
    BUDGET_EXHAUSTED = 8


_TERMINAL_PLAN_STATUSES = {
    PlanStatus.COMPLETED,
    PlanStatus.FAILED,
    PlanStatus.CANCELLED,
    PlanStatus.BUDGET_EXHAUSTED,
}


@dataclass(frozen=True)
class ReasoningBudget:
    max_transitions: int = 64
    max_retries_per_step: int = 2
    max_memory_queries: int = 8
    max_memory_hits: int = 8

    def __post_init__(self) -> None:
        if self.max_transitions <= 0:
            raise ValueError("max_transitions must be positive")
        if self.max_retries_per_step < 0:
            raise ValueError("max_retries_per_step must be non-negative")
        if self.max_memory_queries < 0:
            raise ValueError("max_memory_queries must be non-negative")
        if self.max_memory_hits <= 0:
            raise ValueError("max_memory_hits must be positive")


@dataclass(frozen=True)
class PlanStepSpec:
    kind: StepKind
    objective: str
    dependencies: tuple[int, ...] = ()
    requires_verification: bool = False
    min_confidence: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", StepKind(self.kind))
        object.__setattr__(self, "dependencies", tuple(self.dependencies))
        if not self.objective.strip():
            raise ValueError("step objective must not be empty")
        if len(set(self.dependencies)) != len(self.dependencies):
            raise ValueError("step dependencies must be unique")
        if any(dep <= 0 for dep in self.dependencies):
            raise ValueError("step dependencies must be positive IDs")
        if not math.isfinite(self.min_confidence) or not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be finite and in [0, 1]")


@dataclass
class PlanStep:
    step_id: int
    spec: PlanStepSpec
    status: StepStatus = StepStatus.PENDING
    attempts: int = 0
    result: str = ""
    confidence: float | None = None
    evidence_record_ids: tuple[int, ...] = ()
    verification_note: str = ""
    failure_reason: str = ""


@dataclass(frozen=True)
class MemoryContextItem:
    record_id: int
    content: str
    source: str
    score: float
    semantic_score: float
    recency_score: float
    importance_score: float


@dataclass(frozen=True)
class MemoryContext:
    items: tuple[MemoryContextItem, ...]

    @property
    def record_ids(self) -> tuple[int, ...]:
        return tuple(item.record_id for item in self.items)


@dataclass
class Plan:
    plan_id: str
    goal: str
    budget: ReasoningBudget
    steps: list[PlanStep]
    created_ns: int
    status: PlanStatus = PlanStatus.READY
    transitions_used: int = 0
    memory_queries_used: int = 0
    paused_reason: str = ""
    terminal_reason: str = ""

    def step(self, step_id: int) -> PlanStep:
        if step_id <= 0 or step_id > len(self.steps):
            raise PlannerError(f"unknown step_id: {step_id}")
        step = self.steps[step_id - 1]
        if step.step_id != step_id:
            raise PlannerError("plan step ordering invariant is broken")
        return step

    def is_terminal(self) -> bool:
        return self.status in _TERMINAL_PLAN_STATUSES


@dataclass(frozen=True)
class StepDirective:
    step_id: int
    kind: StepKind
    objective: str
    external_required: bool


def _spec_payload(goal: str, specs: Sequence[PlanStepSpec], budget: ReasoningBudget) -> dict:
    return {
        "goal": goal,
        "budget": {
            "max_transitions": budget.max_transitions,
            "max_retries_per_step": budget.max_retries_per_step,
            "max_memory_queries": budget.max_memory_queries,
            "max_memory_hits": budget.max_memory_hits,
        },
        "steps": [
            {
                "kind": int(spec.kind),
                "objective": spec.objective,
                "dependencies": list(spec.dependencies),
                "requires_verification": spec.requires_verification,
                "min_confidence": spec.min_confidence,
            }
            for spec in specs
        ],
    }


def _canonical_json_bytes(value: dict) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def compute_plan_id(goal: str, specs: Sequence[PlanStepSpec], budget: ReasoningBudget) -> str:
    if not goal.strip():
        raise ValueError("goal must not be empty")
    return hashlib.sha256(
        _canonical_json_bytes(_spec_payload(goal, specs, budget))
    ).hexdigest()


def _validate_specs(specs: Sequence[PlanStepSpec]) -> None:
    if not specs:
        raise ValueError("plan requires at least one step")
    for index, spec in enumerate(specs, start=1):
        if any(dep >= index for dep in spec.dependencies):
            raise ValueError(f"step {index} dependencies must reference earlier steps")


def create_plan(
    goal: str,
    specs: Sequence[PlanStepSpec],
    *,
    budget: ReasoningBudget | None = None,
    created_ns: int | None = None,
) -> Plan:
    if not goal.strip():
        raise ValueError("goal must not be empty")
    specs_tuple = tuple(specs)
    _validate_specs(specs_tuple)
    resolved_budget = budget or ReasoningBudget()
    return Plan(
        plan_id=compute_plan_id(goal, specs_tuple, resolved_budget),
        goal=goal,
        budget=resolved_budget,
        steps=[
            PlanStep(step_id=index, spec=spec)
            for index, spec in enumerate(specs_tuple, start=1)
        ],
        created_ns=time.time_ns() if created_ns is None else created_ns,
    )


def _validate_confidence(confidence: float) -> float:
    value = float(confidence)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("confidence must be finite and in [0, 1]")
    return value


class PlanController:
    """Deterministic bounded state machine for reasoning/planning orchestration."""

    def __init__(self, plan: Plan) -> None:
        self.plan = plan
        self._validate_runtime()

    @classmethod
    def create(
        cls,
        goal: str,
        specs: Sequence[PlanStepSpec],
        *,
        budget: ReasoningBudget | None = None,
        created_ns: int | None = None,
    ) -> "PlanController":
        return cls(create_plan(goal, specs, budget=budget, created_ns=created_ns))

    def _validate_runtime(self) -> None:
        plan = self.plan
        if plan.created_ns < 0:
            raise PlannerError("created_ns must be non-negative")
        if plan.transitions_used < 0 or plan.transitions_used > plan.budget.max_transitions:
            raise PlannerError("transitions_used is outside budget")
        if plan.memory_queries_used < 0 or plan.memory_queries_used > plan.budget.max_memory_queries:
            raise PlannerError("memory_queries_used is outside budget")
        specs = tuple(step.spec for step in plan.steps)
        _validate_specs(specs)
        if plan.plan_id != compute_plan_id(plan.goal, specs, plan.budget):
            raise PlannerError("plan_id does not match immutable plan definition")
        if [step.step_id for step in plan.steps] != list(range(1, len(plan.steps) + 1)):
            raise PlannerError("step IDs must be dense and ordered")

        running = sum(step.status == StepStatus.RUNNING for step in plan.steps)
        waiting_external = sum(
            step.status == StepStatus.WAITING_EXTERNAL for step in plan.steps
        )
        if running > 1:
            raise PlannerError("at most one internal step may be running")
        if waiting_external > 1:
            raise PlannerError("at most one external step may be waiting")
        if plan.status == PlanStatus.COMPLETED and not all(
            step.status == StepStatus.SUCCEEDED for step in plan.steps
        ):
            raise PlannerError("completed plan contains unfinished steps")
        if plan.status == PlanStatus.FAILED and not any(
            step.status == StepStatus.FAILED for step in plan.steps
        ):
            raise PlannerError("failed plan has no failed step")
        if plan.status == PlanStatus.WAITING_EXTERNAL and waiting_external != 1:
            raise PlannerError("WAITING_EXTERNAL plan must have exactly one waiting step")
        if plan.status == PlanStatus.PAUSED and running:
            raise PlannerError("paused plan cannot contain a running step")

        for step in plan.steps:
            if step.attempts < 0:
                raise PlannerError("step attempts must be non-negative")
            if step.confidence is not None:
                _validate_confidence(step.confidence)
            if step.status == StepStatus.WAITING_EXTERNAL and step.spec.kind != StepKind.EXTERNAL:
                raise PlannerError("only EXTERNAL steps may wait for external results")
            if step.status in {StepStatus.WAITING_VERIFICATION, StepStatus.SUCCEEDED} and (
                not step.result or step.confidence is None
            ):
                raise PlannerError("completed candidate state requires result and confidence")
            if any(record_id <= 0 for record_id in step.evidence_record_ids):
                raise PlannerError("evidence record IDs must be positive")
            if len(set(step.evidence_record_ids)) != len(step.evidence_record_ids):
                raise PlannerError("evidence record IDs must be unique")

    def _ensure_mutable(self) -> None:
        if self.plan.is_terminal():
            raise PlannerError(f"plan is terminal: {self.plan.status.name}")
        if self.plan.status == PlanStatus.PAUSED:
            raise PlannerError("plan is paused")

    def _consume_transition(self) -> None:
        if self.plan.transitions_used >= self.plan.budget.max_transitions:
            reason = "reasoning transition budget exhausted"
            for step in self.plan.steps:
                if step.status in {
                    StepStatus.RUNNING,
                    StepStatus.WAITING_VERIFICATION,
                    StepStatus.WAITING_EXTERNAL,
                }:
                    step.status = StepStatus.CANCELLED
                    step.failure_reason = reason
            self.plan.status = PlanStatus.BUDGET_EXHAUSTED
            self.plan.terminal_reason = reason
            raise PlannerError(reason)
        self.plan.transitions_used += 1

    def _dependencies_succeeded(self, step: PlanStep) -> bool:
        return all(
            self.plan.step(dep).status == StepStatus.SUCCEEDED
            for dep in step.spec.dependencies
        )

    def _has_active_step(self) -> bool:
        return any(
            step.status in {
                StepStatus.RUNNING,
                StepStatus.WAITING_VERIFICATION,
                StepStatus.WAITING_EXTERNAL,
            }
            for step in self.plan.steps
        )

    def _refresh_plan_status(self) -> None:
        plan = self.plan
        if plan.is_terminal() or plan.status == PlanStatus.PAUSED:
            return
        if any(step.status == StepStatus.FAILED for step in plan.steps):
            plan.status = PlanStatus.FAILED
            if not plan.terminal_reason:
                plan.terminal_reason = "a plan step failed"
        elif all(step.status == StepStatus.SUCCEEDED for step in plan.steps):
            plan.status = PlanStatus.COMPLETED
        elif any(step.status == StepStatus.WAITING_EXTERNAL for step in plan.steps):
            plan.status = PlanStatus.WAITING_EXTERNAL
        else:
            plan.status = PlanStatus.RUNNING if self._has_active_step() else PlanStatus.READY

    def ready_steps(self) -> tuple[PlanStep, ...]:
        if self.plan.is_terminal() or self.plan.status == PlanStatus.PAUSED:
            return ()
        if self._has_active_step():
            return ()
        return tuple(
            step for step in self.plan.steps
            if step.status == StepStatus.PENDING and self._dependencies_succeeded(step)
        )

    def next_directive(self) -> StepDirective | None:
        ready = self.ready_steps()
        if not ready:
            self._refresh_plan_status()
            return None
        step = ready[0]
        return StepDirective(
            step.step_id,
            step.spec.kind,
            step.spec.objective,
            step.spec.kind == StepKind.EXTERNAL,
        )

    def begin_step(self, step_id: int) -> StepDirective:
        self._ensure_mutable()
        if self._has_active_step():
            raise PlannerError("another step is already active")
        step = self.plan.step(step_id)
        if step.status != StepStatus.PENDING:
            raise PlannerError("step is not pending")
        if not self._dependencies_succeeded(step):
            raise PlannerError("step dependencies are not satisfied")
        self._consume_transition()
        step.attempts += 1
        step.result = ""
        step.confidence = None
        step.evidence_record_ids = ()
        step.verification_note = ""
        step.failure_reason = ""
        if step.spec.kind == StepKind.EXTERNAL:
            step.status = StepStatus.WAITING_EXTERNAL
            self.plan.status = PlanStatus.WAITING_EXTERNAL
        else:
            step.status = StepStatus.RUNNING
            self.plan.status = PlanStatus.RUNNING
        return StepDirective(
            step.step_id,
            step.spec.kind,
            step.spec.objective,
            step.spec.kind == StepKind.EXTERNAL,
        )

    def _finish_result(
        self,
        step: PlanStep,
        result: str,
        confidence: float,
        evidence_record_ids: Iterable[int],
    ) -> None:
        if not result:
            raise ValueError("step result must not be empty")
        confidence_value = _validate_confidence(confidence)
        evidence = tuple(int(record_id) for record_id in evidence_record_ids)
        if any(record_id <= 0 for record_id in evidence):
            raise ValueError("evidence record IDs must be positive")
        if len(set(evidence)) != len(evidence):
            raise ValueError("evidence record IDs must be unique")
        step.result = result
        step.confidence = confidence_value
        step.evidence_record_ids = evidence
        step.status = (
            StepStatus.WAITING_VERIFICATION
            if step.spec.requires_verification
            or confidence_value < step.spec.min_confidence
            else StepStatus.SUCCEEDED
        )
        self._refresh_plan_status()

    def complete_step(
        self,
        step_id: int,
        *,
        result: str,
        confidence: float,
        evidence_record_ids: Iterable[int] = (),
    ) -> None:
        self._ensure_mutable()
        step = self.plan.step(step_id)
        if step.status != StepStatus.RUNNING:
            raise PlannerError("step is not running")
        self._consume_transition()
        self._finish_result(step, result, confidence, evidence_record_ids)

    def record_external_result(
        self,
        step_id: int,
        *,
        result: str,
        confidence: float,
        evidence_record_ids: Iterable[int] = (),
    ) -> None:
        self._ensure_mutable()
        step = self.plan.step(step_id)
        if step.status != StepStatus.WAITING_EXTERNAL:
            raise PlannerError("step is not waiting for an external result")
        self._consume_transition()
        self._finish_result(step, result, confidence, evidence_record_ids)

    def _retry_or_fail(self, step: PlanStep, reason: str) -> None:
        retries_used = max(0, step.attempts - 1)
        if retries_used < self.plan.budget.max_retries_per_step:
            step.status = StepStatus.PENDING
            step.result = ""
            step.confidence = None
            step.evidence_record_ids = ()
            step.failure_reason = reason
        else:
            step.status = StepStatus.FAILED
            step.failure_reason = reason
            self.plan.status = PlanStatus.FAILED
            self.plan.terminal_reason = (
                f"step {step.step_id} exhausted retry budget: {reason}"
            )
        self._refresh_plan_status()

    def verify_step(self, step_id: int, *, passed: bool, note: str = "") -> None:
        self._ensure_mutable()
        step = self.plan.step(step_id)
        if step.status != StepStatus.WAITING_VERIFICATION:
            raise PlannerError("step is not waiting for verification")
        self._consume_transition()
        step.verification_note = note
        if passed:
            step.status = StepStatus.SUCCEEDED
            self._refresh_plan_status()
        else:
            self._retry_or_fail(step, note or "verification failed")

    def fail_step(
        self,
        step_id: int,
        *,
        reason: str,
        retryable: bool = True,
    ) -> None:
        self._ensure_mutable()
        if not reason:
            raise ValueError("failure reason must not be empty")
        step = self.plan.step(step_id)
        if step.status not in {StepStatus.RUNNING, StepStatus.WAITING_EXTERNAL}:
            raise PlannerError("step is not active")
        self._consume_transition()
        if retryable:
            self._retry_or_fail(step, reason)
        else:
            step.status = StepStatus.FAILED
            step.failure_reason = reason
            self.plan.status = PlanStatus.FAILED
            self.plan.terminal_reason = f"step {step.step_id} failed: {reason}"

    def retrieve_context(
        self,
        journal: MemoryJournal,
        query_vector: Sequence[float],
        *,
        top_k: int = 5,
        kinds: Iterable[MemoryKind] | None = None,
        semantic_weight: float = 1.0,
        recency_weight: float = 0.0,
        importance_weight: float = 0.0,
        now_ns: int | None = None,
        recency_half_life_ns: int = 86_400_000_000_000,
    ) -> MemoryContext:
        self._ensure_mutable()
        if self.plan.memory_queries_used >= self.plan.budget.max_memory_queries:
            raise PlannerError("memory query budget exhausted")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        self.plan.memory_queries_used += 1
        hits = journal.retrieve(
            query_vector,
            top_k=min(top_k, self.plan.budget.max_memory_hits),
            kinds=kinds,
            semantic_weight=semantic_weight,
            recency_weight=recency_weight,
            importance_weight=importance_weight,
            now_ns=now_ns,
            recency_half_life_ns=recency_half_life_ns,
        )
        return MemoryContext(
            tuple(
                MemoryContextItem(
                    hit.record.record_id,
                    hit.record.content,
                    hit.record.source,
                    hit.score,
                    hit.semantic_score,
                    hit.recency_score,
                    hit.importance_score,
                )
                for hit in hits
            )
        )

    def pause(self, reason: str) -> None:
        if self.plan.is_terminal():
            raise PlannerError("terminal plan cannot be paused")
        if not reason:
            raise ValueError("pause reason must not be empty")
        if self.plan.status == PlanStatus.PAUSED:
            return
        for step in self.plan.steps:
            if step.status == StepStatus.RUNNING:
                step.status = StepStatus.PENDING
                step.result = ""
                step.confidence = None
                step.evidence_record_ids = ()
                step.failure_reason = "interrupted before durable result"
        self.plan.paused_reason = reason
        self.plan.status = PlanStatus.PAUSED

    def resume(self) -> None:
        if self.plan.is_terminal():
            raise PlannerError("terminal plan cannot be resumed")
        if self.plan.status != PlanStatus.PAUSED:
            raise PlannerError("plan is not paused")
        self.plan.paused_reason = ""
        self.plan.status = PlanStatus.READY
        self._refresh_plan_status()

    def cancel(self, reason: str = "cancelled") -> None:
        if self.plan.is_terminal():
            raise PlannerError("plan is already terminal")
        for step in self.plan.steps:
            if step.status not in {StepStatus.SUCCEEDED, StepStatus.FAILED}:
                step.status = StepStatus.CANCELLED
        self.plan.status = PlanStatus.CANCELLED
        self.plan.terminal_reason = reason

    def checkpoint_bytes(self) -> bytes:
        return encode_checkpoint(self.plan)


def _step_to_dict(step: PlanStep) -> dict:
    return {
        "step_id": step.step_id,
        "spec": {
            "kind": int(step.spec.kind),
            "objective": step.spec.objective,
            "dependencies": list(step.spec.dependencies),
            "requires_verification": step.spec.requires_verification,
            "min_confidence": step.spec.min_confidence,
        },
        "status": int(step.status),
        "attempts": step.attempts,
        "result": step.result,
        "confidence": step.confidence,
        "evidence_record_ids": list(step.evidence_record_ids),
        "verification_note": step.verification_note,
        "failure_reason": step.failure_reason,
    }


def _plan_to_payload(plan: Plan) -> dict:
    if any(step.status == StepStatus.RUNNING for step in plan.steps):
        raise PlannerError(
            "checkpoint requires a safe point; pause an in-flight internal step first"
        )
    return {
        "plan_id": plan.plan_id,
        "goal": plan.goal,
        "created_ns": plan.created_ns,
        "status": int(plan.status),
        "transitions_used": plan.transitions_used,
        "memory_queries_used": plan.memory_queries_used,
        "paused_reason": plan.paused_reason,
        "terminal_reason": plan.terminal_reason,
        "budget": {
            "max_transitions": plan.budget.max_transitions,
            "max_retries_per_step": plan.budget.max_retries_per_step,
            "max_memory_queries": plan.budget.max_memory_queries,
            "max_memory_hits": plan.budget.max_memory_hits,
        },
        "steps": [_step_to_dict(step) for step in plan.steps],
    }


def encode_checkpoint(plan: Plan) -> bytes:
    PlanController(plan)._validate_runtime()
    payload = _canonical_json_bytes(_plan_to_payload(plan))
    if len(payload) > MAX_CHECKPOINT_PAYLOAD:
        raise PlannerError("checkpoint payload exceeds format limit")
    return _CHECKPOINT_HEADER.pack(
        CHECKPOINT_MAGIC,
        CHECKPOINT_VERSION,
        len(payload),
        hashlib.sha256(payload).digest(),
    ) + payload


def decode_checkpoint(blob: bytes) -> Plan:
    if len(blob) < _CHECKPOINT_HEADER.size:
        raise CheckpointCorruptionError("checkpoint is shorter than header")
    magic, version, payload_size, expected_digest = _CHECKPOINT_HEADER.unpack_from(blob)
    if magic != CHECKPOINT_MAGIC:
        raise CheckpointCorruptionError("bad VN97PLN1 magic")
    if version != CHECKPOINT_VERSION:
        raise CheckpointCorruptionError(f"unsupported VN97PLN1 version: {version}")
    if payload_size > MAX_CHECKPOINT_PAYLOAD:
        raise CheckpointCorruptionError("checkpoint payload exceeds format limit")
    if len(blob) != _CHECKPOINT_HEADER.size + payload_size:
        raise CheckpointCorruptionError("checkpoint length does not match header")
    payload = blob[_CHECKPOINT_HEADER.size :]
    if hashlib.sha256(payload).digest() != expected_digest:
        raise CheckpointCorruptionError("checkpoint SHA-256 mismatch")
    try:
        raw = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CheckpointCorruptionError("checkpoint payload is not canonical JSON") from exc

    try:
        budget_raw = raw["budget"]
        budget = ReasoningBudget(
            max_transitions=int(budget_raw["max_transitions"]),
            max_retries_per_step=int(budget_raw["max_retries_per_step"]),
            max_memory_queries=int(budget_raw["max_memory_queries"]),
            max_memory_hits=int(budget_raw["max_memory_hits"]),
        )
        steps: list[PlanStep] = []
        for step_raw in raw["steps"]:
            spec_raw = step_raw["spec"]
            spec = PlanStepSpec(
                kind=StepKind(int(spec_raw["kind"])),
                objective=str(spec_raw["objective"]),
                dependencies=tuple(int(v) for v in spec_raw["dependencies"]),
                requires_verification=bool(spec_raw["requires_verification"]),
                min_confidence=float(spec_raw["min_confidence"]),
            )
            confidence_raw = step_raw["confidence"]
            steps.append(
                PlanStep(
                    step_id=int(step_raw["step_id"]),
                    spec=spec,
                    status=StepStatus(int(step_raw["status"])),
                    attempts=int(step_raw["attempts"]),
                    result=str(step_raw["result"]),
                    confidence=None if confidence_raw is None else float(confidence_raw),
                    evidence_record_ids=tuple(
                        int(v) for v in step_raw["evidence_record_ids"]
                    ),
                    verification_note=str(step_raw["verification_note"]),
                    failure_reason=str(step_raw["failure_reason"]),
                )
            )
        plan = Plan(
            plan_id=str(raw["plan_id"]),
            goal=str(raw["goal"]),
            budget=budget,
            steps=steps,
            created_ns=int(raw["created_ns"]),
            status=PlanStatus(int(raw["status"])),
            transitions_used=int(raw["transitions_used"]),
            memory_queries_used=int(raw["memory_queries_used"]),
            paused_reason=str(raw["paused_reason"]),
            terminal_reason=str(raw["terminal_reason"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise CheckpointCorruptionError("checkpoint payload fields are invalid") from exc

    PlanController(plan)
    if _canonical_json_bytes(_plan_to_payload(plan)) != payload:
        raise CheckpointCorruptionError("checkpoint JSON is not canonical")
    return plan


def save_checkpoint(plan: Plan, path: os.PathLike[str] | str) -> None:
    blob = encode_checkpoint(plan)
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{file_path.name}.",
        suffix=".tmp",
        dir=str(file_path.parent),
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(blob)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, file_path)
        try:
            directory_fd = os.open(file_path.parent, os.O_RDONLY)
        except (AttributeError, OSError):
            directory_fd = None
        if directory_fd is not None:
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def load_checkpoint(path: os.PathLike[str] | str) -> Plan:
    return decode_checkpoint(Path(path).read_bytes())
