from __future__ import annotations

from dataclasses import dataclass
import json
import secrets
import threading
import time
from typing import Mapping, Protocol, runtime_checkable

from .authority import (
    ApprovalAuthority,
    ApprovalToken,
    CapabilityScope,
    ExternalActionRequest,
    TypedCapabilityRegistry,
)


class ExternalIntentError(RuntimeError):
    pass


class ExternalIntentContractError(ExternalIntentError):
    pass


class ApprovalSessionError(ExternalIntentError):
    pass


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _strict_object(text: str) -> dict[str, object]:
    duplicates: list[str] = []

    def pairs_hook(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                duplicates.append(key)
            result[key] = value
        return result

    def reject_constant(value: str) -> object:
        raise ValueError(f"invalid JSON constant: {value}")

    try:
        value = json.loads(
            text,
            object_pairs_hook=pairs_hook,
            parse_constant=reject_constant,
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise ExternalIntentContractError("intent payload_json is not strict JSON") from exc
    if duplicates:
        raise ExternalIntentContractError("intent payload_json contains duplicate keys")
    if not isinstance(value, dict):
        raise ExternalIntentContractError("intent payload_json root must be an object")
    return value


@dataclass(frozen=True)
class ExternalIntentLimits:
    max_capabilities: int = 32
    max_scope_entries: int = 16
    max_scope_value_utf8_bytes: int = 4096
    max_payload_utf8_bytes: int = 64 * 1024
    max_goal_utf8_bytes: int = 32 * 1024
    max_objective_utf8_bytes: int = 4096

    def __post_init__(self) -> None:
        values = (
            self.max_capabilities,
            self.max_scope_entries,
            self.max_scope_value_utf8_bytes,
            self.max_payload_utf8_bytes,
            self.max_goal_utf8_bytes,
            self.max_objective_utf8_bytes,
        )
        if any(value <= 0 for value in values):
            raise ValueError("all external intent limits must be positive")


@dataclass(frozen=True)
class ExternalCapabilityView:
    capability_id: str
    required_scope_keys: tuple[str, ...]
    optional_scope_keys: tuple[str, ...]
    approval_required: bool
    max_payload_utf8_bytes: int


@dataclass(frozen=True)
class ExternalIntentRequest:
    plan_id: str
    goal: str
    step_id: int
    objective: str
    capabilities: tuple[ExternalCapabilityView, ...]


@dataclass(frozen=True)
class ExternalIntent:
    capability_id: str
    scope: tuple[tuple[str, str], ...]
    payload_json: str = "{}"

    def __post_init__(self) -> None:
        if not isinstance(self.capability_id, str) or not self.capability_id:
            raise ValueError("capability_id must be non-empty")
        normalized: list[tuple[str, str]] = []
        for item in self.scope:
            if not isinstance(item, tuple) or len(item) != 2:
                raise ValueError("intent scope entries must be key/value tuples")
            key, value = item
            if not isinstance(key, str) or not isinstance(value, str):
                raise ValueError("intent scope keys and values must be strings")
            if not key or not value:
                raise ValueError("intent scope keys and values must be non-empty")
            normalized.append((key, value))
        keys = [key for key, _ in normalized]
        if len(set(keys)) != len(keys):
            raise ValueError("intent scope keys must be unique")
        object.__setattr__(self, "scope", tuple(sorted(normalized)))
        payload = _strict_object(self.payload_json)
        object.__setattr__(self, "payload_json", _canonical_json(payload))

    @classmethod
    def create(
        cls,
        *,
        capability_id: str,
        scope: Mapping[str, str],
        payload: Mapping[str, object] | None = None,
    ) -> "ExternalIntent":
        return cls(
            capability_id=capability_id,
            scope=tuple((key, value) for key, value in scope.items()),
            payload_json=_canonical_json(dict(payload or {})),
        )

    def payload_object(self) -> dict[str, object]:
        return _strict_object(self.payload_json)


@runtime_checkable
class ExternalIntentBackend(Protocol):
    def propose_external_intent(self, request: ExternalIntentRequest) -> ExternalIntent:
        ...


class ExternalIntentBinder:
    """Bind untrusted cognition intent to a trusted immutable M6A action request."""

    def __init__(
        self,
        registry: TypedCapabilityRegistry,
        allowed_capability_ids: tuple[str, ...],
        *,
        limits: ExternalIntentLimits | None = None,
    ) -> None:
        if not registry.sealed:
            raise ValueError("capability registry must be sealed before intent binding")
        self._registry = registry
        self.limits = limits or ExternalIntentLimits()
        ids = tuple(str(value) for value in allowed_capability_ids)
        if not ids:
            raise ValueError("at least one external capability must be allowed")
        if len(ids) > self.limits.max_capabilities:
            raise ValueError("allowed capability catalog exceeds limit")
        if len(set(ids)) != len(ids):
            raise ValueError("allowed capability IDs must be unique")
        views: list[ExternalCapabilityView] = []
        for capability_id in ids:
            descriptor = registry.descriptor(capability_id)
            views.append(
                ExternalCapabilityView(
                    capability_id=descriptor.capability_id,
                    required_scope_keys=tuple(sorted(descriptor.required_scope_keys)),
                    optional_scope_keys=tuple(sorted(descriptor.optional_scope_keys)),
                    approval_required=descriptor.approval_required,
                    max_payload_utf8_bytes=min(
                        descriptor.max_payload_utf8_bytes,
                        self.limits.max_payload_utf8_bytes,
                    ),
                )
            )
        self._views = tuple(views)
        self._allowed = frozenset(ids)

    @property
    def capabilities(self) -> tuple[ExternalCapabilityView, ...]:
        return self._views

    def _waiting_step(self, controller: object) -> object:
        plan = getattr(controller, "plan", None)
        if plan is None or getattr(getattr(plan, "status", None), "name", "") != "WAITING_EXTERNAL":
            raise ExternalIntentError("plan is not WAITING_EXTERNAL")
        waiting = [
            step
            for step in getattr(plan, "steps", ())
            if getattr(getattr(step, "status", None), "name", "") == "WAITING_EXTERNAL"
        ]
        if len(waiting) != 1:
            raise ExternalIntentError("plan must contain exactly one waiting external step")
        step = waiting[0]
        spec = getattr(step, "spec", None)
        if getattr(getattr(spec, "kind", None), "name", "") != "EXTERNAL":
            raise ExternalIntentError("waiting step is not EXTERNAL")
        return step

    def request_for_backend(self, controller: object) -> ExternalIntentRequest:
        step = self._waiting_step(controller)
        plan = controller.plan
        goal = str(plan.goal)
        objective = str(step.spec.objective)
        if len(goal.encode("utf-8")) > self.limits.max_goal_utf8_bytes:
            raise ExternalIntentError("goal exceeds external intent byte limit")
        if len(objective.encode("utf-8")) > self.limits.max_objective_utf8_bytes:
            raise ExternalIntentError("objective exceeds external intent byte limit")
        return ExternalIntentRequest(
            plan_id=str(plan.plan_id),
            goal=goal,
            step_id=int(step.step_id),
            objective=objective,
            capabilities=self._views,
        )

    def _bind_snapshot(
        self,
        backend_request: ExternalIntentRequest,
        intent: object,
    ) -> ExternalActionRequest:
        if not isinstance(intent, ExternalIntent):
            raise ExternalIntentContractError("backend must return ExternalIntent")
        if intent.capability_id not in self._allowed:
            raise ExternalIntentContractError("backend selected capability outside trusted catalog")
        if len(intent.scope) > self.limits.max_scope_entries:
            raise ExternalIntentContractError("intent scope exceeds entry limit")
        for key, value in intent.scope:
            if not key or len(value.encode("utf-8")) > self.limits.max_scope_value_utf8_bytes:
                raise ExternalIntentContractError("intent scope entry is invalid or too large")
        if len(intent.payload_json.encode("utf-8")) > self.limits.max_payload_utf8_bytes:
            raise ExternalIntentContractError("intent payload exceeds byte limit")
        try:
            scope = CapabilityScope(intent.scope)
            request = ExternalActionRequest(
                plan_id=backend_request.plan_id,
                step_id=backend_request.step_id,
                objective=backend_request.objective,
                capability_id=intent.capability_id,
                scope=scope,
                payload_json=intent.payload_json,
            )
            self._registry.validate(request)
        except Exception as exc:
            raise ExternalIntentContractError("intent violates registered capability contract") from exc
        return request

    def bind(self, controller: object, intent: object) -> ExternalActionRequest:
        return self._bind_snapshot(self.request_for_backend(controller), intent)

    def propose_and_bind(
        self,
        controller: object,
        backend: ExternalIntentBackend,
    ) -> ExternalActionRequest:
        request = self.request_for_backend(controller)
        try:
            raw = backend.propose_external_intent(request)
        except Exception as exc:
            raise ExternalIntentError(
                f"external intent backend raised {type(exc).__name__}"
            ) from exc
        current = self.request_for_backend(controller)
        if current != request:
            raise ExternalIntentError(
                "waiting external step changed during intent proposal"
            )
        return self._bind_snapshot(request, raw)


@dataclass(frozen=True)
class ApprovalPrompt:
    prompt_id: str
    principal: str
    request_digest: str
    plan_id: str
    step_id: int
    objective: str
    capability_id: str
    scope: tuple[tuple[str, str], ...]
    payload_json: str
    created_ns: int
    expires_ns: int

    @property
    def presentation_json(self) -> str:
        return _canonical_json(
            {
                "request_digest": self.request_digest,
                "plan_id": self.plan_id,
                "step_id": self.step_id,
                "objective": self.objective,
                "capability_id": self.capability_id,
                "scope": dict(self.scope),
                "payload": _strict_object(self.payload_json),
            }
        )


@dataclass(frozen=True)
class ExternalApprovalLimits:
    max_pending_prompts: int = 32
    max_prompt_ttl_ns: int = 300_000_000_000
    max_approval_ttl_ns: int = 60_000_000_000

    def __post_init__(self) -> None:
        if self.max_pending_prompts <= 0:
            raise ValueError("max_pending_prompts must be positive")
        if self.max_prompt_ttl_ns <= 0 or self.max_approval_ttl_ns <= 0:
            raise ValueError("approval TTL limits must be positive")


@dataclass(frozen=True)
class _PendingApproval:
    prompt: ApprovalPrompt
    request: ExternalActionRequest


class ExternalApprovalCoordinator:
    """One-shot trusted approval sessions bound to an exact M6A request digest."""

    def __init__(
        self,
        authority: ApprovalAuthority,
        *,
        limits: ExternalApprovalLimits | None = None,
    ) -> None:
        self._authority = authority
        self.limits = limits or ExternalApprovalLimits()
        self._pending: dict[str, _PendingApproval] = {}
        self._lock = threading.Lock()

    def _prune_expired_locked(self, now_ns: int) -> None:
        expired = [
            prompt_id
            for prompt_id, pending in self._pending.items()
            if now_ns >= pending.prompt.expires_ns
        ]
        for prompt_id in expired:
            del self._pending[prompt_id]

    def create_prompt(
        self,
        request: ExternalActionRequest,
        *,
        principal: str,
        prompt_ttl_ns: int,
        now_ns: int | None = None,
    ) -> ApprovalPrompt:
        if not isinstance(request, ExternalActionRequest):
            raise TypeError("request must be ExternalActionRequest")
        if not isinstance(principal, str) or not principal or len(principal.encode("utf-8")) > 256:
            raise ValueError("principal must be non-empty and bounded")
        allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._:-/")
        if any(char not in allowed for char in principal):
            raise ValueError("principal contains unsupported characters")
        if prompt_ttl_ns <= 0 or prompt_ttl_ns > self.limits.max_prompt_ttl_ns:
            raise ValueError("prompt_ttl_ns exceeds configured approval limit")
        current = time.time_ns() if now_ns is None else int(now_ns)
        if current < 0:
            raise ValueError("now_ns must be non-negative")
        prompt = ApprovalPrompt(
            prompt_id=secrets.token_hex(16),
            principal=principal,
            request_digest=request.request_digest,
            plan_id=request.plan_id,
            step_id=request.step_id,
            objective=request.objective,
            capability_id=request.capability_id,
            scope=request.scope.entries,
            payload_json=request.payload_json,
            created_ns=current,
            expires_ns=current + int(prompt_ttl_ns),
        )
        with self._lock:
            self._prune_expired_locked(current)
            if len(self._pending) >= self.limits.max_pending_prompts:
                raise ApprovalSessionError("too many pending approval prompts")
            self._pending[prompt.prompt_id] = _PendingApproval(prompt, request)
        return prompt

    def resolve(
        self,
        prompt: ApprovalPrompt,
        *,
        approved: bool,
        approval_ttl_ns: int,
        now_ns: int | None = None,
    ) -> ApprovalToken | None:
        if not isinstance(approved, bool):
            raise TypeError("approved must be bool")
        current = time.time_ns() if now_ns is None else int(now_ns)
        with self._lock:
            pending = self._pending.pop(prompt.prompt_id, None)
        if pending is None or pending.prompt != prompt:
            raise ApprovalSessionError("approval prompt is unknown, stale or already consumed")
        if current < prompt.created_ns or current >= prompt.expires_ns:
            raise ApprovalSessionError("approval prompt is outside its validity window")
        if not approved:
            return None
        if approval_ttl_ns <= 0 or approval_ttl_ns > self.limits.max_approval_ttl_ns:
            raise ValueError("approval_ttl_ns exceeds configured approval limit")
        return self._authority.approve(
            pending.request,
            principal=prompt.principal,
            ttl_ns=approval_ttl_ns,
            now_ns=current,
        )
