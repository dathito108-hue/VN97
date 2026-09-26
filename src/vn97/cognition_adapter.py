from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Protocol, runtime_checkable

import torch

from .cognition import (
    CognitionBackend,
    CognitionContractError,
    DependencyResult,
    MemoryQuery,
    MemoryQueryRequest,
    PlanDraft,
    PlanDraftRequest,
    StepProposal,
    StepReasoningRequest,
    VerificationDecision,
    VerificationRequest,
)
from .external_intent import ExternalIntent, ExternalIntentRequest
from .memory import MemoryKind
from .model import VN97LanguageCore
from .planner import MemoryContext, PlanStepSpec, StepKind
from .tokenizer import VN97Tokenizer


_PROTOCOL = "VN97COG1"


class VN97InferenceError(RuntimeError):
    pass


class VN97InferenceContractError(VN97InferenceError):
    pass


class VN97CognitionOutputError(CognitionContractError):
    pass


@dataclass(frozen=True)
class VN97InferenceLimits:
    max_prompt_tokens: int = 4096
    max_output_utf8_bytes: int = 64 * 1024
    repetition_penalty: float = 1.0
    no_repeat_ngram_size: int = 0

    def __post_init__(self) -> None:
        if self.max_prompt_tokens <= 0:
            raise ValueError("max_prompt_tokens must be positive")
        if self.max_output_utf8_bytes <= 0:
            raise ValueError("max_output_utf8_bytes must be positive")
        if (
            not math.isfinite(self.repetition_penalty)
            or self.repetition_penalty < 1.0
        ):
            raise ValueError(
                "repetition_penalty must be finite and >= 1.0"
            )
        if (
            type(self.no_repeat_ngram_size) is not int
            or self.no_repeat_ngram_size < 0
            or self.no_repeat_ngram_size > 32
        ):
            raise ValueError(
                "no_repeat_ngram_size must be in [0, 32]"
            )


@dataclass(frozen=True)
class VN97CognitionAdapterConfig:
    max_prompt_utf8_bytes: int = 128 * 1024
    plan_new_tokens: int = 768
    memory_query_new_tokens: int = 384
    step_new_tokens: int = 1024
    verification_new_tokens: int = 384
    external_intent_new_tokens: int = 512

    def __post_init__(self) -> None:
        values = (
            self.max_prompt_utf8_bytes,
            self.plan_new_tokens,
            self.memory_query_new_tokens,
            self.step_new_tokens,
            self.verification_new_tokens,
            self.external_intent_new_tokens,
        )
        if any(value <= 0 for value in values):
            raise ValueError("all cognition adapter limits must be positive")


@runtime_checkable
class VN97InferenceEngine(Protocol):
    def generate_text(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
    ) -> str:
        ...

    def embed_text(
        self,
        text: str,
        *,
        vector_dim: int,
    ) -> tuple[float, ...]:
        ...


_CHAT_ROLE_MARKERS = (
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
)
_CHAT_ASSISTANT_MARKER = "<|assistant|>"


def recover_chat_response_text(
    raw_text: str,
) -> str:
    """Recover the assistant payload from legacy textual role-marker leakage.

    P3/P4C historically supervised textual chat role markers. P4D stopped
    supervising them for new examples, but legacy replay can still cause the
    model to emit another assistant boundary before the intended payload.

    This function is deliberately post-generation and weight-neutral. It uses
    the last emitted assistant marker as the response boundary, then truncates
    at any subsequent textual role marker.
    """
    if not isinstance(raw_text, str):
        raise TypeError(
            "raw_text must be a string"
        )

    text = raw_text
    boundary = text.rfind(
        _CHAT_ASSISTANT_MARKER
    )
    if boundary >= 0:
        text = text[
            boundary
            + len(
                _CHAT_ASSISTANT_MARKER
            ):
        ]

    cut_positions = [
        position
        for marker in _CHAT_ROLE_MARKERS
        if (
            position := text.find(
                marker
            )
        )
        >= 0
    ]
    if cut_positions:
        text = text[
            :min(cut_positions)
        ]

    return text.strip()


def _no_repeat_banned_tokens(
    generated: list[int],
    ngram_size: int,
) -> set[int]:
    if ngram_size <= 0:
        return set()
    if ngram_size == 1:
        return set(generated)
    if len(generated) < ngram_size:
        return set()

    prefix_size = ngram_size - 1
    current_prefix = tuple(
        generated[-prefix_size:]
    )
    banned: set[int] = set()
    limit = len(generated) - ngram_size + 1
    for start in range(limit):
        prefix = tuple(
            generated[
                start:
                start + prefix_size
            ]
        )
        if prefix == current_prefix:
            banned.add(
                generated[
                    start + prefix_size
                ]
            )
    return banned


def _select_greedy_token(
    logits: torch.Tensor,
    generated: list[int],
    *,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
) -> int:
    scores = logits[:, -1].clone()

    if repetition_penalty > 1.0 and generated:
        for token_id in set(generated):
            value = scores[0, token_id]
            scores[0, token_id] = torch.where(
                value < 0,
                value * repetition_penalty,
                value / repetition_penalty,
            )

    banned = _no_repeat_banned_tokens(
        generated,
        no_repeat_ngram_size,
    )
    for token_id in banned:
        scores[0, token_id] = -torch.inf

    if not bool(torch.isfinite(scores).any()):
        raise VN97InferenceContractError(
            "generation constraints removed every token"
        )
    return int(
        scores.argmax(dim=-1).item()
    )


class TorchVN97InferenceEngine:
    """Reference VN97 inference engine over VN97TK1 + VN97LanguageCore."""

    def __init__(
        self,
        model: VN97LanguageCore,
        tokenizer: VN97Tokenizer,
        *,
        limits: VN97InferenceLimits | None = None,
    ) -> None:
        if model.config.vocab_size != tokenizer.vocab_size:
            raise VN97InferenceContractError(
                "model vocab_size must exactly match VN97TK1 tokenizer vocab_size"
            )
        self.model = model
        self.tokenizer = tokenizer
        self.limits = limits or VN97InferenceLimits()

    def _device(self) -> torch.device:
        try:
            return next(self.model.parameters()).device
        except StopIteration as exc:
            raise VN97InferenceContractError(
                "VN97LanguageCore has no parameters"
            ) from exc

    def _encode_prompt(self, text: str) -> list[int]:
        token_ids = self.tokenizer.encode(
            text,
            add_bos=True,
            add_text_tag=True,
        )
        if len(token_ids) > self.limits.max_prompt_tokens:
            raise VN97InferenceContractError(
                "prompt exceeds VN97 inference token budget"
            )
        return token_ids

    @torch.inference_mode()
    def generate_text(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
    ) -> str:
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        token_ids = self._encode_prompt(prompt)
        if not token_ids:
            raise VN97InferenceContractError("encoded prompt is empty")

        device = self._device()
        input_ids = torch.tensor(
            [token_ids],
            dtype=torch.long,
            device=device,
        )
        self.model.eval()
        logits, states = self.model(input_ids)
        generated: list[int] = []

        for _ in range(max_new_tokens):
            token = _select_greedy_token(
                logits,
                generated,
                repetition_penalty=
                    self.limits.repetition_penalty,
                no_repeat_ngram_size=
                    self.limits.no_repeat_ngram_size,
            )
            if token == self.tokenizer.eos_id:
                break
            if token < 0 or token >= self.tokenizer.vocab_size:
                raise VN97InferenceContractError(
                    "model generated token outside tokenizer vocabulary"
                )
            generated.append(token)
            next_token = torch.tensor(
                [[token]],
                dtype=torch.long,
                device=device,
            )
            logits, states = self.model(
                next_token,
                states,
            )

        try:
            text = self.tokenizer.decode(
                generated,
                skip_control=True,
                errors="strict",
            )
        except UnicodeDecodeError as exc:
            raise VN97InferenceContractError(
                "model output is not valid UTF-8 transport"
            ) from exc
        if (
            len(text.encode("utf-8"))
            > self.limits.max_output_utf8_bytes
        ):
            raise VN97InferenceContractError(
                "model output exceeds UTF-8 byte budget"
            )
        return text

    @torch.inference_mode()
    def embed_text(
        self,
        text: str,
        *,
        vector_dim: int,
    ) -> tuple[float, ...]:
        if vector_dim != self.model.config.d_model:
            raise VN97InferenceContractError(
                "reference VN97 retrieval embedding requires "
                "journal vector_dim == model d_model"
            )
        token_ids = self._encode_prompt(text)
        device = self._device()
        input_ids = torch.tensor(
            [token_ids],
            dtype=torch.long,
            device=device,
        )
        self.model.eval()
        hidden, _ = self.model.forward_hidden(
            input_ids
        )
        vector = (
            hidden[0, -1]
            .detach()
            .to(dtype=torch.float32, device="cpu")
        )
        if not bool(torch.isfinite(vector).all()):
            raise VN97InferenceError(
                "VN97 retrieval embedding contains non-finite values"
            )
        norm = float(torch.linalg.vector_norm(vector).item())
        if not math.isfinite(norm) or norm == 0.0:
            raise VN97InferenceError(
                "VN97 retrieval embedding has zero/invalid norm"
            )
        return tuple(float(value) for value in vector.tolist())


def _json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _dependency_payload(
    dependencies: tuple[DependencyResult, ...],
) -> list[dict]:
    return [
        {
            "step_id": item.step_id,
            "kind": item.kind.name,
            "objective": item.objective,
            "result": item.result,
            "confidence": item.confidence,
            "evidence_record_ids": list(
                item.evidence_record_ids
            ),
        }
        for item in dependencies
    ]


def _memory_context_payload(
    context: MemoryContext,
) -> list[dict]:
    return [
        {
            "record_id": item.record_id,
            "content": item.content,
            "source": item.source,
            "score": item.score,
            "semantic_score": item.semantic_score,
            "recency_score": item.recency_score,
            "importance_score": item.importance_score,
        }
        for item in context.items
    ]


def _step_spec_payload(
    spec: PlanStepSpec,
) -> dict:
    return {
        "kind": spec.kind.name,
        "objective": spec.objective,
        "dependencies": list(spec.dependencies),
        "requires_verification": spec.requires_verification,
        "min_confidence": spec.min_confidence,
    }


def _strict_object(text: str) -> dict:
    duplicates: list[str] = []

    def pairs_hook(
        pairs: list[tuple[str, object]],
    ) -> dict:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                duplicates.append(key)
            result[key] = value
        return result

    def reject_constant(value: str):
        raise ValueError(
            f"non-finite JSON constant: {value}"
        )

    try:
        value = json.loads(
            text,
            object_pairs_hook=pairs_hook,
            parse_constant=reject_constant,
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise VN97CognitionOutputError(
            "VN97 cognition output is not strict JSON"
        ) from exc
    if duplicates:
        raise VN97CognitionOutputError(
            "VN97 cognition output contains duplicate JSON keys"
        )
    if not isinstance(value, dict):
        raise VN97CognitionOutputError(
            "VN97 cognition output root must be a JSON object"
        )
    return value


def _exact_keys(
    value: dict,
    expected: set[str],
    *,
    label: str,
) -> None:
    keys = set(value)
    if keys != expected:
        missing = sorted(expected - keys)
        extra = sorted(keys - expected)
        raise VN97CognitionOutputError(
            f"{label} keys mismatch; missing={missing}, extra={extra}"
        )


def _strict_int(
    value: object,
    *,
    label: str,
    minimum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise VN97CognitionOutputError(
            f"{label} must be an integer"
        )
    if minimum is not None and value < minimum:
        raise VN97CognitionOutputError(
            f"{label} must be >= {minimum}"
        )
    return value


def _strict_float01(
    value: object,
    *,
    label: str,
) -> float:
    if isinstance(value, bool) or not isinstance(
        value,
        (int, float),
    ):
        raise VN97CognitionOutputError(
            f"{label} must be numeric"
        )
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise VN97CognitionOutputError(
            f"{label} must be finite and in [0, 1]"
        )
    return result


def _strict_nonnegative_float(
    value: object,
    *,
    label: str,
) -> float:
    if isinstance(value, bool) or not isinstance(
        value,
        (int, float),
    ):
        raise VN97CognitionOutputError(
            f"{label} must be numeric"
        )
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise VN97CognitionOutputError(
            f"{label} must be finite and non-negative"
        )
    return result


class VN97CognitionAdapter(CognitionBackend):
    """CognitionBackend implemented through VN97-native text/embedding inference."""

    def __init__(
        self,
        engine: VN97InferenceEngine,
        *,
        config: VN97CognitionAdapterConfig | None = None,
    ) -> None:
        self.engine = engine
        self.config = config or VN97CognitionAdapterConfig()

    def _prompt(
        self,
        *,
        operation: str,
        schema: str,
        request: object,
    ) -> str:
        prompt = (
            f"{_PROTOCOL}\n"
            f"operation={operation}\n"
            "Return exactly one UTF-8 JSON object. "
            "No markdown fences. No prose outside JSON.\n"
            f"schema={schema}\n"
            "request="
            + _json_bytes(request).decode("utf-8")
        )
        if (
            len(prompt.encode("utf-8"))
            > self.config.max_prompt_utf8_bytes
        ):
            raise VN97CognitionOutputError(
                "VN97 cognition prompt exceeds UTF-8 byte budget"
            )
        return prompt

    def _generate_object(
        self,
        *,
        operation: str,
        schema: str,
        request: object,
        max_new_tokens: int,
    ) -> dict:
        prompt = self._prompt(
            operation=operation,
            schema=schema,
            request=request,
        )
        text = self.engine.generate_text(
            prompt,
            max_new_tokens=max_new_tokens,
        )
        return _strict_object(text)

    def propose_plan(
        self,
        request: PlanDraftRequest,
    ) -> PlanDraft:
        request_payload = {
            "goal": request.goal,
            "max_steps": request.max_steps,
            "previous_plan_id": request.previous_plan_id,
            "previous_steps": [
                _step_spec_payload(spec)
                for spec in request.previous_steps
            ],
            "feedback": request.feedback,
        }
        root = self._generate_object(
            operation="plan",
            schema=(
                '{"steps":[{"kind":"REASON|RETRIEVE|VERIFY|RESPOND|EXTERNAL",'
                '"objective":"string","dependencies":[1],'
                '"requires_verification":false,"min_confidence":0.0}]}'
            ),
            request=request_payload,
            max_new_tokens=self.config.plan_new_tokens,
        )
        _exact_keys(
            root,
            {"steps"},
            label="plan",
        )
        raw_steps = root["steps"]
        if not isinstance(raw_steps, list) or not raw_steps:
            raise VN97CognitionOutputError(
                "plan steps must be a non-empty array"
            )

        steps: list[PlanStepSpec] = []
        for index, raw in enumerate(
            raw_steps,
            start=1,
        ):
            if not isinstance(raw, dict):
                raise VN97CognitionOutputError(
                    f"plan step {index} must be an object"
                )
            _exact_keys(
                raw,
                {
                    "kind",
                    "objective",
                    "dependencies",
                    "requires_verification",
                    "min_confidence",
                },
                label=f"plan step {index}",
            )
            kind_raw = raw["kind"]
            if not isinstance(kind_raw, str):
                raise VN97CognitionOutputError(
                    f"plan step {index} kind must be a string"
                )
            try:
                kind = StepKind[kind_raw]
            except KeyError as exc:
                raise VN97CognitionOutputError(
                    f"plan step {index} has unknown kind"
                ) from exc
            objective = raw["objective"]
            if not isinstance(objective, str):
                raise VN97CognitionOutputError(
                    f"plan step {index} objective must be a string"
                )
            dependencies_raw = raw["dependencies"]
            if not isinstance(dependencies_raw, list):
                raise VN97CognitionOutputError(
                    f"plan step {index} dependencies must be an array"
                )
            dependencies = tuple(
                _strict_int(
                    value,
                    label=f"plan step {index} dependency",
                    minimum=1,
                )
                for value in dependencies_raw
            )
            requires_verification = raw[
                "requires_verification"
            ]
            if not isinstance(
                requires_verification,
                bool,
            ):
                raise VN97CognitionOutputError(
                    f"plan step {index} requires_verification must be bool"
                )
            min_confidence = _strict_float01(
                raw["min_confidence"],
                label=f"plan step {index} min_confidence",
            )
            try:
                steps.append(
                    PlanStepSpec(
                        kind=kind,
                        objective=objective,
                        dependencies=dependencies,
                        requires_verification=requires_verification,
                        min_confidence=min_confidence,
                    )
                )
            except ValueError as exc:
                raise VN97CognitionOutputError(
                    f"plan step {index} is invalid"
                ) from exc
        return PlanDraft(tuple(steps))

    def memory_query(
        self,
        request: MemoryQueryRequest,
    ) -> MemoryQuery:
        request_payload = {
            "plan_id": request.plan_id,
            "goal": request.goal,
            "step_id": request.step_id,
            "objective": request.objective,
            "attempt": request.attempt,
            "previous_failure": request.previous_failure,
            "dependencies": _dependency_payload(
                request.dependencies
            ),
            "vector_dim": request.vector_dim,
        }
        root = self._generate_object(
            operation="memory_query",
            schema=(
                '{"query":"string","top_k":5,'
                '"kinds":["EPISODIC","SEMANTIC"]|null,'
                '"semantic_weight":1.0,"recency_weight":0.0,'
                '"importance_weight":0.0,'
                '"recency_half_life_ns":86400000000000}'
            ),
            request=request_payload,
            max_new_tokens=self.config.memory_query_new_tokens,
        )
        _exact_keys(
            root,
            {
                "query",
                "top_k",
                "kinds",
                "semantic_weight",
                "recency_weight",
                "importance_weight",
                "recency_half_life_ns",
            },
            label="memory query",
        )
        query_text = root["query"]
        if not isinstance(query_text, str) or not query_text.strip():
            raise VN97CognitionOutputError(
                "memory query text must be a non-empty string"
            )
        top_k = _strict_int(
            root["top_k"],
            label="memory query top_k",
            minimum=1,
        )
        kinds_raw = root["kinds"]
        kinds: tuple[MemoryKind, ...] | None
        if kinds_raw is None:
            kinds = None
        else:
            if not isinstance(kinds_raw, list):
                raise VN97CognitionOutputError(
                    "memory query kinds must be null or an array"
                )
            parsed_kinds: list[MemoryKind] = []
            for value in kinds_raw:
                if not isinstance(value, str):
                    raise VN97CognitionOutputError(
                        "memory query kind must be a string"
                    )
                try:
                    parsed_kinds.append(
                        MemoryKind[value]
                    )
                except KeyError as exc:
                    raise VN97CognitionOutputError(
                        "memory query contains unknown kind"
                    ) from exc
            kinds = tuple(parsed_kinds)

        semantic_weight = _strict_nonnegative_float(
            root["semantic_weight"],
            label="semantic_weight",
        )
        recency_weight = _strict_nonnegative_float(
            root["recency_weight"],
            label="recency_weight",
        )
        importance_weight = _strict_nonnegative_float(
            root["importance_weight"],
            label="importance_weight",
        )
        if (
            semantic_weight
            + recency_weight
            + importance_weight
            <= 0.0
        ):
            raise VN97CognitionOutputError(
                "memory query weights must contain a positive value"
            )
        half_life = _strict_int(
            root["recency_half_life_ns"],
            label="recency_half_life_ns",
            minimum=1,
        )

        try:
            vector = self.engine.embed_text(
                query_text,
                vector_dim=request.vector_dim,
            )
        except VN97InferenceContractError as exc:
            raise VN97CognitionOutputError(
                str(exc)
            ) from exc
        return MemoryQuery(
            vector=vector,
            top_k=top_k,
            kinds=kinds,
            semantic_weight=semantic_weight,
            recency_weight=recency_weight,
            importance_weight=importance_weight,
            recency_half_life_ns=half_life,
        )

    def propose_step(
        self,
        request: StepReasoningRequest,
    ) -> StepProposal:
        request_payload = {
            "plan_id": request.plan_id,
            "goal": request.goal,
            "step_id": request.step_id,
            "kind": request.kind.name,
            "objective": request.objective,
            "attempt": request.attempt,
            "previous_failure": request.previous_failure,
            "dependencies": _dependency_payload(
                request.dependencies
            ),
            "memory_context": _memory_context_payload(
                request.memory_context
            ),
            "context_truncated": request.context_truncated,
        }
        root = self._generate_object(
            operation="step",
            schema='{"result":"string","confidence":0.0}',
            request=request_payload,
            max_new_tokens=self.config.step_new_tokens,
        )
        _exact_keys(
            root,
            {"result", "confidence"},
            label="step proposal",
        )
        result = root["result"]
        if not isinstance(result, str) or not result:
            raise VN97CognitionOutputError(
                "step proposal result must be a non-empty string"
            )
        confidence = _strict_float01(
            root["confidence"],
            label="step proposal confidence",
        )
        return StepProposal(
            result=result,
            confidence=confidence,
        )

    def verify_step(
        self,
        request: VerificationRequest,
    ) -> VerificationDecision:
        request_payload = {
            "plan_id": request.plan_id,
            "goal": request.goal,
            "step_id": request.step_id,
            "kind": request.kind.name,
            "objective": request.objective,
            "attempt": request.attempt,
            "candidate": request.candidate,
            "confidence": request.confidence,
            "dependencies": _dependency_payload(
                request.dependencies
            ),
            "evidence_record_ids": list(
                request.evidence_record_ids
            ),
        }
        root = self._generate_object(
            operation="verify",
            schema='{"passed":true,"note":"string"}',
            request=request_payload,
            max_new_tokens=self.config.verification_new_tokens,
        )
        _exact_keys(
            root,
            {"passed", "note"},
            label="verification",
        )
        passed = root["passed"]
        note = root["note"]
        if not isinstance(passed, bool):
            raise VN97CognitionOutputError(
                "verification passed must be bool"
            )
        if not isinstance(note, str):
            raise VN97CognitionOutputError(
                "verification note must be a string"
            )
        return VerificationDecision(
            passed=passed,
            note=note,
        )

    def propose_external_intent(
        self,
        request: ExternalIntentRequest,
    ) -> ExternalIntent:
        capability_catalog = [
            {
                "capability_id": item.capability_id,
                "required_scope_keys": list(item.required_scope_keys),
                "optional_scope_keys": list(item.optional_scope_keys),
                "approval_required": item.approval_required,
                "max_payload_utf8_bytes": item.max_payload_utf8_bytes,
            }
            for item in request.capabilities
        ]
        root = self._generate_object(
            operation="external_intent",
            schema=(
                '{"capability_id":"string","scope":{"key":"string"},'
                '"payload":{"key":"json-value"}}'
            ),
            request={
                "plan_id": request.plan_id,
                "goal": request.goal,
                "step_id": request.step_id,
                "objective": request.objective,
                "capabilities": capability_catalog,
            },
            max_new_tokens=self.config.external_intent_new_tokens,
        )
        _exact_keys(
            root,
            {"capability_id", "scope", "payload"},
            label="external intent",
        )
        capability_id = root["capability_id"]
        scope_raw = root["scope"]
        payload = root["payload"]
        if not isinstance(capability_id, str) or not capability_id:
            raise VN97CognitionOutputError(
                "external intent capability_id must be a non-empty string"
            )
        if not isinstance(scope_raw, dict):
            raise VN97CognitionOutputError(
                "external intent scope must be an object"
            )
        scope: dict[str, str] = {}
        for key, value in scope_raw.items():
            if not isinstance(key, str) or not key:
                raise VN97CognitionOutputError(
                    "external intent scope keys must be non-empty strings"
                )
            if not isinstance(value, str) or not value:
                raise VN97CognitionOutputError(
                    "external intent scope values must be non-empty strings"
                )
            scope[key] = value
        if not isinstance(payload, dict):
            raise VN97CognitionOutputError(
                "external intent payload must be an object"
            )
        try:
            payload_json = _json_bytes(payload).decode("utf-8")
            return ExternalIntent(
                capability_id=capability_id,
                scope=tuple(scope.items()),
                payload_json=payload_json,
            )
        except (TypeError, ValueError) as exc:
            raise VN97CognitionOutputError(
                "external intent is invalid"
            ) from exc

