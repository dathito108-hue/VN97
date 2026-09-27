from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import re
from typing import Sequence

import torch

from ..tokenizer import VN97Tokenizer
from ..training import (
    VN97ChatMessage,
    encode_chat_completion_prompt,
)
from .evaluation import EvaluationDomain
from .model import VN97R2Model
from .runtime import (
    CognitionMode,
    CognitionRequest,
    VN97R2ExecutionPolicy,
)


_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


@dataclass(frozen=True)
class R2PilotProbe:
    domain: EvaluationDomain
    messages: tuple[VN97ChatMessage, ...]
    expected: str
    requires_external_write: bool = False

    def __post_init__(self) -> None:
        if not self.messages:
            raise ValueError("pilot probe messages must not be empty")
        if self.messages[-1].role == "assistant":
            raise ValueError(
                "pilot probe prompt must end before assistant content"
            )
        if not isinstance(self.expected, str) or not self.expected.strip():
            raise ValueError("pilot probe expected text must be non-empty")
        if (
            self.requires_external_write
            and self.domain is not EvaluationDomain.TOOL_ACTION
        ):
            raise ValueError(
                "external-write pilot probes must use tool_action domain"
            )


@dataclass(frozen=True)
class R2PilotProbeResult:
    index: int
    domain: str
    generated_sha256: str
    expected_sha256: str
    valid_utf8: bool
    nonempty: bool
    lexical_target_f1: float
    structured_valid: bool
    exact_match: bool
    instruction_following: bool
    tool_call_correct: bool
    authority_route_correct: bool | None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class R2PilotProbeMetrics:
    tasks: int
    natural_language_tasks: int
    structured_tasks: int
    tool_action_tasks: int
    external_write_tasks: int
    generation_success_rate: float
    lexical_target_f1: float
    instruction_following_rate: float
    structured_valid_rate: float
    tool_call_correct_rate: float
    authority_route_correct_rate: float
    protocol_exact_match_rate: float

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class R2PilotGateThresholds:
    generation_success_rate: float = 0.80
    lexical_target_f1: float = 0.50
    instruction_following_rate: float = 0.60
    structured_valid_rate: float = 0.90
    tool_call_correct_rate: float = 0.80
    authority_route_correct_rate: float = 1.0
    protocol_exact_match_rate: float = 0.80


@dataclass(frozen=True)
class R2PilotGateDecision:
    passed: bool
    reasons: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _normalize_text(value: str) -> str:
    return " ".join(value.casefold().split())


def lexical_token_f1(expected: str, generated: str) -> float:
    expected_tokens = _TOKEN_RE.findall(_normalize_text(expected))
    generated_tokens = _TOKEN_RE.findall(_normalize_text(generated))
    if not expected_tokens and not generated_tokens:
        return 1.0
    if not expected_tokens or not generated_tokens:
        return 0.0

    expected_counts: dict[str, int] = {}
    generated_counts: dict[str, int] = {}
    for token in expected_tokens:
        expected_counts[token] = expected_counts.get(token, 0) + 1
    for token in generated_tokens:
        generated_counts[token] = generated_counts.get(token, 0) + 1

    overlap = sum(
        min(count, generated_counts.get(token, 0))
        for token, count in expected_counts.items()
    )
    if overlap <= 0:
        return 0.0
    precision = overlap / len(generated_tokens)
    recall = overlap / len(expected_tokens)
    return 2.0 * precision * recall / (precision + recall)


def _canonical_json_text(value: str) -> str | None:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return None
    return json.dumps(
        parsed,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@torch.inference_mode()
def evaluate_pilot_probes(
    model: VN97R2Model,
    tokenizer: VN97Tokenizer,
    probes: Sequence[R2PilotProbe],
    *,
    max_new_tokens: int = 96,
    profile: str = "deep",
    device: str | torch.device = "cpu",
) -> tuple[R2PilotProbeMetrics, tuple[R2PilotProbeResult, ...]]:
    if not probes:
        raise ValueError("pilot probes must not be empty")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")

    resolved = torch.device(device)
    model.to(resolved)
    model.eval()
    policy = VN97R2ExecutionPolicy(model.config)

    results: list[R2PilotProbeResult] = []
    generation_success = 0
    instruction_following = 0
    lexical_sum = 0.0

    natural_count = 0
    structured_count = 0
    tool_count = 0
    external_write_count = 0
    structured_valid_count = 0
    protocol_exact_count = 0
    tool_correct_count = 0
    authority_correct_count = 0

    for index, probe in enumerate(probes):
        prompt_ids = encode_chat_completion_prompt(
            tokenizer,
            probe.messages,
        )
        generated_ids = model.generate_greedy(
            prompt_ids,
            max_new_tokens=max_new_tokens,
            eos_id=tokenizer.eos_id,
            profile=profile,
            device=resolved,
        )
        payload: list[int] = []
        for token_id in generated_ids:
            if token_id == tokenizer.eos_id:
                break
            payload.append(token_id)

        valid_utf8 = True
        try:
            generated_text = tokenizer.decode(
                payload,
                errors="strict",
            )
        except (UnicodeDecodeError, ValueError):
            valid_utf8 = False
            try:
                generated_text = tokenizer.decode(
                    payload,
                    errors="replace",
                )
            except ValueError:
                generated_text = ""

        normalized_generated = _normalize_text(generated_text)
        nonempty = bool(normalized_generated)
        success = valid_utf8 and nonempty
        generation_success += int(success)

        lexical = lexical_token_f1(
            probe.expected,
            generated_text,
        )
        lexical_sum += lexical

        structured_valid = False
        exact_match = False
        tool_correct = False
        authority_correct: bool | None = None

        if probe.domain is EvaluationDomain.NATURAL_LANGUAGE:
            natural_count += 1
            follows = success and lexical >= 0.50
        else:
            structured_count += 1
            expected_json = _canonical_json_text(probe.expected)
            generated_json = _canonical_json_text(generated_text)
            if expected_json is None:
                raise ValueError(
                    "structured/tool pilot expected output must be valid JSON"
                )
            structured_valid = generated_json is not None
            structured_valid_count += int(structured_valid)
            exact_match = (
                generated_json is not None
                and generated_json == expected_json
            )
            protocol_exact_count += int(exact_match)
            follows = success and exact_match

            if probe.domain is EvaluationDomain.TOOL_ACTION:
                tool_count += 1
                tool_correct = exact_match
                tool_correct_count += int(tool_correct)

        if probe.requires_external_write:
            external_write_count += 1
            decision = policy.decide(
                CognitionRequest(
                    requires_external_write=True,
                )
            )
            authority_correct = (
                decision.mode is CognitionMode.DEEP
                and decision.active_layers == model.config.n_layers
            )
            authority_correct_count += int(authority_correct)

        instruction_following += int(follows)
        results.append(
            R2PilotProbeResult(
                index=index,
                domain=probe.domain.value,
                generated_sha256=_sha_text(generated_text),
                expected_sha256=_sha_text(probe.expected),
                valid_utf8=valid_utf8,
                nonempty=nonempty,
                lexical_target_f1=lexical,
                structured_valid=structured_valid,
                exact_match=exact_match,
                instruction_following=follows,
                tool_call_correct=tool_correct,
                authority_route_correct=authority_correct,
            )
        )

    tasks = len(probes)
    metrics = R2PilotProbeMetrics(
        tasks=tasks,
        natural_language_tasks=natural_count,
        structured_tasks=structured_count,
        tool_action_tasks=tool_count,
        external_write_tasks=external_write_count,
        generation_success_rate=generation_success / tasks,
        lexical_target_f1=lexical_sum / tasks,
        instruction_following_rate=instruction_following / tasks,
        structured_valid_rate=(
            structured_valid_count / structured_count
            if structured_count
            else 1.0
        ),
        tool_call_correct_rate=(
            tool_correct_count / tool_count
            if tool_count
            else 0.0
        ),
        authority_route_correct_rate=(
            authority_correct_count / external_write_count
            if external_write_count
            else 0.0
        ),
        protocol_exact_match_rate=(
            protocol_exact_count / structured_count
            if structured_count
            else 1.0
        ),
    )
    return metrics, tuple(results)


def evaluate_pilot_gate(
    metrics: R2PilotProbeMetrics,
    *,
    thresholds: R2PilotGateThresholds | None = None,
) -> R2PilotGateDecision:
    t = thresholds or R2PilotGateThresholds()
    reasons: list[str] = []

    if metrics.natural_language_tasks <= 0:
        reasons.append("natural_language_probe_missing")
    if metrics.tool_action_tasks <= 0:
        reasons.append("tool_action_probe_missing")
    if metrics.external_write_tasks <= 0:
        reasons.append("external_write_probe_missing")
    if metrics.generation_success_rate < t.generation_success_rate:
        reasons.append("generation_success_below_pilot_threshold")
    if metrics.lexical_target_f1 < t.lexical_target_f1:
        reasons.append("target_alignment_below_pilot_threshold")
    if metrics.instruction_following_rate < t.instruction_following_rate:
        reasons.append("instruction_following_below_pilot_threshold")
    if metrics.structured_valid_rate < t.structured_valid_rate:
        reasons.append("structured_validity_below_pilot_threshold")
    if metrics.tool_call_correct_rate < t.tool_call_correct_rate:
        reasons.append("tool_call_correctness_below_pilot_threshold")
    if (
        metrics.authority_route_correct_rate
        < t.authority_route_correct_rate
    ):
        reasons.append("authority_route_below_pilot_threshold")
    if (
        metrics.protocol_exact_match_rate
        < t.protocol_exact_match_rate
    ):
        reasons.append("protocol_exact_match_below_pilot_threshold")

    return R2PilotGateDecision(
        passed=not reasons,
        reasons=tuple(reasons),
    )
