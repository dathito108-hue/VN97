from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch

from .cognition_adapter import (
    TorchVN97InferenceEngine,
    VN97InferenceLimits,
)
from .deployment_checkpoint import (
    VN97LoadedDeploymentCheckpoint,
    load_deployment_checkpoint_file,
)
from .p3_language_campaign import (
    CANDIDATES,
    candidate_ids,
    profile_sha256,
)
from .tokenizer import VN97Tokenizer, VN97TokenizerPackage
from .training_cli import _read_bounded_regular_file


P4_PROFILE_ID = "vn97-p4-task-level-evaluation-v1"
P4_TASK_SCHEMA = "VN97P4TASK1"
P4_REPORT_SCHEMA = "VN97P4EVAL1"

P4_CATEGORIES = (
    "instruction_following",
    "reasoning_planning",
    "memory_use",
    "structured_cognition",
    "tool_intent",
    "authority_behavior",
)

P4_SCORING_KINDS = (
    "exact_text",
    "contains_all",
    "json_exact",
)

_CHAT_USER_MARKER = "\n<|user|>\n"
_CHAT_ASSISTANT_MARKER = "\n<|assistant|>\n"

_P3_FINAL_FILES = {
    "SHA256SUMS",
    "campaign-report.json",
    "model.vn97ck1",
    "model.vn97mi1",
    "p3-run.vn97p3run1.json",
    "tokenizer.vn97tk1",
}
_P3_HASHED_FILES = _P3_FINAL_FILES - {"SHA256SUMS"}
_MAX_JSON_BYTES = 16 * 1024 * 1024
_MAX_SUITE_BYTES = 32 * 1024 * 1024
_MAX_MODEL_BYTES = 512 * 1024 * 1024
_MAX_TASKS = 10_000
_MAX_PROMPT_BYTES = 128 * 1024
_MAX_EXPECTED_TEXT_BYTES = 64 * 1024


class VN97P4EvaluationError(RuntimeError):
    pass


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _strict_json_value(
    data: bytes,
    *,
    label: str,
    require_canonical: bool = True,
) -> Any:
    duplicates: list[str] = []

    def pairs_hook(
        pairs: list[tuple[str, object]],
    ) -> dict[str, object]:
        out: dict[str, object] = {}
        for key, value in pairs:
            if key in out:
                duplicates.append(key)
            out[key] = value
        return out

    try:
        text = data.decode("utf-8", errors="strict")
        trailing_newline = text.endswith("\n")
        body = text[:-1] if trailing_newline else text
        value = json.loads(
            body,
            object_pairs_hook=pairs_hook,
            parse_constant=lambda raw: (
                _ for _ in ()
            ).throw(ValueError(raw)),
        )
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        ValueError,
    ) as exc:
        raise VN97P4EvaluationError(
            f"{label} must be strict UTF-8 JSON"
        ) from exc

    if duplicates:
        raise VN97P4EvaluationError(
            f"{label} contains duplicate JSON keys"
        )
    if require_canonical:
        expected = _canonical_json(value).decode("utf-8")
        if trailing_newline:
            expected += "\n"
        if expected != text:
            raise VN97P4EvaluationError(
                f"{label} must use canonical JSON"
            )
    return value


def _require_sha256(
    value: object,
    *,
    label: str,
) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(
            ch not in "0123456789abcdef"
            for ch in value
        )
    ):
        raise VN97P4EvaluationError(
            f"{label} must be lowercase SHA-256"
        )
    return value


def _bounded_file(
    path: Path,
    *,
    max_bytes: int,
) -> bytes:
    try:
        return _read_bounded_regular_file(
            path,
            max_bytes=max_bytes,
        )
    except (OSError, ValueError) as exc:
        raise VN97P4EvaluationError(
            f"could not safely read {path.name}"
        ) from exc


@dataclass(frozen=True)
class VN97P3FinalArtifact:
    root: Path
    p3_run_sha256: str
    corpus_manifest_id: str
    corpus_manifest_sha256: str
    selected_candidate_id: str
    checkpoint_sha256: str
    tokenizer_sha256: str
    model_image_sha256: str
    model_image_bytes: int
    checkpoint: VN97LoadedDeploymentCheckpoint
    tokenizer_package: VN97TokenizerPackage


def _parse_sha256sums(
    data: bytes,
) -> dict[str, str]:
    try:
        text = data.decode("ascii", errors="strict")
    except UnicodeDecodeError as exc:
        raise VN97P4EvaluationError(
            "SHA256SUMS must be ASCII"
        ) from exc
    if not text.endswith("\n"):
        raise VN97P4EvaluationError(
            "SHA256SUMS must end with a newline"
        )

    result: dict[str, str] = {}
    for line in text.splitlines():
        if len(line) < 67:
            raise VN97P4EvaluationError(
                "SHA256SUMS contains a malformed line"
            )
        digest = line[:64]
        rest = line[64:]
        if not rest.startswith("  "):
            raise VN97P4EvaluationError(
                "SHA256SUMS must use sha256sum text format"
            )
        filename = rest[2:]
        _require_sha256(
            digest,
            label="SHA256SUMS digest",
        )
        if (
            not filename
            or "/" in filename
            or "\\" in filename
            or filename in result
        ):
            raise VN97P4EvaluationError(
                "SHA256SUMS contains an invalid filename"
            )
        result[filename] = digest

    if set(result) != _P3_HASHED_FILES:
        raise VN97P4EvaluationError(
            "SHA256SUMS file set does not match canonical P3 final bundle"
        )
    return result


def verify_p3_final_artifact(
    root: Path,
) -> VN97P3FinalArtifact:
    if root.is_symlink():
        raise VN97P4EvaluationError(
            "P3 final root must not be a symlink"
        )
    try:
        resolved = root.resolve(strict=True)
    except FileNotFoundError as exc:
        raise VN97P4EvaluationError(
            "P3 final root does not exist"
        ) from exc
    if not resolved.is_dir():
        raise VN97P4EvaluationError(
            "P3 final root must be a directory"
        )

    names = {
        item.name
        for item in resolved.iterdir()
    }
    if names != _P3_FINAL_FILES:
        raise VN97P4EvaluationError(
            "P3 final directory has unexpected file set"
        )
    if any(
        (resolved / name).is_symlink()
        for name in names
    ):
        raise VN97P4EvaluationError(
            "P3 final directory must not contain symlinks"
        )

    sums = _parse_sha256sums(
        _bounded_file(
            resolved / "SHA256SUMS",
            max_bytes=64 * 1024,
        )
    )

    raw_files: dict[str, bytes] = {}
    for name in sorted(_P3_HASHED_FILES):
        limit = (
            _MAX_MODEL_BYTES
            if name in {
                "model.vn97ck1",
                "model.vn97mi1",
            }
            else _MAX_JSON_BYTES
        )
        data = _bounded_file(
            resolved / name,
            max_bytes=limit,
        )
        actual = _sha256(data)
        if actual != sums[name]:
            raise VN97P4EvaluationError(
                f"P3 final SHA256SUMS mismatch: {name}"
            )
        raw_files[name] = data

    p3_run = _strict_json_value(
        raw_files[
            "p3-run.vn97p3run1.json"
        ],
        label="VN97P3RUN1",
    )
    if not isinstance(p3_run, dict):
        raise VN97P4EvaluationError(
            "VN97P3RUN1 root must be an object"
        )
    expected_run_keys = {
        "campaign_report_sha256",
        "checkpoint_sha256",
        "corpus_manifest_id",
        "corpus_manifest_sha256",
        "device",
        "model_image_bytes",
        "model_image_sha256",
        "p2_baseline_run_identity",
        "profile_sha256",
        "release_mean_loss",
        "release_top1_accuracy",
        "schema",
        "selected_candidate_id",
        "tokenizer_sha256",
        "validation_mean_loss",
        "validation_top1_accuracy",
    }
    if set(p3_run) != expected_run_keys:
        raise VN97P4EvaluationError(
            "VN97P3RUN1 key set is invalid"
        )
    if p3_run.get("schema") != "VN97P3RUN1":
        raise VN97P4EvaluationError(
            "P3 run schema mismatch"
        )
    if p3_run.get("profile_sha256") != profile_sha256():
        raise VN97P4EvaluationError(
            "P3 run profile identity mismatch"
        )

    selected_candidate_id = p3_run.get(
        "selected_candidate_id"
    )
    if (
        not isinstance(selected_candidate_id, str)
        or len(selected_candidate_id) != 16
        or any(
            ch not in "0123456789abcdef"
            for ch in selected_candidate_id
        )
    ):
        raise VN97P4EvaluationError(
            "P3 selected candidate ID is invalid"
        )

    corpus_manifest_id = _require_sha256(
        p3_run.get(
            "corpus_manifest_id"
        ),
        label="P3 corpus manifest ID",
    )
    corpus_manifest_sha256 = _require_sha256(
        p3_run.get(
            "corpus_manifest_sha256"
        ),
        label="P3 corpus manifest SHA-256",
    )
    campaign_sha = _require_sha256(
        p3_run.get(
            "campaign_report_sha256"
        ),
        label="P3 campaign report SHA-256",
    )
    checkpoint_sha = _require_sha256(
        p3_run.get("checkpoint_sha256"),
        label="P3 checkpoint SHA-256",
    )
    tokenizer_sha = _require_sha256(
        p3_run.get("tokenizer_sha256"),
        label="P3 tokenizer SHA-256",
    )
    model_image_sha = _require_sha256(
        p3_run.get("model_image_sha256"),
        label="P3 model-image SHA-256",
    )

    if (
        _sha256(
            raw_files["campaign-report.json"]
        )
        != campaign_sha
    ):
        raise VN97P4EvaluationError(
            "P3 campaign report does not match VN97P3RUN1"
        )
    if (
        _sha256(
            raw_files["tokenizer.vn97tk1"]
        )
        != tokenizer_sha
    ):
        raise VN97P4EvaluationError(
            "P3 tokenizer does not match VN97P3RUN1"
        )
    if (
        _sha256(
            raw_files["model.vn97mi1"]
        )
        != model_image_sha
    ):
        raise VN97P4EvaluationError(
            "P3 model image does not match VN97P3RUN1"
        )

    image_bytes = p3_run.get(
        "model_image_bytes"
    )
    if (
        type(image_bytes) is not int
        or image_bytes <= 0
        or image_bytes
        != len(
            raw_files[
                "model.vn97mi1"
            ]
        )
    ):
        raise VN97P4EvaluationError(
            "P3 model image byte count mismatch"
        )

    campaign = _strict_json_value(
        raw_files["campaign-report.json"],
        label="VN97CAMP2",
    )
    if (
        not isinstance(campaign, dict)
        or campaign.get("schema")
        != "VN97CAMP2"
        or campaign.get(
            "selected_candidate_id"
        )
        != selected_candidate_id
        or campaign.get(
            "selected_checkpoint_sha256"
        )
        != checkpoint_sha
        or campaign.get(
            "tokenizer_sha256"
        )
        != tokenizer_sha
    ):
        raise VN97P4EvaluationError(
            "VN97CAMP2 does not bind the same P3 winner"
        )

    checkpoint = (
        load_deployment_checkpoint_file(
            resolved / "model.vn97ck1"
        )
    )
    if (
        checkpoint.checkpoint_sha256
        != checkpoint_sha
    ):
        raise VN97P4EvaluationError(
            "P3 checkpoint does not match VN97P3RUN1"
        )

    try:
        tokenizer_package = (
            VN97TokenizerPackage.from_bytes(
                raw_files[
                    "tokenizer.vn97tk1"
                ]
            )
        )
    except (TypeError, ValueError) as exc:
        raise VN97P4EvaluationError(
            "P3 tokenizer package is invalid"
        ) from exc

    if (
        checkpoint.config.vocab_size
        != tokenizer_package.vocab_size
    ):
        raise VN97P4EvaluationError(
            "P3 checkpoint/tokenizer vocabulary mismatch"
        )

    ids = candidate_ids()
    if selected_candidate_id not in ids:
        raise VN97P4EvaluationError(
            "P3 selected candidate is outside the frozen candidate set"
        )
    candidate = CANDIDATES[
        ids.index(
            selected_candidate_id
        )
    ]
    if (
        checkpoint.config.d_model
        != int(candidate["d_model"])
        or checkpoint.config.n_layers
        != int(candidate["n_layers"])
        or checkpoint.config.d_state
        != int(candidate["d_state"])
        or checkpoint.config.embedding_rank
        != candidate["embedding_rank"]
    ):
        raise VN97P4EvaluationError(
            "P3 checkpoint geometry does not match the selected candidate"
        )

    for field in (
        "validation_mean_loss",
        "validation_top1_accuracy",
        "release_mean_loss",
        "release_top1_accuracy",
    ):
        value = p3_run.get(field)
        if (
            isinstance(value, bool)
            or not isinstance(
                value,
                (int, float),
            )
            or not math.isfinite(
                float(value)
            )
        ):
            raise VN97P4EvaluationError(
                f"P3 run {field} is invalid"
            )
    for field in (
        "validation_top1_accuracy",
        "release_top1_accuracy",
    ):
        value = float(
            p3_run[field]
        )
        if not 0.0 <= value <= 1.0:
            raise VN97P4EvaluationError(
                f"P3 run {field} must be in [0, 1]"
            )

    return VN97P3FinalArtifact(
        root=resolved,
        p3_run_sha256=_sha256(
            raw_files[
                "p3-run.vn97p3run1.json"
            ]
        ),
        corpus_manifest_id=
            corpus_manifest_id,
        corpus_manifest_sha256=
            corpus_manifest_sha256,
        selected_candidate_id=
            selected_candidate_id,
        checkpoint_sha256=
            checkpoint_sha,
        tokenizer_sha256=
            tokenizer_sha,
        model_image_sha256=
            model_image_sha,
        model_image_bytes=
            image_bytes,
        checkpoint=checkpoint,
        tokenizer_package=
            tokenizer_package,
    )


@dataclass(frozen=True)
class VN97P4Task:
    task_id: str
    category: str
    prompt: str
    max_new_tokens: int
    scoring_kind: str
    expected: object
    case_sensitive: bool = True

    def __post_init__(self) -> None:
        if (
            not self.task_id
            or len(self.task_id) > 128
            or any(
                ch not in (
                    "abcdefghijklmnopqrstuvwxyz"
                    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                    "0123456789-_."
                )
                for ch in self.task_id
            )
        ):
            raise VN97P4EvaluationError(
                "P4 task_id is invalid"
            )
        if self.category not in P4_CATEGORIES:
            raise VN97P4EvaluationError(
                "P4 task category is invalid"
            )
        prompt_bytes = self.prompt.encode(
            "utf-8"
        )
        if (
            not self.prompt
            or len(prompt_bytes)
            > _MAX_PROMPT_BYTES
        ):
            raise VN97P4EvaluationError(
                "P4 task prompt is empty or too large"
            )
        if (
            type(self.max_new_tokens) is not int
            or not 1
            <= self.max_new_tokens
            <= 4096
        ):
            raise VN97P4EvaluationError(
                "P4 task max_new_tokens must be in [1, 4096]"
            )
        if self.scoring_kind not in P4_SCORING_KINDS:
            raise VN97P4EvaluationError(
                "P4 task scoring kind is invalid"
            )


@dataclass(frozen=True)
class VN97P4TaskSuite:
    tasks: tuple[VN97P4Task, ...]
    suite_sha256: str


def render_p4_chat_prompt(
    prompt: str,
) -> str:
    if not isinstance(prompt, str) or not prompt:
        raise VN97P4EvaluationError(
            "P4 chat prompt must be non-empty text"
        )
    return (
        _CHAT_USER_MARKER
        + prompt
        + "\n"
        + _CHAT_ASSISTANT_MARKER
    )


def _task_from_object(
    value: object,
) -> VN97P4Task:
    if not isinstance(value, dict):
        raise VN97P4EvaluationError(
            "P4 task line root must be an object"
        )
    if set(value) != {
        "category",
        "max_new_tokens",
        "prompt",
        "schema",
        "scoring",
        "task_id",
    }:
        raise VN97P4EvaluationError(
            "P4 task key set is invalid"
        )
    if value.get("schema") != P4_TASK_SCHEMA:
        raise VN97P4EvaluationError(
            "P4 task schema mismatch"
        )
    scoring = value.get("scoring")
    if not isinstance(scoring, dict):
        raise VN97P4EvaluationError(
            "P4 task scoring must be an object"
        )
    kind = scoring.get("kind")
    case_sensitive = True
    expected: object

    if kind in {
        "exact_text",
        "json_exact",
    }:
        if set(scoring) != {
            "expected",
            "kind",
        }:
            raise VN97P4EvaluationError(
                f"P4 {kind} scoring key set is invalid"
            )
        expected = scoring.get("expected")
        if kind == "exact_text":
            if not isinstance(expected, str):
                raise VN97P4EvaluationError(
                    "P4 exact_text expected value must be a string"
                )
            if (
                len(
                    expected.encode(
                        "utf-8"
                    )
                )
                > _MAX_EXPECTED_TEXT_BYTES
            ):
                raise VN97P4EvaluationError(
                    "P4 exact_text expected value is too large"
                )
    elif kind == "contains_all":
        if set(scoring) != {
            "case_sensitive",
            "expected",
            "kind",
        }:
            raise VN97P4EvaluationError(
                "P4 contains_all scoring key set is invalid"
            )
        expected = scoring.get("expected")
        raw_case = scoring.get(
            "case_sensitive"
        )
        if (
            not isinstance(expected, list)
            or not expected
            or any(
                not isinstance(item, str)
                or not item
                or len(
                    item.encode(
                        "utf-8"
                    )
                )
                > _MAX_EXPECTED_TEXT_BYTES
                for item in expected
            )
            or type(raw_case) is not bool
        ):
            raise VN97P4EvaluationError(
                "P4 contains_all scoring values are invalid"
            )
        case_sensitive = raw_case
        expected = tuple(expected)
    else:
        raise VN97P4EvaluationError(
            "P4 task scoring kind is invalid"
        )

    task_id = value.get("task_id")
    category = value.get("category")
    prompt = value.get("prompt")
    max_new_tokens = value.get(
        "max_new_tokens"
    )
    if (
        not isinstance(task_id, str)
        or not isinstance(category, str)
        or not isinstance(prompt, str)
    ):
        raise VN97P4EvaluationError(
            "P4 task text fields are invalid"
        )

    return VN97P4Task(
        task_id=task_id,
        category=category,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        scoring_kind=str(kind),
        expected=expected,
        case_sensitive=case_sensitive,
    )


def load_p4_task_suite(
    path: Path,
) -> VN97P4TaskSuite:
    data = _bounded_file(
        path,
        max_bytes=_MAX_SUITE_BYTES,
    )
    if not data or not data.endswith(b"\n"):
        raise VN97P4EvaluationError(
            "P4 task suite must be non-empty canonical JSONL ending in newline"
        )

    tasks: list[VN97P4Task] = []
    ids: set[str] = set()
    categories: set[str] = set()

    for index, raw_line in enumerate(
        data.splitlines(
            keepends=True
        ),
        start=1,
    ):
        if (
            not raw_line.endswith(b"\n")
            or raw_line == b"\n"
        ):
            raise VN97P4EvaluationError(
                f"P4 task line {index} is malformed"
            )
        body = raw_line[:-1]
        value = _strict_json_value(
            body,
            label=f"P4 task line {index}",
        )
        if (
            _canonical_json(value)
            != body
        ):
            raise VN97P4EvaluationError(
                f"P4 task line {index} is not canonical JSON"
            )
        task = _task_from_object(
            value
        )
        if task.task_id in ids:
            raise VN97P4EvaluationError(
                "P4 task suite contains duplicate task_id"
            )
        ids.add(task.task_id)
        categories.add(task.category)
        tasks.append(task)
        if len(tasks) > _MAX_TASKS:
            raise VN97P4EvaluationError(
                "P4 task suite exceeds maximum task count"
            )

    missing = set(P4_CATEGORIES) - categories
    if missing:
        raise VN97P4EvaluationError(
            "P4 task suite must cover every canonical category; "
            f"missing={sorted(missing)}"
        )

    return VN97P4TaskSuite(
        tasks=tuple(tasks),
        suite_sha256=_sha256(data),
    )


def _parse_generated_json(
    text: str,
) -> object:
    data = text.strip().encode(
        "utf-8"
    )
    if not data:
        raise VN97P4EvaluationError(
            "generated JSON response is empty"
        )
    return _strict_json_value(
        data,
        label="generated P4 JSON",
        require_canonical=False,
    )


def score_p4_output(
    task: VN97P4Task,
    output: str,
) -> bool:
    if task.scoring_kind == "exact_text":
        assert isinstance(
            task.expected,
            str,
        )
        return (
            output.strip()
            == task.expected.strip()
        )

    if task.scoring_kind == "contains_all":
        assert isinstance(
            task.expected,
            tuple,
        )
        haystack = (
            output
            if task.case_sensitive
            else output.casefold()
        )
        needles = (
            task.expected
            if task.case_sensitive
            else tuple(
                item.casefold()
                for item in task.expected
            )
        )
        return all(
            item in haystack
            for item in needles
        )

    if task.scoring_kind == "json_exact":
        try:
            generated = (
                _parse_generated_json(
                    output
                )
            )
        except VN97P4EvaluationError:
            return False
        return generated == task.expected

    raise AssertionError(
        "unreachable P4 scoring kind"
    )


@dataclass(frozen=True)
class VN97P4TaskResult:
    task_id: str
    category: str
    scoring_kind: str
    passed: bool
    output_sha256: str
    output_utf8_bytes: int
    error_type: str

    def canonical_object(
        self,
    ) -> dict[str, object]:
        return {
            "category": self.category,
            "error_type": self.error_type,
            "output_sha256":
                self.output_sha256,
            "output_utf8_bytes":
                self.output_utf8_bytes,
            "passed": self.passed,
            "scoring_kind":
                self.scoring_kind,
            "task_id": self.task_id,
        }


@dataclass(frozen=True)
class VN97P4EvaluationReport:
    artifact: VN97P3FinalArtifact
    suite: VN97P4TaskSuite
    device: str
    results: tuple[
        VN97P4TaskResult,
        ...
    ]

    def canonical_object(
        self,
    ) -> dict[str, object]:
        category_rows: dict[
            str,
            dict[str, object],
        ] = {}
        for category in P4_CATEGORIES:
            subset = [
                result
                for result in self.results
                if result.category
                == category
            ]
            passed = sum(
                1
                for result in subset
                if result.passed
            )
            category_rows[
                category
            ] = {
                "pass_rate":
                    passed
                    / len(subset),
                "passed": passed,
                "tasks": len(subset),
            }

        passed = sum(
            1
            for result in self.results
            if result.passed
        )
        identity = {
            "categories": category_rows,
            "checkpoint_sha256":
                self.artifact.checkpoint_sha256,
            "device": self.device,
            "failed_tasks":
                len(self.results)
                - passed,
            "model_image_sha256":
                self.artifact.model_image_sha256,
            "p3_run_sha256":
                self.artifact.p3_run_sha256,
            "p3_selected_candidate_id":
                self.artifact.selected_candidate_id,
            "pass_rate":
                passed
                / len(self.results),
            "passed_tasks":
                passed,
            "profile_id":
                P4_PROFILE_ID,
            "results": [
                result.canonical_object()
                for result
                in self.results
            ],
            "schema":
                P4_REPORT_SCHEMA,
            "status":
                "MEASURED",
            "suite_sha256":
                self.suite.suite_sha256,
            "task_count":
                len(self.results),
            "tokenizer_sha256":
                self.artifact.tokenizer_sha256,
        }
        evaluation_id = _sha256(
            b"VN97P4EVAL1\0"
            + _canonical_json(
                identity
            )
        )
        return {
            **identity,
            "evaluation_id":
                evaluation_id,
        }

    def to_bytes(self) -> bytes:
        return (
            _canonical_json(
                self.canonical_object()
            )
            + b"\n"
        )


def evaluate_p4_task_suite(
    *,
    p3_dir: Path,
    suite_path: Path,
    device: str,
    max_prompt_tokens: int = 4096,
) -> VN97P4EvaluationReport:
    artifact = verify_p3_final_artifact(
        p3_dir
    )
    suite = load_p4_task_suite(
        suite_path
    )
    if max_prompt_tokens <= 0:
        raise VN97P4EvaluationError(
            "max_prompt_tokens must be positive"
        )

    if device == "auto":
        resolved = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
    else:
        try:
            resolved = torch.device(
                device
            )
        except RuntimeError as exc:
            raise VN97P4EvaluationError(
                "P4 device is invalid"
            ) from exc
    if (
        resolved.type == "cuda"
        and not torch.cuda.is_available()
    ):
        raise VN97P4EvaluationError(
            "requested CUDA device is unavailable"
        )

    model = artifact.checkpoint.model
    model.to(resolved)
    tokenizer = VN97Tokenizer(
        artifact.tokenizer_package
    )
    engine = TorchVN97InferenceEngine(
        model,
        tokenizer,
        limits=VN97InferenceLimits(
            max_prompt_tokens=
                max_prompt_tokens,
        ),
    )

    results: list[
        VN97P4TaskResult
    ] = []
    for task in suite.tasks:
        error_type = ""
        output = ""
        try:
            output = engine.generate_text(
                render_p4_chat_prompt(
                    task.prompt
                ),
                max_new_tokens=
                    task.max_new_tokens,
            )
            passed = score_p4_output(
                task,
                output,
            )
        except Exception as exc:
            passed = False
            error_type = type(exc).__name__

        output_bytes = output.encode(
            "utf-8"
        )
        results.append(
            VN97P4TaskResult(
                task_id=task.task_id,
                category=task.category,
                scoring_kind=
                    task.scoring_kind,
                passed=passed,
                output_sha256=
                    _sha256(
                        output_bytes
                    ),
                output_utf8_bytes=
                    len(output_bytes),
                error_type=
                    error_type,
            )
        )

    if len(results) != len(suite.tasks):
        raise VN97P4EvaluationError(
            "P4 evaluator did not produce one result per task"
        )

    return VN97P4EvaluationReport(
        artifact=artifact,
        suite=suite,
        device=str(resolved),
        results=tuple(results),
    )
