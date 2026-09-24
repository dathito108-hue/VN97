from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

from .campaign import VN97CampaignCandidate
from .evaluation import VN97EvaluationResult


class VN97P3KaggleError(RuntimeError):
    pass


def canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def require_sha256(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(ch not in "0123456789abcdef" for ch in value)
    ):
        raise VN97P3KaggleError(
            f"{label} must be lowercase SHA-256"
        )
    return value


def candidate_from_object(
    raw: object,
) -> VN97CampaignCandidate:
    if (
        not isinstance(raw, dict)
        or set(raw)
        != {
            "d_model",
            "d_state",
            "embedding_rank",
            "learning_rate",
            "n_layers",
            "seed",
        }
    ):
        raise VN97P3KaggleError(
            "P3 candidate object is invalid"
        )
    if any(
        type(raw[key]) is not int
        for key in (
            "d_model",
            "d_state",
            "n_layers",
            "seed",
        )
    ):
        raise VN97P3KaggleError(
            "P3 candidate integer fields are invalid"
        )
    rank = raw["embedding_rank"]
    if rank is not None and type(rank) is not int:
        raise VN97P3KaggleError(
            "P3 candidate embedding rank is invalid"
        )
    learning_rate = raw["learning_rate"]
    if (
        isinstance(learning_rate, bool)
        or not isinstance(
            learning_rate,
            (int, float),
        )
    ):
        raise VN97P3KaggleError(
            "P3 candidate learning rate is invalid"
        )
    return VN97CampaignCandidate(
        d_model=raw["d_model"],
        n_layers=raw["n_layers"],
        d_state=raw["d_state"],
        embedding_rank=rank,
        seed=raw["seed"],
        learning_rate=float(learning_rate),
    )


def evaluation_object(
    result: VN97EvaluationResult,
) -> dict[str, object]:
    return {
        "mean_loss": result.mean_loss,
        "target_tokens": result.target_tokens,
        "top1_accuracy": result.top1_accuracy,
        "windows": result.windows,
    }


@dataclass(frozen=True)
class VN97P3CandidateResult:
    candidate_index: int
    candidate: VN97CampaignCandidate
    corpus_manifest_id: str
    corpus_manifest_sha256: str
    training_dataset_sha256: str
    validation_dataset_sha256: str
    release_split_sha256: str
    tokenizer_sha256: str
    profile_sha256: str
    parameter_count: int
    mobile_footprint: dict[str, object]
    status: str
    training_steps: int
    training_target_tokens: int
    training_mean_loss: float
    training_final_loss: float
    evaluation: VN97EvaluationResult
    checkpoint_sha256: str | None
    device: str

    def __post_init__(self) -> None:
        if self.candidate_index < 0:
            raise VN97P3KaggleError(
                "candidate index must be non-negative"
            )
        for value, label in (
            (self.corpus_manifest_id, "corpus manifest ID"),
            (
                self.corpus_manifest_sha256,
                "corpus manifest SHA-256",
            ),
            (
                self.training_dataset_sha256,
                "training dataset SHA-256",
            ),
            (
                self.validation_dataset_sha256,
                "validation dataset SHA-256",
            ),
            (
                self.release_split_sha256,
                "release split SHA-256",
            ),
            (self.tokenizer_sha256, "tokenizer SHA-256"),
            (self.profile_sha256, "profile SHA-256"),
        ):
            require_sha256(value, label=label)
        if self.parameter_count <= 0:
            raise VN97P3KaggleError(
                "parameter count must be positive"
            )
        if not isinstance(
            self.mobile_footprint,
            dict,
        ):
            raise VN97P3KaggleError(
                "mobile footprint must be an object"
            )
        if self.status not in {
            "ELIGIBLE",
            "REJECTED_QUALITY",
        }:
            raise VN97P3KaggleError(
                "candidate status is invalid"
            )
        if (
            self.training_steps <= 0
            or self.training_target_tokens <= 0
        ):
            raise VN97P3KaggleError(
                "candidate training metrics must contain work"
            )
        for value, label in (
            (
                self.training_mean_loss,
                "training mean loss",
            ),
            (
                self.training_final_loss,
                "training final loss",
            ),
        ):
            if (
                not math.isfinite(value)
                or value < 0.0
            ):
                raise VN97P3KaggleError(
                    f"{label} must be finite and non-negative"
                )
        if not self.device.startswith("cuda"):
            raise VN97P3KaggleError(
                "P3 candidate result requires CUDA"
            )
        if self.status == "ELIGIBLE":
            require_sha256(
                self.checkpoint_sha256,
                label="candidate checkpoint SHA-256",
            )
        elif self.checkpoint_sha256 is not None:
            raise VN97P3KaggleError(
                "rejected candidate must not publish a promoted checkpoint"
            )

    @property
    def candidate_id(self) -> str:
        return self.candidate.candidate_id

    def canonical_body(self) -> dict[str, object]:
        return {
            "candidate": self.candidate.canonical_object(),
            "candidate_id": self.candidate_id,
            "candidate_index": self.candidate_index,
            "checkpoint_sha256": self.checkpoint_sha256,
            "corpus_manifest_id": self.corpus_manifest_id,
            "corpus_manifest_sha256":
                self.corpus_manifest_sha256,
            "device": self.device,
            "evaluation": evaluation_object(
                self.evaluation
            ),
            "mobile_footprint": self.mobile_footprint,
            "parameter_count": self.parameter_count,
            "profile_sha256": self.profile_sha256,
            "release_split_sha256":
                self.release_split_sha256,
            "schema": "VN97P3CAND1",
            "status": self.status,
            "tokenizer_sha256": self.tokenizer_sha256,
            "training": {
                "final_loss": self.training_final_loss,
                "mean_loss": self.training_mean_loss,
                "steps": self.training_steps,
                "target_tokens":
                    self.training_target_tokens,
            },
            "training_dataset_sha256":
                self.training_dataset_sha256,
            "validation_dataset_sha256":
                self.validation_dataset_sha256,
        }

    def canonical_object(self) -> dict[str, object]:
        body = self.canonical_body()
        body["result_id"] = hashlib.sha256(
            b"VN97P3CAND1\0"
            + canonical_json(body)
        ).hexdigest()
        return body

    def to_bytes(self) -> bytes:
        return canonical_json(
            self.canonical_object()
        ) + b"\n"


def parse_candidate_result(
    data: bytes,
) -> VN97P3CandidateResult:
    duplicates: list[str] = []

    def hook(pairs):
        out = {}
        for key, value in pairs:
            if key in out:
                duplicates.append(key)
            out[key] = value
        return out

    try:
        text = data.decode(
            "utf-8",
            errors="strict",
        )
        body_text = (
            text[:-1]
            if text.endswith("\n")
            else text
        )
        value = json.loads(
            body_text,
            object_pairs_hook=hook,
            parse_constant=lambda raw: (
                _ for _ in ()
            ).throw(ValueError(raw)),
        )
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        ValueError,
    ) as exc:
        raise VN97P3KaggleError(
            "VN97P3CAND1 must be strict UTF-8 JSON"
        ) from exc
    if duplicates or not isinstance(value, dict):
        raise VN97P3KaggleError(
            "VN97P3CAND1 must be one object without duplicate keys"
        )
    expected_keys = {
        "candidate",
        "candidate_id",
        "candidate_index",
        "checkpoint_sha256",
        "corpus_manifest_id",
        "corpus_manifest_sha256",
        "device",
        "evaluation",
        "mobile_footprint",
        "parameter_count",
        "profile_sha256",
        "release_split_sha256",
        "result_id",
        "schema",
        "status",
        "tokenizer_sha256",
        "training",
        "training_dataset_sha256",
        "validation_dataset_sha256",
    }
    if set(value) != expected_keys:
        raise VN97P3KaggleError(
            "VN97P3CAND1 fields are invalid"
        )
    if value["schema"] != "VN97P3CAND1":
        raise VN97P3KaggleError(
            "candidate result schema mismatch"
        )
    result_id = require_sha256(
        value["result_id"],
        label="candidate result ID",
    )
    identity = dict(value)
    identity.pop("result_id")
    expected_result_id = hashlib.sha256(
        b"VN97P3CAND1\0"
        + canonical_json(identity)
    ).hexdigest()
    if result_id != expected_result_id:
        raise VN97P3KaggleError(
            "candidate result identity mismatch"
        )
    expected_text = (
        canonical_json(value).decode("utf-8")
        + ("\n" if text.endswith("\n") else "")
    )
    if text != expected_text:
        raise VN97P3KaggleError(
            "VN97P3CAND1 must use canonical JSON"
        )

    evaluation = value["evaluation"]
    training = value["training"]
    if (
        not isinstance(evaluation, dict)
        or set(evaluation)
        != {
            "mean_loss",
            "target_tokens",
            "top1_accuracy",
            "windows",
        }
        or not isinstance(training, dict)
        or set(training)
        != {
            "final_loss",
            "mean_loss",
            "steps",
            "target_tokens",
        }
    ):
        raise VN97P3KaggleError(
            "candidate metrics are invalid"
        )

    candidate = candidate_from_object(
        value["candidate"]
    )
    if value["candidate_id"] != candidate.candidate_id:
        raise VN97P3KaggleError(
            "candidate ID does not match candidate object"
        )

    return VN97P3CandidateResult(
        candidate_index=int(
            value["candidate_index"]
        ),
        candidate=candidate,
        corpus_manifest_id=str(
            value["corpus_manifest_id"]
        ),
        corpus_manifest_sha256=str(
            value["corpus_manifest_sha256"]
        ),
        training_dataset_sha256=str(
            value["training_dataset_sha256"]
        ),
        validation_dataset_sha256=str(
            value["validation_dataset_sha256"]
        ),
        release_split_sha256=str(
            value["release_split_sha256"]
        ),
        tokenizer_sha256=str(
            value["tokenizer_sha256"]
        ),
        profile_sha256=str(
            value["profile_sha256"]
        ),
        parameter_count=int(
            value["parameter_count"]
        ),
        mobile_footprint=dict(
            value["mobile_footprint"]
        ),
        status=str(value["status"]),
        training_steps=int(
            training["steps"]
        ),
        training_target_tokens=int(
            training["target_tokens"]
        ),
        training_mean_loss=float(
            training["mean_loss"]
        ),
        training_final_loss=float(
            training["final_loss"]
        ),
        evaluation=VN97EvaluationResult(
            windows=int(
                evaluation["windows"]
            ),
            target_tokens=int(
                evaluation["target_tokens"]
            ),
            mean_loss=float(
                evaluation["mean_loss"]
            ),
            top1_accuracy=float(
                evaluation["top1_accuracy"]
            ),
        ),
        checkpoint_sha256=(
            None
            if value["checkpoint_sha256"] is None
            else str(
                value["checkpoint_sha256"]
            )
        ),
        device=str(value["device"]),
    )
