from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Mapping, Sequence

from .production_contract import R2_PRODUCTION_STAGES
from .production_corpus_scale import verify_r2d6_corpus_index


R2D8_DEFINITION_SCHEMA = "VN97R2D8DEF1"
R2D8_PLAN_SCHEMA = "VN97R2D8PLAN1"
R2D8_ALLOWED_FAMILIES = {
    "dense_pretrain": ("language", "reasoning"),
    "instruction_reasoning": (
        "instruction",
        "language",
        "reasoning",
    ),
    "tool_action": (
        "action",
        "instruction",
        "language",
        "reasoning",
        "tool",
    ),
    "capability": (
        "action",
        "capability",
        "instruction",
        "language",
        "reasoning",
        "tool",
    ),
}


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_sha256(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(ch not in "0123456789abcdef" for ch in value)
    ):
        raise ValueError(f"{label} must be lowercase SHA-256")
    return value


@dataclass(frozen=True)
class R2D8CurriculumDefinition:
    stage: str
    epochs: int
    seed: int
    family_weights: tuple[tuple[str, float], ...]
    primary_family_by_manifest: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if self.stage not in R2_PRODUCTION_STAGES:
            raise ValueError("R2-D8 stage is invalid")
        if self.epochs <= 0:
            raise ValueError("R2-D8 epochs must be positive")
        if self.seed < 0:
            raise ValueError("R2-D8 seed must be non-negative")

        allowed = set(R2D8_ALLOWED_FAMILIES[self.stage])
        names = [name for name, _ in self.family_weights]
        if not names or names != sorted(set(names)):
            raise ValueError(
                "R2-D8 family weights must be sorted and unique"
            )
        total = 0.0
        for name, weight in self.family_weights:
            if name not in allowed:
                raise ValueError(
                    f"R2-D8 family {name!r} is not allowed for {self.stage}"
                )
            if not math.isfinite(weight) or weight <= 0.0:
                raise ValueError(
                    "R2-D8 family weights must be finite and positive"
                )
            total += weight
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("R2-D8 family weights must sum to 1")

        manifest_ids = [
            manifest_id
            for manifest_id, _ in self.primary_family_by_manifest
        ]
        if manifest_ids != sorted(set(manifest_ids)):
            raise ValueError(
                "R2-D8 manifest assignments must be sorted and unique"
            )
        for manifest_id, family in self.primary_family_by_manifest:
            _require_sha256(
                manifest_id,
                label="R2-D8 source manifest id",
            )
            if family not in names:
                raise ValueError(
                    "R2-D8 assigned primary family has zero curriculum weight"
                )

    @property
    def weights(self) -> dict[str, float]:
        return dict(self.family_weights)

    @property
    def assignments(self) -> dict[str, str]:
        return dict(self.primary_family_by_manifest)

    def canonical_object(self) -> dict[str, object]:
        return {
            "schema": R2D8_DEFINITION_SCHEMA,
            "stage": self.stage,
            "epochs": self.epochs,
            "seed": self.seed,
            "family_weights": dict(self.family_weights),
            "primary_family_by_manifest": dict(
                self.primary_family_by_manifest
            ),
        }


def load_r2d8_definition(path: Path) -> R2D8CurriculumDefinition:
    try:
        payload = json.loads(
            path.resolve(strict=True).read_text(
                encoding="utf-8",
                errors="strict",
            ),
            parse_constant=lambda raw: (
                _ for _ in ()
            ).throw(ValueError(raw)),
        )
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        ValueError,
    ) as exc:
        raise ValueError(
            "R2-D8 definition must be strict UTF-8 JSON"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError("R2-D8 definition must be an object")
    if set(payload) != {
        "schema",
        "stage",
        "epochs",
        "seed",
        "family_weights",
        "primary_family_by_manifest",
    }:
        raise ValueError("R2-D8 definition fields are invalid")
    if payload.get("schema") != R2D8_DEFINITION_SCHEMA:
        raise ValueError("R2-D8 definition schema mismatch")
    raw_weights = payload.get("family_weights")
    raw_assignments = payload.get("primary_family_by_manifest")
    if (
        not isinstance(raw_weights, dict)
        or not isinstance(raw_assignments, dict)
    ):
        raise ValueError("R2-D8 weights/assignments must be objects")
    weights: list[tuple[str, float]] = []
    for key, value in raw_weights.items():
        if not isinstance(key, str):
            raise ValueError("R2-D8 family names must be strings")
        weights.append((key, float(value)))
    assignments: list[tuple[str, str]] = []
    for key, value in raw_assignments.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise ValueError(
                "R2-D8 manifest assignments must be string-to-string"
            )
        assignments.append((key, value))
    return R2D8CurriculumDefinition(
        stage=str(payload.get("stage")),
        epochs=int(payload.get("epochs", 0)),
        seed=int(payload.get("seed", -1)),
        family_weights=tuple(sorted(weights)),
        primary_family_by_manifest=tuple(sorted(assignments)),
    )


def _training_shards(
    index: Mapping[str, object],
) -> tuple[dict[str, object], ...]:
    raw = index.get("shards")
    if not isinstance(raw, list):
        raise ValueError("R2-D8 D6 shard list is invalid")
    output = [
        item
        for item in raw
        if isinstance(item, dict) and item.get("split") == "training"
    ]
    if not output:
        raise ValueError("R2-D8 requires training shards")
    output.sort(
        key=lambda item: (
            str(item.get("source_manifest_id")),
            str(item.get("shard_id")),
        )
    )
    return tuple(output)


def _family_seed(seed: int, epoch: int, family: str) -> int:
    digest = hashlib.sha256(
        b"VN97R2D8FAMILY\0"
        + str(seed).encode("ascii")
        + b"\0"
        + str(epoch).encode("ascii")
        + b"\0"
        + family.encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def _validate_definition_against_index(
    definition: R2D8CurriculumDefinition,
    index: Mapping[str, object],
) -> dict[str, tuple[dict[str, object], ...]]:
    if index.get("release_held_out") is not True:
        raise ValueError("R2-D8 requires release_held_out=true")

    weights = definition.weights
    assignments = definition.assignments
    by_family: dict[str, list[dict[str, object]]] = {
        family: [] for family in weights
    }
    seen_manifests: set[str] = set()

    for shard in _training_shards(index):
        manifest_id = _require_sha256(
            shard.get("source_manifest_id"),
            label="R2-D8 source manifest id",
        )
        shard_id = _require_sha256(
            shard.get("shard_id"),
            label="R2-D8 shard id",
        )
        _ = shard_id
        if manifest_id not in assignments:
            raise ValueError(
                "R2-D8 every training source manifest needs a primary family"
            )
        primary = assignments[manifest_id]
        raw_families = shard.get("task_families")
        if (
            not isinstance(raw_families, list)
            or primary not in raw_families
        ):
            raise ValueError(
                "R2-D8 primary family must be declared by the D6 source"
            )
        if int(shard.get("target_tokens", 0)) <= 0:
            raise ValueError("R2-D8 shard target-token count is invalid")
        by_family[primary].append(shard)
        seen_manifests.add(manifest_id)

    if set(assignments) != seen_manifests:
        raise ValueError(
            "R2-D8 assignments must exactly match D6 training manifests"
        )
    for family, shards in by_family.items():
        if not shards:
            raise ValueError(
                f"R2-D8 weighted family {family!r} has no training shard"
            )
        shards.sort(key=lambda item: str(item["shard_id"]))
    return {
        family: tuple(shards)
        for family, shards in sorted(by_family.items())
    }


def _compile_epoch(
    *,
    by_family: Mapping[str, Sequence[dict[str, object]]],
    weights: Mapping[str, float],
    token_budget: int,
    seed: int,
    epoch: int,
) -> dict[str, object]:
    if token_budget <= 0:
        raise ValueError("R2-D8 epoch token budget must be positive")

    family_orders: dict[str, list[dict[str, object]]] = {}
    family_positions: dict[str, int] = {}
    family_cycles: dict[str, int] = {}
    family_tokens = {family: 0 for family in weights}
    visits = {family: 0 for family in weights}

    for family, shards in by_family.items():
        order = list(shards)
        random.Random(
            _family_seed(seed, epoch, family)
        ).shuffle(order)
        family_orders[family] = order
        family_positions[family] = 0
        family_cycles[family] = 0

    def next_shard(family: str) -> dict[str, object]:
        order = family_orders[family]
        position = family_positions[family]
        if position >= len(order):
            family_cycles[family] += 1
            order = list(by_family[family])
            random.Random(
                _family_seed(
                    seed + family_cycles[family],
                    epoch,
                    family,
                )
            ).shuffle(order)
            family_orders[family] = order
            position = 0
        shard = order[position]
        family_positions[family] = position + 1
        return shard

    schedule: list[str] = []
    total_tokens = 0
    max_visits = 100_000

    while total_tokens < token_budget or any(
        visits[family] == 0 for family in weights
    ):
        if len(schedule) >= max_visits:
            raise RuntimeError(
                "R2-D8 epoch schedule exceeded visit safety bound"
            )

        # Weighted fair scheduling: the family furthest behind its ideal
        # normalized progress receives the next shard visit.
        family = min(
            weights,
            key=lambda name: (
                family_tokens[name] / weights[name],
                name,
            ),
        )
        shard = next_shard(family)
        tokens = int(shard["target_tokens"])
        family_tokens[family] += tokens
        visits[family] += 1
        total_tokens += tokens
        schedule.append(str(shard["shard_id"]))

    realized = {
        family: family_tokens[family] / total_tokens
        for family in weights
    }
    body = {
        "epoch": epoch,
        "family_target_tokens": family_tokens,
        "family_visits": visits,
        "realized_weights": realized,
        "shard_ids": schedule,
        "target_tokens": total_tokens,
        "token_budget": token_budget,
    }
    body["order_digest"] = _sha256(
        b"VN97R2D8ORDER1\0"
        + _canonical_json(schedule)
    )
    return body


def compile_r2d8_plan(
    index: Mapping[str, object],
    definition: R2D8CurriculumDefinition,
) -> dict[str, object]:
    d6_index_id = _require_sha256(
        index.get("index_id"),
        label="R2-D8 D6 index id",
    )
    by_family = _validate_definition_against_index(
        definition,
        index,
    )
    split_totals = index.get("split_totals")
    if not isinstance(split_totals, Mapping):
        raise ValueError("R2-D8 D6 split totals are missing")
    training = split_totals.get("training")
    if not isinstance(training, Mapping):
        raise ValueError("R2-D8 D6 training totals are missing")
    token_budget = int(training.get("target_tokens", 0))
    if token_budget <= 0:
        raise ValueError("R2-D8 D6 training target tokens are invalid")

    epochs = [
        _compile_epoch(
            by_family=by_family,
            weights=definition.weights,
            token_budget=token_budget,
            seed=definition.seed,
            epoch=epoch,
        )
        for epoch in range(definition.epochs)
    ]
    body = {
        "schema": R2D8_PLAN_SCHEMA,
        "stage": definition.stage,
        "d6_index_id": d6_index_id,
        "epochs": definition.epochs,
        "seed": definition.seed,
        "family_weights": definition.weights,
        "primary_family_by_manifest": definition.assignments,
        "epoch_token_budget": token_budget,
        "release_held_out": True,
        "epoch_schedules": epochs,
    }
    payload = dict(body)
    payload["plan_id"] = _sha256(
        b"VN97R2D8PLAN1\0" + _canonical_json(body)
    )
    return payload


def verify_r2d8_plan(
    plan: Mapping[str, object],
    index: Mapping[str, object],
) -> dict[str, object]:
    if plan.get("schema") != R2D8_PLAN_SCHEMA:
        raise ValueError("R2-D8 plan schema mismatch")
    plan_id = _require_sha256(
        plan.get("plan_id"),
        label="R2-D8 plan id",
    )
    body = dict(plan)
    body.pop("plan_id", None)
    expected_id = _sha256(
        b"VN97R2D8PLAN1\0" + _canonical_json(body)
    )
    if plan_id != expected_id:
        raise ValueError("R2-D8 plan identity mismatch")
    if plan.get("d6_index_id") != index.get("index_id"):
        raise ValueError("R2-D8 D6 index identity mismatch")
    if plan.get("release_held_out") is not True:
        raise ValueError("R2-D8 release holdout is not locked")

    raw_weights = plan.get("family_weights")
    raw_assignments = plan.get("primary_family_by_manifest")
    if (
        not isinstance(raw_weights, dict)
        or not isinstance(raw_assignments, dict)
    ):
        raise ValueError("R2-D8 plan weights/assignments are invalid")
    definition = R2D8CurriculumDefinition(
        stage=str(plan.get("stage")),
        epochs=int(plan.get("epochs", 0)),
        seed=int(plan.get("seed", -1)),
        family_weights=tuple(
            sorted(
                (str(name), float(weight))
                for name, weight in raw_weights.items()
            )
        ),
        primary_family_by_manifest=tuple(
            sorted(
                (str(manifest), str(family))
                for manifest, family in raw_assignments.items()
            )
        ),
    )
    rebuilt = compile_r2d8_plan(index, definition)
    if rebuilt != dict(plan):
        raise ValueError(
            "R2-D8 plan does not match deterministic recompilation"
        )
    return rebuilt


def load_r2d8_plan(
    path: Path,
    index: Mapping[str, object],
) -> dict[str, object]:
    try:
        payload = json.loads(
            path.resolve(strict=True).read_text(
                encoding="utf-8",
                errors="strict",
            ),
            parse_constant=lambda raw: (
                _ for _ in ()
            ).throw(ValueError(raw)),
        )
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        ValueError,
    ) as exc:
        raise ValueError("R2-D8 plan must be strict UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("R2-D8 plan must be an object")
    return verify_r2d8_plan(payload, index)


def curriculum_epoch_shards(
    plan: Mapping[str, object],
    index: Mapping[str, object],
    *,
    epoch: int,
) -> tuple[dict[str, object], ...]:
    verified = verify_r2d8_plan(plan, index)
    schedules = verified["epoch_schedules"]
    if not isinstance(schedules, list) or not 0 <= epoch < len(schedules):
        raise ValueError("R2-D8 epoch is out of range")
    raw_shards = index.get("shards")
    if not isinstance(raw_shards, list):
        raise ValueError("R2-D8 D6 shard list is invalid")
    by_id = {
        str(item["shard_id"]): item
        for item in raw_shards
        if isinstance(item, dict) and item.get("split") == "training"
    }
    schedule = schedules[epoch]
    if not isinstance(schedule, dict):
        raise ValueError("R2-D8 epoch schedule is invalid")
    ids = schedule.get("shard_ids")
    if not isinstance(ids, list) or not ids:
        raise ValueError("R2-D8 epoch shard order is invalid")
    try:
        return tuple(by_id[str(shard_id)] for shard_id in ids)
    except KeyError as exc:
        raise ValueError(
            "R2-D8 plan references an unknown training shard"
        ) from exc
