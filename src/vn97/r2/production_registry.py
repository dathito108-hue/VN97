from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import shutil
from typing import Iterator, Mapping, Sequence

from ..training import (
    encode_chat_completion_messages,
    make_training_windows,
)
from ..training_cli import _atomic_write, _load_records
from .config import r2_mobile_1b_config
from .data_bridge import load_vn97tk1
from .pilot_contract import chat_record_digest
from .production_acquisition import (
    R2D9_ALLOWED_FAMILIES,
    verify_r2d9_campaign,
)
from .production_corpus_scale import (
    R2D6_DEFAULT_MIN_TOKENS_PER_PARAMETER,
    R2D6_DEFAULT_TARGET_TOKENS_PER_PARAMETER,
    verify_r2d6_corpus_index,
)
from .production_source_adapter import verify_r2d10_pack


R2D11_DEFINITION_SCHEMA = "VN97R2D11DEF1"
R2D11_REGISTRY_SCHEMA = "VN97R2D11REG1"
R2D11_LEDGER_SCHEMA = "VN97R2D11LEDGER1"
R2D11_DIGEST_SCHEMA = "VN97R2D11DIGEST1"


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"R2-D11 evidence must be a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _require_sha256(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(ch not in "0123456789abcdef" for ch in value)
    ):
        raise ValueError(f"{label} must be lowercase SHA-256")
    return value


def _strict_json_bytes(data: bytes, *, label: str) -> dict[str, object]:
    duplicates: list[str] = []

    def hook(pairs):
        out: dict[str, object] = {}
        for key, value in pairs:
            if key in out:
                duplicates.append(key)
            out[key] = value
        return out

    try:
        value = json.loads(
            data.decode("utf-8", errors="strict"),
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
        raise ValueError(f"{label} must be strict UTF-8 JSON") from exc
    if duplicates or not isinstance(value, dict):
        raise ValueError(
            f"{label} must be one object without duplicate keys"
        )
    return value


def _safe_relative_path(raw: object, *, label: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"{label} must be a non-empty relative path")
    path = Path(raw)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{label} must remain relative")
    return path


@dataclass(frozen=True)
class R2D11RegistryDefinition:
    sequence_length: int
    family_weights: tuple[tuple[str, float], ...]

    def __post_init__(self) -> None:
        if self.sequence_length < 64:
            raise ValueError(
                "R2-D11 sequence_length must be at least 64"
            )
        names = [name for name, _ in self.family_weights]
        if not names or names != sorted(set(names)):
            raise ValueError(
                "R2-D11 family weights must be sorted and unique"
            )
        total = 0.0
        for name, weight in self.family_weights:
            if name not in R2D9_ALLOWED_FAMILIES:
                raise ValueError(
                    f"R2-D11 family {name!r} is not canonical"
                )
            if not math.isfinite(weight) or weight <= 0.0:
                raise ValueError(
                    "R2-D11 family weights must be finite and positive"
                )
            total += weight
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("R2-D11 family weights must sum to 1")

    @property
    def weights(self) -> dict[str, float]:
        return dict(self.family_weights)

    def canonical_object(self) -> dict[str, object]:
        return {
            "schema": R2D11_DEFINITION_SCHEMA,
            "sequence_length": self.sequence_length,
            "family_weights": self.weights,
        }

    def fingerprint(self) -> str:
        return _sha256_bytes(
            b"VN97R2D11DEF1\0"
            + _canonical_json(self.canonical_object())
        )


def load_r2d11_definition(path: Path) -> R2D11RegistryDefinition:
    if path.is_symlink():
        raise ValueError("R2-D11 definition must not be a symlink")
    payload = _strict_json_bytes(
        path.resolve(strict=True).read_bytes(),
        label="R2-D11 definition",
    )
    if set(payload) != {
        "schema",
        "sequence_length",
        "family_weights",
    }:
        raise ValueError("R2-D11 definition fields are invalid")
    if payload.get("schema") != R2D11_DEFINITION_SCHEMA:
        raise ValueError("R2-D11 definition schema mismatch")
    if type(payload.get("sequence_length")) is not int:
        raise ValueError("R2-D11 sequence_length must be an integer")
    raw_weights = payload.get("family_weights")
    if not isinstance(raw_weights, dict):
        raise ValueError("R2-D11 family_weights must be an object")
    weights: list[tuple[str, float]] = []
    for name, weight in raw_weights.items():
        if (
            not isinstance(name, str)
            or not isinstance(weight, (int, float))
            or isinstance(weight, bool)
        ):
            raise ValueError("R2-D11 family weight entry is invalid")
        weights.append((name, float(weight)))
    return R2D11RegistryDefinition(
        sequence_length=int(payload["sequence_length"]),
        family_weights=tuple(sorted(weights)),
    )


def _registry_id(body: Mapping[str, object]) -> str:
    return _sha256_bytes(
        b"VN97R2D11REG1\0"
        + _canonical_json(dict(body))
    )


def _snapshot_id(body: Mapping[str, object]) -> str:
    return _sha256_bytes(
        b"VN97R2D11LEDGER1\0"
        + _canonical_json(dict(body))
    )


def _family_requirements(
    *,
    parameter_count: int,
    family_weights: Mapping[str, float],
    tokens_per_parameter: float,
) -> dict[str, int]:
    return {
        family: int(
            math.ceil(
                parameter_count
                * tokens_per_parameter
                * weight
            )
        )
        for family, weight in sorted(family_weights.items())
    }


def init_r2d11_registry(
    *,
    definition_path: Path,
    tokenizer_path: Path,
    registry_dir: Path,
) -> dict[str, object]:
    definition = load_r2d11_definition(definition_path)
    if tokenizer_path.is_symlink():
        raise ValueError("R2-D11 tokenizer must not be a symlink")
    tokenizer_source = tokenizer_path.resolve(strict=True)
    tokenizer = load_vn97tk1(tokenizer_source)
    tokenizer_sha = _sha256_file(tokenizer_source)
    config = r2_mobile_1b_config(tokenizer.vocab_size)
    parameter_count = config.estimated_parameter_count()

    if registry_dir.exists():
        if (
            registry_dir.is_symlink()
            or not registry_dir.is_dir()
            or any(registry_dir.iterdir())
        ):
            raise ValueError(
                "R2-D11 registry-dir must be new or empty"
            )
    else:
        registry_dir.mkdir(parents=True, exist_ok=False)
    root = registry_dir.resolve(strict=True)
    (root / "ledger").mkdir()
    (root / "digests").mkdir()
    (root / "evidence").mkdir()

    tokenizer_target = root / "tokenizer.vn97tk1"
    shutil.copyfile(tokenizer_source, tokenizer_target)
    if _sha256_file(tokenizer_target) != tokenizer_sha:
        raise IOError("R2-D11 tokenizer copy identity mismatch")

    body = {
        "schema": R2D11_REGISTRY_SCHEMA,
        "definition_fingerprint": definition.fingerprint(),
        "sequence_length": definition.sequence_length,
        "family_weights": definition.weights,
        "tokenizer_sha256": tokenizer_sha,
        "vocab_size": tokenizer.vocab_size,
        "architecture_fingerprint": config.fingerprint(),
        "parameter_count": parameter_count,
        "floor_tokens_per_parameter": (
            R2D6_DEFAULT_MIN_TOKENS_PER_PARAMETER
        ),
        "target_tokens_per_parameter": (
            R2D6_DEFAULT_TARGET_TOKENS_PER_PARAMETER
        ),
        "floor_required_tokens_by_family": _family_requirements(
            parameter_count=parameter_count,
            family_weights=definition.weights,
            tokens_per_parameter=(
                R2D6_DEFAULT_MIN_TOKENS_PER_PARAMETER
            ),
        ),
        "target_required_tokens_by_family": _family_requirements(
            parameter_count=parameter_count,
            family_weights=definition.weights,
            tokens_per_parameter=(
                R2D6_DEFAULT_TARGET_TOKENS_PER_PARAMETER
            ),
        ),
    }
    payload = dict(body)
    payload["registry_id"] = _registry_id(body)
    _atomic_write(
        root / "registry.json",
        _canonical_json(payload) + b"\n",
    )

    state = _empty_state(payload)
    _write_snapshot(
        root,
        generation=0,
        previous_ledger_sha256=None,
        event={"type": "genesis"},
        state=state,
        registry_id=str(payload["registry_id"]),
    )
    verify_r2d11_registry(root)
    return payload


def _load_registry_config(root: Path) -> dict[str, object]:
    path = root / "registry.json"
    payload = _strict_json_bytes(
        path.read_bytes(),
        label="R2-D11 registry config",
    )
    if payload.get("schema") != R2D11_REGISTRY_SCHEMA:
        raise ValueError("R2-D11 registry schema mismatch")
    registry_id = _require_sha256(
        payload.get("registry_id"),
        label="R2-D11 registry_id",
    )
    body = dict(payload)
    body.pop("registry_id", None)
    if registry_id != _registry_id(body):
        raise ValueError("R2-D11 registry identity mismatch")

    tokenizer_path = root / "tokenizer.vn97tk1"
    if tokenizer_path.is_symlink() or not tokenizer_path.is_file():
        raise ValueError("R2-D11 tokenizer path is invalid")
    if _sha256_file(tokenizer_path) != payload.get(
        "tokenizer_sha256"
    ):
        raise ValueError("R2-D11 tokenizer SHA-256 mismatch")
    tokenizer = load_vn97tk1(tokenizer_path)
    config = r2_mobile_1b_config(tokenizer.vocab_size)
    if tokenizer.vocab_size != int(payload.get("vocab_size", -1)):
        raise ValueError("R2-D11 tokenizer vocab mismatch")
    if config.fingerprint() != payload.get(
        "architecture_fingerprint"
    ):
        raise ValueError("R2-D11 architecture fingerprint mismatch")
    if config.estimated_parameter_count() != int(
        payload.get("parameter_count", -1)
    ):
        raise ValueError("R2-D11 parameter count mismatch")
    if float(payload.get("floor_tokens_per_parameter", -1.0)) != (
        R2D6_DEFAULT_MIN_TOKENS_PER_PARAMETER
    ):
        raise ValueError("R2-D11 scale floor policy mismatch")
    if float(payload.get("target_tokens_per_parameter", -1.0)) != (
        R2D6_DEFAULT_TARGET_TOKENS_PER_PARAMETER
    ):
        raise ValueError("R2-D11 scale target policy mismatch")

    weights = payload.get("family_weights")
    if not isinstance(weights, dict) or not weights:
        raise ValueError("R2-D11 family weights are missing")
    definition = R2D11RegistryDefinition(
        sequence_length=int(payload.get("sequence_length", 0)),
        family_weights=tuple(
            sorted(
                (str(name), float(weight))
                for name, weight in weights.items()
            )
        ),
    )
    expected_floor = _family_requirements(
        parameter_count=config.estimated_parameter_count(),
        family_weights=definition.weights,
        tokens_per_parameter=R2D6_DEFAULT_MIN_TOKENS_PER_PARAMETER,
    )
    expected_target = _family_requirements(
        parameter_count=config.estimated_parameter_count(),
        family_weights=definition.weights,
        tokens_per_parameter=R2D6_DEFAULT_TARGET_TOKENS_PER_PARAMETER,
    )
    if payload.get("floor_required_tokens_by_family") != expected_floor:
        raise ValueError("R2-D11 family floor requirements mismatch")
    if payload.get("target_required_tokens_by_family") != expected_target:
        raise ValueError("R2-D11 family target requirements mismatch")
    return payload


def _empty_state(config: Mapping[str, object]) -> dict[str, object]:
    weights = config["family_weights"]
    assert isinstance(weights, dict)
    zero = {family: 0 for family in sorted(weights)}
    return _state_with_progress(
        config,
        admitted_batches=[],
        attached_batches=[],
        admitted_unique_records_by_family=dict(zero),
        admitted_normalized_target_tokens_by_family=dict(zero),
        training_target_tokens_by_family=dict(zero),
    )


def _state_with_progress(
    config: Mapping[str, object],
    *,
    admitted_batches: Sequence[str],
    attached_batches: Sequence[str],
    admitted_unique_records_by_family: Mapping[str, int],
    admitted_normalized_target_tokens_by_family: Mapping[str, int],
    training_target_tokens_by_family: Mapping[str, int],
) -> dict[str, object]:
    weights = config["family_weights"]
    assert isinstance(weights, dict)
    floor_required = config["floor_required_tokens_by_family"]
    target_required = config["target_required_tokens_by_family"]
    assert isinstance(floor_required, dict)
    assert isinstance(target_required, dict)

    progress: dict[str, dict[str, object]] = {}
    for family in sorted(weights):
        training = int(training_target_tokens_by_family.get(family, 0))
        floor_need = int(floor_required[family])
        target_need = int(target_required[family])
        progress[family] = {
            "weight": float(weights[family]),
            "training_target_tokens": training,
            "floor_required_tokens": floor_need,
            "floor_missing_tokens": max(0, floor_need - training),
            "floor_passed": training >= floor_need,
            "target_required_tokens": target_need,
            "target_missing_tokens": max(0, target_need - training),
            "target_passed": training >= target_need,
        }

    parameter_count = int(config["parameter_count"])
    training_total = sum(
        int(value)
        for value in training_target_tokens_by_family.values()
    )
    global_floor = int(
        math.ceil(
            parameter_count
            * float(config["floor_tokens_per_parameter"])
        )
    )
    global_target = int(
        math.ceil(
            parameter_count
            * float(config["target_tokens_per_parameter"])
        )
    )
    return {
        "admitted_batches": sorted(admitted_batches),
        "attached_batches": sorted(attached_batches),
        "admitted_unique_records_by_family": {
            family: int(admitted_unique_records_by_family.get(family, 0))
            for family in sorted(weights)
        },
        "admitted_normalized_target_tokens_by_family": {
            family: int(
                admitted_normalized_target_tokens_by_family.get(
                    family,
                    0,
                )
            )
            for family in sorted(weights)
        },
        "training_target_tokens_by_family": {
            family: int(
                training_target_tokens_by_family.get(family, 0)
            )
            for family in sorted(weights)
        },
        "family_progress": progress,
        "global_training_target_tokens": training_total,
        "global_tokens_per_parameter": (
            training_total / parameter_count
        ),
        "global_floor_required_tokens": global_floor,
        "global_floor_missing_tokens": max(
            0,
            global_floor - training_total,
        ),
        "global_floor_passed": training_total >= global_floor,
        "all_family_floor_passed": all(
            bool(value["floor_passed"])
            for value in progress.values()
        ),
        "production_floor_ready": (
            training_total >= global_floor
            and all(
                bool(value["floor_passed"])
                for value in progress.values()
            )
        ),
        "global_target_required_tokens": global_target,
        "global_target_missing_tokens": max(
            0,
            global_target - training_total,
        ),
        "global_target_passed": training_total >= global_target,
        "all_family_target_passed": all(
            bool(value["target_passed"])
            for value in progress.values()
        ),
    }


def _ledger_files(root: Path) -> list[Path]:
    ledger = root / "ledger"
    if ledger.is_symlink() or not ledger.is_dir():
        raise ValueError("R2-D11 ledger directory is invalid")
    paths = sorted(ledger.glob("*.json"))
    if not paths:
        raise ValueError("R2-D11 ledger is empty")
    expected = [
        ledger / f"{index:06d}.json"
        for index in range(len(paths))
    ]
    if paths != expected:
        raise ValueError("R2-D11 ledger generations are not contiguous")
    return paths


def _write_snapshot(
    root: Path,
    *,
    generation: int,
    previous_ledger_sha256: str | None,
    event: Mapping[str, object],
    state: Mapping[str, object],
    registry_id: str,
) -> dict[str, object]:
    body = {
        "schema": R2D11_LEDGER_SCHEMA,
        "registry_id": registry_id,
        "generation": generation,
        "previous_ledger_sha256": previous_ledger_sha256,
        "event": dict(event),
        "state": dict(state),
    }
    payload = dict(body)
    payload["snapshot_id"] = _snapshot_id(body)
    target = root / "ledger" / f"{generation:06d}.json"
    if target.exists() or target.is_symlink():
        raise ValueError("R2-D11 ledger generation already exists")
    _atomic_write(target, _canonical_json(payload) + b"\n")
    return payload


def _read_snapshot(path: Path) -> dict[str, object]:
    payload = _strict_json_bytes(
        path.read_bytes(),
        label=f"R2-D11 ledger {path.name}",
    )
    if payload.get("schema") != R2D11_LEDGER_SCHEMA:
        raise ValueError("R2-D11 ledger schema mismatch")
    snapshot_id = _require_sha256(
        payload.get("snapshot_id"),
        label="R2-D11 snapshot_id",
    )
    body = dict(payload)
    body.pop("snapshot_id", None)
    if snapshot_id != _snapshot_id(body):
        raise ValueError("R2-D11 ledger snapshot identity mismatch")
    return payload


def _digest_entries(path: Path) -> Iterator[dict[str, str]]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("R2-D11 digest shard path is invalid")
    previous = ""
    try:
        with path.open(
            "r",
            encoding="utf-8",
            errors="strict",
        ) as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = _strict_json_bytes(
                    line.encode("utf-8"),
                    label=f"R2-D11 digest line {line_number}",
                )
                if set(value) != {
                    "schema",
                    "digest",
                    "family",
                    "source_id",
                }:
                    raise ValueError(
                        "R2-D11 digest entry fields are invalid"
                    )
                if value.get("schema") != R2D11_DIGEST_SCHEMA:
                    raise ValueError("R2-D11 digest schema mismatch")
                digest = _require_sha256(
                    value.get("digest"),
                    label="R2-D11 record digest",
                )
                family = value.get("family")
                source_id = value.get("source_id")
                if (
                    not isinstance(family, str)
                    or family not in R2D9_ALLOWED_FAMILIES
                    or not isinstance(source_id, str)
                    or not source_id
                ):
                    raise ValueError(
                        "R2-D11 digest family/source is invalid"
                    )
                if digest <= previous:
                    raise ValueError(
                        "R2-D11 digest shard must be strictly sorted"
                    )
                previous = digest
                yield {
                    "digest": digest,
                    "family": family,
                    "source_id": source_id,
                }
    except UnicodeDecodeError as exc:
        raise ValueError("R2-D11 digest shard is not UTF-8") from exc


def _digest_overlap(
    left: Path,
    right_entries: Sequence[dict[str, str]],
) -> str | None:
    right_digests = {item["digest"] for item in right_entries}
    for item in _digest_entries(left):
        if item["digest"] in right_digests:
            return item["digest"]
    return None


def _load_state_from_events(
    config: Mapping[str, object],
    snapshots: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    weights = config["family_weights"]
    assert isinstance(weights, dict)
    admitted_batches: list[str] = []
    attached_batches: list[str] = []
    admitted_records = {family: 0 for family in weights}
    admitted_tokens = {family: 0 for family in weights}
    training_tokens = {family: 0 for family in weights}

    for index, snapshot in enumerate(snapshots):
        event = snapshot.get("event")
        if not isinstance(event, Mapping):
            raise ValueError("R2-D11 ledger event is invalid")
        event_type = event.get("type")
        if index == 0:
            if event_type != "genesis":
                raise ValueError("R2-D11 first event must be genesis")
            continue
        if event_type == "admit_pack":
            pack_id = _require_sha256(
                event.get("pack_id"),
                label="R2-D11 admitted pack_id",
            )
            if pack_id in admitted_batches:
                raise ValueError("R2-D11 pack admitted more than once")
            admitted_batches.append(pack_id)
            family_records = event.get(
                "unique_records_by_family"
            )
            family_tokens = event.get(
                "normalized_target_tokens_by_family"
            )
            if (
                not isinstance(family_records, Mapping)
                or not isinstance(family_tokens, Mapping)
            ):
                raise ValueError(
                    "R2-D11 admission family evidence is invalid"
                )
            if set(family_records) != set(weights) or set(family_tokens) != set(weights):
                raise ValueError(
                    "R2-D11 admission family evidence keys mismatch"
                )
            for family in weights:
                record_value = int(family_records[family])
                token_value = int(family_tokens[family])
                if record_value < 0 or token_value < 0:
                    raise ValueError(
                        "R2-D11 admission counts/tokens must be non-negative"
                    )
                admitted_records[family] += record_value
                admitted_tokens[family] += token_value
        elif event_type == "attach_campaign":
            pack_id = _require_sha256(
                event.get("pack_id"),
                label="R2-D11 attached pack_id",
            )
            if pack_id not in admitted_batches:
                raise ValueError(
                    "R2-D11 campaign attached before pack admission"
                )
            if pack_id in attached_batches:
                raise ValueError(
                    "R2-D11 pack campaign attached more than once"
                )
            attached_batches.append(pack_id)
            family_tokens = event.get(
                "training_target_tokens_by_family"
            )
            if not isinstance(family_tokens, Mapping):
                raise ValueError(
                    "R2-D11 attached token evidence is invalid"
                )
            if set(family_tokens) != set(weights):
                raise ValueError(
                    "R2-D11 attached token family keys mismatch"
                )
            for family in weights:
                token_value = int(family_tokens[family])
                if token_value < 0:
                    raise ValueError(
                        "R2-D11 attached tokens must be non-negative"
                    )
                training_tokens[family] += token_value
        else:
            raise ValueError("R2-D11 ledger event type is invalid")

    return _state_with_progress(
        config,
        admitted_batches=admitted_batches,
        attached_batches=attached_batches,
        admitted_unique_records_by_family=admitted_records,
        admitted_normalized_target_tokens_by_family=admitted_tokens,
        training_target_tokens_by_family=training_tokens,
    )


def verify_r2d11_registry(
    registry_dir: Path,
) -> dict[str, object]:
    if registry_dir.is_symlink():
        raise ValueError("R2-D11 registry-dir must not be a symlink")
    root = registry_dir.resolve(strict=True)
    config = _load_registry_config(root)
    registry_id = str(config["registry_id"])

    paths = _ledger_files(root)
    snapshots: list[dict[str, object]] = []
    previous_sha: str | None = None
    digest_files: list[Path] = []

    for generation, path in enumerate(paths):
        snapshot = _read_snapshot(path)
        if snapshot.get("registry_id") != registry_id:
            raise ValueError("R2-D11 ledger registry ID mismatch")
        if int(snapshot.get("generation", -1)) != generation:
            raise ValueError("R2-D11 ledger generation mismatch")
        if snapshot.get("previous_ledger_sha256") != previous_sha:
            raise ValueError("R2-D11 ledger chain mismatch")
        event = snapshot.get("event")
        if not isinstance(event, dict):
            raise ValueError("R2-D11 ledger event is invalid")

        if event.get("type") == "admit_pack":
            digest_relative = _safe_relative_path(
                event.get("digest_path"),
                label="R2-D11 digest_path",
            )
            digest_path = root / digest_relative
            if _sha256_file(digest_path) != event.get(
                "digest_sha256"
            ):
                raise ValueError("R2-D11 digest shard hash mismatch")
            count = sum(1 for _ in _digest_entries(digest_path))
            if count != int(event.get("unique_records", -1)):
                raise ValueError(
                    "R2-D11 digest shard record count mismatch"
                )
            for prior in digest_files:
                overlap = _digest_overlap(
                    prior,
                    tuple(_digest_entries(digest_path)),
                )
                if overlap is not None:
                    raise ValueError(
                        "R2-D11 cross-batch digest overlap detected"
                    )
            digest_files.append(digest_path)

            for label, key in (
                ("D10 pack evidence", "d10_pack_evidence_path"),
                ("D10 lock evidence", "d10_lock_evidence_path"),
            ):
                relative = _safe_relative_path(
                    event.get(key),
                    label=label,
                )
                evidence = root / relative
                expected_key = (
                    "d10_pack_evidence_sha256"
                    if key == "d10_pack_evidence_path"
                    else "d10_lock_evidence_sha256"
                )
                if _sha256_file(evidence) != event.get(expected_key):
                    raise ValueError(
                        f"R2-D11 {label} hash mismatch"
                    )

        elif event.get("type") == "attach_campaign":
            for label, path_key, sha_key in (
                (
                    "D9 campaign evidence",
                    "d9_campaign_evidence_path",
                    "d9_campaign_evidence_sha256",
                ),
                (
                    "D6 index evidence",
                    "d6_index_evidence_path",
                    "d6_index_evidence_sha256",
                ),
            ):
                relative = _safe_relative_path(
                    event.get(path_key),
                    label=label,
                )
                evidence = root / relative
                if _sha256_file(evidence) != event.get(sha_key):
                    raise ValueError(
                        f"R2-D11 {label} hash mismatch"
                    )

        snapshots.append(snapshot)
        previous_sha = _sha256_file(path)

    recomputed_state = _load_state_from_events(
        config,
        snapshots,
    )
    latest_state = snapshots[-1].get("state")
    if latest_state != recomputed_state:
        raise ValueError("R2-D11 cumulative registry state mismatch")
    return {
        "config": config,
        "latest_snapshot": snapshots[-1],
        "state": recomputed_state,
    }


def _normalized_target_tokens(
    records: Sequence[object],
    *,
    tokenizer,
    sequence_length: int,
) -> int:
    total = 0
    for record in records:
        example = encode_chat_completion_messages(
            tokenizer,
            record,
        )
        windows = make_training_windows(
            example,
            sequence_length=sequence_length,
            stride=sequence_length,
            pad_token_id=tokenizer.pad_id,
        )
        total += sum(item.target_tokens for item in windows)
    if total <= 0:
        raise ValueError(
            "R2-D11 normalized source produced no target tokens"
        )
    return total


def admit_r2d11_pack(
    *,
    registry_dir: Path,
    pack_dir: Path,
) -> dict[str, object]:
    verified_registry = verify_r2d11_registry(registry_dir)
    root = registry_dir.resolve(strict=True)
    config = verified_registry["config"]
    state = verified_registry["state"]
    assert isinstance(config, dict)
    assert isinstance(state, dict)

    if pack_dir.is_symlink():
        raise ValueError("R2-D11 pack-dir must not be a symlink")
    pack_root = pack_dir.resolve(strict=True)
    pack = verify_r2d10_pack(pack_root)
    pack_id = _require_sha256(
        pack.get("pack_id"),
        label="R2-D11 D10 pack_id",
    )
    if pack_id in state["admitted_batches"]:
        raise ValueError("R2-D11 D10 pack is already admitted")

    tokenizer = load_vn97tk1(root / "tokenizer.vn97tk1")
    sequence_length = int(config["sequence_length"])
    weights = config["family_weights"]
    assert isinstance(weights, dict)

    receipts = pack.get("receipts")
    if not isinstance(receipts, list) or not receipts:
        raise ValueError("R2-D11 D10 receipts are missing")

    unique: dict[str, dict[str, object]] = {}
    for receipt in sorted(
        receipts,
        key=lambda item: str(item.get("source_id"))
        if isinstance(item, dict)
        else "",
    ):
        if not isinstance(receipt, dict):
            raise ValueError("R2-D11 D10 receipt is invalid")
        family = str(receipt.get("family"))
        if family not in weights:
            raise ValueError(
                f"R2-D11 family {family!r} is outside registry allocation"
            )
        source_id = str(receipt.get("source_id"))
        relative = _safe_relative_path(
            receipt.get("normalized_path"),
            label="R2-D11 normalized_path",
        )
        path = pack_root / relative
        records, _ = _load_records(
            [path],
            mode="chat",
            max_input_bytes=int(
                receipt.get("normalized_bytes", 0)
            ),
            max_examples=int(
                receipt.get("normalized_records", 0)
            ),
        )
        if len(records) != int(
            receipt.get("normalized_records", -1)
        ):
            raise ValueError(
                "R2-D11 normalized record count mismatch"
            )

        for record in records:
            digest = chat_record_digest(record)
            token_count = _normalized_target_tokens(
                (record,),
                tokenizer=tokenizer,
                sequence_length=sequence_length,
            )
            prior = unique.get(digest)
            if prior is not None:
                if prior["family"] != family:
                    raise ValueError(
                        "R2-D11 duplicate record crosses task families"
                    )
                if source_id < str(prior["source_id"]):
                    prior["source_id"] = source_id
                continue
            unique[digest] = {
                "digest": digest,
                "family": family,
                "source_id": source_id,
                "target_tokens": token_count,
            }

    if not unique:
        raise ValueError("R2-D11 pack contains no unique records")
    entries = [
        {
            "digest": str(value["digest"]),
            "family": str(value["family"]),
            "source_id": str(value["source_id"]),
        }
        for _, value in sorted(unique.items())
    ]

    latest = verified_registry["latest_snapshot"]
    assert isinstance(latest, dict)
    snapshots = _ledger_files(root)
    for snapshot_path in snapshots[1:]:
        snapshot = _read_snapshot(snapshot_path)
        event = snapshot.get("event")
        if (
            isinstance(event, dict)
            and event.get("type") == "admit_pack"
        ):
            old_path = root / _safe_relative_path(
                event.get("digest_path"),
                label="R2-D11 prior digest_path",
            )
            overlap = _digest_overlap(old_path, entries)
            if overlap is not None:
                raise ValueError(
                    "R2-D11 pack overlaps a previously admitted record"
                )

    digest_relative = Path("digests") / f"{pack_id}.jsonl"
    digest_path = root / digest_relative
    digest_bytes = b"".join(
        _canonical_json(
            {
                "schema": R2D11_DIGEST_SCHEMA,
                **entry,
            }
        )
        + b"\n"
        for entry in entries
    )
    if digest_path.exists() or digest_path.is_symlink():
        raise ValueError("R2-D11 digest shard already exists")
    _atomic_write(digest_path, digest_bytes)

    evidence_dir = root / "evidence" / pack_id
    evidence_dir.mkdir(parents=False, exist_ok=False)
    d10_pack_source = pack_root / "r2d10-source-pack.json"
    d10_lock_source = pack_root / "source-lock.vn97r2d10.json"
    d10_pack_target = evidence_dir / "r2d10-source-pack.json"
    d10_lock_target = evidence_dir / "source-lock.vn97r2d10.json"
    _atomic_write(d10_pack_target, d10_pack_source.read_bytes())
    _atomic_write(d10_lock_target, d10_lock_source.read_bytes())

    records_by_family = {family: 0 for family in weights}
    tokens_by_family = {family: 0 for family in weights}
    for value in unique.values():
        family = str(value["family"])
        records_by_family[family] += 1
        tokens_by_family[family] += int(value["target_tokens"])

    event = {
        "type": "admit_pack",
        "pack_id": pack_id,
        "digest_path": str(digest_relative).replace("\\", "/"),
        "digest_sha256": _sha256_file(digest_path),
        "unique_records": len(entries),
        "unique_records_by_family": records_by_family,
        "normalized_target_tokens_by_family": tokens_by_family,
        "d10_pack_evidence_path": str(
            d10_pack_target.relative_to(root)
        ).replace("\\", "/"),
        "d10_pack_evidence_sha256": _sha256_file(
            d10_pack_target
        ),
        "d10_lock_evidence_path": str(
            d10_lock_target.relative_to(root)
        ).replace("\\", "/"),
        "d10_lock_evidence_sha256": _sha256_file(
            d10_lock_target
        ),
    }

    new_state = _state_from_event_append(
        config,
        state,
        event,
    )
    generation = int(latest["generation"]) + 1
    previous_path = root / "ledger" / (
        f"{generation - 1:06d}.json"
    )
    snapshot = _write_snapshot(
        root,
        generation=generation,
        previous_ledger_sha256=_sha256_file(previous_path),
        event=event,
        state=new_state,
        registry_id=str(config["registry_id"]),
    )
    verify_r2d11_registry(root)
    return snapshot


def _state_from_event_append(
    config: Mapping[str, object],
    current: Mapping[str, object],
    event: Mapping[str, object],
) -> dict[str, object]:
    admitted = list(current["admitted_batches"])
    attached = list(current["attached_batches"])
    admitted_records = dict(
        current["admitted_unique_records_by_family"]
    )
    admitted_tokens = dict(
        current["admitted_normalized_target_tokens_by_family"]
    )
    training_tokens = dict(
        current["training_target_tokens_by_family"]
    )
    weights = config["family_weights"]
    assert isinstance(weights, dict)

    if event["type"] == "admit_pack":
        pack_id = str(event["pack_id"])
        admitted.append(pack_id)
        source_records = event["unique_records_by_family"]
        source_tokens = event[
            "normalized_target_tokens_by_family"
        ]
        assert isinstance(source_records, Mapping)
        assert isinstance(source_tokens, Mapping)
        for family in weights:
            admitted_records[family] = int(
                admitted_records.get(family, 0)
            ) + int(source_records.get(family, 0))
            admitted_tokens[family] = int(
                admitted_tokens.get(family, 0)
            ) + int(source_tokens.get(family, 0))
    elif event["type"] == "attach_campaign":
        pack_id = str(event["pack_id"])
        attached.append(pack_id)
        source_tokens = event[
            "training_target_tokens_by_family"
        ]
        assert isinstance(source_tokens, Mapping)
        for family in weights:
            training_tokens[family] = int(
                training_tokens.get(family, 0)
            ) + int(source_tokens.get(family, 0))
    else:
        raise ValueError("R2-D11 append event type is invalid")

    return _state_with_progress(
        config,
        admitted_batches=admitted,
        attached_batches=attached,
        admitted_unique_records_by_family=admitted_records,
        admitted_normalized_target_tokens_by_family=admitted_tokens,
        training_target_tokens_by_family=training_tokens,
    )


def _bind_d10_to_d9(
    d10: Mapping[str, object],
    d9: Mapping[str, object],
) -> None:
    d10_receipts = d10.get("receipts")
    d9_receipts = d9.get("source_receipts")
    if (
        not isinstance(d10_receipts, list)
        or not isinstance(d9_receipts, list)
    ):
        raise ValueError("R2-D11 D10/D9 receipts are missing")
    d10_map = {
        str(item["source_id"]): item
        for item in d10_receipts
        if isinstance(item, dict)
    }
    d9_map = {
        str(item["source_id"]): item
        for item in d9_receipts
        if isinstance(item, dict)
    }
    if set(d10_map) != set(d9_map):
        raise ValueError("R2-D11 D10/D9 source ID sets differ")

    for source_id in sorted(d10_map):
        left = d10_map[source_id]
        right = d9_map[source_id]
        checks = (
            ("origin", "origin"),
            ("revision", "revision"),
            ("license", "license"),
            ("family", "family"),
            ("normalized_sha256", "sha256"),
            ("normalized_records", "input_records"),
        )
        for left_key, right_key in checks:
            if left.get(left_key) != right.get(right_key):
                raise ValueError(
                    f"R2-D11 D10/D9 source binding mismatch: {source_id}"
                )


def _bind_d9_to_d6(
    d9: Mapping[str, object],
    d6: Mapping[str, object],
) -> dict[str, str]:
    seals = d9.get("seals")
    source_manifests = d6.get("source_manifests")
    if (
        not isinstance(seals, list)
        or not isinstance(source_manifests, list)
    ):
        raise ValueError("R2-D11 D9/D6 manifest evidence is missing")

    d9_map: dict[str, str] = {}
    for item in seals:
        if not isinstance(item, dict):
            raise ValueError("R2-D11 D9 seal entry is invalid")
        manifest_id = _require_sha256(
            item.get("manifest_id"),
            label="R2-D11 D9 seal manifest ID",
        )
        family = str(item.get("family"))
        if family not in R2D9_ALLOWED_FAMILIES:
            raise ValueError("R2-D11 D9 seal family is invalid")
        d9_map[manifest_id] = family

    d6_map: dict[str, str] = {}
    for item in source_manifests:
        if not isinstance(item, dict):
            raise ValueError(
                "R2-D11 D6 source manifest entry is invalid"
            )
        manifest_id = _require_sha256(
            item.get("manifest_id"),
            label="R2-D11 D6 source manifest ID",
        )
        families = item.get("task_families")
        if (
            not isinstance(families, list)
            or len(families) != 1
            or not isinstance(families[0], str)
        ):
            raise ValueError(
                "R2-D11 incremental D6 source requires one family"
            )
        d6_map[manifest_id] = families[0]

    if d9_map != d6_map:
        raise ValueError("R2-D11 D9/D6 manifest-family maps differ")
    if sorted(d9_map) != sorted(
        d6.get("corpus_manifest_ids", [])
    ):
        raise ValueError("R2-D11 D6 corpus manifest ID set mismatch")
    return d9_map


def attach_r2d11_campaign(
    *,
    registry_dir: Path,
    pack_dir: Path,
    campaign_dir: Path,
    d6_package_dir: Path,
) -> dict[str, object]:
    verified_registry = verify_r2d11_registry(registry_dir)
    root = registry_dir.resolve(strict=True)
    config = verified_registry["config"]
    state = verified_registry["state"]
    latest = verified_registry["latest_snapshot"]
    assert isinstance(config, dict)
    assert isinstance(state, dict)
    assert isinstance(latest, dict)

    if pack_dir.is_symlink():
        raise ValueError("R2-D11 pack-dir must not be a symlink")
    if campaign_dir.is_symlink():
        raise ValueError("R2-D11 campaign-dir must not be a symlink")
    if d6_package_dir.is_symlink():
        raise ValueError("R2-D11 D6 package-dir must not be a symlink")
    d10 = verify_r2d10_pack(pack_dir.resolve(strict=True))
    pack_id = _require_sha256(
        d10.get("pack_id"),
        label="R2-D11 attached pack_id",
    )
    if pack_id not in state["admitted_batches"]:
        raise ValueError(
            "R2-D11 pack must be admitted before campaign attachment"
        )
    if pack_id in state["attached_batches"]:
        raise ValueError(
            "R2-D11 pack already has an attached campaign"
        )

    d9 = verify_r2d9_campaign(
        campaign_dir.resolve(strict=True)
    )
    d6 = verify_r2d6_corpus_index(
        d6_package_dir.resolve(strict=True)
    )
    _bind_d10_to_d9(d10, d9)
    manifest_family = _bind_d9_to_d6(d9, d6)

    admitted_event = None
    for ledger_path in _ledger_files(root):
        snapshot = _read_snapshot(ledger_path)
        event = snapshot.get("event")
        if (
            isinstance(event, dict)
            and event.get("type") == "admit_pack"
            and event.get("pack_id") == pack_id
        ):
            admitted_event = event
            break
    if admitted_event is None:
        raise RuntimeError("R2-D11 admitted event is missing")
    if int(d9.get("global_unique_records", -1)) != int(
        admitted_event["unique_records"]
    ):
        raise ValueError(
            "R2-D11 D9 unique-record count differs from admission"
        )

    if d6.get("release_held_out") is not True:
        raise ValueError("R2-D11 D6 release holdout is not locked")
    scale = d6.get("scale")
    if not isinstance(scale, dict):
        raise ValueError("R2-D11 D6 scale evidence is missing")
    if (
        float(scale.get("minimum_tokens_per_parameter", -1.0))
        != R2D6_DEFAULT_MIN_TOKENS_PER_PARAMETER
        or float(scale.get("target_tokens_per_parameter", -1.0))
        != R2D6_DEFAULT_TARGET_TOKENS_PER_PARAMETER
    ):
        raise ValueError("R2-D11 D6 scale policy must remain canonical 8/20")
    if d6.get("tokenizer_sha256") != config.get(
        "tokenizer_sha256"
    ):
        raise ValueError("R2-D11 D6 tokenizer identity mismatch")
    if d6.get("architecture_fingerprint") != config.get(
        "architecture_fingerprint"
    ):
        raise ValueError("R2-D11 D6 architecture identity mismatch")
    if int(d6.get("sequence_length", -1)) != int(
        config["sequence_length"]
    ):
        raise ValueError("R2-D11 D6 sequence length mismatch")

    weights = config["family_weights"]
    assert isinstance(weights, dict)
    training_tokens = {family: 0 for family in weights}
    raw_shards = d6.get("shards")
    if not isinstance(raw_shards, list):
        raise ValueError("R2-D11 D6 shard list is missing")
    for shard in raw_shards:
        if (
            not isinstance(shard, dict)
            or shard.get("split") != "training"
        ):
            continue
        manifest_id = str(shard.get("source_manifest_id"))
        family = manifest_family.get(manifest_id)
        if family is None:
            raise ValueError(
                "R2-D11 D6 training shard has unknown source manifest"
            )
        if family not in weights:
            raise ValueError(
                f"R2-D11 D6 family {family!r} is outside registry allocation"
            )
        training_tokens[family] += int(
            shard.get("target_tokens", 0)
        )

    if sum(training_tokens.values()) <= 0:
        raise ValueError(
            "R2-D11 attached campaign has no training target tokens"
        )

    evidence_dir = root / "evidence" / pack_id
    if not evidence_dir.is_dir():
        raise RuntimeError("R2-D11 pack evidence directory is missing")
    d9_source = (
        campaign_dir.resolve(strict=True)
        / "r2d9-campaign.json"
    )
    d6_source = (
        d6_package_dir.resolve(strict=True)
        / "r2d6-corpus-index.json"
    )
    d9_target = evidence_dir / "r2d9-campaign.json"
    d6_target = evidence_dir / "r2d6-corpus-index.json"
    if d9_target.exists() or d6_target.exists():
        raise ValueError(
            "R2-D11 campaign evidence already exists"
        )
    _atomic_write(d9_target, d9_source.read_bytes())
    _atomic_write(d6_target, d6_source.read_bytes())

    event = {
        "type": "attach_campaign",
        "pack_id": pack_id,
        "campaign_id": _require_sha256(
            d9.get("campaign_id"),
            label="R2-D11 D9 campaign_id",
        ),
        "d6_index_id": _require_sha256(
            d6.get("index_id"),
            label="R2-D11 D6 index_id",
        ),
        "training_target_tokens_by_family": training_tokens,
        "d9_campaign_evidence_path": str(
            d9_target.relative_to(root)
        ).replace("\\", "/"),
        "d9_campaign_evidence_sha256": _sha256_file(
            d9_target
        ),
        "d6_index_evidence_path": str(
            d6_target.relative_to(root)
        ).replace("\\", "/"),
        "d6_index_evidence_sha256": _sha256_file(
            d6_target
        ),
    }
    new_state = _state_from_event_append(
        config,
        state,
        event,
    )
    generation = int(latest["generation"]) + 1
    previous_path = root / "ledger" / (
        f"{generation - 1:06d}.json"
    )
    snapshot = _write_snapshot(
        root,
        generation=generation,
        previous_ledger_sha256=_sha256_file(previous_path),
        event=event,
        state=new_state,
        registry_id=str(config["registry_id"]),
    )
    verify_r2d11_registry(root)
    return snapshot
