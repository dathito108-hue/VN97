from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
from typing import Sequence

from ..training import (
    encode_chat_completion_messages,
    make_training_windows,
)
from ..training_cli import _atomic_write, _load_records
from .config import r2_mobile_1b_config
from .data_bridge import load_vn97tk1
from .pilot_contract import chat_record_digest
from .production_campaign import (
    R2D5_MAX_SPLIT_BYTES,
    R2D5_SPLITS,
    verify_r2d5_corpus,
)
from .production_contract import assert_r2_production_scale


R2D6_DEFINITION_SCHEMA = "VN97R2D6DEF1"
R2D6_INDEX_SCHEMA = "VN97R2D6CORPUS1"
R2D6_DEFAULT_MIN_TOKENS_PER_PARAMETER = 8.0
R2D6_DEFAULT_TARGET_TOKENS_PER_PARAMETER = 20.0


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
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


def _safe_relative_path(raw: object, *, label: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"{label} must be a non-empty relative path")
    path = Path(raw)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{label} must stay relative to the definition")
    return path


@dataclass(frozen=True)
class R2D6CorpusInput:
    path: Path
    task_families: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.task_families:
            raise ValueError("R2-D6 corpus input needs task families")
        if any(not item for item in self.task_families):
            raise ValueError("R2-D6 task families must be non-empty")
        if tuple(sorted(set(self.task_families))) != self.task_families:
            raise ValueError(
                "R2-D6 task families must be sorted and unique"
            )


@dataclass(frozen=True)
class R2D6ShardEvidence:
    shard_id: str
    source_manifest_id: str
    source_manifest_sha256: str
    split: str
    filename: str
    sha256: str
    bytes: int
    records: int
    input_tokens: int
    target_tokens: int
    windows: int
    task_families: tuple[str, ...]
    source_licenses: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_sha256(self.shard_id, label="R2-D6 shard_id")
        _require_sha256(
            self.source_manifest_id,
            label="R2-D6 source manifest id",
        )
        _require_sha256(
            self.source_manifest_sha256,
            label="R2-D6 source manifest SHA-256",
        )
        _require_sha256(self.sha256, label="R2-D6 shard SHA-256")
        if self.split not in R2D5_SPLITS:
            raise ValueError("R2-D6 split is invalid")
        if (
            not self.filename
            or Path(self.filename).is_absolute()
            or ".." in Path(self.filename).parts
        ):
            raise ValueError("R2-D6 shard filename must be relative")
        for name in (
            "bytes",
            "records",
            "input_tokens",
            "target_tokens",
            "windows",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"R2-D6 {name} must be positive")
        if not self.task_families or any(
            not item for item in self.task_families
        ):
            raise ValueError("R2-D6 task families are invalid")
        if not self.source_licenses or any(
            not item for item in self.source_licenses
        ):
            raise ValueError("R2-D6 source licenses are invalid")

    def canonical_object(self) -> dict[str, object]:
        value = asdict(self)
        value["task_families"] = list(self.task_families)
        value["source_licenses"] = list(self.source_licenses)
        return value


@dataclass(frozen=True)
class R2D6ScaleEvidence:
    parameter_count: int
    training_input_tokens: int
    training_target_tokens: int
    validation_target_tokens: int
    release_target_tokens: int
    tokens_per_parameter: float
    minimum_tokens_per_parameter: float
    target_tokens_per_parameter: float
    scale_floor_passed: bool
    scale_target_passed: bool

    def __post_init__(self) -> None:
        if self.parameter_count <= 0:
            raise ValueError("R2-D6 parameter_count must be positive")
        for name in (
            "training_input_tokens",
            "training_target_tokens",
            "validation_target_tokens",
            "release_target_tokens",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"R2-D6 {name} must be positive")
        if (
            not math.isfinite(self.tokens_per_parameter)
            or self.tokens_per_parameter <= 0.0
        ):
            raise ValueError("R2-D6 tokens_per_parameter must be positive")
        if (
            not math.isfinite(self.minimum_tokens_per_parameter)
            or self.minimum_tokens_per_parameter <= 0.0
        ):
            raise ValueError(
                "R2-D6 minimum_tokens_per_parameter must be positive"
            )
        if (
            not math.isfinite(self.target_tokens_per_parameter)
            or self.target_tokens_per_parameter
            < self.minimum_tokens_per_parameter
        ):
            raise ValueError(
                "R2-D6 target token ratio must be >= minimum"
            )

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def load_r2d6_definition(path: Path) -> tuple[R2D6CorpusInput, ...]:
    definition = path.resolve(strict=True)
    try:
        value = json.loads(
            definition.read_text(encoding="utf-8", errors="strict")
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("R2-D6 definition must be UTF-8 JSON") from exc
    if (
        not isinstance(value, dict)
        or set(value) != {"schema", "corpora"}
        or value.get("schema") != R2D6_DEFINITION_SCHEMA
        or not isinstance(value.get("corpora"), list)
        or not value["corpora"]
    ):
        raise ValueError("R2-D6 definition schema is invalid")

    root = definition.parent
    inputs: list[R2D6CorpusInput] = []
    seen_paths: set[Path] = set()
    for item in value["corpora"]:
        if (
            not isinstance(item, dict)
            or set(item) != {"path", "task_families"}
            or not isinstance(item.get("task_families"), list)
        ):
            raise ValueError("R2-D6 corpus definition entry is invalid")
        relative = _safe_relative_path(
            item["path"],
            label="R2-D6 corpus path",
        )
        resolved = (root / relative).resolve(strict=True)
        try:
            resolved.relative_to(root.resolve(strict=True))
        except ValueError as exc:
            raise ValueError(
                "R2-D6 corpus path escapes definition directory"
            ) from exc
        if resolved in seen_paths:
            raise ValueError("R2-D6 definition repeats a corpus directory")
        seen_paths.add(resolved)

        families = tuple(
            sorted(
                {
                    str(value)
                    for value in item["task_families"]
                    if isinstance(value, str) and value
                }
            )
        )
        if len(families) != len(item["task_families"]):
            raise ValueError(
                "R2-D6 task families must be unique non-empty strings"
            )
        inputs.append(
            R2D6CorpusInput(
                path=resolved,
                task_families=families,
            )
        )
    return tuple(inputs)


def _measure_records(
    records,
    *,
    tokenizer,
    sequence_length: int,
) -> tuple[int, int, int]:
    input_tokens = 0
    target_tokens = 0
    windows = 0
    for record in records:
        example = encode_chat_completion_messages(
            tokenizer,
            record,
        )
        sample_windows = make_training_windows(
            example,
            sequence_length=sequence_length,
            stride=sequence_length,
            pad_token_id=tokenizer.pad_id,
        )
        input_tokens += len(example.token_ids)
        target_tokens += sum(
            item.target_tokens for item in sample_windows
        )
        windows += len(sample_windows)
    if input_tokens <= 0 or target_tokens <= 0 or windows <= 0:
        raise RuntimeError("R2-D6 shard produced no usable token evidence")
    return input_tokens, target_tokens, windows


def _copy_verified(
    source: Path,
    target: Path,
    *,
    expected_sha256: str,
) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        raise ValueError(f"R2-D6 target already exists: {target}")
    shutil.copyfile(source, target)
    if _sha256_file(target) != expected_sha256:
        target.unlink(missing_ok=True)
        raise IOError("R2-D6 copied shard identity mismatch")


def _index_identity(body: dict[str, object]) -> str:
    return hashlib.sha256(
        b"VN97R2D6CORPUS1\0" + _canonical_json(body)
    ).hexdigest()


def build_r2d6_corpus_index(
    *,
    corpus_inputs: Sequence[R2D6CorpusInput],
    tokenizer_path: Path,
    output_dir: Path,
    sequence_length: int = 128,
    minimum_tokens_per_parameter: float = (
        R2D6_DEFAULT_MIN_TOKENS_PER_PARAMETER
    ),
    target_tokens_per_parameter: float = (
        R2D6_DEFAULT_TARGET_TOKENS_PER_PARAMETER
    ),
    max_split_bytes: int = R2D5_MAX_SPLIT_BYTES,
) -> dict[str, object]:
    if not corpus_inputs:
        raise ValueError("R2-D6 requires at least one sealed corpus")
    if sequence_length < 64:
        raise ValueError("R2-D6 sequence_length must be at least 64")
    if (
        not math.isfinite(minimum_tokens_per_parameter)
        or minimum_tokens_per_parameter <= 0.0
    ):
        raise ValueError("R2-D6 minimum token ratio must be positive")
    if (
        not math.isfinite(target_tokens_per_parameter)
        or target_tokens_per_parameter < minimum_tokens_per_parameter
    ):
        raise ValueError("R2-D6 target token ratio must be >= minimum")

    tokenizer_resolved = tokenizer_path.resolve(strict=True)
    tokenizer_sha = _sha256_file(tokenizer_resolved)
    tokenizer = load_vn97tk1(tokenizer_resolved)
    config = r2_mobile_1b_config(tokenizer.vocab_size)
    parameter_count = assert_r2_production_scale(config)

    if output_dir.exists():
        if (
            output_dir.is_symlink()
            or not output_dir.is_dir()
            or any(output_dir.iterdir())
        ):
            raise ValueError("R2-D6 output-dir must be new or empty")
    else:
        output_dir.mkdir(parents=True, exist_ok=False)
    output = output_dir.resolve(strict=True)
    (output / "shards").mkdir()
    (output / "manifests").mkdir()
    _copy_verified(
        tokenizer_resolved,
        output / "tokenizer.vn97tk1",
        expected_sha256=tokenizer_sha,
    )

    verified = []
    manifest_ids: set[str] = set()
    for item in corpus_inputs:
        evidence = verify_r2d5_corpus(
            item.path,
            max_split_bytes=max_split_bytes,
        )
        if evidence.manifest_id in manifest_ids:
            raise ValueError(
                "R2-D6 repeats a sealed corpus manifest identity"
            )
        manifest_ids.add(evidence.manifest_id)
        verified.append((evidence.manifest_id, item, evidence))
    verified.sort(key=lambda item: item[0])

    global_records: dict[str, tuple[str, str]] = {}
    shards: list[R2D6ShardEvidence] = []
    split_totals = {
        split: {
            "bytes": 0,
            "records": 0,
            "input_tokens": 0,
            "target_tokens": 0,
            "windows": 0,
        }
        for split in R2D5_SPLITS
    }

    for corpus_index, (_, item, evidence) in enumerate(verified):
        root = item.path.resolve(strict=True)
        manifest_name = (
            f"corpus-{corpus_index:05d}.vn97corpus1.json"
        )
        _copy_verified(
            root / "corpus.vn97corpus1.json",
            output / "manifests" / manifest_name,
            expected_sha256=evidence.manifest_sha256,
        )

        for split in R2D5_SPLITS:
            identity = evidence.split_identity[split]
            path = root / f"{split}.jsonl"
            records, _ = _load_records(
                [path],
                mode="chat",
                max_input_bytes=max_split_bytes,
                max_examples=int(identity["records"]),
            )
            for record in records:
                digest = chat_record_digest(record)
                prior = global_records.get(digest)
                if prior is not None:
                    raise ValueError(
                        "R2-D6 cross-corpus record overlap: "
                        f"{prior[0]}/{prior[1]} vs "
                        f"{evidence.manifest_id}/{split}"
                    )
                global_records[digest] = (
                    evidence.manifest_id,
                    split,
                )

            (
                input_tokens,
                target_tokens,
                windows,
            ) = _measure_records(
                records,
                tokenizer=tokenizer,
                sequence_length=sequence_length,
            )

            filename = f"{split}-{corpus_index:05d}.jsonl"
            target_path = output / "shards" / filename
            expected_sha = str(identity["sha256"])
            _copy_verified(
                path,
                target_path,
                expected_sha256=expected_sha,
            )
            shard_id = hashlib.sha256(
                b"VN97R2D6SHARD1\0"
                + evidence.manifest_id.encode("ascii")
                + b"\0"
                + split.encode("ascii")
                + b"\0"
                + expected_sha.encode("ascii")
            ).hexdigest()
            shard = R2D6ShardEvidence(
                shard_id=shard_id,
                source_manifest_id=evidence.manifest_id,
                source_manifest_sha256=evidence.manifest_sha256,
                split=split,
                filename=f"shards/{filename}",
                sha256=expected_sha,
                bytes=int(identity["bytes"]),
                records=int(identity["records"]),
                input_tokens=input_tokens,
                target_tokens=target_tokens,
                windows=windows,
                task_families=item.task_families,
                source_licenses=evidence.source_licenses,
            )
            shards.append(shard)

            totals = split_totals[split]
            totals["bytes"] += shard.bytes
            totals["records"] += shard.records
            totals["input_tokens"] += shard.input_tokens
            totals["target_tokens"] += shard.target_tokens
            totals["windows"] += shard.windows

    train = split_totals["training"]
    validation = split_totals["validation"]
    release = split_totals["release"]
    ratio = train["target_tokens"] / parameter_count
    scale = R2D6ScaleEvidence(
        parameter_count=parameter_count,
        training_input_tokens=train["input_tokens"],
        training_target_tokens=train["target_tokens"],
        validation_target_tokens=validation["target_tokens"],
        release_target_tokens=release["target_tokens"],
        tokens_per_parameter=ratio,
        minimum_tokens_per_parameter=minimum_tokens_per_parameter,
        target_tokens_per_parameter=target_tokens_per_parameter,
        scale_floor_passed=ratio >= minimum_tokens_per_parameter,
        scale_target_passed=ratio >= target_tokens_per_parameter,
    )

    body: dict[str, object] = {
        "architecture_fingerprint": config.fingerprint(),
        "corpus_manifest_ids": sorted(manifest_ids),
        "release_held_out": True,
        "scale": scale.as_dict(),
        "schema": R2D6_INDEX_SCHEMA,
        "sequence_length": sequence_length,
        "shards": [
            item.canonical_object()
            for item in sorted(
                shards,
                key=lambda value: (
                    value.split,
                    value.source_manifest_id,
                    value.shard_id,
                ),
            )
        ],
        "split_totals": split_totals,
        "tokenizer_sha256": tokenizer_sha,
    }
    index_id = _index_identity(body)
    payload = dict(body)
    payload["index_id"] = index_id
    _atomic_write(
        output / "r2d6-corpus-index.json",
        _canonical_json(payload) + b"\n",
    )
    return payload


def verify_r2d6_corpus_index(package_dir: Path) -> dict[str, object]:
    if package_dir.is_symlink():
        raise ValueError("R2-D6 package-dir must not be a symlink")
    root = package_dir.resolve(strict=True)
    try:
        payload = json.loads(
            (root / "r2d6-corpus-index.json").read_text(
                encoding="utf-8",
                errors="strict",
            )
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("R2-D6 index must be UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("R2-D6 index must be an object")
    if payload.get("schema") != R2D6_INDEX_SCHEMA:
        raise ValueError("R2-D6 index schema mismatch")
    index_id = _require_sha256(
        payload.get("index_id"),
        label="R2-D6 index_id",
    )
    body = dict(payload)
    body.pop("index_id", None)
    if index_id != _index_identity(body):
        raise ValueError("R2-D6 index identity mismatch")
    if payload.get("release_held_out") is not True:
        raise ValueError("R2-D6 release split must remain held out")

    tokenizer_sha = _require_sha256(
        payload.get("tokenizer_sha256"),
        label="R2-D6 tokenizer SHA-256",
    )
    tokenizer_path = root / "tokenizer.vn97tk1"
    if _sha256_file(tokenizer_path) != tokenizer_sha:
        raise ValueError("R2-D6 tokenizer hash mismatch")
    tokenizer = load_vn97tk1(tokenizer_path)
    config = r2_mobile_1b_config(tokenizer.vocab_size)
    assert_r2_production_scale(config)
    if payload.get("architecture_fingerprint") != config.fingerprint():
        raise ValueError("R2-D6 architecture fingerprint mismatch")

    raw_shards = payload.get("shards")
    if not isinstance(raw_shards, list) or not raw_shards:
        raise ValueError("R2-D6 shard list is invalid")
    shards: list[R2D6ShardEvidence] = []
    seen_ids: set[str] = set()
    for raw in raw_shards:
        if not isinstance(raw, dict):
            raise ValueError("R2-D6 shard entry is invalid")
        normalized = dict(raw)
        normalized["task_families"] = tuple(
            normalized.get("task_families", ())
        )
        normalized["source_licenses"] = tuple(
            normalized.get("source_licenses", ())
        )
        shard = R2D6ShardEvidence(**normalized)
        if shard.shard_id in seen_ids:
            raise ValueError("R2-D6 shard IDs are not unique")
        seen_ids.add(shard.shard_id)
        shard_path = root / shard.filename
        if shard_path.is_symlink() or not shard_path.is_file():
            raise ValueError("R2-D6 shard path is invalid")
        if shard_path.stat().st_size != shard.bytes:
            raise ValueError("R2-D6 shard byte identity mismatch")
        if _sha256_file(shard_path) != shard.sha256:
            raise ValueError("R2-D6 shard SHA-256 mismatch")
        shards.append(shard)

    expected_totals = {
        split: {
            "bytes": 0,
            "records": 0,
            "input_tokens": 0,
            "target_tokens": 0,
            "windows": 0,
        }
        for split in R2D5_SPLITS
    }
    for shard in shards:
        totals = expected_totals[shard.split]
        totals["bytes"] += shard.bytes
        totals["records"] += shard.records
        totals["input_tokens"] += shard.input_tokens
        totals["target_tokens"] += shard.target_tokens
        totals["windows"] += shard.windows
    if payload.get("split_totals") != expected_totals:
        raise ValueError("R2-D6 split totals mismatch")

    scale_raw = payload.get("scale")
    if not isinstance(scale_raw, dict):
        raise ValueError("R2-D6 scale evidence is missing")
    scale = R2D6ScaleEvidence(**scale_raw)
    training = expected_totals["training"]
    validation = expected_totals["validation"]
    release = expected_totals["release"]
    expected_ratio = training["target_tokens"] / scale.parameter_count
    if scale.parameter_count != config.estimated_parameter_count():
        raise ValueError("R2-D6 scale parameter count mismatch")
    if scale.training_input_tokens != training["input_tokens"]:
        raise ValueError("R2-D6 training input-token total mismatch")
    if scale.training_target_tokens != training["target_tokens"]:
        raise ValueError("R2-D6 training target-token total mismatch")
    if scale.validation_target_tokens != validation["target_tokens"]:
        raise ValueError("R2-D6 validation target-token total mismatch")
    if scale.release_target_tokens != release["target_tokens"]:
        raise ValueError("R2-D6 release target-token total mismatch")
    if not math.isclose(
        scale.tokens_per_parameter,
        expected_ratio,
        rel_tol=1e-12,
        abs_tol=0.0,
    ):
        raise ValueError("R2-D6 token ratio mismatch")
    if (
        scale.scale_floor_passed
        != (expected_ratio >= scale.minimum_tokens_per_parameter)
    ):
        raise ValueError("R2-D6 scale floor decision mismatch")
    if (
        scale.scale_target_passed
        != (expected_ratio >= scale.target_tokens_per_parameter)
    ):
        raise ValueError("R2-D6 scale target decision mismatch")

    return payload
