from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import shutil
from typing import Mapping, Sequence

from ..training_cli import _atomic_write
from .production_contract import R2_PRODUCTION_STAGES
from .production_corpus_scale import (
    R2D6_DEFAULT_MIN_TOKENS_PER_PARAMETER,
    R2D6_DEFAULT_TARGET_TOKENS_PER_PARAMETER,
    verify_r2d6_corpus_index,
)
from .production_curriculum import R2D8_ALLOWED_FAMILIES
from .production_registry import verify_r2d11_generation


R2D12_MOUNT_DEFINITION_SCHEMA = "VN97R2D12MOUNTDEF1"
R2D12_MOUNTS_SCHEMA = "VN97R2D12MOUNTS1"
R2D12_VIEW_SCHEMA = "VN97R2D12VIEW1"
R2D12_INDEX_SCHEMA = "VN97R2D12INDEX1"


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


def _require_sha256(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(ch not in "0123456789abcdef" for ch in value)
    ):
        raise ValueError(f"{label} must be lowercase SHA-256")
    return value


def _strict_json(path: Path, *, label: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular non-symlink file")
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
            path.read_text(encoding="utf-8", errors="strict"),
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


def _sha256_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"R2-D12 file is invalid: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _index_identity(body: Mapping[str, object]) -> str:
    return _sha256_bytes(
        b"VN97R2D12INDEX1\0"
        + _canonical_json(dict(body))
    )


def _view_identity(body: Mapping[str, object]) -> str:
    return _sha256_bytes(
        b"VN97R2D12VIEW1\0"
        + _canonical_json(dict(body))
    )


def load_r2d12_mount_definition(
    path: Path,
) -> dict[str, str]:
    payload = _strict_json(
        path.resolve(strict=True),
        label="R2-D12 mount definition",
    )
    if set(payload) != {"schema", "packages"}:
        raise ValueError("R2-D12 mount definition fields are invalid")
    if payload.get("schema") != R2D12_MOUNT_DEFINITION_SCHEMA:
        raise ValueError("R2-D12 mount definition schema mismatch")
    raw = payload.get("packages")
    if not isinstance(raw, dict) or not raw:
        raise ValueError("R2-D12 mount package map is invalid")

    output: dict[str, str] = {}
    for pack_id, relative in raw.items():
        _require_sha256(pack_id, label="R2-D12 mount pack_id")
        safe = _safe_relative_path(
            relative,
            label="R2-D12 mount package path",
        )
        output[pack_id] = safe.as_posix()
    return dict(sorted(output.items()))


def _attached_d6_ids(
    generation: Mapping[str, object],
) -> dict[str, str]:
    state = generation.get("state")
    events = generation.get("events")
    if not isinstance(state, Mapping) or not isinstance(events, list):
        raise ValueError("R2-D12 frozen D11 generation is invalid")
    admitted = state.get("admitted_batches")
    attached = state.get("attached_batches")
    if (
        not isinstance(admitted, list)
        or not isinstance(attached, list)
        or not attached
    ):
        raise ValueError("R2-D12 generation has no attached batches")
    if admitted != attached:
        raise RuntimeError(
            "R2-D12 freeze requires every admitted pack to be attached"
        )

    found: dict[str, str] = {}
    for event in events:
        if (
            isinstance(event, Mapping)
            and event.get("type") == "attach_campaign"
        ):
            pack_id = _require_sha256(
                event.get("pack_id"),
                label="R2-D12 attached pack_id",
            )
            d6_index_id = _require_sha256(
                event.get("d6_index_id"),
                label="R2-D12 attached D6 index_id",
            )
            if pack_id in found:
                raise ValueError(
                    "R2-D12 pack has duplicate attach events"
                )
            found[pack_id] = d6_index_id
    if sorted(found) != attached:
        raise ValueError(
            "R2-D12 attach-event set does not match frozen registry state"
        )
    return dict(sorted(found.items()))


def _zero_split_totals() -> dict[str, dict[str, int]]:
    return {
        split: {
            "bytes": 0,
            "records": 0,
            "input_tokens": 0,
            "target_tokens": 0,
            "windows": 0,
        }
        for split in ("training", "validation", "release")
    }


def _add_split_totals(
    target: dict[str, dict[str, int]],
    source: Mapping[str, object],
) -> None:
    for split in ("training", "validation", "release"):
        raw = source.get(split)
        if not isinstance(raw, Mapping):
            raise ValueError(
                f"R2-D12 D6 split totals missing for {split}"
            )
        for key in (
            "bytes",
            "records",
            "input_tokens",
            "target_tokens",
            "windows",
        ):
            value = int(raw.get(key, -1))
            if value <= 0:
                raise ValueError(
                    f"R2-D12 D6 split total {split}/{key} is invalid"
                )
            target[split][key] += value


def _scale_from_totals(
    *,
    parameter_count: int,
    split_totals: Mapping[str, Mapping[str, int]],
) -> dict[str, object]:
    train = split_totals["training"]
    validation = split_totals["validation"]
    release = split_totals["release"]
    target_tokens = int(train["target_tokens"])
    ratio = target_tokens / parameter_count
    return {
        "parameter_count": parameter_count,
        "training_input_tokens": int(train["input_tokens"]),
        "training_target_tokens": target_tokens,
        "validation_target_tokens": int(
            validation["target_tokens"]
        ),
        "release_target_tokens": int(release["target_tokens"]),
        "tokens_per_parameter": ratio,
        "minimum_tokens_per_parameter": (
            R2D6_DEFAULT_MIN_TOKENS_PER_PARAMETER
        ),
        "target_tokens_per_parameter": (
            R2D6_DEFAULT_TARGET_TOKENS_PER_PARAMETER
        ),
        "scale_floor_passed": (
            ratio >= R2D6_DEFAULT_MIN_TOKENS_PER_PARAMETER
        ),
        "scale_target_passed": (
            ratio >= R2D6_DEFAULT_TARGET_TOKENS_PER_PARAMETER
        ),
    }


def _combine_indexes(
    *,
    registry_config: Mapping[str, object],
    batch_indexes: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    tokenizer_sha = _require_sha256(
        registry_config.get("tokenizer_sha256"),
        label="R2-D12 registry tokenizer SHA-256",
    )
    architecture = _require_sha256(
        registry_config.get("architecture_fingerprint"),
        label="R2-D12 registry architecture fingerprint",
    )
    sequence_length = int(
        registry_config.get("sequence_length", -1)
    )
    parameter_count = int(
        registry_config.get("parameter_count", -1)
    )
    if sequence_length < 64 or parameter_count <= 0:
        raise ValueError("R2-D12 registry production contract is invalid")

    manifests: list[dict[str, object]] = []
    shards: list[dict[str, object]] = []
    manifest_ids: set[str] = set()
    shard_ids: set[str] = set()
    split_totals = _zero_split_totals()

    for pack_id, index in sorted(batch_indexes.items()):
        if index.get("release_held_out") is not True:
            raise ValueError("R2-D12 batch release holdout is not locked")
        if index.get("tokenizer_sha256") != tokenizer_sha:
            raise ValueError("R2-D12 batch tokenizer identity mismatch")
        if index.get("architecture_fingerprint") != architecture:
            raise ValueError(
                "R2-D12 batch architecture identity mismatch"
            )
        if int(index.get("sequence_length", -1)) != sequence_length:
            raise ValueError(
                "R2-D12 batch sequence length mismatch"
            )
        scale = index.get("scale")
        if not isinstance(scale, Mapping):
            raise ValueError("R2-D12 batch scale evidence is missing")
        if (
            int(scale.get("parameter_count", -1))
            != parameter_count
            or float(
                scale.get(
                    "minimum_tokens_per_parameter",
                    float("nan"),
                )
            )
            != R2D6_DEFAULT_MIN_TOKENS_PER_PARAMETER
            or float(
                scale.get(
                    "target_tokens_per_parameter",
                    float("nan"),
                )
            )
            != R2D6_DEFAULT_TARGET_TOKENS_PER_PARAMETER
        ):
            raise ValueError(
                "R2-D12 batch production scale contract mismatch"
            )

        raw_manifests = index.get("source_manifests")
        raw_shards = index.get("shards")
        if (
            not isinstance(raw_manifests, list)
            or not isinstance(raw_shards, list)
        ):
            raise ValueError("R2-D12 batch D6 metadata is invalid")

        local_manifest_ids: set[str] = set()
        for raw in raw_manifests:
            if not isinstance(raw, dict):
                raise ValueError(
                    "R2-D12 batch source manifest entry is invalid"
                )
            manifest_id = _require_sha256(
                raw.get("manifest_id"),
                label="R2-D12 source manifest ID",
            )
            if manifest_id in manifest_ids:
                raise ValueError(
                    "R2-D12 source manifest repeats across batches"
                )
            families = raw.get("task_families")
            if (
                not isinstance(families, list)
                or len(families) != 1
                or not isinstance(families[0], str)
                or not families[0]
            ):
                raise ValueError(
                    "R2-D12 virtual source manifests require one family"
                )
            manifest_ids.add(manifest_id)
            local_manifest_ids.add(manifest_id)
            item = dict(raw)
            item["batch_pack_id"] = pack_id
            manifests.append(item)

        if sorted(local_manifest_ids) != sorted(
            index.get("corpus_manifest_ids", [])
        ):
            raise ValueError(
                "R2-D12 batch corpus-manifest set mismatch"
            )

        for raw in raw_shards:
            if not isinstance(raw, dict):
                raise ValueError("R2-D12 batch shard entry is invalid")
            shard_id = _require_sha256(
                raw.get("shard_id"),
                label="R2-D12 shard ID",
            )
            if shard_id in shard_ids:
                raise ValueError(
                    "R2-D12 shard identity repeats across batches"
                )
            manifest_id = _require_sha256(
                raw.get("source_manifest_id"),
                label="R2-D12 shard source manifest ID",
            )
            if manifest_id not in local_manifest_ids:
                raise ValueError(
                    "R2-D12 shard references another batch manifest"
                )
            shard_ids.add(shard_id)
            item = dict(raw)
            item["batch_pack_id"] = pack_id
            shards.append(item)

        totals = index.get("split_totals")
        if not isinstance(totals, Mapping):
            raise ValueError("R2-D12 batch split totals are missing")
        _add_split_totals(split_totals, totals)

    body: dict[str, object] = {
        "schema": R2D12_INDEX_SCHEMA,
        "architecture_fingerprint": architecture,
        "corpus_manifest_ids": sorted(manifest_ids),
        "release_held_out": True,
        "scale": _scale_from_totals(
            parameter_count=parameter_count,
            split_totals=split_totals,
        ),
        "source_manifests": sorted(
            manifests,
            key=lambda item: (
                str(item["batch_pack_id"]),
                str(item["manifest_id"]),
            ),
        ),
        "sequence_length": sequence_length,
        "shards": sorted(
            shards,
            key=lambda item: (
                str(item["split"]),
                str(item["source_manifest_id"]),
                str(item["shard_id"]),
            ),
        ),
        "split_totals": split_totals,
        "tokenizer_sha256": tokenizer_sha,
    }
    payload = dict(body)
    payload["index_id"] = _index_identity(body)
    return payload


def build_r2d12_view(
    *,
    registry_dir: Path,
    generation: int,
    workspace_root: Path,
    mount_definition_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    if workspace_root.is_symlink():
        raise ValueError("R2-D12 workspace root must not be a symlink")
    workspace = workspace_root.resolve(strict=True)
    if not workspace.is_dir():
        raise ValueError("R2-D12 workspace root must be a directory")

    frozen = verify_r2d11_generation(
        registry_dir,
        generation,
    )
    config = frozen["config"]
    state = frozen["state"]
    snapshot = frozen["snapshot"]
    assert isinstance(config, Mapping)
    assert isinstance(state, Mapping)
    assert isinstance(snapshot, Mapping)

    expected = _attached_d6_ids(frozen)
    mounts = load_r2d12_mount_definition(
        mount_definition_path
    )
    if sorted(mounts) != sorted(expected):
        raise ValueError(
            "R2-D12 mount pack set must equal frozen attached pack set"
        )

    batch_indexes: dict[str, dict[str, object]] = {}
    normalized_mounts: dict[str, str] = {}
    for pack_id, relative in sorted(mounts.items()):
        path = workspace / _safe_relative_path(
            relative,
            label="R2-D12 D6 package path",
        )
        if path.is_symlink():
            raise ValueError("R2-D12 D6 package must not be a symlink")
        package = path.resolve(strict=True)
        try:
            package.relative_to(workspace)
        except ValueError as exc:
            raise ValueError(
                "R2-D12 D6 package escapes workspace root"
            ) from exc
        index = verify_r2d6_corpus_index(package)
        if index.get("index_id") != expected[pack_id]:
            raise ValueError(
                "R2-D12 mounted D6 package index does not match D11 evidence"
            )
        batch_indexes[pack_id] = index
        normalized_mounts[pack_id] = relative

    virtual_index = _combine_indexes(
        registry_config=config,
        batch_indexes=batch_indexes,
    )

    if output_dir.exists():
        if (
            output_dir.is_symlink()
            or not output_dir.is_dir()
            or any(output_dir.iterdir())
        ):
            raise ValueError("R2-D12 output-dir must be new or empty")
    else:
        output_dir.mkdir(parents=True, exist_ok=False)
    output = output_dir.resolve(strict=True)

    registry_root = registry_dir.resolve(strict=True)
    tokenizer_source = registry_root / "tokenizer.vn97tk1"
    tokenizer_target = output / "tokenizer.vn97tk1"
    shutil.copyfile(tokenizer_source, tokenizer_target)
    if _sha256_file(tokenizer_target) != config.get(
        "tokenizer_sha256"
    ):
        raise IOError("R2-D12 tokenizer copy identity mismatch")

    batches = [
        {
            "pack_id": pack_id,
            "d6_index_id": expected[pack_id],
        }
        for pack_id in sorted(expected)
    ]
    body = {
        "schema": R2D12_VIEW_SCHEMA,
        "registry_id": _require_sha256(
            config.get("registry_id"),
            label="R2-D12 registry ID",
        ),
        "registry_generation": generation,
        "registry_snapshot_id": _require_sha256(
            snapshot.get("snapshot_id"),
            label="R2-D12 registry snapshot ID",
        ),
        "registry_ledger_sha256": _require_sha256(
            frozen.get("ledger_sha256"),
            label="R2-D12 registry ledger SHA-256",
        ),
        "attached_batches": batches,
        "virtual_index": virtual_index,
    }
    view = dict(body)
    view["view_id"] = _view_identity(body)
    _atomic_write(
        output / "r2d12-view.json",
        _canonical_json(view) + b"\n",
    )

    mounts_payload = {
        "schema": R2D12_MOUNTS_SCHEMA,
        "view_id": view["view_id"],
        "packages": normalized_mounts,
    }
    _atomic_write(
        output / "r2d12-mounts.json",
        _canonical_json(mounts_payload) + b"\n",
    )
    verify_r2d12_view(
        output,
        workspace_root=workspace,
    )
    return view


def _load_view_file(view_dir: Path) -> dict[str, object]:
    view = _strict_json(
        view_dir / "r2d12-view.json",
        label="R2-D12 view",
    )
    if view.get("schema") != R2D12_VIEW_SCHEMA:
        raise ValueError("R2-D12 view schema mismatch")
    view_id = _require_sha256(
        view.get("view_id"),
        label="R2-D12 view ID",
    )
    body = dict(view)
    body.pop("view_id", None)
    if view_id != _view_identity(body):
        raise ValueError("R2-D12 view identity mismatch")
    return view


def _load_mounts_file(
    view_dir: Path,
    *,
    view_id: str,
) -> dict[str, str]:
    payload = _strict_json(
        view_dir / "r2d12-mounts.json",
        label="R2-D12 mounts",
    )
    if (
        payload.get("schema") != R2D12_MOUNTS_SCHEMA
        or payload.get("view_id") != view_id
    ):
        raise ValueError("R2-D12 mounts identity mismatch")
    raw = payload.get("packages")
    if not isinstance(raw, dict) or not raw:
        raise ValueError("R2-D12 mounts package map is invalid")
    mounts: dict[str, str] = {}
    for pack_id, path in raw.items():
        _require_sha256(pack_id, label="R2-D12 mounts pack ID")
        mounts[pack_id] = _safe_relative_path(
            path,
            label="R2-D12 mounts package path",
        ).as_posix()
    return dict(sorted(mounts.items()))


def verify_r2d12_view(
    view_dir: Path,
    *,
    workspace_root: Path,
) -> dict[str, object]:
    if view_dir.is_symlink():
        raise ValueError("R2-D12 view-dir must not be a symlink")
    view_root = view_dir.resolve(strict=True)
    if workspace_root.is_symlink():
        raise ValueError("R2-D12 workspace root must not be a symlink")
    workspace = workspace_root.resolve(strict=True)

    view = _load_view_file(view_root)
    view_id = str(view["view_id"])
    virtual_index = view.get("virtual_index")
    batches = view.get("attached_batches")
    if (
        not isinstance(virtual_index, dict)
        or not isinstance(batches, list)
        or not batches
    ):
        raise ValueError("R2-D12 view payload is incomplete")

    index_id = _require_sha256(
        virtual_index.get("index_id"),
        label="R2-D12 virtual index ID",
    )
    body = dict(virtual_index)
    body.pop("index_id", None)
    if index_id != _index_identity(body):
        raise ValueError("R2-D12 virtual index identity mismatch")

    tokenizer_sha = _require_sha256(
        virtual_index.get("tokenizer_sha256"),
        label="R2-D12 tokenizer SHA-256",
    )
    if _sha256_file(
        view_root / "tokenizer.vn97tk1"
    ) != tokenizer_sha:
        raise ValueError("R2-D12 tokenizer identity mismatch")

    expected: dict[str, str] = {}
    for item in batches:
        if not isinstance(item, dict) or set(item) != {
            "pack_id",
            "d6_index_id",
        }:
            raise ValueError("R2-D12 batch freeze entry is invalid")
        pack_id = _require_sha256(
            item.get("pack_id"),
            label="R2-D12 frozen pack ID",
        )
        index = _require_sha256(
            item.get("d6_index_id"),
            label="R2-D12 frozen D6 index ID",
        )
        if pack_id in expected:
            raise ValueError("R2-D12 frozen pack repeats")
        expected[pack_id] = index

    mounts = _load_mounts_file(
        view_root,
        view_id=view_id,
    )
    if sorted(mounts) != sorted(expected):
        raise ValueError(
            "R2-D12 runtime mounts do not cover the frozen batch set"
        )

    batch_indexes: dict[str, dict[str, object]] = {}
    shard_roots: dict[str, Path] = {}
    for pack_id, relative in sorted(mounts.items()):
        candidate = workspace / _safe_relative_path(
            relative,
            label="R2-D12 mounted package path",
        )
        if candidate.is_symlink():
            raise ValueError("R2-D12 mounted D6 package is a symlink")
        package = candidate.resolve(strict=True)
        try:
            package.relative_to(workspace)
        except ValueError as exc:
            raise ValueError(
                "R2-D12 mounted D6 package escapes workspace"
            ) from exc
        index = verify_r2d6_corpus_index(package)
        if index.get("index_id") != expected[pack_id]:
            raise ValueError(
                "R2-D12 mounted D6 index differs from frozen evidence"
            )
        batch_indexes[pack_id] = index
        shard_roots[pack_id] = package

    rebuilt = _combine_indexes(
        registry_config={
            "tokenizer_sha256": virtual_index["tokenizer_sha256"],
            "architecture_fingerprint": virtual_index[
                "architecture_fingerprint"
            ],
            "sequence_length": virtual_index["sequence_length"],
            "parameter_count": virtual_index["scale"][
                "parameter_count"
            ],
        },
        batch_indexes=batch_indexes,
    )
    if rebuilt != virtual_index:
        raise ValueError(
            "R2-D12 virtual index does not match mounted D6 packages"
        )
    return {
        "view": view,
        "index": virtual_index,
        "shard_roots": shard_roots,
        "tokenizer_path": view_root / "tokenizer.vn97tk1",
    }


def project_r2d12_index(
    index: Mapping[str, object],
    *,
    families: Sequence[str],
) -> dict[str, object]:
    selected = tuple(sorted(set(families)))
    if not selected or len(selected) != len(tuple(families)):
        raise ValueError(
            "R2-D12 projection families must be unique and non-empty"
        )
    if any(
        family not in {
            item
            for values in R2D8_ALLOWED_FAMILIES.values()
            for item in values
        }
        for family in selected
    ):
        raise ValueError("R2-D12 projection family is not canonical")

    manifests = index.get("source_manifests")
    shards = index.get("shards")
    if not isinstance(manifests, list) or not isinstance(shards, list):
        raise ValueError("R2-D12 virtual index metadata is invalid")

    selected_manifests: list[dict[str, object]] = []
    manifest_ids: set[str] = set()
    for raw in manifests:
        if not isinstance(raw, dict):
            raise ValueError("R2-D12 source manifest entry is invalid")
        raw_families = raw.get("task_families")
        if (
            not isinstance(raw_families, list)
            or len(raw_families) != 1
        ):
            raise ValueError(
                "R2-D12 source manifest must have one task family"
            )
        if raw_families[0] in selected:
            manifest_id = _require_sha256(
                raw.get("manifest_id"),
                label="R2-D12 projected manifest ID",
            )
            manifest_ids.add(manifest_id)
            selected_manifests.append(dict(raw))
    if not selected_manifests:
        raise ValueError(
            "R2-D12 projection contains no source manifests"
        )

    selected_shards = [
        dict(raw)
        for raw in shards
        if isinstance(raw, dict)
        and raw.get("source_manifest_id") in manifest_ids
    ]
    if not selected_shards:
        raise ValueError("R2-D12 projection contains no shards")

    split_totals = _zero_split_totals()
    for shard in selected_shards:
        split = str(shard.get("split"))
        if split not in split_totals:
            raise ValueError("R2-D12 projected shard split is invalid")
        for key in (
            "bytes",
            "records",
            "input_tokens",
            "target_tokens",
            "windows",
        ):
            value = int(shard.get(key, -1))
            if value <= 0:
                raise ValueError(
                    "R2-D12 projected shard evidence is invalid"
                )
            split_totals[split][key] += value

    parameter_count = int(index["scale"]["parameter_count"])
    body: dict[str, object] = {
        "schema": R2D12_INDEX_SCHEMA,
        "architecture_fingerprint": index[
            "architecture_fingerprint"
        ],
        "corpus_manifest_ids": sorted(manifest_ids),
        "release_held_out": True,
        "scale": _scale_from_totals(
            parameter_count=parameter_count,
            split_totals=split_totals,
        ),
        "source_manifests": sorted(
            selected_manifests,
            key=lambda item: (
                str(item["batch_pack_id"]),
                str(item["manifest_id"]),
            ),
        ),
        "sequence_length": index["sequence_length"],
        "shards": sorted(
            selected_shards,
            key=lambda item: (
                str(item["split"]),
                str(item["source_manifest_id"]),
                str(item["shard_id"]),
            ),
        ),
        "split_totals": split_totals,
        "tokenizer_sha256": index["tokenizer_sha256"],
    }
    payload = dict(body)
    payload["index_id"] = _index_identity(body)
    return payload


def projection_for_stage(
    index: Mapping[str, object],
    *,
    stage: str,
    family_weights: Mapping[str, object],
) -> dict[str, object]:
    if stage not in R2_PRODUCTION_STAGES:
        raise ValueError("R2-D12 projection stage is invalid")
    allowed = set(R2D8_ALLOWED_FAMILIES[stage])
    families = tuple(sorted(str(item) for item in family_weights))
    if not families or any(item not in allowed for item in families):
        raise ValueError(
            "R2-D12 projection families are invalid for the stage"
        )
    return project_r2d12_index(
        index,
        families=families,
    )


def resolve_r2d12_shard(
    shard: Mapping[str, object],
    shard_roots: Mapping[str, Path],
) -> Path:
    pack_id = _require_sha256(
        shard.get("batch_pack_id"),
        label="R2-D12 shard batch pack ID",
    )
    root = shard_roots.get(pack_id)
    if root is None:
        raise ValueError("R2-D12 shard batch is not mounted")
    relative = _safe_relative_path(
        shard.get("filename"),
        label="R2-D12 shard filename",
    )
    candidate = root / relative
    if candidate.is_symlink():
        raise ValueError("R2-D12 shard must not be a symlink")
    path = candidate.resolve(strict=True)
    try:
        path.relative_to(root.resolve(strict=True))
    except ValueError as exc:
        raise ValueError("R2-D12 shard escapes mounted D6 package") from exc
    if not path.is_file():
        raise ValueError("R2-D12 shard path is not a regular file")
    return path
