from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

from ..corpus_io import (
    atomic_write,
    load_records,
    read_bounded_regular_file,
)
from ..production_corpus import prepare_corpus
from .pilot_contract import chat_record_digest
from .production_campaign import verify_r2d5_corpus
from .production_curriculum import R2D8_ALLOWED_FAMILIES


R2D9_DEFINITION_SCHEMA = "VN97R2D9DEF1"
R2D9_CAMPAIGN_SCHEMA = "VN97R2D9CAMPAIGN1"
R2D9_PROFILE_ID = "vn97-production-intelligence-v1"
R2D9_SPLITS = ("training", "validation", "release")
R2D9_ALLOWED_FAMILIES = tuple(
    sorted(
        {
            family
            for families in R2D8_ALLOWED_FAMILIES.values()
            for family in families
        }
    )
)


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


def _strict_json_object(data: bytes, *, label: str) -> dict[str, object]:
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


def _require_sha256(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(ch not in "0123456789abcdef" for ch in value)
    ):
        raise ValueError(f"{label} must be lowercase SHA-256")
    return value


def _bounded_text(
    value: object,
    *,
    label: str,
    max_bytes: int,
) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be non-empty text")
    encoded = value.encode("utf-8")
    if len(encoded) > max_bytes:
        raise ValueError(f"{label} is too long")
    return value


def _safe_relative_path(raw: object, *, label: str) -> Path:
    value = _bounded_text(
        raw,
        label=label,
        max_bytes=4096,
    )
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{label} must be a safe relative path")
    return path


@dataclass(frozen=True)
class R2D9SourceSpec:
    source_id: str
    origin: str
    revision: str
    license: str
    family: str
    path: str
    expected_sha256: str
    expected_records: int
    max_bytes: int

    def __post_init__(self) -> None:
        _bounded_text(
            self.source_id,
            label="source_id",
            max_bytes=256,
        )
        _bounded_text(
            self.origin,
            label="origin",
            max_bytes=4096,
        )
        _bounded_text(
            self.revision,
            label="revision",
            max_bytes=512,
        )
        _bounded_text(
            self.license,
            label="license",
            max_bytes=512,
        )
        if self.family not in R2D9_ALLOWED_FAMILIES:
            raise ValueError("R2-D9 source family is not canonical")
        _safe_relative_path(
            self.path,
            label="source path",
        )
        _require_sha256(
            self.expected_sha256,
            label="source expected_sha256",
        )
        if self.expected_records <= 0:
            raise ValueError("source expected_records must be positive")
        if self.max_bytes <= 0:
            raise ValueError("source max_bytes must be positive")

    def canonical_object(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class R2D9CampaignDefinition:
    shard_target_training_records: int
    validation_fraction: float
    release_fraction: float
    sources: tuple[R2D9SourceSpec, ...]

    def __post_init__(self) -> None:
        if self.shard_target_training_records <= 0:
            raise ValueError(
                "shard_target_training_records must be positive"
            )
        if (
            not math.isfinite(self.validation_fraction)
            or not math.isfinite(self.release_fraction)
            or self.validation_fraction <= 0.0
            or self.release_fraction <= 0.0
            or self.validation_fraction >= 0.25
            or self.release_fraction >= 0.25
            or self.validation_fraction + self.release_fraction >= 0.5
        ):
            raise ValueError("R2-D9 holdout fractions are invalid")
        if not self.sources:
            raise ValueError("R2-D9 campaign needs sources")
        ids = [item.source_id for item in self.sources]
        if ids != sorted(set(ids)):
            raise ValueError(
                "R2-D9 source IDs must be sorted and unique"
            )

    def canonical_object(self) -> dict[str, object]:
        return {
            "schema": R2D9_DEFINITION_SCHEMA,
            "profile_id": R2D9_PROFILE_ID,
            "shard_target_training_records": (
                self.shard_target_training_records
            ),
            "validation_fraction": self.validation_fraction,
            "release_fraction": self.release_fraction,
            "sources": [
                item.canonical_object()
                for item in self.sources
            ],
        }

    def fingerprint(self) -> str:
        return _sha256_bytes(
            b"VN97R2D9DEF1\0"
            + _canonical_json(self.canonical_object())
        )


@dataclass(frozen=True)
class _AcceptedRecord:
    digest: str
    source_id: str
    origin: str
    revision: str
    license: str
    family: str
    raw_source_sha256: str
    raw_source_bytes: int
    messages: tuple[object, ...]


def load_r2d9_definition(path: Path) -> R2D9CampaignDefinition:
    payload = _strict_json_object(
        path.resolve(strict=True).read_bytes(),
        label="R2-D9 definition",
    )
    if set(payload) != {
        "schema",
        "profile_id",
        "shard_target_training_records",
        "validation_fraction",
        "release_fraction",
        "sources",
    }:
        raise ValueError("R2-D9 definition fields are invalid")
    if payload.get("schema") != R2D9_DEFINITION_SCHEMA:
        raise ValueError("R2-D9 definition schema mismatch")
    if payload.get("profile_id") != R2D9_PROFILE_ID:
        raise ValueError("R2-D9 profile_id mismatch")
    raw_sources = payload.get("sources")
    if not isinstance(raw_sources, list) or not raw_sources:
        raise ValueError("R2-D9 source list is invalid")

    required = {
        "source_id",
        "origin",
        "revision",
        "license",
        "license_approved",
        "family",
        "path",
        "expected_sha256",
        "expected_records",
        "max_bytes",
    }
    sources: list[R2D9SourceSpec] = []
    for raw in raw_sources:
        if not isinstance(raw, dict) or set(raw) != required:
            raise ValueError("R2-D9 source fields are invalid")
        if raw.get("license_approved") is not True:
            raise ValueError(
                "R2-D9 every source requires license_approved=true"
            )
        for key in (
            "source_id",
            "origin",
            "revision",
            "license",
            "family",
            "path",
            "expected_sha256",
        ):
            if not isinstance(raw.get(key), str):
                raise ValueError(
                    f"R2-D9 source {key} must be a string"
                )
        if (
            not isinstance(raw.get("expected_records"), int)
            or isinstance(raw.get("expected_records"), bool)
            or not isinstance(raw.get("max_bytes"), int)
            or isinstance(raw.get("max_bytes"), bool)
        ):
            raise ValueError(
                "R2-D9 source expected_records/max_bytes must be integers"
            )
        sources.append(
            R2D9SourceSpec(
                source_id=raw["source_id"],
                origin=raw["origin"],
                revision=raw["revision"],
                license=raw["license"],
                family=raw["family"],
                path=raw["path"],
                expected_sha256=raw["expected_sha256"],
                expected_records=raw["expected_records"],
                max_bytes=raw["max_bytes"],
            )
        )
    shard_target = payload["shard_target_training_records"]
    validation_fraction = payload["validation_fraction"]
    release_fraction = payload["release_fraction"]
    if (
        not isinstance(shard_target, int)
        or isinstance(shard_target, bool)
        or not isinstance(validation_fraction, (int, float))
        or isinstance(validation_fraction, bool)
        or not isinstance(release_fraction, (int, float))
        or isinstance(release_fraction, bool)
    ):
        raise ValueError("R2-D9 campaign numeric fields are invalid")
    return R2D9CampaignDefinition(
        shard_target_training_records=shard_target,
        validation_fraction=float(validation_fraction),
        release_fraction=float(release_fraction),
        sources=tuple(
            sorted(sources, key=lambda item: item.source_id)
        ),
    )


def _resolve_source(definition_path: Path, spec: R2D9SourceSpec) -> Path:
    root = definition_path.parent.resolve(strict=True)
    relative = _safe_relative_path(
        spec.path,
        label="source path",
    )
    candidate = root / relative
    if candidate.is_symlink():
        raise ValueError(
            "R2-D9 source must be a regular non-symlink file"
        )
    path = candidate.resolve(strict=True)
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            "R2-D9 source path escapes definition directory"
        ) from exc
    if not path.is_file():
        raise ValueError("R2-D9 source must be a regular non-symlink file")
    return path


def _partition_counts(
    total: int,
    *,
    validation_fraction: float,
    release_fraction: float,
) -> tuple[int, int, int]:
    if total < 3:
        raise ValueError(
            "R2-D9 family needs at least 3 globally unique records"
        )
    validation = max(
        1,
        int(round(total * validation_fraction)),
    )
    release = max(
        1,
        int(round(total * release_fraction)),
    )
    while validation + release >= total:
        if validation >= release and validation > 1:
            validation -= 1
        elif release > 1:
            release -= 1
        else:
            break
    training = total - validation - release
    if training <= 0:
        raise ValueError("R2-D9 family has no training records")
    return training, validation, release


def _balanced_chunks(
    values: Sequence[_AcceptedRecord],
    count: int,
) -> tuple[tuple[_AcceptedRecord, ...], ...]:
    if count <= 0 or count > len(values):
        raise ValueError("R2-D9 chunk count is invalid")
    buckets: list[list[_AcceptedRecord]] = [
        [] for _ in range(count)
    ]
    for index, value in enumerate(values):
        buckets[index % count].append(value)
    if any(not bucket for bucket in buckets):
        raise RuntimeError("R2-D9 produced an empty split chunk")
    return tuple(tuple(bucket) for bucket in buckets)


def _seal_rows_for_split(
    records: Sequence[_AcceptedRecord],
    *,
    split: str,
    seal_index: int,
    raw_by_source: Mapping[str, bytes],
) -> list[
    tuple[str, str, str, str, str, bytes, Sequence[object]]
]:
    grouped: dict[str, list[object]] = {}
    metadata: dict[str, _AcceptedRecord] = {}
    for record in records:
        grouped.setdefault(record.source_id, []).append(record.messages)
        metadata.setdefault(record.source_id, record)

    rows = []
    for source_id in sorted(grouped):
        record = metadata[source_id]
        derived_source_id = (
            "r2d9-"
            + hashlib.sha256(
                source_id.encode("utf-8")
            ).hexdigest()
            + f"-{split}-{seal_index:05d}"
        )
        rows.append(
            (
                derived_source_id,
                record.origin,
                record.license,
                split,
                "chat",
                raw_by_source[source_id],
                tuple(grouped[source_id]),
            )
        )
    return rows


def build_r2d9_campaign(
    *,
    definition_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    definition_file = definition_path.resolve(strict=True)
    definition = load_r2d9_definition(definition_file)

    if output_dir.exists():
        if (
            output_dir.is_symlink()
            or not output_dir.is_dir()
            or any(output_dir.iterdir())
        ):
            raise ValueError("R2-D9 output-dir must be new or empty")
    else:
        output_dir.mkdir(parents=True, exist_ok=False)
    output = output_dir.resolve(strict=True)
    seals_root = output / "seals"
    seals_root.mkdir()

    accepted_by_family: dict[str, list[_AcceptedRecord]] = {}
    digest_owner: dict[str, tuple[str, str]] = {}
    raw_by_source: dict[str, bytes] = {}
    source_receipts: list[dict[str, object]] = []
    physical_sources: set[tuple[int, int]] = set()

    for spec in definition.sources:
        path = _resolve_source(definition_file, spec)
        info = path.stat()
        physical_identity = (
            int(info.st_dev),
            int(info.st_ino),
        )
        if physical_identity in physical_sources:
            raise ValueError(
                "R2-D9 source files must be physically distinct"
            )
        physical_sources.add(physical_identity)
        raw = read_bounded_regular_file(
            path,
            max_bytes=spec.max_bytes,
        )
        sha = _sha256_bytes(raw)
        if sha != spec.expected_sha256:
            raise ValueError(
                f"R2-D9 source SHA-256 mismatch: {spec.source_id}"
            )
        records, _ = load_records(
            [path],
            mode="chat",
            max_input_bytes=spec.max_bytes,
            max_examples=spec.expected_records,
        )
        if len(records) != spec.expected_records:
            raise ValueError(
                f"R2-D9 source record-count mismatch: {spec.source_id}"
            )

        raw_by_source[spec.source_id] = raw
        kept = 0
        duplicate_same_family = 0
        for messages in records:
            digest = chat_record_digest(messages)
            prior = digest_owner.get(digest)
            if prior is not None:
                prior_source, prior_family = prior
                if prior_family != spec.family:
                    raise ValueError(
                        "R2-D9 duplicate record crosses task families: "
                        f"{prior_source}/{prior_family} vs "
                        f"{spec.source_id}/{spec.family}"
                    )
                duplicate_same_family += 1
                continue
            digest_owner[digest] = (
                spec.source_id,
                spec.family,
            )
            accepted_by_family.setdefault(
                spec.family,
                [],
            ).append(
                _AcceptedRecord(
                    digest=digest,
                    source_id=spec.source_id,
                    origin=spec.origin,
                    revision=spec.revision,
                    license=spec.license,
                    family=spec.family,
                    raw_source_sha256=sha,
                    raw_source_bytes=len(raw),
                    messages=tuple(messages),
                )
            )
            kept += 1

        source_receipts.append(
            {
                "source_id": spec.source_id,
                "origin": spec.origin,
                "revision": spec.revision,
                "license": spec.license,
                "license_approved": True,
                "family": spec.family,
                "sha256": sha,
                "bytes": len(raw),
                "input_records": len(records),
                "accepted_records": kept,
                "duplicate_records": duplicate_same_family,
            }
        )

    if not accepted_by_family:
        raise RuntimeError("R2-D9 campaign accepted no records")

    seals: list[dict[str, object]] = []
    d6_corpora: list[dict[str, object]] = []
    assignment_map: dict[str, str] = {}

    for family in sorted(accepted_by_family):
        unique = sorted(
            accepted_by_family[family],
            key=lambda item: item.digest,
        )
        train_count, validation_count, release_count = _partition_counts(
            len(unique),
            validation_fraction=definition.validation_fraction,
            release_fraction=definition.release_fraction,
        )
        training = unique[:train_count]
        validation = unique[
            train_count : train_count + validation_count
        ]
        release = unique[train_count + validation_count :]

        desired_seals = math.ceil(
            train_count
            / definition.shard_target_training_records
        )
        if (
            validation_count < desired_seals
            or release_count < desired_seals
        ):
            raise RuntimeError(
                "R2-D9 holdout volume is too small for requested shard "
                f"granularity: family={family} desired_seals={desired_seals} "
                f"validation={validation_count} release={release_count}"
            )

        train_chunks = _balanced_chunks(training, desired_seals)
        validation_chunks = _balanced_chunks(
            validation,
            desired_seals,
        )
        release_chunks = _balanced_chunks(
            release,
            desired_seals,
        )

        for seal_index in range(desired_seals):
            source_rows = []
            source_rows.extend(
                _seal_rows_for_split(
                    train_chunks[seal_index],
                    split="training",
                    seal_index=seal_index,
                    raw_by_source=raw_by_source,
                )
            )
            source_rows.extend(
                _seal_rows_for_split(
                    validation_chunks[seal_index],
                    split="validation",
                    seal_index=seal_index,
                    raw_by_source=raw_by_source,
                )
            )
            source_rows.extend(
                _seal_rows_for_split(
                    release_chunks[seal_index],
                    split="release",
                    seal_index=seal_index,
                    raw_by_source=raw_by_source,
                )
            )

            manifest, prepared = prepare_corpus(
                source_rows,
                profile_id=R2D9_PROFILE_ID,
            )
            seal_name = f"{family}-{seal_index:05d}"
            seal_dir = seals_root / seal_name
            seal_dir.mkdir()
            for split in R2D9_SPLITS:
                atomic_write(
                    seal_dir / f"{split}.jsonl",
                    prepared[split].bytes,
                )
            atomic_write(
                seal_dir / "corpus.vn97corpus1.json",
                manifest.to_bytes(),
            )
            verified = verify_r2d5_corpus(seal_dir)
            if verified.manifest_id != manifest.manifest_id:
                raise RuntimeError("R2-D9 post-seal identity mismatch")

            relative = f"seals/{seal_name}"
            seals.append(
                {
                    "family": family,
                    "corpus_path": relative,
                    "manifest_id": manifest.manifest_id,
                    "manifest_sha256": verified.manifest_sha256,
                    "training_records": manifest.training.records,
                    "validation_records": manifest.validation.records,
                    "release_records": manifest.release.records,
                    "source_licenses": list(
                        verified.source_licenses
                    ),
                }
            )
            d6_corpora.append(
                {
                    "path": relative,
                    "task_families": [family],
                }
            )
            assignment_map[manifest.manifest_id] = family

    d6_definition = {
        "schema": "VN97R2D6DEF1",
        "corpora": sorted(
            d6_corpora,
            key=lambda item: item["path"],
        ),
    }
    atomic_write(
        output / "r2d6-definition.json",
        json.dumps(
            d6_definition,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        ).encode("utf-8") + b"\n",
    )
    atomic_write(
        output / "r2d8-primary-family.json",
        json.dumps(
            dict(sorted(assignment_map.items())),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        ).encode("utf-8") + b"\n",
    )

    family_totals = {}
    for family in sorted(accepted_by_family):
        family_seals = [
            item for item in seals if item["family"] == family
        ]
        family_totals[family] = {
            "unique_records": len(accepted_by_family[family]),
            "seals": len(family_seals),
            "training_records": sum(
                int(item["training_records"])
                for item in family_seals
            ),
            "validation_records": sum(
                int(item["validation_records"])
                for item in family_seals
            ),
            "release_records": sum(
                int(item["release_records"])
                for item in family_seals
            ),
        }

    body = {
        "schema": R2D9_CAMPAIGN_SCHEMA,
        "definition_fingerprint": definition.fingerprint(),
        "profile_id": R2D9_PROFILE_ID,
        "global_unique_records": len(digest_owner),
        "source_receipts": sorted(
            source_receipts,
            key=lambda item: str(item["source_id"]),
        ),
        "family_totals": family_totals,
        "seals": sorted(
            seals,
            key=lambda item: (
                str(item["family"]),
                str(item["manifest_id"]),
            ),
        ),
        "d6_definition_sha256": _sha256_bytes(
            (output / "r2d6-definition.json").read_bytes()
        ),
        "d8_primary_family_sha256": _sha256_bytes(
            (output / "r2d8-primary-family.json").read_bytes()
        ),
    }
    payload = dict(body)
    payload["campaign_id"] = _sha256_bytes(
        b"VN97R2D9CAMPAIGN1\0" + _canonical_json(body)
    )
    atomic_write(
        output / "r2d9-campaign.json",
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        ).encode("utf-8") + b"\n",
    )
    return payload


def verify_r2d9_campaign(output_dir: Path) -> dict[str, object]:
    root = output_dir.resolve(strict=True)
    payload = _strict_json_object(
        (root / "r2d9-campaign.json").read_bytes(),
        label="R2-D9 campaign",
    )
    if payload.get("schema") != R2D9_CAMPAIGN_SCHEMA:
        raise ValueError("R2-D9 campaign schema mismatch")
    campaign_id = _require_sha256(
        payload.get("campaign_id"),
        label="R2-D9 campaign_id",
    )
    body = dict(payload)
    body.pop("campaign_id", None)
    expected = _sha256_bytes(
        b"VN97R2D9CAMPAIGN1\0" + _canonical_json(body)
    )
    if campaign_id != expected:
        raise ValueError("R2-D9 campaign identity mismatch")

    if _sha256_bytes(
        (root / "r2d6-definition.json").read_bytes()
    ) != payload.get("d6_definition_sha256"):
        raise ValueError("R2-D9 D6 definition hash mismatch")
    if _sha256_bytes(
        (root / "r2d8-primary-family.json").read_bytes()
    ) != payload.get("d8_primary_family_sha256"):
        raise ValueError(
            "R2-D9 D8 primary-family map hash mismatch"
        )

    raw_seals = payload.get("seals")
    if not isinstance(raw_seals, list) or not raw_seals:
        raise ValueError("R2-D9 seal list is invalid")
    seen_manifests: set[str] = set()
    total_records = 0
    assignments: dict[str, str] = {}
    d6_entries = []
    for raw in raw_seals:
        if not isinstance(raw, dict):
            raise ValueError("R2-D9 seal entry is invalid")
        family = str(raw.get("family"))
        if family not in R2D9_ALLOWED_FAMILIES:
            raise ValueError("R2-D9 seal family is invalid")
        relative = _safe_relative_path(
            raw.get("corpus_path"),
            label="seal corpus_path",
        )
        seal = (root / relative).resolve(strict=True)
        try:
            seal.relative_to(root)
        except ValueError as exc:
            raise ValueError("R2-D9 seal path escapes campaign") from exc

        verified = verify_r2d5_corpus(seal)
        manifest_id = _require_sha256(
            raw.get("manifest_id"),
            label="R2-D9 seal manifest_id",
        )
        if verified.manifest_id != manifest_id:
            raise ValueError("R2-D9 seal manifest identity mismatch")
        if manifest_id in seen_manifests:
            raise ValueError("R2-D9 duplicate sealed manifest")
        seen_manifests.add(manifest_id)
        if verified.manifest_sha256 != raw.get("manifest_sha256"):
            raise ValueError("R2-D9 seal manifest hash mismatch")

        split_counts = verified.split_identity
        for split in R2D9_SPLITS:
            expected_count = int(raw.get(f"{split}_records", -1))
            if int(split_counts[split]["records"]) != expected_count:
                raise ValueError(
                    f"R2-D9 seal {split} count mismatch"
                )
            total_records += expected_count

        assignments[manifest_id] = family
        d6_entries.append(
            {
                "path": str(relative).replace("\\", "/"),
                "task_families": [family],
            }
        )

    d6_expected = {
        "schema": "VN97R2D6DEF1",
        "corpora": sorted(
            d6_entries,
            key=lambda item: item["path"],
        ),
    }
    d6_actual = _strict_json_object(
        (root / "r2d6-definition.json").read_bytes(),
        label="R2-D9 D6 definition",
    )
    if d6_actual != d6_expected:
        raise ValueError("R2-D9 D6 definition content mismatch")

    assignments_actual = _strict_json_object(
        (root / "r2d8-primary-family.json").read_bytes(),
        label="R2-D9 D8 primary-family map",
    )
    if assignments_actual != dict(sorted(assignments.items())):
        raise ValueError("R2-D9 D8 family map content mismatch")

    family_totals = payload.get("family_totals")
    if not isinstance(family_totals, dict):
        raise ValueError("R2-D9 family totals are missing")

    recomputed_family_totals: dict[str, dict[str, int]] = {}
    for raw in raw_seals:
        family = str(raw["family"])
        totals = recomputed_family_totals.setdefault(
            family,
            {
                "unique_records": 0,
                "seals": 0,
                "training_records": 0,
                "validation_records": 0,
                "release_records": 0,
            },
        )
        totals["seals"] += 1
        for split in R2D9_SPLITS:
            value = int(raw[f"{split}_records"])
            totals[f"{split}_records"] += value
            totals["unique_records"] += value

    if family_totals != recomputed_family_totals:
        raise ValueError("R2-D9 family totals mismatch")

    global_unique = int(payload.get("global_unique_records", -1))
    if sum(
        value["unique_records"]
        for value in recomputed_family_totals.values()
    ) != global_unique:
        raise ValueError("R2-D9 global unique-record total mismatch")
    if total_records != global_unique:
        raise ValueError("R2-D9 sealed record total mismatch")

    receipts = payload.get("source_receipts")
    if not isinstance(receipts, list) or not receipts:
        raise ValueError("R2-D9 source receipts are missing")
    receipt_ids: set[str] = set()
    accepted_total = 0
    for receipt in receipts:
        if not isinstance(receipt, dict):
            raise ValueError("R2-D9 source receipt is invalid")
        source_id = str(receipt.get("source_id", ""))
        if not source_id or source_id in receipt_ids:
            raise ValueError("R2-D9 source receipt IDs are invalid")
        receipt_ids.add(source_id)
        if receipt.get("license_approved") is not True:
            raise ValueError("R2-D9 source license approval is invalid")
        if str(receipt.get("family")) not in R2D9_ALLOWED_FAMILIES:
            raise ValueError("R2-D9 source receipt family is invalid")
        _require_sha256(
            receipt.get("sha256"),
            label="R2-D9 source receipt SHA-256",
        )
        input_records = int(receipt.get("input_records", -1))
        accepted = int(receipt.get("accepted_records", -1))
        duplicates = int(receipt.get("duplicate_records", -1))
        if (
            input_records <= 0
            or accepted < 0
            or duplicates < 0
            or accepted + duplicates != input_records
        ):
            raise ValueError("R2-D9 source receipt counts are invalid")
        accepted_total += accepted
    if accepted_total != global_unique:
        raise ValueError(
            "R2-D9 source accepted-record total mismatch"
        )
    return payload
