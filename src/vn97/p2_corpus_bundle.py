from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import stat


class VN97P2CorpusBundleError(RuntimeError):
    pass


_PROFILE_ID = "vn97-production-intelligence-v1"
_SPLITS = ("training", "validation", "release")
_MAX_JSON_BYTES = 4 * 1024 * 1024
_MAX_SPLIT_BYTES = 128 * 1024 * 1024


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _require_sha256(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(ch not in "0123456789abcdef" for ch in value)
    ):
        raise VN97P2CorpusBundleError(
            f"{label} must be lowercase SHA-256"
        )
    return value


def _read_regular(
    path: Path,
    *,
    max_bytes: int,
    label: str,
) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise VN97P2CorpusBundleError(
            f"{label} could not be opened safely"
        ) from exc
    try:
        info = os.fstat(fd)
        if (
            not stat.S_ISREG(info.st_mode)
            or not 0 < info.st_size <= max_bytes
        ):
            raise VN97P2CorpusBundleError(
                f"{label} size/type is invalid"
            )
        out = bytearray()
        while len(out) < info.st_size:
            chunk = os.read(
                fd,
                min(1024 * 1024, info.st_size - len(out)),
            )
            if not chunk:
                break
            out.extend(chunk)
        after = os.fstat(fd)
        if (
            len(out) != info.st_size
            or after.st_size != info.st_size
            or after.st_dev != info.st_dev
            or after.st_ino != info.st_ino
        ):
            raise VN97P2CorpusBundleError(
                f"{label} changed while being read"
            )
        return bytes(out)
    finally:
        os.close(fd)


def _strict_canonical_json(
    path: Path,
    *,
    label: str,
) -> tuple[dict[str, object], bytes]:
    data = _read_regular(
        path,
        max_bytes=_MAX_JSON_BYTES,
        label=label,
    )
    duplicates: list[str] = []

    def hook(pairs):
        output = {}
        for key, value in pairs:
            if key in output:
                duplicates.append(key)
            output[key] = value
        return output

    try:
        text = data.decode("utf-8", errors="strict")
        body = text[:-1] if text.endswith("\n") else text
        value = json.loads(
            body,
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
        raise VN97P2CorpusBundleError(
            f"{label} must be strict UTF-8 JSON"
        ) from exc
    if duplicates or not isinstance(value, dict):
        raise VN97P2CorpusBundleError(
            f"{label} must be one object without duplicate keys"
        )
    canonical = _canonical_json(value).decode("utf-8")
    expected = canonical + ("\n" if text.endswith("\n") else "")
    if text != expected:
        raise VN97P2CorpusBundleError(
            f"{label} must use canonical JSON"
        )
    return value, data


def _verify_identity(
    value: dict[str, object],
    *,
    id_key: str,
    prefix: bytes,
    label: str,
) -> str:
    identity = _require_sha256(
        value.get(id_key),
        label=f"{label} identity",
    )
    body = dict(value)
    body.pop(id_key)
    expected = hashlib.sha256(
        prefix + _canonical_json(body)
    ).hexdigest()
    if identity != expected:
        raise VN97P2CorpusBundleError(
            f"{label} identity mismatch"
        )
    return identity


def _resolve_definition_source(
    definition_path: Path,
    relative: object,
) -> Path:
    if not isinstance(relative, str) or not relative:
        raise VN97P2CorpusBundleError(
            "corpus definition source path is invalid"
        )
    raw = Path(relative)
    if raw.is_absolute() or ".." in raw.parts:
        raise VN97P2CorpusBundleError(
            "corpus definition source path must stay relative"
        )
    target = definition_path.parent / raw
    if target.is_symlink():
        raise VN97P2CorpusBundleError(
            "corpus definition source must not be a symlink"
        )
    try:
        root = definition_path.parent.resolve(strict=True)
        resolved = target.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as exc:
        raise VN97P2CorpusBundleError(
            "corpus definition source escapes intake directory"
        ) from exc
    return resolved


@dataclass(frozen=True)
class VN97P2CorpusBundleReceipt:
    repository_commit: str
    fetch_receipt_sha256: str
    source_summary_sha256: str
    definition_sha256: str
    corpus_manifest_id: str
    corpus_manifest_sha256: str
    split_identity: dict[str, dict[str, object]]

    def __post_init__(self) -> None:
        if (
            len(self.repository_commit) != 40
            or any(
                ch not in "0123456789abcdef"
                for ch in self.repository_commit
            )
        ):
            raise VN97P2CorpusBundleError(
                "repository_commit must be 40 lowercase hex"
            )
        for value, label in (
            (self.fetch_receipt_sha256, "fetch receipt SHA-256"),
            (self.source_summary_sha256, "source summary SHA-256"),
            (self.definition_sha256, "definition SHA-256"),
            (self.corpus_manifest_id, "corpus manifest ID"),
            (self.corpus_manifest_sha256, "corpus manifest SHA-256"),
        ):
            _require_sha256(value, label=label)
        if set(self.split_identity) != set(_SPLITS):
            raise VN97P2CorpusBundleError(
                "bundle split identity map is invalid"
            )

    def canonical_object(self) -> dict[str, object]:
        body = {
            "corpus_manifest_id": self.corpus_manifest_id,
            "corpus_manifest_sha256": self.corpus_manifest_sha256,
            "definition_sha256": self.definition_sha256,
            "fetch_receipt_sha256": self.fetch_receipt_sha256,
            "repository_commit": self.repository_commit,
            "schema": "VN97P2BUNDLE1",
            "source_summary_sha256": self.source_summary_sha256,
            "splits": self.split_identity,
        }
        body["bundle_id"] = hashlib.sha256(
            b"VN97P2BUNDLE1\0"
            + _canonical_json(body)
        ).hexdigest()
        return body

    def to_bytes(self) -> bytes:
        return _canonical_json(
            self.canonical_object()
        ) + b"\n"


def verify_p2_corpus_chain(
    *,
    fetch_receipt_path: Path,
    source_summary_path: Path,
    definition_path: Path,
    corpus_dir: Path,
    repository_commit: str,
) -> VN97P2CorpusBundleReceipt:
    fetch, fetch_bytes = _strict_canonical_json(
        fetch_receipt_path,
        label="VN97P2FETCH1 receipt",
    )
    if (
        set(fetch) != {
            "definition_sha256",
            "receipt_id",
            "schema",
            "sources",
        }
        or fetch.get("schema") != "VN97P2FETCH1"
        or not isinstance(fetch.get("sources"), list)
    ):
        raise VN97P2CorpusBundleError(
            "VN97P2FETCH1 receipt fields are invalid"
        )
    _verify_identity(
        fetch,
        id_key="receipt_id",
        prefix=b"VN97P2FETCH1\0",
        label="VN97P2FETCH1",
    )
    fetch_sources: dict[str, str] = {}
    for raw in fetch["sources"]:
        if (
            not isinstance(raw, dict)
            or not isinstance(raw.get("source_id"), str)
        ):
            raise VN97P2CorpusBundleError(
                "VN97P2FETCH1 source entry is invalid"
            )
        source_id = raw["source_id"]
        if source_id in fetch_sources:
            raise VN97P2CorpusBundleError(
                "VN97P2FETCH1 source IDs are not unique"
            )
        fetch_sources[source_id] = _require_sha256(
            raw.get("sha256"),
            label=f"{source_id} fetched SHA-256",
        )

    summary, summary_bytes = _strict_canonical_json(
        source_summary_path,
        label="VN97P2SRC1 summary",
    )
    if summary.get("schema") != "VN97P2SRC1":
        raise VN97P2CorpusBundleError(
            "source summary schema must be VN97P2SRC1"
        )
    _verify_identity(
        summary,
        id_key="summary_id",
        prefix=b"VN97P2SRC1\0",
        label="VN97P2SRC1",
    )
    raw_map = summary.get("raw_source_sha256")
    if not isinstance(raw_map, dict):
        raise VN97P2CorpusBundleError(
            "VN97P2SRC1 raw source map is invalid"
        )
    if set(raw_map) != set(fetch_sources):
        raise VN97P2CorpusBundleError(
            "VN97P2SRC1 raw source set does not match VN97P2FETCH1"
        )
    for source_id, expected_sha in fetch_sources.items():
        if raw_map.get(source_id) != expected_sha:
            raise VN97P2CorpusBundleError(
                f"raw source identity mismatch: {source_id}"
            )

    definition, definition_bytes = _strict_canonical_json(
        definition_path,
        label="VN97CORPUSDEF1 definition",
    )
    definition_sha256 = hashlib.sha256(
        definition_bytes
    ).hexdigest()
    if summary.get("definition_sha256") != definition_sha256:
        raise VN97P2CorpusBundleError(
            "VN97P2SRC1 definition hash mismatch"
        )
    if (
        set(definition) != {"profile_id", "schema", "sources"}
        or definition.get("schema") != "VN97CORPUSDEF1"
        or definition.get("profile_id") != _PROFILE_ID
        or not isinstance(definition.get("sources"), list)
    ):
        raise VN97P2CorpusBundleError(
            "VN97CORPUSDEF1 definition is invalid"
        )

    if corpus_dir.is_symlink():
        raise VN97P2CorpusBundleError(
            "VN97CORPUS1 directory must not be a symlink"
        )
    corpus_root = corpus_dir.resolve(strict=True)
    if not corpus_root.is_dir():
        raise VN97P2CorpusBundleError(
            "VN97CORPUS1 path must be a directory"
        )
    expected_entries = {
        "corpus.vn97corpus1.json",
        "training.jsonl",
        "validation.jsonl",
        "release.jsonl",
    }
    if {item.name for item in corpus_root.iterdir()} != expected_entries:
        raise VN97P2CorpusBundleError(
            "VN97CORPUS1 directory contains an unexpected entry set"
        )

    manifest_path = corpus_root / "corpus.vn97corpus1.json"
    manifest, manifest_bytes = _strict_canonical_json(
        manifest_path,
        label="VN97CORPUS1 manifest",
    )
    if (
        set(manifest) != {
            "manifest_id",
            "profile_id",
            "schema",
            "sources",
            "splits",
        }
        or manifest.get("schema") != "VN97CORPUS1"
        or manifest.get("profile_id") != _PROFILE_ID
        or not isinstance(manifest.get("sources"), list)
        or not isinstance(manifest.get("splits"), dict)
    ):
        raise VN97P2CorpusBundleError(
            "VN97CORPUS1 manifest fields are invalid"
        )
    manifest_id = _verify_identity(
        manifest,
        id_key="manifest_id",
        prefix=b"VN97CORPUS1\0",
        label="VN97CORPUS1",
    )

    definition_sources = definition["sources"]
    manifest_sources = manifest["sources"]
    manifest_by_id: dict[str, dict[str, object]] = {}
    for raw in manifest_sources:
        if (
            not isinstance(raw, dict)
            or not isinstance(raw.get("source_id"), str)
        ):
            raise VN97P2CorpusBundleError(
                "VN97CORPUS1 source entry is invalid"
            )
        source_id = raw["source_id"]
        if source_id in manifest_by_id:
            raise VN97P2CorpusBundleError(
                "VN97CORPUS1 source IDs are not unique"
            )
        manifest_by_id[source_id] = raw

    if {
        raw.get("source_id")
        for raw in definition_sources
        if isinstance(raw, dict)
    } != set(manifest_by_id):
        raise VN97P2CorpusBundleError(
            "VN97CORPUSDEF1/VN97CORPUS1 source sets differ"
        )

    for raw in definition_sources:
        if not isinstance(raw, dict):
            raise VN97P2CorpusBundleError(
                "VN97CORPUSDEF1 source entry is invalid"
            )
        required = {
            "license",
            "license_approved",
            "mode",
            "origin",
            "path",
            "source_id",
            "split",
        }
        if set(raw) != required or raw.get("license_approved") is not True:
            raise VN97P2CorpusBundleError(
                "VN97CORPUSDEF1 source fields are invalid"
            )
        source_id = raw["source_id"]
        manifest_source = manifest_by_id[source_id]
        for key in (
            "license",
            "mode",
            "origin",
            "source_id",
            "split",
        ):
            if manifest_source.get(key) != raw.get(key):
                raise VN97P2CorpusBundleError(
                    f"corpus source metadata mismatch: {source_id}:{key}"
                )
        if manifest_source.get("license_approved") is not True:
            raise VN97P2CorpusBundleError(
                f"corpus source lost license approval: {source_id}"
            )
        source_path = _resolve_definition_source(
            definition_path,
            raw["path"],
        )
        source_bytes = _read_regular(
            source_path,
            max_bytes=_MAX_SPLIT_BYTES,
            label=f"intake source {source_id}",
        )
        if (
            manifest_source.get("content_bytes") != len(source_bytes)
            or manifest_source.get("content_sha256")
            != hashlib.sha256(source_bytes).hexdigest()
        ):
            raise VN97P2CorpusBundleError(
                f"sealed corpus source identity mismatch: {source_id}"
            )

    splits = manifest["splits"]
    if set(splits) != set(_SPLITS):
        raise VN97P2CorpusBundleError(
            "VN97CORPUS1 split map is invalid"
        )
    split_identity: dict[str, dict[str, object]] = {}
    for split in _SPLITS:
        spec = splits[split]
        if (
            not isinstance(spec, dict)
            or set(spec) != {"bytes", "mode", "records", "sha256"}
            or spec.get("mode") != "chat"
            or type(spec.get("bytes")) is not int
            or type(spec.get("records")) is not int
            or spec["bytes"] <= 0
            or spec["records"] <= 0
        ):
            raise VN97P2CorpusBundleError(
                f"{split} split contract is invalid"
            )
        expected_sha = _require_sha256(
            spec.get("sha256"),
            label=f"{split} split SHA-256",
        )
        data = _read_regular(
            corpus_root / f"{split}.jsonl",
            max_bytes=_MAX_SPLIT_BYTES,
            label=f"{split} split",
        )
        if (
            len(data) != spec["bytes"]
            or hashlib.sha256(data).hexdigest()
            != expected_sha
        ):
            raise VN97P2CorpusBundleError(
                f"{split} split identity mismatch"
            )
        records = sum(
            1 for line in data.splitlines() if line.strip()
        )
        if records != spec["records"]:
            raise VN97P2CorpusBundleError(
                f"{split} split record count mismatch"
            )
        split_identity[split] = {
            "bytes": len(data),
            "records": records,
            "sha256": expected_sha,
        }

    return VN97P2CorpusBundleReceipt(
        repository_commit=repository_commit,
        fetch_receipt_sha256=hashlib.sha256(
            fetch_bytes
        ).hexdigest(),
        source_summary_sha256=hashlib.sha256(
            summary_bytes
        ).hexdigest(),
        definition_sha256=definition_sha256,
        corpus_manifest_id=manifest_id,
        corpus_manifest_sha256=hashlib.sha256(
            manifest_bytes
        ).hexdigest(),
        split_identity=split_identity,
    )
