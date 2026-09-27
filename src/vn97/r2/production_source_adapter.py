from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Callable, Mapping, Sequence

from ..corpus_io import atomic_write, read_bounded_regular_file
from .production_acquisition import (
    R2D9_ALLOWED_FAMILIES,
    R2D9_DEFINITION_SCHEMA,
    R2D9_PROFILE_ID,
)


R2D10_LOCK_SCHEMA = "VN97R2D10LOCK1"
R2D10_PACK_SCHEMA = "VN97R2D10PACK1"
R2D10_ADAPTERS = (
    "action_trace",
    "capability_demo",
    "dolly",
    "gsm8k",
    "instruction_io",
    "messages",
    "prompt_response",
    "question_answer",
    "tool_trace",
)
_MAX_MESSAGE_BYTES = 64 * 1024
_MAX_CONVERSATION_BYTES = 256 * 1024


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


def _bounded_text(
    value: object,
    *,
    label: str,
    allow_empty: bool = False,
) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be text")
    value = value.strip()
    if not value and not allow_empty:
        raise ValueError(f"{label} must be non-empty")
    if len(value.encode("utf-8")) > _MAX_MESSAGE_BYTES:
        raise ValueError(f"{label} exceeds message byte bound")
    return value


def _strict_json(data: bytes, *, label: str) -> object:
    duplicates: list[str] = []

    def hook(pairs):
        out = {}
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
    if duplicates:
        raise ValueError(f"{label} contains duplicate object keys")
    return value


def _safe_relative_path(raw: object, *, label: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"{label} must be a non-empty relative path")
    path = Path(raw)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{label} must stay relative to the lockfile")
    return path


@dataclass(frozen=True)
class R2D10SourceLock:
    source_id: str
    origin: str
    revision: str
    license: str
    family: str
    adapter: str
    path: str
    expected_sha256: str
    expected_records: int
    max_bytes: int

    def __post_init__(self) -> None:
        for value, label, bound in (
            (self.source_id, "source_id", 256),
            (self.origin, "origin", 4096),
            (self.revision, "revision", 1024),
            (self.license, "license", 512),
            (self.path, "path", 4096),
        ):
            if (
                not isinstance(value, str)
                or not value
                or len(value.encode("utf-8")) > bound
            ):
                raise ValueError(f"R2-D10 {label} is outside bounds")
        if self.family not in R2D9_ALLOWED_FAMILIES:
            raise ValueError("R2-D10 family is not canonical")
        if self.adapter not in R2D10_ADAPTERS:
            raise ValueError("R2-D10 adapter is unsupported")
        _safe_relative_path(self.path, label="source path")
        _require_sha256(
            self.expected_sha256,
            label="R2-D10 expected_sha256",
        )
        if self.expected_records <= 0 or self.max_bytes <= 0:
            raise ValueError(
                "R2-D10 expected_records/max_bytes must be positive"
            )

    def canonical_object(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class R2D10SourcePackLock:
    sources: tuple[R2D10SourceLock, ...]

    def __post_init__(self) -> None:
        if not self.sources:
            raise ValueError("R2-D10 source pack is empty")
        ids = [item.source_id for item in self.sources]
        if ids != sorted(set(ids)):
            raise ValueError(
                "R2-D10 source IDs must be sorted and unique"
            )

    def canonical_object(self) -> dict[str, object]:
        return {
            "schema": R2D10_LOCK_SCHEMA,
            "profile_id": R2D9_PROFILE_ID,
            "sources": [
                item.canonical_object()
                for item in self.sources
            ],
        }

    def fingerprint(self) -> str:
        return _sha256_bytes(
            b"VN97R2D10LOCK1\0"
            + _canonical_json(self.canonical_object())
        )


def load_r2d10_lock(path: Path) -> R2D10SourcePackLock:
    root = path.resolve(strict=True)
    if root.is_symlink():
        raise ValueError("R2-D10 lockfile must not be a symlink")
    payload = _strict_json(
        root.read_bytes(),
        label="R2-D10 lockfile",
    )
    if (
        not isinstance(payload, dict)
        or set(payload) != {"schema", "profile_id", "sources"}
        or payload.get("schema") != R2D10_LOCK_SCHEMA
        or payload.get("profile_id") != R2D9_PROFILE_ID
        or not isinstance(payload.get("sources"), list)
    ):
        raise ValueError("R2-D10 lockfile schema is invalid")

    required = {
        "source_id",
        "origin",
        "revision",
        "license",
        "license_approved",
        "family",
        "adapter",
        "path",
        "expected_sha256",
        "expected_records",
        "max_bytes",
    }
    sources: list[R2D10SourceLock] = []
    for raw in payload["sources"]:
        if not isinstance(raw, dict) or set(raw) != required:
            raise ValueError("R2-D10 source fields are invalid")
        if raw.get("license_approved") is not True:
            raise ValueError(
                "R2-D10 every source requires license_approved=true"
            )
        for key in (
            "source_id",
            "origin",
            "revision",
            "license",
            "family",
            "adapter",
            "path",
            "expected_sha256",
        ):
            if not isinstance(raw.get(key), str):
                raise ValueError(
                    f"R2-D10 source {key} must be a string"
                )
        if (
            type(raw.get("expected_records")) is not int
            or type(raw.get("max_bytes")) is not int
        ):
            raise ValueError(
                "R2-D10 expected_records/max_bytes must be integers"
            )
        sources.append(
            R2D10SourceLock(
                source_id=raw["source_id"],
                origin=raw["origin"],
                revision=raw["revision"],
                license=raw["license"],
                family=raw["family"],
                adapter=raw["adapter"],
                path=raw["path"],
                expected_sha256=raw["expected_sha256"],
                expected_records=raw["expected_records"],
                max_bytes=raw["max_bytes"],
            )
        )
    return R2D10SourcePackLock(
        sources=tuple(
            sorted(sources, key=lambda item: item.source_id)
        )
    )


def _resolve_source(lock_path: Path, source: R2D10SourceLock) -> Path:
    root = lock_path.parent.resolve(strict=True)
    relative = _safe_relative_path(
        source.path,
        label="source path",
    )
    candidate = root / relative
    if candidate.is_symlink():
        raise ValueError("R2-D10 source must not be a symlink")
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            "R2-D10 source path escapes lockfile directory"
        ) from exc
    if not resolved.is_file():
        raise ValueError("R2-D10 source must be a regular file")
    return resolved


def _load_raw_jsonl(
    path: Path,
    *,
    max_bytes: int,
    expected_records: int,
) -> tuple[list[dict[str, object]], bytes]:
    raw = read_bounded_regular_file(
        path,
        max_bytes=max_bytes,
    )
    records: list[dict[str, object]] = []
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ValueError("R2-D10 raw source is not UTF-8") from exc
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        value = _strict_json(
            line.encode("utf-8"),
            label=f"R2-D10 raw record {line_number}",
        )
        if not isinstance(value, dict):
            raise ValueError(
                f"R2-D10 raw record must be an object at line {line_number}"
            )
        records.append(value)
    if len(records) != expected_records:
        raise ValueError(
            "R2-D10 source record-count mismatch: "
            f"expected={expected_records} observed={len(records)}"
        )
    return records, raw


def _validate_messages(
    messages: Sequence[Mapping[str, object]],
) -> tuple[dict[str, str], ...]:
    if not messages:
        raise ValueError("R2-D10 conversation must not be empty")
    out: list[dict[str, str]] = []
    total = 0
    for raw in messages:
        if not isinstance(raw, Mapping) or set(raw) != {
            "role",
            "content",
        }:
            raise ValueError(
                "R2-D10 messages must contain exactly role/content"
            )
        role = raw["role"]
        if role not in {"system", "user", "assistant"}:
            raise ValueError("R2-D10 chat role is invalid")
        content = _bounded_text(
            raw["content"],
            label="message content",
        )
        total += len(content.encode("utf-8"))
        if total > _MAX_CONVERSATION_BYTES:
            raise ValueError(
                "R2-D10 conversation exceeds byte bound"
            )
        out.append(
            {"role": str(role), "content": content}
        )
    if out[-1]["role"] != "assistant":
        raise ValueError(
            "R2-D10 conversation must end with assistant target"
        )
    if not any(item["role"] == "user" for item in out):
        raise ValueError(
            "R2-D10 conversation must contain a user turn"
        )
    return tuple(out)


def _simple_pair(
    user: object,
    assistant: object,
) -> tuple[dict[str, str], ...]:
    return _validate_messages(
        (
            {
                "role": "user",
                "content": _bounded_text(user, label="user content"),
            },
            {
                "role": "assistant",
                "content": _bounded_text(
                    assistant,
                    label="assistant content",
                ),
            },
        )
    )


def _json_text(value: object, *, label: str) -> str:
    if isinstance(value, str):
        return _bounded_text(value, label=label)
    try:
        encoded = _canonical_json(value).decode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is not JSON-serializable") from exc
    return _bounded_text(encoded, label=label)


def adapt_r2d10_record(
    adapter: str,
    raw: Mapping[str, object],
) -> tuple[dict[str, str], ...]:
    if adapter == "messages":
        if set(raw) != {"messages"} or not isinstance(
            raw["messages"], list
        ):
            raise ValueError(
                "messages adapter requires exactly {messages:[...]}"
            )
        return _validate_messages(raw["messages"])

    if adapter == "dolly":
        if set(raw) != {
            "instruction",
            "context",
            "response",
            "category",
        }:
            raise ValueError("dolly record fields are invalid")
        instruction = _bounded_text(
            raw["instruction"],
            label="dolly instruction",
        )
        context = _bounded_text(
            raw["context"],
            label="dolly context",
            allow_empty=True,
        )
        user = (
            instruction
            if not context
            else f"{instruction}\n\nContext:\n{context}"
        )
        return _simple_pair(user, raw["response"])

    if adapter == "gsm8k" or adapter == "question_answer":
        if set(raw) != {"question", "answer"}:
            raise ValueError(
                f"{adapter} requires exactly question/answer"
            )
        return _simple_pair(raw["question"], raw["answer"])

    if adapter == "prompt_response":
        if set(raw) != {"prompt", "response"}:
            raise ValueError(
                "prompt_response requires exactly prompt/response"
            )
        return _simple_pair(raw["prompt"], raw["response"])

    if adapter == "instruction_io":
        if set(raw) != {"instruction", "input", "output"}:
            raise ValueError(
                "instruction_io requires instruction/input/output"
            )
        instruction = _bounded_text(
            raw["instruction"],
            label="instruction",
        )
        input_text = _bounded_text(
            raw["input"],
            label="instruction input",
            allow_empty=True,
        )
        user = (
            instruction
            if not input_text
            else f"{instruction}\n\nInput:\n{input_text}"
        )
        return _simple_pair(user, raw["output"])

    if adapter == "tool_trace":
        if set(raw) != {
            "prompt",
            "tool_name",
            "arguments",
            "result",
            "response",
        }:
            raise ValueError("tool_trace fields are invalid")
        tool_name = _bounded_text(
            raw["tool_name"],
            label="tool name",
        )
        arguments = _json_text(
            raw["arguments"],
            label="tool arguments",
        )
        result = _json_text(
            raw["result"],
            label="tool result",
        )
        return _validate_messages(
            (
                {
                    "role": "user",
                    "content": _bounded_text(
                        raw["prompt"],
                        label="tool prompt",
                    ),
                },
                {
                    "role": "assistant",
                    "content": (
                        "[TOOL_CALL]\n"
                        + _canonical_json(
                            {
                                "arguments": json.loads(arguments)
                                if not isinstance(raw["arguments"], str)
                                else arguments,
                                "name": tool_name,
                            }
                        ).decode("utf-8")
                    ),
                },
                {
                    "role": "system",
                    "content": "[TOOL_RESULT]\n" + result,
                },
                {
                    "role": "assistant",
                    "content": _bounded_text(
                        raw["response"],
                        label="tool response",
                    ),
                },
            )
        )

    if adapter == "action_trace":
        if set(raw) != {
            "observation",
            "action",
            "result",
            "response",
        }:
            raise ValueError("action_trace fields are invalid")
        action = _json_text(
            raw["action"],
            label="action",
        )
        result = _json_text(
            raw["result"],
            label="action result",
        )
        return _validate_messages(
            (
                {
                    "role": "user",
                    "content": _bounded_text(
                        raw["observation"],
                        label="observation",
                    ),
                },
                {
                    "role": "assistant",
                    "content": "[ACTION]\n" + action,
                },
                {
                    "role": "system",
                    "content": "[ACTION_RESULT]\n" + result,
                },
                {
                    "role": "assistant",
                    "content": _bounded_text(
                        raw["response"],
                        label="action response",
                    ),
                },
            )
        )

    if adapter == "capability_demo":
        if set(raw) != {
            "capability",
            "instruction",
            "response",
        }:
            raise ValueError("capability_demo fields are invalid")
        return _validate_messages(
            (
                {
                    "role": "system",
                    "content": (
                        "[CAPABILITY]\n"
                        + _bounded_text(
                            raw["capability"],
                            label="capability",
                        )
                    ),
                },
                {
                    "role": "user",
                    "content": _bounded_text(
                        raw["instruction"],
                        label="capability instruction",
                    ),
                },
                {
                    "role": "assistant",
                    "content": _bounded_text(
                        raw["response"],
                        label="capability response",
                    ),
                },
            )
        )

    raise ValueError("R2-D10 adapter is unsupported")


def _render_chat_jsonl(
    records: Sequence[Sequence[Mapping[str, str]]],
) -> bytes:
    return b"".join(
        _canonical_json({"messages": list(messages)}) + b"\n"
        for messages in records
    )


def _normalized_filename(source_id: str) -> str:
    return (
        hashlib.sha256(source_id.encode("utf-8")).hexdigest()
        + ".chat.jsonl"
    )


def build_r2d10_pack(
    *,
    lock_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    lock_file = lock_path.resolve(strict=True)
    lock = load_r2d10_lock(lock_file)

    if output_dir.exists():
        if (
            output_dir.is_symlink()
            or not output_dir.is_dir()
            or any(output_dir.iterdir())
        ):
            raise ValueError("R2-D10 output-dir must be new or empty")
    else:
        output_dir.mkdir(parents=True, exist_ok=False)
    output = output_dir.resolve(strict=True)
    normalized_dir = output / "normalized"
    normalized_dir.mkdir()

    lock_bytes = (
        json.dumps(
            {
                **lock.canonical_object(),
                "sources": [
                    {
                        **item.canonical_object(),
                        "license_approved": True,
                    }
                    for item in lock.sources
                ],
            },
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    atomic_write(
        output / "source-lock.vn97r2d10.json",
        lock_bytes,
    )

    receipts: list[dict[str, object]] = []
    d9_sources: list[dict[str, object]] = []
    physical: set[tuple[int, int]] = set()

    for source in lock.sources:
        source_path = _resolve_source(lock_file, source)
        stat = source_path.stat()
        identity = (int(stat.st_dev), int(stat.st_ino))
        if identity in physical:
            raise ValueError(
                "R2-D10 sources must be physically distinct files"
            )
        physical.add(identity)

        raw_records, raw_bytes = _load_raw_jsonl(
            source_path,
            max_bytes=source.max_bytes,
            expected_records=source.expected_records,
        )
        raw_sha = _sha256_bytes(raw_bytes)
        if raw_sha != source.expected_sha256:
            raise ValueError(
                f"R2-D10 source SHA-256 mismatch: {source.source_id}"
            )

        normalized_records = [
            adapt_r2d10_record(source.adapter, raw)
            for raw in raw_records
        ]
        normalized = _render_chat_jsonl(normalized_records)
        filename = _normalized_filename(source.source_id)
        target = normalized_dir / filename
        atomic_write(target, normalized)
        normalized_sha = _sha256_bytes(normalized)

        receipt = {
            "source_id": source.source_id,
            "origin": source.origin,
            "revision": source.revision,
            "license": source.license,
            "license_approved": True,
            "family": source.family,
            "adapter": source.adapter,
            "raw_sha256": raw_sha,
            "raw_bytes": len(raw_bytes),
            "raw_records": len(raw_records),
            "normalized_path": f"normalized/{filename}",
            "normalized_sha256": normalized_sha,
            "normalized_bytes": len(normalized),
            "normalized_records": len(normalized_records),
        }
        receipts.append(receipt)

        d9_sources.append(
            {
                "source_id": source.source_id,
                "origin": source.origin,
                "revision": source.revision,
                "license": source.license,
                "license_approved": True,
                "family": source.family,
                "path": f"normalized/{filename}",
                "expected_sha256": normalized_sha,
                "expected_records": len(normalized_records),
                "max_bytes": len(normalized),
            }
        )

    d9_definition = {
        "schema": R2D9_DEFINITION_SCHEMA,
        "profile_id": R2D9_PROFILE_ID,
        "shard_target_training_records": 10000,
        "validation_fraction": 0.01,
        "release_fraction": 0.01,
        "sources": sorted(
            d9_sources,
            key=lambda item: str(item["source_id"]),
        ),
    }
    d9_bytes = (
        json.dumps(
            d9_definition,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    atomic_write(
        output / "r2d9-definition.json",
        d9_bytes,
    )

    body = {
        "schema": R2D10_PACK_SCHEMA,
        "lock_fingerprint": lock.fingerprint(),
        "lock_sha256": _sha256_bytes(lock_bytes),
        "receipts": sorted(
            receipts,
            key=lambda item: str(item["source_id"]),
        ),
        "r2d9_definition_sha256": _sha256_bytes(d9_bytes),
    }
    payload = dict(body)
    payload["pack_id"] = _sha256_bytes(
        b"VN97R2D10PACK1\0" + _canonical_json(body)
    )
    atomic_write(
        output / "r2d10-source-pack.json",
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n",
    )
    return payload


def verify_r2d10_pack(output_dir: Path) -> dict[str, object]:
    root = output_dir.resolve(strict=True)
    payload = _strict_json(
        (root / "r2d10-source-pack.json").read_bytes(),
        label="R2-D10 source pack",
    )
    if not isinstance(payload, dict):
        raise ValueError("R2-D10 pack must be an object")
    if payload.get("schema") != R2D10_PACK_SCHEMA:
        raise ValueError("R2-D10 pack schema mismatch")
    pack_id = _require_sha256(
        payload.get("pack_id"),
        label="R2-D10 pack_id",
    )
    body = dict(payload)
    body.pop("pack_id", None)
    if pack_id != _sha256_bytes(
        b"VN97R2D10PACK1\0" + _canonical_json(body)
    ):
        raise ValueError("R2-D10 pack identity mismatch")

    lock_path = root / "source-lock.vn97r2d10.json"
    lock = load_r2d10_lock(lock_path)
    if lock.fingerprint() != payload.get("lock_fingerprint"):
        raise ValueError("R2-D10 lock fingerprint mismatch")
    if _sha256_bytes(lock_path.read_bytes()) != payload.get(
        "lock_sha256"
    ):
        raise ValueError("R2-D10 lock SHA-256 mismatch")

    receipts = payload.get("receipts")
    if not isinstance(receipts, list) or not receipts:
        raise ValueError("R2-D10 receipts are invalid")
    receipt_map = {
        str(item.get("source_id")): item
        for item in receipts
        if isinstance(item, dict)
    }
    if len(receipt_map) != len(receipts):
        raise ValueError("R2-D10 receipt source IDs are not unique")

    expected_d9_sources = []
    for source in lock.sources:
        receipt = receipt_map.get(source.source_id)
        if not isinstance(receipt, dict):
            raise ValueError("R2-D10 source receipt is missing")
        if receipt.get("family") != source.family:
            raise ValueError("R2-D10 receipt family mismatch")
        if receipt.get("adapter") != source.adapter:
            raise ValueError("R2-D10 receipt adapter mismatch")
        if receipt.get("raw_sha256") != source.expected_sha256:
            raise ValueError("R2-D10 raw source identity mismatch")
        if int(receipt.get("raw_records", -1)) != source.expected_records:
            raise ValueError("R2-D10 raw source record count mismatch")

        relative = _safe_relative_path(
            receipt.get("normalized_path"),
            label="normalized_path",
        )
        normalized = (root / relative).resolve(strict=True)
        try:
            normalized.relative_to(root)
        except ValueError as exc:
            raise ValueError(
                "R2-D10 normalized path escapes pack"
            ) from exc
        if normalized.is_symlink() or not normalized.is_file():
            raise ValueError(
                "R2-D10 normalized output must be a regular file"
            )
        data = normalized.read_bytes()
        normalized_sha = _sha256_bytes(data)
        if normalized_sha != receipt.get("normalized_sha256"):
            raise ValueError("R2-D10 normalized SHA-256 mismatch")
        if len(data) != int(receipt.get("normalized_bytes", -1)):
            raise ValueError("R2-D10 normalized byte count mismatch")

        records, _ = _load_raw_jsonl(
            normalized,
            max_bytes=len(data),
            expected_records=int(
                receipt.get("normalized_records", -1)
            ),
        )
        for record in records:
            adapt_r2d10_record("messages", record)

        expected_d9_sources.append(
            {
                "source_id": source.source_id,
                "origin": source.origin,
                "revision": source.revision,
                "license": source.license,
                "license_approved": True,
                "family": source.family,
                "path": str(relative).replace("\\", "/"),
                "expected_sha256": normalized_sha,
                "expected_records": len(records),
                "max_bytes": len(data),
            }
        )

    d9_path = root / "r2d9-definition.json"
    d9_bytes = d9_path.read_bytes()
    if _sha256_bytes(d9_bytes) != payload.get(
        "r2d9_definition_sha256"
    ):
        raise ValueError("R2-D10 D9 definition SHA-256 mismatch")
    d9 = _strict_json(
        d9_bytes,
        label="R2-D10 D9 definition",
    )
    expected_d9 = {
        "schema": R2D9_DEFINITION_SCHEMA,
        "profile_id": R2D9_PROFILE_ID,
        "shard_target_training_records": 10000,
        "validation_fraction": 0.01,
        "release_fraction": 0.01,
        "sources": sorted(
            expected_d9_sources,
            key=lambda item: str(item["source_id"]),
        ),
    }
    if d9 != expected_d9:
        raise ValueError("R2-D10 D9 handoff content mismatch")
    return payload
