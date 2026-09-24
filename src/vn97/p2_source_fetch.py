from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import urllib.parse


class VN97P2SourceFetchError(RuntimeError):
    pass


_ALLOWED_SOURCE_HOSTS = {
    "huggingface.co",
    "raw.githubusercontent.com",
}
_ALLOWED_REDIRECT_SUFFIXES = (
    ".huggingface.co",
    ".hf.co",
    ".xethub.hf.co",
)
_MAX_LOCK_BYTES = 256 * 1024
_MAX_SOURCES = 16


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _require_sha256(
    value: object,
    *,
    label: str,
    allow_none: bool = False,
) -> str | None:
    if value is None and allow_none:
        return None
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(ch not in "0123456789abcdef" for ch in value)
    ):
        raise VN97P2SourceFetchError(
            f"{label} must be lowercase SHA-256"
        )
    return value


def _bounded_text(
    value: object,
    *,
    label: str,
    max_bytes: int,
) -> str:
    if not isinstance(value, str):
        raise VN97P2SourceFetchError(
            f"{label} must be text"
        )
    encoded = value.encode("utf-8")
    if not encoded or len(encoded) > max_bytes:
        raise VN97P2SourceFetchError(
            f"{label} must be 1..{max_bytes} UTF-8 bytes"
        )
    return value


def is_allowed_download_url(url: str) -> bool:
    try:
        parsed = urllib.parse.urlparse(url)
    except ValueError:
        return False
    if parsed.scheme != "https" or not parsed.hostname:
        return False
    host = parsed.hostname.lower()
    return (
        host in _ALLOWED_SOURCE_HOSTS
        or any(
            host.endswith(suffix)
            for suffix in _ALLOWED_REDIRECT_SUFFIXES
        )
    )


@dataclass(frozen=True)
class VN97P2FetchSource:
    source_id: str
    origin: str
    license: str
    revision: str
    download_url: str
    filename: str
    max_bytes: int
    expected_bytes: int | None
    expected_records: int
    expected_sha256: str | None

    def __post_init__(self) -> None:
        _bounded_text(
            self.source_id,
            label="source_id",
            max_bytes=128,
        )
        _bounded_text(
            self.origin,
            label="origin",
            max_bytes=1024,
        )
        _bounded_text(
            self.license,
            label="license",
            max_bytes=256,
        )
        _bounded_text(
            self.filename,
            label="filename",
            max_bytes=256,
        )
        if (
            "/" in self.filename
            or "\\" in self.filename
            or self.filename in {".", ".."}
        ):
            raise VN97P2SourceFetchError(
                "source filename must be one basename"
            )
        if (
            len(self.revision) != 40
            or any(
                ch not in "0123456789abcdef"
                for ch in self.revision
            )
        ):
            raise VN97P2SourceFetchError(
                "source revision must be a 40-hex commit"
            )
        if not is_allowed_download_url(
            self.download_url
        ):
            raise VN97P2SourceFetchError(
                "source download URL is outside allowed HTTPS hosts"
            )
        if self.revision not in self.download_url:
            raise VN97P2SourceFetchError(
                "source download URL must contain exact pinned revision"
            )
        if self.max_bytes <= 0:
            raise VN97P2SourceFetchError(
                "source max_bytes must be positive"
            )
        if self.expected_bytes is not None:
            if (
                type(self.expected_bytes) is not int
                or not 0 < self.expected_bytes <= self.max_bytes
            ):
                raise VN97P2SourceFetchError(
                    "source expected_bytes is invalid"
                )
        if (
            type(self.expected_records) is not int
            or self.expected_records <= 0
        ):
            raise VN97P2SourceFetchError(
                "source expected_records must be positive"
            )
        _require_sha256(
            self.expected_sha256,
            label="source expected_sha256",
            allow_none=True,
        )

    def canonical_object(
        self,
    ) -> dict[str, object]:
        return {
            "download_url": self.download_url,
            "expected_bytes": self.expected_bytes,
            "expected_records": self.expected_records,
            "expected_sha256": self.expected_sha256,
            "filename": self.filename,
            "license": self.license,
            "max_bytes": self.max_bytes,
            "origin": self.origin,
            "revision": self.revision,
            "source_id": self.source_id,
        }


@dataclass(frozen=True)
class VN97P2FetchDefinition:
    sources: tuple[VN97P2FetchSource, ...]
    definition_sha256: str

    def __post_init__(self) -> None:
        if not 1 <= len(self.sources) <= _MAX_SOURCES:
            raise VN97P2SourceFetchError(
                "P2 source count is outside bounds"
            )
        ids = tuple(
            item.source_id for item in self.sources
        )
        filenames = tuple(
            item.filename for item in self.sources
        )
        if len(set(ids)) != len(ids):
            raise VN97P2SourceFetchError(
                "P2 source IDs must be unique"
            )
        if len(set(filenames)) != len(filenames):
            raise VN97P2SourceFetchError(
                "P2 source filenames must be unique"
            )
        _require_sha256(
            self.definition_sha256,
            label="fetch definition SHA-256",
        )


def load_fetch_definition(
    path: Path,
) -> VN97P2FetchDefinition:
    if path.is_symlink():
        raise VN97P2SourceFetchError(
            "fetch definition must not be a symlink"
        )
    data = path.read_bytes()
    if not 0 < len(data) <= _MAX_LOCK_BYTES:
        raise VN97P2SourceFetchError(
            "fetch definition size is outside bounds"
        )
    duplicates: list[str] = []

    def hook(pairs):
        out = {}
        for key, value in pairs:
            if key in out:
                duplicates.append(key)
            out[key] = value
        return out

    try:
        root = json.loads(
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
        raise VN97P2SourceFetchError(
            "fetch definition must be strict UTF-8 JSON"
        ) from exc
    if duplicates:
        raise VN97P2SourceFetchError(
            "fetch definition contains duplicate keys"
        )
    if (
        not isinstance(root, dict)
        or set(root) != {"schema", "sources"}
        or root["schema"] != "VN97P2FETCHDEF1"
        or not isinstance(root["sources"], list)
    ):
        raise VN97P2SourceFetchError(
            "fetch definition schema is invalid"
        )

    required = {
        "download_url",
        "expected_bytes",
        "expected_records",
        "expected_sha256",
        "filename",
        "license",
        "max_bytes",
        "origin",
        "revision",
        "source_id",
    }
    sources: list[VN97P2FetchSource] = []
    for raw in root["sources"]:
        if (
            not isinstance(raw, dict)
            or set(raw) != required
        ):
            raise VN97P2SourceFetchError(
                "fetch source fields are invalid"
            )
        if (
            type(raw["max_bytes"]) is not int
            or type(raw["expected_records"])
            is not int
            or (
                raw["expected_bytes"] is not None
                and type(raw["expected_bytes"])
                is not int
            )
        ):
            raise VN97P2SourceFetchError(
                "fetch source numeric fields are invalid"
            )
        sources.append(
            VN97P2FetchSource(
                source_id=raw["source_id"],
                origin=raw["origin"],
                license=raw["license"],
                revision=raw["revision"],
                download_url=raw["download_url"],
                filename=raw["filename"],
                max_bytes=raw["max_bytes"],
                expected_bytes=raw["expected_bytes"],
                expected_records=raw[
                    "expected_records"
                ],
                expected_sha256=raw[
                    "expected_sha256"
                ],
            )
        )

    return VN97P2FetchDefinition(
        sources=tuple(sources),
        definition_sha256=hashlib.sha256(
            data
        ).hexdigest(),
    )


@dataclass(frozen=True)
class VN97P2FetchedSource:
    source_id: str
    revision: str
    filename: str
    bytes: int
    records: int
    sha256: str
    final_url: str

    def __post_init__(self) -> None:
        if self.bytes <= 0 or self.records <= 0:
            raise VN97P2SourceFetchError(
                "fetched source size/records must be positive"
            )
        _require_sha256(
            self.sha256,
            label="fetched source SHA-256",
        )
        if not is_allowed_download_url(
            self.final_url
        ):
            raise VN97P2SourceFetchError(
                "fetched source final URL is outside allowed HTTPS hosts"
            )

    def canonical_object(
        self,
    ) -> dict[str, object]:
        return {
            "bytes": self.bytes,
            "filename": self.filename,
            "final_url": self.final_url,
            "records": self.records,
            "revision": self.revision,
            "sha256": self.sha256,
            "source_id": self.source_id,
        }


def build_fetch_receipt(
    definition: VN97P2FetchDefinition,
    fetched: tuple[VN97P2FetchedSource, ...],
) -> bytes:
    if tuple(
        item.source_id for item in fetched
    ) != tuple(
        item.source_id
        for item in definition.sources
    ):
        raise VN97P2SourceFetchError(
            "fetched source order/identity does not match definition"
        )
    body = {
        "definition_sha256":
            definition.definition_sha256,
        "schema": "VN97P2FETCH1",
        "sources": [
            item.canonical_object()
            for item in fetched
        ],
    }
    body["receipt_id"] = hashlib.sha256(
        b"VN97P2FETCH1\0"
        + _canonical_json(body)
    ).hexdigest()
    return _canonical_json(body) + b"\n"
