from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import stat
import struct
import zlib

MAGIC = b"VN97CAP1"
VERSION = 1
HEADER_SIZE = 96
ENTRY_SIZE = 64
MAX_SECTIONS = 64
MAX_MANIFEST_BYTES = 64 * 1024
MAX_PACKAGE_BYTES = 512 * 1024 * 1024
TYPE_MANIFEST = 1
TYPE_DATA = 2
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
_ID_RE = re.compile(r"^[a-z][a-z0-9._-]{0,127}$")
_ROLE_RE = re.compile(r"^[a-z][a-z0-9._-]{0,63}$")
_FORMAT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_KINDS = frozenset({"weights", "tokenizer", "memory", "avatar", "multimodal", "knowledge", "composite"})
_FORBIDDEN_ROLES = frozenset({"code", "executable", "plugin", "shared-library", "native-library", "script"})

class CapabilityPackageError(RuntimeError): pass
class CapabilityPackageFormatError(CapabilityPackageError): pass
class CapabilityPackageIntegrityError(CapabilityPackageError): pass
class CapabilityManifestError(CapabilityPackageError): pass
class CapabilityStageError(CapabilityPackageError): pass

def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise CapabilityManifestError("manifest is not canonicalizable JSON") from exc

def _strict_json_object(data: bytes) -> dict[str, object]:
    duplicates: list[str] = []
    def pairs_hook(pairs: list[tuple[str, object]]) -> dict[str, object]:
        out: dict[str, object] = {}
        for key, value in pairs:
            if key in out: duplicates.append(key)
            out[key] = value
        return out
    try:
        text = data.decode("utf-8", errors="strict")
        value = json.loads(text, object_pairs_hook=pairs_hook, parse_constant=lambda v: (_ for _ in ()).throw(ValueError(v)))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise CapabilityManifestError("manifest is not strict UTF-8 JSON") from exc
    if duplicates or not isinstance(value, dict):
        raise CapabilityManifestError("manifest must be an object without duplicate keys")
    if _canonical_json(value) != data:
        raise CapabilityManifestError("manifest bytes must use canonical JSON encoding")
    return value

def _exact_keys(value: dict[str, object], keys: set[str], label: str) -> None:
    if set(value) != keys: raise CapabilityManifestError(f"{label} keys are not exact")

def _text(value: object, label: str, max_bytes: int) -> str:
    if not isinstance(value, str) or not value or len(value.encode()) > max_bytes:
        raise CapabilityManifestError(f"{label} must be a non-empty bounded string")
    return value

@dataclass(frozen=True)
class CapabilityDataSection:
    role: str
    format: str
    data: bytes
    def __post_init__(self) -> None:
        if not _ROLE_RE.fullmatch(self.role) or self.role in _FORBIDDEN_ROLES:
            raise ValueError("section role is unsupported or executable")
        if not _FORMAT_RE.fullmatch(self.format): raise ValueError("section format is invalid")
        if not isinstance(self.data, bytes) or not self.data: raise ValueError("section data must be non-empty bytes")

@dataclass(frozen=True)
class CapabilitySource:
    origin: str
    source_sha256: str
    license: str
    def __post_init__(self) -> None:
        if not self.origin or len(self.origin.encode()) > 1024: raise ValueError("origin must be bounded")
        if not _SHA_RE.fullmatch(self.source_sha256): raise ValueError("source_sha256 must be lowercase SHA-256")
        if not self.license or len(self.license.encode()) > 128: raise ValueError("license must be bounded")

@dataclass(frozen=True)
class CapabilityManifest:
    capability_id: str
    capability_version: int
    kind: str
    source: CapabilitySource
    sections: tuple[dict[str, object], ...]

@dataclass(frozen=True)
class ParsedCapabilityPackage:
    manifest: CapabilityManifest
    package_sha256: str
    sections: tuple[bytes, ...]

@dataclass(frozen=True)
class StagedCapability:
    manifest: CapabilityManifest
    package_sha256: str
    path: Path


def _validate_manifest(data: bytes, data_sections: tuple[bytes, ...]) -> CapabilityManifest:
    if not data or len(data) > MAX_MANIFEST_BYTES: raise CapabilityManifestError("manifest byte size is invalid")
    root = _strict_json_object(data)
    _exact_keys(root, {"schema","capability_id","capability_version","kind","source","sections"}, "manifest")
    if root["schema"] != "VN97CAP1": raise CapabilityManifestError("manifest schema mismatch")
    capability_id = _text(root["capability_id"], "capability_id", 128)
    if not _ID_RE.fullmatch(capability_id): raise CapabilityManifestError("capability_id is invalid")
    version = root["capability_version"]
    if type(version) is not int or version <= 0 or version > 0xffffffff: raise CapabilityManifestError("capability_version is invalid")
    kind = _text(root["kind"], "kind", 32)
    if kind not in _KINDS: raise CapabilityManifestError("capability kind is unsupported")
    source = root["source"]
    if not isinstance(source, dict): raise CapabilityManifestError("source must be an object")
    _exact_keys(source, {"origin","source_sha256","license"}, "source")
    try:
        source_obj = CapabilitySource(
            _text(source["origin"], "origin", 1024),
            _text(source["source_sha256"], "source_sha256", 64),
            _text(source["license"], "license", 128),
        )
    except ValueError as exc: raise CapabilityManifestError(str(exc)) from exc
    raw_sections = root["sections"]
    if not isinstance(raw_sections, list) or len(raw_sections) != len(data_sections):
        raise CapabilityManifestError("manifest section list does not match package")
    seen_roles: set[str] = set()
    normalized: list[dict[str, object]] = []
    for pos, (raw, section) in enumerate(zip(raw_sections, data_sections), start=1):
        if not isinstance(raw, dict): raise CapabilityManifestError("section metadata must be an object")
        _exact_keys(raw, {"index","role","format","size","sha256"}, "section")
        if raw["index"] != pos: raise CapabilityManifestError("section index mismatch")
        role = _text(raw["role"], "role", 64)
        fmt = _text(raw["format"], "format", 64)
        if not _ROLE_RE.fullmatch(role) or role in _FORBIDDEN_ROLES: raise CapabilityManifestError("section role is unsupported or executable")
        if role in seen_roles: raise CapabilityManifestError("section roles must be unique")
        seen_roles.add(role)
        if not _FORMAT_RE.fullmatch(fmt): raise CapabilityManifestError("section format is invalid")
        size = raw["size"]
        if type(size) is not int or size != len(section): raise CapabilityManifestError("section size mismatch")
        digest = raw["sha256"]
        if not isinstance(digest, str) or not _SHA_RE.fullmatch(digest) or digest != hashlib.sha256(section).hexdigest():
            raise CapabilityManifestError("section SHA-256 mismatch")
        normalized.append(dict(raw))
    return CapabilityManifest(capability_id, version, kind, source_obj, tuple(normalized))


def build_capability_package(*, capability_id: str, capability_version: int, kind: str, source: CapabilitySource, sections: tuple[CapabilityDataSection, ...]) -> bytes:
    if not sections or len(sections) + 1 > MAX_SECTIONS: raise ValueError("package data section count is invalid")
    if len({s.role for s in sections}) != len(sections): raise ValueError("section roles must be unique")
    manifest_obj = {
        "schema":"VN97CAP1", "capability_id":capability_id, "capability_version":capability_version,
        "kind":kind,
        "source":{"origin":source.origin,"source_sha256":source.source_sha256,"license":source.license},
        "sections":[{"index":i,"role":s.role,"format":s.format,"size":len(s.data),"sha256":hashlib.sha256(s.data).hexdigest()} for i,s in enumerate(sections,1)],
    }
    manifest = _canonical_json(manifest_obj)
    data_sections = tuple(s.data for s in sections)
    _validate_manifest(manifest, data_sections)
    all_sections = (manifest,) + data_sections
    count = len(all_sections)
    payload_offset = HEADER_SIZE + count * ENTRY_SIZE
    cursor = payload_offset
    entries = bytearray()
    payload = bytearray()
    for i, section in enumerate(all_sections):
        section_type = TYPE_MANIFEST if i == 0 else TYPE_DATA
        entries += struct.pack("<IIQQ", section_type, 0, cursor, len(section))
        entries += hashlib.sha256(section).digest()
        entries += struct.pack("<Q", 0)
        payload += section
        cursor += len(section)
    total_size = cursor
    if total_size > MAX_PACKAGE_BYTES: raise ValueError("package exceeds byte limit")
    content = bytes(entries + payload)
    header = bytearray(HEADER_SIZE)
    struct.pack_into("<8sHHIIIQQQII", header, 0, MAGIC, VERSION, HEADER_SIZE, 0, count, ENTRY_SIZE, HEADER_SIZE, payload_offset, total_size, 0, 0)
    header[56:88] = hashlib.sha256(content).digest()
    struct.pack_into("<I", header, 88, zlib.crc32(header[:88]) & 0xffffffff)
    struct.pack_into("<I", header, 92, 0)
    return bytes(header) + content


def parse_capability_package(blob: bytes, *, max_package_bytes: int = MAX_PACKAGE_BYTES) -> ParsedCapabilityPackage:
    if not isinstance(blob, bytes): raise TypeError("blob must be bytes")
    if max_package_bytes < HEADER_SIZE or len(blob) > max_package_bytes: raise CapabilityPackageFormatError("package size is outside configured bounds")
    if len(blob) < HEADER_SIZE: raise CapabilityPackageFormatError("package is shorter than header")
    magic, version, header_size, flags, count, entry_size, table_offset, payload_offset, total_size, manifest_index, reserved = struct.unpack_from("<8sHHIIIQQQII", blob, 0)
    if magic != MAGIC or version != VERSION or header_size != HEADER_SIZE: raise CapabilityPackageFormatError("package magic/version/header mismatch")
    if flags != 0 or reserved != 0 or struct.unpack_from("<I", blob, 92)[0] != 0: raise CapabilityPackageFormatError("reserved package fields are nonzero")
    if count < 2 or count > MAX_SECTIONS or entry_size != ENTRY_SIZE or manifest_index != 0: raise CapabilityPackageFormatError("package section table header is invalid")
    expected_payload = HEADER_SIZE + count * ENTRY_SIZE
    if table_offset != HEADER_SIZE or payload_offset != expected_payload or total_size != len(blob): raise CapabilityPackageFormatError("package offsets/length are invalid")
    if struct.unpack_from("<I", blob, 88)[0] != (zlib.crc32(blob[:88]) & 0xffffffff): raise CapabilityPackageIntegrityError("package header CRC mismatch")
    if blob[56:88] != hashlib.sha256(blob[HEADER_SIZE:]).digest(): raise CapabilityPackageIntegrityError("package content SHA-256 mismatch")
    cursor = payload_offset
    sections: list[bytes] = []
    for index in range(count):
        off = HEADER_SIZE + index * ENTRY_SIZE
        section_type, section_flags, section_offset, section_size = struct.unpack_from("<IIQQ", blob, off)
        digest = blob[off+24:off+56]
        entry_reserved = struct.unpack_from("<Q", blob, off+56)[0]
        expected_type = TYPE_MANIFEST if index == 0 else TYPE_DATA
        if section_type != expected_type or section_flags != 0 or entry_reserved != 0: raise CapabilityPackageFormatError("section entry type/flags are invalid")
        if section_offset != cursor or section_size <= 0 or section_size > len(blob) - cursor: raise CapabilityPackageFormatError("section layout is invalid")
        section = blob[cursor:cursor+section_size]
        if hashlib.sha256(section).digest() != digest: raise CapabilityPackageIntegrityError("section SHA-256 mismatch")
        if index == 0 and section_size > MAX_MANIFEST_BYTES: raise CapabilityPackageFormatError("manifest exceeds byte limit")
        sections.append(section)
        cursor += section_size
    if cursor != len(blob): raise CapabilityPackageFormatError("package contains trailing or hidden bytes")
    manifest = _validate_manifest(sections[0], tuple(sections[1:]))
    return ParsedCapabilityPackage(manifest, hashlib.sha256(blob).hexdigest(), tuple(sections[1:]))


class CapabilityStager:
    def __init__(self, root: Path, *, max_package_bytes: int = MAX_PACKAGE_BYTES) -> None:
        root = Path(root)
        if not root.is_absolute() or not root.exists() or not root.is_dir() or root.is_symlink(): raise ValueError("stage root must be an existing absolute non-symlink directory")
        if max_package_bytes < HEADER_SIZE: raise ValueError("max_package_bytes is invalid")
        self.root = root
        self.max_package_bytes = max_package_bytes

    def stage(self, blob: bytes) -> StagedCapability:
        parsed = parse_capability_package(blob, max_package_bytes=self.max_package_bytes)
        target_name = parsed.package_sha256 + ".vn97cap1"
        try: import fcntl
        except ImportError as exc: raise CapabilityStageError("platform lacks advisory locking") from exc
        root_fd = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        temp_name = ".vn97cap-" + secrets.token_hex(12) + ".tmp"
        try:
            fcntl.flock(root_fd, fcntl.LOCK_EX)
            try:
                existing = os.open(target_name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=root_fd)
            except FileNotFoundError:
                existing = -1
            if existing >= 0:
                try:
                    info = os.fstat(existing)
                    if not stat.S_ISREG(info.st_mode) or info.st_size != len(blob): raise CapabilityStageError("staged digest path is not the expected regular file")
                    digest = hashlib.sha256()
                    while True:
                        chunk = os.read(existing, 65536)
                        if not chunk: break
                        digest.update(chunk)
                    if digest.hexdigest() != parsed.package_sha256: raise CapabilityStageError("staged digest path content mismatch")
                finally: os.close(existing)
                return StagedCapability(parsed.manifest, parsed.package_sha256, self.root / target_name)
            fd = os.open(temp_name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=root_fd)
            try:
                view = memoryview(blob); written = 0
                while written < len(view):
                    n = os.write(fd, view[written:])
                    if n <= 0: raise CapabilityStageError("short stage write")
                    written += n
                os.fsync(fd)
            finally: os.close(fd)
            try:
                os.link(temp_name, target_name, src_dir_fd=root_fd, dst_dir_fd=root_fd, follow_symlinks=False)
            except FileExistsError:
                raise CapabilityStageError("digest target appeared during staging")
            finally:
                try: os.unlink(temp_name, dir_fd=root_fd)
                except FileNotFoundError: pass
            os.fsync(root_fd)
            return StagedCapability(parsed.manifest, parsed.package_sha256, self.root / target_name)
        except OSError as exc:
            if isinstance(exc, CapabilityStageError): raise
            raise CapabilityStageError("capability staging failed") from exc
        finally:
            try: os.unlink(temp_name, dir_fd=root_fd)
            except OSError: pass
            os.close(root_fd)
