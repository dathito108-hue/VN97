from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Protocol

from .capability_package import (
    MAX_PACKAGE_BYTES,
    CapabilityManifest,
    CapabilityPackageError,
    ParsedCapabilityPackage,
    StagedCapability,
    parse_capability_package,
)

SIG_SCHEMA = "VN97SIG1"
SIG_ALGORITHM = "ed25519"
_ID_RE = re.compile(r"^[a-z][a-z0-9._-]{0,127}$")
_ROLE_RE = re.compile(r"^[a-z][a-z0-9._-]{0,63}$")
_FORMAT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
_SIG_RE = re.compile(r"^[0-9a-f]{128}$")
_KINDS = frozenset({"weights","tokenizer","memory","avatar","multimodal","knowledge","composite"})


class CapabilityTrustError(RuntimeError): pass
class SignatureEnvelopeError(CapabilityTrustError): pass
class UntrustedCapabilityError(CapabilityTrustError): pass
class TrustBackendUnavailable(CapabilityTrustError): pass
class CompatibilityError(RuntimeError): pass
class CompatibilityAmbiguityError(CompatibilityError): pass
class LossyAdaptationDeniedError(CompatibilityError): pass


def _canon(value: object) -> bytes:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",",":"), allow_nan=False).encode()
    except (TypeError, ValueError) as exc:
        raise SignatureEnvelopeError("value is not canonical JSON") from exc


def _strict_object(data: bytes) -> dict[str, object]:
    duplicates: list[str] = []
    def hook(pairs):
        out = {}
        for key, value in pairs:
            if key in out: duplicates.append(key)
            out[key] = value
        return out
    try:
        value = json.loads(
            data.decode("utf-8", errors="strict"),
            object_pairs_hook=hook,
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise SignatureEnvelopeError("signature envelope is not strict UTF-8 JSON") from exc
    if duplicates or not isinstance(value, dict) or _canon(value) != data:
        raise SignatureEnvelopeError("signature envelope must be canonical object JSON")
    return value


@dataclass(frozen=True)
class SignatureEnvelope:
    key_id: str
    package_sha256: str
    capability_id: str
    capability_version: int
    signature: bytes
    algorithm: str = SIG_ALGORITHM

    def __post_init__(self) -> None:
        if not _ID_RE.fullmatch(self.key_id): raise ValueError("key_id is invalid")
        if self.algorithm != SIG_ALGORITHM: raise ValueError("unsupported signature algorithm")
        if not _SHA_RE.fullmatch(self.package_sha256): raise ValueError("package_sha256 is invalid")
        if not _ID_RE.fullmatch(self.capability_id): raise ValueError("capability_id is invalid")
        if type(self.capability_version) is not int or not 1 <= self.capability_version <= 0xffffffff:
            raise ValueError("capability_version is invalid")
        if not isinstance(self.signature, bytes) or len(self.signature) != 64:
            raise ValueError("signature must be exactly 64 bytes")

    def claims(self) -> dict[str, object]:
        return {
            "algorithm": self.algorithm,
            "capability_id": self.capability_id,
            "capability_version": self.capability_version,
            "key_id": self.key_id,
            "package_sha256": self.package_sha256,
            "schema": SIG_SCHEMA,
        }

    def signing_message(self) -> bytes:
        return b"VN97CAP1-SIGNATURE-V1\0" + _canon(self.claims())

    def to_bytes(self) -> bytes:
        value = self.claims()
        value["signature"] = self.signature.hex()
        return _canon(value)


def parse_signature_envelope(data: bytes) -> SignatureEnvelope:
    if not isinstance(data, bytes): raise TypeError("signature envelope must be bytes")
    if not data or len(data) > 16 * 1024: raise SignatureEnvelopeError("signature envelope size is invalid")
    root = _strict_object(data)
    keys = {"schema","algorithm","key_id","package_sha256","capability_id","capability_version","signature"}
    if set(root) != keys or root["schema"] != SIG_SCHEMA:
        raise SignatureEnvelopeError("signature envelope schema/keys mismatch")
    version = root["capability_version"]
    signature = root["signature"]
    if (
        not isinstance(root["algorithm"], str)
        or not isinstance(root["key_id"], str)
        or not isinstance(root["package_sha256"], str)
        or not isinstance(root["capability_id"], str)
        or type(version) is not int
        or not isinstance(signature, str)
        or not _SIG_RE.fullmatch(signature)
    ):
        raise SignatureEnvelopeError("signature envelope field type/encoding mismatch")
    try:
        return SignatureEnvelope(
            key_id=root["key_id"],
            algorithm=root["algorithm"],
            package_sha256=root["package_sha256"],
            capability_id=root["capability_id"],
            capability_version=version,
            signature=bytes.fromhex(signature),
        )
    except ValueError as exc:
        raise SignatureEnvelopeError(str(exc)) from exc


class SignatureVerifier(Protocol):
    def verify(self, *, public_key: bytes, message: bytes, signature: bytes) -> bool: ...


class Ed25519Verifier:
    def verify(self, *, public_key: bytes, message: bytes, signature: bytes) -> bool:
        if len(public_key) != 32 or len(signature) != 64: return False
        try:
            from cryptography.exceptions import InvalidSignature
            from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
        except ImportError as exc:
            raise TrustBackendUnavailable("optional Ed25519 backend is unavailable") from exc
        try:
            Ed25519PublicKey.from_public_bytes(public_key).verify(signature, message)
        except (InvalidSignature, ValueError):
            return False
        return True


@dataclass(frozen=True)
class TrustedPublisherKey:
    key_id: str
    public_key: bytes
    capability_prefixes: tuple[str, ...]
    allowed_kinds: frozenset[str]
    min_version: int = 1
    max_version: int = 0xffffffff
    revoked: bool = False
    algorithm: str = SIG_ALGORITHM

    def __post_init__(self) -> None:
        if not _ID_RE.fullmatch(self.key_id): raise ValueError("key_id is invalid")
        if self.algorithm != SIG_ALGORITHM or not isinstance(self.public_key, bytes) or len(self.public_key) != 32:
            raise ValueError("trusted key must be 32-byte Ed25519")
        if not self.capability_prefixes or any(not _ID_RE.fullmatch(v) for v in self.capability_prefixes):
            raise ValueError("capability_prefixes are invalid")
        if not self.allowed_kinds or not self.allowed_kinds.issubset(_KINDS):
            raise ValueError("allowed_kinds are invalid")
        if type(self.min_version) is not int or type(self.max_version) is not int or not 1 <= self.min_version <= self.max_version <= 0xffffffff:
            raise ValueError("trusted version range is invalid")

    def permits(self, manifest: CapabilityManifest) -> bool:
        scoped = any(
            manifest.capability_id == prefix or manifest.capability_id.startswith(prefix + ".")
            for prefix in self.capability_prefixes
        )
        return (
            not self.revoked
            and scoped
            and manifest.kind in self.allowed_kinds
            and self.min_version <= manifest.capability_version <= self.max_version
        )


class CapabilityTrustStore:
    def __init__(self, keys: tuple[TrustedPublisherKey, ...]) -> None:
        if not keys: raise ValueError("trust store must not be empty")
        self._keys = {key.key_id: key for key in keys}
        if len(self._keys) != len(keys): raise ValueError("duplicate trust key_id")

    def require(self, key_id: str) -> TrustedPublisherKey:
        try: return self._keys[key_id]
        except KeyError as exc: raise UntrustedCapabilityError("publisher key is not trusted") from exc


@dataclass(frozen=True)
class VerifiedCapability:
    staged: StagedCapability
    parsed: ParsedCapabilityPackage
    publisher_key_id: str
    signature_sha256: str


def _read_staged(staged: StagedCapability, root: Path, max_bytes: int) -> bytes:
    root = Path(root)
    if not root.is_absolute() or not root.is_dir() or root.is_symlink():
        raise UntrustedCapabilityError("stage root is unsafe")
    name = staged.package_sha256 + ".vn97cap1"
    if staged.path.parent != root or staged.path.name != name:
        raise UntrustedCapabilityError("staged path is outside content-addressed root")
    root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=root_fd)
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= max_bytes:
                raise UntrustedCapabilityError("staged package file is invalid")
            out = bytearray()
            while len(out) < info.st_size:
                chunk = os.read(fd, min(65536, info.st_size - len(out)))
                if not chunk: break
                out.extend(chunk)
            if len(out) != info.st_size: raise UntrustedCapabilityError("staged package read was truncated")
            return bytes(out)
        finally:
            os.close(fd)
    except OSError as exc:
        raise UntrustedCapabilityError("staged package is unavailable or unsafe") from exc
    finally:
        os.close(root_fd)


def verify_staged_capability(
    staged: StagedCapability,
    signature_envelope: bytes,
    *,
    stage_root: Path,
    trust_store: CapabilityTrustStore,
    verifier: SignatureVerifier,
    max_package_bytes: int = MAX_PACKAGE_BYTES,
) -> VerifiedCapability:
    if not _SHA_RE.fullmatch(staged.package_sha256): raise UntrustedCapabilityError("staged digest is invalid")
    if max_package_bytes <= 0: raise ValueError("max_package_bytes must be positive")
    blob = _read_staged(staged, stage_root, max_package_bytes)
    try:
        parsed = parse_capability_package(blob, max_package_bytes=max_package_bytes)
    except CapabilityPackageError as exc:
        raise UntrustedCapabilityError("staged package failed revalidation") from exc
    if parsed.package_sha256 != staged.package_sha256 or parsed.manifest != staged.manifest:
        raise UntrustedCapabilityError("staged package identity changed after staging")

    envelope = parse_signature_envelope(signature_envelope)
    manifest = parsed.manifest
    if (
        envelope.package_sha256 != parsed.package_sha256
        or envelope.capability_id != manifest.capability_id
        or envelope.capability_version != manifest.capability_version
    ):
        raise UntrustedCapabilityError("signature is not bound to this package")
    key = trust_store.require(envelope.key_id)
    if key.algorithm != envelope.algorithm or not key.permits(manifest):
        raise UntrustedCapabilityError("publisher key scope does not permit this package")
    if not verifier.verify(public_key=key.public_key, message=envelope.signing_message(), signature=envelope.signature):
        raise UntrustedCapabilityError("signature verification failed")
    return VerifiedCapability(staged, parsed, key.key_id, hashlib.sha256(signature_envelope).hexdigest())


@dataclass(frozen=True)
class FormatRule:
    role: str
    accepted_formats: tuple[str, ...]

    def __post_init__(self) -> None:
        if not _ROLE_RE.fullmatch(self.role): raise ValueError("format role is invalid")
        if not self.accepted_formats or len(set(self.accepted_formats)) != len(self.accepted_formats):
            raise ValueError("accepted_formats must be non-empty and unique")
        if any(not _FORMAT_RE.fullmatch(v) for v in self.accepted_formats):
            raise ValueError("accepted format is invalid")


@dataclass(frozen=True)
class CompatibilityProfile:
    profile_id: str
    runtime_api_version: int
    supported_kinds: frozenset[str]
    min_capability_version: int
    max_capability_version: int
    format_rules: tuple[FormatRule, ...]

    def __post_init__(self) -> None:
        if not _ID_RE.fullmatch(self.profile_id): raise ValueError("profile_id is invalid")
        if type(self.runtime_api_version) is not int or self.runtime_api_version <= 0:
            raise ValueError("runtime_api_version must be positive")
        if not self.supported_kinds or not self.supported_kinds.issubset(_KINDS):
            raise ValueError("supported_kinds are invalid")
        if type(self.min_capability_version) is not int or type(self.max_capability_version) is not int or not 1 <= self.min_capability_version <= self.max_capability_version <= 0xffffffff:
            raise ValueError("profile version range is invalid")
        roles = tuple(rule.role for rule in self.format_rules)
        if not roles or len(set(roles)) != len(roles): raise ValueError("format rules must be non-empty and unique")

    def rule(self, role: str) -> FormatRule | None:
        return next((rule for rule in self.format_rules if rule.role == role), None)

    def fingerprint(self) -> str:
        value = {
            "format_rules":[{"accepted_formats":list(r.accepted_formats),"role":r.role} for r in self.format_rules],
            "max_capability_version":self.max_capability_version,
            "min_capability_version":self.min_capability_version,
            "profile_id":self.profile_id,
            "runtime_api_version":self.runtime_api_version,
            "supported_kinds":sorted(self.supported_kinds),
        }
        return hashlib.sha256(_canon(value)).hexdigest()


@dataclass(frozen=True)
class AdapterSpec:
    adapter_id: str
    role: str
    input_format: str
    output_format: str
    lossy: bool = False

    def __post_init__(self) -> None:
        if not _ID_RE.fullmatch(self.adapter_id) or not _ROLE_RE.fullmatch(self.role):
            raise ValueError("adapter id/role is invalid")
        if not _FORMAT_RE.fullmatch(self.input_format) or not _FORMAT_RE.fullmatch(self.output_format):
            raise ValueError("adapter format is invalid")
        if self.input_format == self.output_format: raise ValueError("adapter must change format")


class CompatibilityDisposition(str, Enum):
    DIRECT = "direct"
    ADAPT_REQUIRED = "adapt_required"


@dataclass(frozen=True)
class SectionPlan:
    section_index: int
    role: str
    source_format: str
    target_format: str
    adapter_id: str | None
    lossy: bool


@dataclass(frozen=True)
class CompatibilityPlan:
    package_sha256: str
    capability_id: str
    capability_version: int
    publisher_key_id: str
    signature_sha256: str
    profile_id: str
    profile_sha256: str
    runtime_api_version: int
    disposition: CompatibilityDisposition
    sections: tuple[SectionPlan, ...]


def plan_capability_compatibility(
    verified: VerifiedCapability,
    profile: CompatibilityProfile,
    *,
    adapters: tuple[AdapterSpec, ...] = (),
    allow_lossy: bool = False,
) -> CompatibilityPlan:
    manifest = verified.parsed.manifest
    if manifest.kind not in profile.supported_kinds:
        raise CompatibilityError("capability kind is unsupported")
    if not profile.min_capability_version <= manifest.capability_version <= profile.max_capability_version:
        raise CompatibilityError("capability version is outside profile range")
    if len({adapter.adapter_id for adapter in adapters}) != len(adapters):
        raise ValueError("adapter IDs must be unique")

    plans: list[SectionPlan] = []
    adapted = False
    for raw in manifest.sections:
        index, role, source = raw.get("index"), raw.get("role"), raw.get("format")
        if type(index) is not int or not isinstance(role, str) or not isinstance(source, str):
            raise CompatibilityError("section metadata is malformed")
        rule = profile.rule(role)
        if rule is None: raise CompatibilityError(f"profile has no rule for role={role}")
        if source in rule.accepted_formats:
            plans.append(SectionPlan(index, role, source, source, None, False))
            continue
        matches = tuple(
            adapter for adapter in adapters
            if adapter.role == role and adapter.input_format == source and adapter.output_format in rule.accepted_formats
        )
        if not matches: raise CompatibilityError(f"no adaptation path for role={role} format={source}")
        if len(matches) != 1: raise CompatibilityAmbiguityError(f"ambiguous adaptation for role={role} format={source}")
        adapter = matches[0]
        if adapter.lossy and not allow_lossy:
            raise LossyAdaptationDeniedError(f"lossy adaptation denied for role={role}")
        adapted = True
        plans.append(SectionPlan(index, role, source, adapter.output_format, adapter.adapter_id, adapter.lossy))

    return CompatibilityPlan(
        package_sha256=verified.parsed.package_sha256,
        capability_id=manifest.capability_id,
        capability_version=manifest.capability_version,
        publisher_key_id=verified.publisher_key_id,
        signature_sha256=verified.signature_sha256,
        profile_id=profile.profile_id,
        profile_sha256=profile.fingerprint(),
        runtime_api_version=profile.runtime_api_version,
        disposition=CompatibilityDisposition.ADAPT_REQUIRED if adapted else CompatibilityDisposition.DIRECT,
        sections=tuple(plans),
    )
