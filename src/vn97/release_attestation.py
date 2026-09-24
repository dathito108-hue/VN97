from __future__ import annotations

from dataclasses import dataclass
import base64
import hashlib
import json
from pathlib import Path
import re
import zipfile


SCHEMA = "VN97APK1"
MAX_RELEASE_MANIFEST_BYTES = 4096
APK_BOOTSTRAP_PACKAGE = "assets/vn97-bootstrap/model.vn97cap1"
APK_BOOTSTRAP_SIGNATURE = "assets/vn97-bootstrap/model.vn97sig1"
APK_BOOTSTRAP_PUBLISHER = "assets/vn97-bootstrap/publisher.ed25519"
APK_RELEASE_MANIFEST = "assets/vn97-release/release.vn97rel1"
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")


class VN97ApkAttestationError(RuntimeError):
    pass


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA_RE.fullmatch(value) is None:
        raise VN97ApkAttestationError(
            f"{label} must be lowercase SHA-256"
        )
    return value


@dataclass(frozen=True)
class VN97ReleaseManifest:
    application_id: str
    version_code: int
    version_name: str
    model_bytes: int
    model_sha256: str
    signature_bytes: int
    signature_sha256: str
    publisher_bytes: int
    publisher_sha256: str

    def __post_init__(self) -> None:
        if self.application_id != "ai.vn97.app":
            raise VN97ApkAttestationError(
                "release manifest application id mismatch"
            )
        if type(self.version_code) is not int or self.version_code <= 0:
            raise VN97ApkAttestationError(
                "release manifest version code is invalid"
            )
        if not self.version_name or any(
            ord(ch) < 0x20 for ch in self.version_name
        ):
            raise VN97ApkAttestationError(
                "release manifest version name is invalid"
            )
        if not 96 <= self.model_bytes <= 512 * 1024 * 1024:
            raise VN97ApkAttestationError(
                "release manifest model byte size is invalid"
            )
        if not 1 <= self.signature_bytes <= 16 * 1024:
            raise VN97ApkAttestationError(
                "release manifest signature byte size is invalid"
            )
        if self.publisher_bytes != 32:
            raise VN97ApkAttestationError(
                "release manifest publisher byte size is invalid"
            )
        _require_sha(self.model_sha256, "model SHA-256")
        _require_sha(self.signature_sha256, "signature SHA-256")
        _require_sha(self.publisher_sha256, "publisher SHA-256")


def parse_release_manifest(data: bytes) -> VN97ReleaseManifest:
    if not 0 < len(data) <= MAX_RELEASE_MANIFEST_BYTES:
        raise VN97ApkAttestationError(
            "VN97REL1 byte size is outside bounds"
        )
    try:
        text = data.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise VN97ApkAttestationError(
            "VN97REL1 must be strict UTF-8"
        ) from exc
    if not text.endswith("\n"):
        raise VN97ApkAttestationError(
            "VN97REL1 must end with newline"
        )
    lines = text.split("\n")
    if len(lines) != 11 or lines[-1] != "":
        raise VN97ApkAttestationError(
            "VN97REL1 line count is invalid"
        )
    if lines[0] != "VN97REL1":
        raise VN97ApkAttestationError(
            "VN97REL1 schema mismatch"
        )
    fields: dict[str, str] = {}
    for line in lines[1:-1]:
        if "=" not in line:
            raise VN97ApkAttestationError(
                "VN97REL1 field is malformed"
            )
        key, value = line.split("=", 1)
        if not key or key in fields:
            raise VN97ApkAttestationError(
                "VN97REL1 fields are duplicated or empty"
            )
        fields[key] = value
    expected = {
        "application_id",
        "version_code",
        "version_name_b64",
        "model_bytes",
        "model_sha256",
        "signature_bytes",
        "signature_sha256",
        "publisher_bytes",
        "publisher_sha256",
    }
    if set(fields) != expected:
        raise VN97ApkAttestationError(
            "VN97REL1 fields mismatch"
        )
    try:
        version_name = base64.b64decode(
            fields["version_name_b64"],
            validate=True,
        ).decode("utf-8", errors="strict")
        return VN97ReleaseManifest(
            application_id=fields["application_id"],
            version_code=int(fields["version_code"]),
            version_name=version_name,
            model_bytes=int(fields["model_bytes"]),
            model_sha256=fields["model_sha256"],
            signature_bytes=int(fields["signature_bytes"]),
            signature_sha256=fields["signature_sha256"],
            publisher_bytes=int(fields["publisher_bytes"]),
            publisher_sha256=fields["publisher_sha256"],
        )
    except (ValueError, UnicodeDecodeError) as exc:
        raise VN97ApkAttestationError(
            "VN97REL1 field encoding is invalid"
        ) from exc


@dataclass(frozen=True)
class VN97ApkAttestation:
    application_id: str
    version_code: int
    version_name: str
    apk_bytes: int
    apk_sha256: str
    signer_certificate_sha256: tuple[str, ...]
    release_candidate_manifest_sha256: str
    bootstrap_release_report_sha256: str
    bootstrap_package_sha256: str
    bootstrap_signature_sha256: str
    bootstrap_publisher_sha256: str
    release_manifest_sha256: str

    def __post_init__(self) -> None:
        if self.application_id != "ai.vn97.app":
            raise VN97ApkAttestationError(
                "attestation application id mismatch"
            )
        if type(self.version_code) is not int or self.version_code <= 0:
            raise VN97ApkAttestationError(
                "attestation version code is invalid"
            )
        if not self.version_name:
            raise VN97ApkAttestationError(
                "attestation version name is empty"
            )
        if type(self.apk_bytes) is not int or self.apk_bytes <= 0:
            raise VN97ApkAttestationError(
                "attestation APK byte size is invalid"
            )
        _require_sha(self.apk_sha256, "APK SHA-256")
        _require_sha(
            self.release_candidate_manifest_sha256,
            "release candidate manifest SHA-256",
        )
        _require_sha(
            self.bootstrap_release_report_sha256,
            "bootstrap release report SHA-256",
        )
        _require_sha(
            self.bootstrap_package_sha256,
            "bootstrap package SHA-256",
        )
        _require_sha(
            self.bootstrap_signature_sha256,
            "bootstrap signature SHA-256",
        )
        _require_sha(
            self.bootstrap_publisher_sha256,
            "bootstrap publisher SHA-256",
        )
        _require_sha(
            self.release_manifest_sha256,
            "release manifest SHA-256",
        )
        if not self.signer_certificate_sha256:
            raise VN97ApkAttestationError(
                "attestation requires APK signer certificate"
            )
        if tuple(sorted(set(self.signer_certificate_sha256))) != (
            self.signer_certificate_sha256
        ):
            raise VN97ApkAttestationError(
                "APK signer certificate identities must be unique and sorted"
            )
        for digest in self.signer_certificate_sha256:
            _require_sha(digest, "APK signer certificate SHA-256")

    def canonical_object(self) -> dict[str, object]:
        return {
            "apk_bytes": self.apk_bytes,
            "apk_sha256": self.apk_sha256,
            "application_id": self.application_id,
            "bootstrap_package_sha256": self.bootstrap_package_sha256,
            "bootstrap_publisher_sha256": self.bootstrap_publisher_sha256,
            "bootstrap_release_report_sha256":
                self.bootstrap_release_report_sha256,
            "bootstrap_signature_sha256": self.bootstrap_signature_sha256,
            "release_candidate_manifest_sha256":
                self.release_candidate_manifest_sha256,
            "release_manifest_sha256": self.release_manifest_sha256,
            "schema": SCHEMA,
            "signer_certificate_sha256":
                list(self.signer_certificate_sha256),
            "version_code": self.version_code,
            "version_name": self.version_name,
        }

    def to_bytes(self) -> bytes:
        return json.dumps(
            self.canonical_object(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")


def verify_apk_payload(
    apk_path: Path,
    *,
    expected_application_id: str,
    expected_version_code: int,
    expected_version_name: str,
    expected_package: bytes,
    expected_signature: bytes,
    expected_publisher: bytes,
) -> tuple[VN97ReleaseManifest, bytes]:
    try:
        archive = zipfile.ZipFile(apk_path, "r")
    except (OSError, zipfile.BadZipFile) as exc:
        raise VN97ApkAttestationError(
            "release APK is not a readable ZIP archive"
        ) from exc
    with archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        if len(names) != len(set(names)):
            raise VN97ApkAttestationError(
                "release APK contains duplicate ZIP entries"
            )
        required = {
            APK_BOOTSTRAP_PACKAGE,
            APK_BOOTSTRAP_SIGNATURE,
            APK_BOOTSTRAP_PUBLISHER,
            APK_RELEASE_MANIFEST,
        }
        if not required.issubset(set(names)):
            missing = sorted(required - set(names))
            raise VN97ApkAttestationError(
                "release APK is missing required assets: "
                + ", ".join(missing)
            )
        package = archive.read(APK_BOOTSTRAP_PACKAGE)
        signature = archive.read(APK_BOOTSTRAP_SIGNATURE)
        publisher = archive.read(APK_BOOTSTRAP_PUBLISHER)
        release_manifest_bytes = archive.read(APK_RELEASE_MANIFEST)

    if package != expected_package:
        raise VN97ApkAttestationError(
            "APK bootstrap package differs from staged signed bytes"
        )
    if signature != expected_signature:
        raise VN97ApkAttestationError(
            "APK bootstrap signature differs from staged signed bytes"
        )
    if publisher != expected_publisher:
        raise VN97ApkAttestationError(
            "APK publisher key differs from staged signed bytes"
        )

    manifest = parse_release_manifest(release_manifest_bytes)
    if manifest.application_id != expected_application_id:
        raise VN97ApkAttestationError(
            "APK release manifest application id mismatch"
        )
    if manifest.version_code != expected_version_code:
        raise VN97ApkAttestationError(
            "APK release manifest version code mismatch"
        )
    if manifest.version_name != expected_version_name:
        raise VN97ApkAttestationError(
            "APK release manifest version name mismatch"
        )
    checks = (
        (
            manifest.model_bytes,
            len(package),
            manifest.model_sha256,
            sha256_bytes(package),
            "model",
        ),
        (
            manifest.signature_bytes,
            len(signature),
            manifest.signature_sha256,
            sha256_bytes(signature),
            "signature",
        ),
        (
            manifest.publisher_bytes,
            len(publisher),
            manifest.publisher_sha256,
            sha256_bytes(publisher),
            "publisher",
        ),
    )
    for declared_bytes, actual_bytes, declared_sha, actual_sha, label in checks:
        if declared_bytes != actual_bytes or declared_sha != actual_sha:
            raise VN97ApkAttestationError(
                f"APK VN97REL1 {label} identity mismatch"
            )
    return manifest, release_manifest_bytes


def parse_aapt_badging(text: str) -> tuple[str, int, str]:
    first = next(
        (
            line
            for line in text.splitlines()
            if line.startswith("package: ")
        ),
        None,
    )
    if first is None:
        raise VN97ApkAttestationError(
            "aapt did not report APK package identity"
        )
    match = re.search(
        r"name='([^']+)'\s+versionCode='([0-9]+)'\s+versionName='([^']*)'",
        first,
    )
    if match is None:
        raise VN97ApkAttestationError(
            "aapt package identity format is unexpected"
        )
    return (
        match.group(1),
        int(match.group(2)),
        match.group(3),
    )


def parse_apksigner_certificate_sha256(text: str) -> tuple[str, ...]:
    values = []
    prefix = "certificate SHA-256 digest:"
    for line in text.splitlines():
        lower = line.lower()
        index = lower.find(prefix.lower())
        if index < 0:
            continue
        value = line[index + len(prefix):].strip().lower()
        if _SHA_RE.fullmatch(value) is None:
            raise VN97ApkAttestationError(
                "apksigner certificate SHA-256 is malformed"
            )
        values.append(value)
    unique = tuple(sorted(set(values)))
    if not unique:
        raise VN97ApkAttestationError(
            "apksigner did not report a signer certificate SHA-256"
        )
    return unique
