from __future__ import annotations

from contextlib import redirect_stdout
from dataclasses import dataclass
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
from typing import Callable, Any
import zipfile

from .capability_package import (
    parse_capability_package,
)
from .capability_trust import (
    Ed25519Verifier,
    parse_signature_envelope,
)
from .production_closure import (
    VN97ProductionClosureReport,
    VN97ReleaseReadinessInputs,
    run_production_closure,
)
from .production_readiness import (
    parse_production_readiness_report,
)
from .production_run_manifest import (
    VN97ProductionRunManifest,
    load_production_run_manifest,
    verify_production_run_inputs,
)
from .release_attestation import (
    APK_BOOTSTRAP_PACKAGE,
    APK_BOOTSTRAP_PUBLISHER,
    APK_BOOTSTRAP_SIGNATURE,
    APK_RELEASE_MANIFEST,
    VN97ApkAttestation,
    parse_aapt_badging,
    parse_apk_attestation,
    parse_apksigner_certificate_sha256,
    parse_release_manifest,
    sha256_bytes,
    sha256_file,
)
from .release_candidate import (
    VN97LoadedReleaseCandidate,
    load_release_candidate_directory,
)


SCHEMA = "VN97FINAL1"
MAX_RECEIPT_BYTES = 256 * 1024
_EXPECTED_RELEASE_ENTRIES = {
    "VN97-production.apk",
    "bootstrap-release.vn97bootrel6.json",
    "production-readiness.vn97ready1",
    "release-attestation.vn97apk1",
}


class VN97ProductionMaterializationError(
    RuntimeError
):
    pass


class VN97ProductionMaterializationBlocked(
    VN97ProductionMaterializationError
):
    def __init__(
        self,
        closure_report:
            VN97ProductionClosureReport,
    ) -> None:
        super().__init__(
            "VN97 production materialization is blocked by "
            + closure_report.phase
        )
        self.closure_report = (
            closure_report
        )


def _require_hex(
    value: object,
    *,
    length: int,
    label: str,
) -> str:
    if (
        not isinstance(value, str)
        or len(value) != length
        or any(
            ch not in
            "0123456789abcdef"
            for ch in value
        )
    ):
        raise VN97ProductionMaterializationError(
            f"{label} must be lowercase hex"
        )
    return value


@dataclass(frozen=True)
class VN97FinalReleaseReceipt:
    manifest_sha256: str
    repository_commit: str
    closure_report_sha256: str
    readiness_report_sha256: str
    release_candidate_manifest_sha256: str
    release_attestation_sha256: str
    bootstrap_release_report_sha256: str
    release_manifest_sha256: str
    application_id: str
    version_code: int
    version_name: str
    apk_bytes: int
    apk_sha256: str
    signer_certificate_sha256:
        tuple[str, ...]

    def __post_init__(self) -> None:
        _require_hex(
            self.manifest_sha256,
            length=64,
            label="manifest SHA-256",
        )
        _require_hex(
            self.repository_commit,
            length=40,
            label="repository commit",
        )
        for value, label in (
            (
                self.closure_report_sha256,
                "closure report SHA-256",
            ),
            (
                self.readiness_report_sha256,
                "readiness report SHA-256",
            ),
            (
                self.release_candidate_manifest_sha256,
                "release candidate manifest SHA-256",
            ),
            (
                self.release_attestation_sha256,
                "release attestation SHA-256",
            ),
            (
                self.bootstrap_release_report_sha256,
                "bootstrap release report SHA-256",
            ),
            (
                self.release_manifest_sha256,
                "release manifest SHA-256",
            ),
            (
                self.apk_sha256,
                "APK SHA-256",
            ),
        ):
            _require_hex(
                value,
                length=64,
                label=label,
            )
        if (
            self.application_id
            != "ai.vn97.app"
        ):
            raise VN97ProductionMaterializationError(
                "final application id mismatch"
            )
        if (
            type(self.version_code)
            is not int
            or self.version_code <= 0
        ):
            raise VN97ProductionMaterializationError(
                "final version code is invalid"
            )
        if (
            not isinstance(
                self.version_name,
                str,
            )
            or not self.version_name
            or any(
                ord(ch) < 0x20
                for ch in
                    self.version_name
            )
        ):
            raise VN97ProductionMaterializationError(
                "final version name is invalid"
            )
        if (
            type(self.apk_bytes)
            is not int
            or self.apk_bytes <= 0
        ):
            raise VN97ProductionMaterializationError(
                "final APK byte count is invalid"
            )
        if (
            not self
            .signer_certificate_sha256
            or tuple(
                sorted(
                    set(
                        self
                        .signer_certificate_sha256
                    )
                )
            )
            != self
            .signer_certificate_sha256
        ):
            raise VN97ProductionMaterializationError(
                "final APK signer identities must be unique, sorted and non-empty"
            )
        for value in (
            self
            .signer_certificate_sha256
        ):
            _require_hex(
                value,
                length=64,
                label="APK signer certificate SHA-256",
            )

    def canonical_object(
        self,
    ) -> dict[str, object]:
        return {
            "apk_bytes":
                self.apk_bytes,
            "apk_sha256":
                self.apk_sha256,
            "application_id":
                self.application_id,
            "bootstrap_release_report_sha256":
                self
                .bootstrap_release_report_sha256,
            "closure_report_sha256":
                self
                .closure_report_sha256,
            "manifest_sha256":
                self.manifest_sha256,
            "readiness_report_sha256":
                self
                .readiness_report_sha256,
            "release_attestation_sha256":
                self
                .release_attestation_sha256,
            "release_candidate_manifest_sha256":
                self
                .release_candidate_manifest_sha256,
            "release_manifest_sha256":
                self
                .release_manifest_sha256,
            "repository_commit":
                self.repository_commit,
            "schema": SCHEMA,
            "signer_certificate_sha256":
                list(
                    self
                    .signer_certificate_sha256
                ),
            "status": "MATERIALIZED",
            "version_code":
                self.version_code,
            "version_name":
                self.version_name,
        }

    def to_bytes(self) -> bytes:
        return json.dumps(
            self.canonical_object(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")


def parse_final_release_receipt(
    data: bytes,
) -> VN97FinalReleaseReceipt:
    if (
        not 0 < len(data)
        <= MAX_RECEIPT_BYTES
    ):
        raise VN97ProductionMaterializationError(
            "VN97FINAL1 byte size is outside bounds"
        )
    duplicates: list[str] = []

    def hook(
        pairs: list[
            tuple[str, Any]
        ],
    ) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                duplicates.append(key)
            result[key] = value
        return result

    try:
        text = data.decode(
            "utf-8",
            errors="strict",
        )
        root = json.loads(
            text,
            object_pairs_hook=hook,
            parse_constant=lambda raw:
                (_ for _ in ())
                .throw(
                    ValueError(raw)
                ),
        )
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        ValueError,
    ) as exc:
        raise VN97ProductionMaterializationError(
            "VN97FINAL1 must be strict UTF-8 JSON"
        ) from exc
    if duplicates or not isinstance(
        root,
        dict,
    ):
        raise VN97ProductionMaterializationError(
            "VN97FINAL1 must be one object without duplicate keys"
        )
    canonical = json.dumps(
        root,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    if canonical != data:
        raise VN97ProductionMaterializationError(
            "VN97FINAL1 must use canonical JSON"
        )
    expected = {
        "apk_bytes",
        "apk_sha256",
        "application_id",
        "bootstrap_release_report_sha256",
        "closure_report_sha256",
        "manifest_sha256",
        "readiness_report_sha256",
        "release_attestation_sha256",
        "release_candidate_manifest_sha256",
        "release_manifest_sha256",
        "repository_commit",
        "schema",
        "signer_certificate_sha256",
        "status",
        "version_code",
        "version_name",
    }
    if (
        set(root) != expected
        or root["schema"] != SCHEMA
        or root["status"]
        != "MATERIALIZED"
    ):
        raise VN97ProductionMaterializationError(
            "VN97FINAL1 schema/keys/status mismatch"
        )
    raw_signers = root[
        "signer_certificate_sha256"
    ]
    if (
        not isinstance(
            raw_signers,
            list,
        )
        or not all(
            isinstance(value, str)
            for value in raw_signers
        )
    ):
        raise VN97ProductionMaterializationError(
            "VN97FINAL1 signer list is invalid"
        )
    try:
        return VN97FinalReleaseReceipt(
            manifest_sha256=
                root["manifest_sha256"],
            repository_commit=
                root["repository_commit"],
            closure_report_sha256=
                root[
                    "closure_report_sha256"
                ],
            readiness_report_sha256=
                root[
                    "readiness_report_sha256"
                ],
            release_candidate_manifest_sha256=
                root[
                    "release_candidate_manifest_sha256"
                ],
            release_attestation_sha256=
                root[
                    "release_attestation_sha256"
                ],
            bootstrap_release_report_sha256=
                root[
                    "bootstrap_release_report_sha256"
                ],
            release_manifest_sha256=
                root[
                    "release_manifest_sha256"
                ],
            application_id=
                root["application_id"],
            version_code=
                root["version_code"],
            version_name=
                root["version_name"],
            apk_bytes=
                root["apk_bytes"],
            apk_sha256=
                root["apk_sha256"],
            signer_certificate_sha256=
                tuple(raw_signers),
        )
    except (
        TypeError,
        ValueError,
        VN97ProductionMaterializationError,
    ) as exc:
        if isinstance(
            exc,
            VN97ProductionMaterializationError,
        ):
            raise
        raise VN97ProductionMaterializationError(
            str(exc)
        ) from exc


@dataclass(frozen=True)
class VN97FinalValidationConfig:
    validation_format: str = "chat"
    validation_sequence_length: int = 256
    validation_stride: int | None = None
    validation_batch_size: int = 4
    validation_device: str = "auto"
    min_validation_accuracy: float = 0.0
    min_validation_target_tokens: int = 64
    speech_validation_max_examples: int = 100_000
    speech_max_frames: int = 1500
    speech_max_target_tokens: int = 512
    min_speech_validation_accuracy: float = 0.0
    min_speech_validation_target_tokens: int = 32
    min_speech_validation_examples: int = 1
    vision_validation_max_examples: int = 100_000
    vision_max_patches: int = 196
    vision_max_target_tokens: int = 512
    min_vision_validation_accuracy: float = 0.0
    min_vision_validation_target_tokens: int = 32
    min_vision_validation_examples: int = 1

    def __post_init__(self) -> None:
        if (
            self.validation_format
            not in {"text", "chat"}
        ):
            raise ValueError(
                "validation_format must be text or chat"
            )
        for value, label in (
            (
                self.validation_sequence_length,
                "validation_sequence_length",
            ),
            (
                self.validation_batch_size,
                "validation_batch_size",
            ),
            (
                self.min_validation_target_tokens,
                "min_validation_target_tokens",
            ),
            (
                self.speech_validation_max_examples,
                "speech_validation_max_examples",
            ),
            (
                self.speech_max_frames,
                "speech_max_frames",
            ),
            (
                self.speech_max_target_tokens,
                "speech_max_target_tokens",
            ),
            (
                self.min_speech_validation_target_tokens,
                "min_speech_validation_target_tokens",
            ),
            (
                self.min_speech_validation_examples,
                "min_speech_validation_examples",
            ),
            (
                self.vision_validation_max_examples,
                "vision_validation_max_examples",
            ),
            (
                self.vision_max_patches,
                "vision_max_patches",
            ),
            (
                self.vision_max_target_tokens,
                "vision_max_target_tokens",
            ),
            (
                self.min_vision_validation_target_tokens,
                "min_vision_validation_target_tokens",
            ),
            (
                self.min_vision_validation_examples,
                "min_vision_validation_examples",
            ),
        ):
            if (
                type(value) is not int
                or value <= 0
            ):
                raise ValueError(
                    f"{label} must be positive integer"
                )
        if (
            self.validation_stride
            is not None
            and (
                type(
                    self.validation_stride
                )
                is not int
                or self
                .validation_stride
                <= 0
            )
        ):
            raise ValueError(
                "validation_stride must be positive when set"
            )
        for value, label in (
            (
                self.min_validation_accuracy,
                "min_validation_accuracy",
            ),
            (
                self.min_speech_validation_accuracy,
                "min_speech_validation_accuracy",
            ),
            (
                self.min_vision_validation_accuracy,
                "min_vision_validation_accuracy",
            ),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(
                    value,
                    (int, float),
                )
                or not 0.0
                <= float(value)
                <= 1.0
            ):
                raise ValueError(
                    f"{label} must be in [0, 1]"
                )
        if (
            not isinstance(
                self.validation_device,
                str,
            )
            or not self.validation_device
        ):
            raise ValueError(
                "validation_device must be non-empty"
            )


ExternalProbe = Callable[
    [Path, str | None, str | None],
    tuple[
        tuple[str, ...],
        tuple[str, int, str],
    ],
]


def _sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(
                1024 * 1024
            )
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _regular_file(
    path: Path,
    *,
    label: str,
    max_bytes: int,
) -> Path:
    try:
        info = os.lstat(path)
    except OSError as exc:
        raise VN97ProductionMaterializationError(
            f"{label} is unavailable"
        ) from exc
    if (
        stat.S_ISLNK(info.st_mode)
        or not stat.S_ISREG(info.st_mode)
        or not 0 < info.st_size
        <= max_bytes
    ):
        raise VN97ProductionMaterializationError(
            f"{label} must be a bounded regular non-symlink file"
        )
    return path.resolve(strict=True)


def _real_directory(
    path: Path,
    *,
    label: str,
) -> Path:
    try:
        info = os.lstat(path)
    except OSError as exc:
        raise VN97ProductionMaterializationError(
            f"{label} is unavailable"
        ) from exc
    if (
        stat.S_ISLNK(info.st_mode)
        or not stat.S_ISDIR(info.st_mode)
    ):
        raise VN97ProductionMaterializationError(
            f"{label} must be a real non-symlink directory"
        )
    return path.resolve(strict=True)


def _verify_materialization_source(
    repository_root: Path,
) -> None:
    current_root = Path(
        __file__
    ).resolve().parent
    for name in (
        "production_materialization.py",
        "production_release_cli.py",
        "production_closure.py",
        "production_readiness.py",
        "release_attestation.py",
    ):
        current = current_root / name
        bound = (
            repository_root
            / "src/vn97"
            / name
        )
        if (
            not current.is_file()
            or not bound.is_file()
            or _sha256_file(current)
            != _sha256_file(bound)
        ):
            raise VN97ProductionMaterializationError(
                "running M19L/release verification source does not match VN97RUN1-bound checkout"
            )


def _read_canonical_json(
    path: Path,
    *,
    label: str,
    max_bytes: int,
) -> tuple[dict[str, Any], bytes]:
    target = _regular_file(
        path,
        label=label,
        max_bytes=max_bytes,
    )
    data = target.read_bytes()
    duplicates: list[str] = []

    def hook(
        pairs: list[
            tuple[str, Any]
        ],
    ) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, value in pairs:
            if key in output:
                duplicates.append(key)
            output[key] = value
        return output

    try:
        root = json.loads(
            data.decode(
                "utf-8",
                errors="strict",
            ),
            object_pairs_hook=hook,
            parse_constant=lambda raw:
                (_ for _ in ())
                .throw(
                    ValueError(raw)
                ),
        )
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        ValueError,
    ) as exc:
        raise VN97ProductionMaterializationError(
            f"{label} must be strict UTF-8 JSON"
        ) from exc
    if duplicates or not isinstance(
        root,
        dict,
    ):
        raise VN97ProductionMaterializationError(
            f"{label} must be one object without duplicate keys"
        )
    canonical = json.dumps(
        root,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    if canonical != data:
        raise VN97ProductionMaterializationError(
            f"{label} must use canonical JSON"
        )
    return root, data


def _extract_apk_release_assets(
    apk_path: Path,
) -> tuple[
    bytes,
    bytes,
    bytes,
    bytes,
]:
    try:
        archive = zipfile.ZipFile(
            apk_path,
            "r",
        )
    except (
        OSError,
        zipfile.BadZipFile,
    ) as exc:
        raise VN97ProductionMaterializationError(
            "final APK is not a readable ZIP archive"
        ) from exc
    with archive:
        names = [
            item.filename
            for item in
                archive.infolist()
        ]
        if len(names) != len(
            set(names)
        ):
            raise VN97ProductionMaterializationError(
                "final APK contains duplicate ZIP entries"
            )
        required = {
            APK_BOOTSTRAP_PACKAGE,
            APK_BOOTSTRAP_SIGNATURE,
            APK_BOOTSTRAP_PUBLISHER,
            APK_RELEASE_MANIFEST,
        }
        if not required.issubset(
            set(names)
        ):
            raise VN97ProductionMaterializationError(
                "final APK is missing canonical release assets"
            )
        return (
            archive.read(
                APK_BOOTSTRAP_PACKAGE
            ),
            archive.read(
                APK_BOOTSTRAP_SIGNATURE
            ),
            archive.read(
                APK_BOOTSTRAP_PUBLISHER
            ),
            archive.read(
                APK_RELEASE_MANIFEST
            ),
        )


def _default_external_probe(
    apk_path: Path,
    apksigner: str | None,
    aapt: str | None,
) -> tuple[
    tuple[str, ...],
    tuple[str, int, str],
]:
    from .production_release_cli import (
        _find_build_tool,
        _run_checked,
    )

    signer = _find_build_tool(
        "apksigner",
        explicit=apksigner,
    )
    aapt_path = _find_build_tool(
        "aapt",
        explicit=aapt,
    )
    cwd = apk_path.parent
    signer_result = _run_checked(
        [
            str(signer),
            "verify",
            "--verbose",
            "--print-certs",
            str(apk_path),
        ],
        cwd=cwd,
        label=
            "M19L final APK signature verification",
    )
    certificates = (
        parse_apksigner_certificate_sha256(
            signer_result.stdout
            + "\n"
            + signer_result.stderr
        )
    )
    aapt_result = _run_checked(
        [
            str(aapt_path),
            "dump",
            "badging",
            str(apk_path),
        ],
        cwd=cwd,
        label=
            "M19L final APK package inspection",
    )
    identity = parse_aapt_badging(
        aapt_result.stdout
    )
    return (
        certificates,
        identity,
    )


def verify_materialized_release(
    *,
    manifest:
        VN97ProductionRunManifest,
    closure:
        VN97ProductionClosureReport,
    candidate:
        VN97LoadedReleaseCandidate,
    output_dir: Path,
    apksigner: str | None,
    aapt: str | None,
    external_probe:
        ExternalProbe | None = None,
) -> VN97FinalReleaseReceipt:
    root = _real_directory(
        output_dir,
        label="final production release directory",
    )
    actual = {
        entry.name
        for entry in root.iterdir()
    }
    if actual != (
        _EXPECTED_RELEASE_ENTRIES
    ):
        raise VN97ProductionMaterializationError(
            "final production release directory entries do not match M19D contract"
        )

    apk = _regular_file(
        root / "VN97-production.apk",
        label="final production APK",
        max_bytes=2 * 1024 * 1024 * 1024,
    )
    attestation_path = _regular_file(
        root /
        "release-attestation.vn97apk1",
        label="VN97APK1 attestation",
        max_bytes=64 * 1024,
    )
    readiness_path = _regular_file(
        root /
        "production-readiness.vn97ready1",
        label="VN97READY1 report",
        max_bytes=256 * 1024,
    )
    bootstrap_path = (
        root /
        "bootstrap-release.vn97bootrel6.json"
    )

    readiness_bytes = (
        readiness_path.read_bytes()
    )
    readiness = (
        parse_production_readiness_report(
            readiness_bytes
        )
    )
    if (
        not readiness.ready
        or readiness.repository_commit
        != manifest.repository_commit
        or readiness.candidate
        is None
        or readiness.candidate
        .manifest_sha256
        != candidate.manifest_sha256
        or readiness.candidate
        .model_image_sha256
        != candidate.manifest
        .model_image_sha256
        or hashlib.sha256(
            readiness_bytes
        ).hexdigest()
        != closure
        .readiness_report_sha256
    ):
        raise VN97ProductionMaterializationError(
            "published VN97READY1 does not match READY M19K/VN97RC1 chain"
        )

    attestation_bytes = (
        attestation_path.read_bytes()
    )
    attestation = (
        parse_apk_attestation(
            attestation_bytes
        )
    )
    if (
        attestation
        .release_candidate_manifest_sha256
        != candidate.manifest_sha256
        or attestation.apk_bytes
        != apk.stat().st_size
        or attestation.apk_sha256
        != sha256_file(apk)
    ):
        raise VN97ProductionMaterializationError(
            "VN97APK1 does not match final APK/VN97RC1"
        )

    bootstrap, bootstrap_bytes = (
        _read_canonical_json(
            bootstrap_path,
            label="VN97BOOTREL6 report",
            max_bytes=1024 * 1024,
        )
    )
    release_info = bootstrap.get(
        "release_candidate"
    )
    if (
        bootstrap.get("schema")
        != "VN97BOOTREL6"
        or not isinstance(
            release_info,
            dict,
        )
        or release_info.get(
            "manifest_sha256"
        )
        != candidate.manifest_sha256
        or bootstrap.get(
            "signed_source_sha256"
        )
        != candidate.manifest_sha256
        or hashlib.sha256(
            bootstrap_bytes
        ).hexdigest()
        != attestation
        .bootstrap_release_report_sha256
    ):
        raise VN97ProductionMaterializationError(
            "VN97BOOTREL6 does not bind exact VN97RC1/VN97APK1 chain"
        )

    (
        package,
        signature,
        publisher,
        release_manifest_bytes,
    ) = _extract_apk_release_assets(
        apk
    )
    if len(publisher) != 32:
        raise VN97ProductionMaterializationError(
            "final APK publisher key must be 32 bytes"
        )

    parsed_package = (
        parse_capability_package(
            package
        )
    )
    envelope = (
        parse_signature_envelope(
            signature
        )
    )
    if (
        parsed_package.manifest
        .source.source_sha256
        != candidate.manifest_sha256
        or envelope.package_sha256
        != parsed_package.package_sha256
        or envelope.capability_id
        != parsed_package.manifest
        .capability_id
        or envelope.capability_version
        != parsed_package.manifest
        .capability_version
        or not Ed25519Verifier()
        .verify(
            public_key=publisher,
            message=
                envelope
                .signing_message(),
            signature=
                envelope.signature,
        )
    ):
        raise VN97ProductionMaterializationError(
            "embedded signed VN97CAP1 chain is invalid"
        )

    release_manifest = (
        parse_release_manifest(
            release_manifest_bytes
        )
    )
    if (
        release_manifest.application_id
        != attestation.application_id
        or release_manifest.version_code
        != attestation.version_code
        or release_manifest.version_name
        != attestation.version_name
        or release_manifest.model_bytes
        != len(package)
        or release_manifest.model_sha256
        != sha256_bytes(package)
        or release_manifest.signature_bytes
        != len(signature)
        or release_manifest.signature_sha256
        != sha256_bytes(signature)
        or release_manifest.publisher_bytes
        != len(publisher)
        or release_manifest.publisher_sha256
        != sha256_bytes(publisher)
        or sha256_bytes(
            release_manifest_bytes
        )
        != attestation
        .release_manifest_sha256
        or sha256_bytes(package)
        != attestation
        .bootstrap_package_sha256
        or sha256_bytes(signature)
        != attestation
        .bootstrap_signature_sha256
        or sha256_bytes(publisher)
        != attestation
        .bootstrap_publisher_sha256
        or bootstrap.get(
            "package_sha256"
        )
        != sha256_bytes(package)
        or bootstrap.get(
            "publisher_public_key_sha256"
        )
        != sha256_bytes(publisher)
    ):
        raise VN97ProductionMaterializationError(
            "final embedded release identities do not match VN97APK1/VN97BOOTREL6"
        )

    probe = (
        external_probe
        or _default_external_probe
    )
    (
        signer_certificates,
        identity,
    ) = probe(
        apk,
        apksigner,
        aapt,
    )
    if (
        signer_certificates
        != attestation
        .signer_certificate_sha256
        or identity
        != (
            attestation.application_id,
            attestation.version_code,
            attestation.version_name,
        )
        or readiness.application_id
        != attestation.application_id
        or readiness.version_code
        != attestation.version_code
        or readiness.version_name
        != attestation.version_name
    ):
        raise VN97ProductionMaterializationError(
            "external APK identity/signature does not match VN97READY1/VN97APK1"
        )

    return VN97FinalReleaseReceipt(
        manifest_sha256=
            manifest.manifest_sha256,
        repository_commit=
            manifest.repository_commit,
        closure_report_sha256=
            hashlib.sha256(
                closure.to_bytes()
            ).hexdigest(),
        readiness_report_sha256=
            hashlib.sha256(
                readiness_bytes
            ).hexdigest(),
        release_candidate_manifest_sha256=
            candidate.manifest_sha256,
        release_attestation_sha256=
            hashlib.sha256(
                attestation_bytes
            ).hexdigest(),
        bootstrap_release_report_sha256=
            hashlib.sha256(
                bootstrap_bytes
            ).hexdigest(),
        release_manifest_sha256=
            hashlib.sha256(
                release_manifest_bytes
            ).hexdigest(),
        application_id=
            attestation.application_id,
        version_code=
            attestation.version_code,
        version_name=
            attestation.version_name,
        apk_bytes=
            attestation.apk_bytes,
        apk_sha256=
            attestation.apk_sha256,
        signer_certificate_sha256=
            attestation
            .signer_certificate_sha256,
    )


def _release_policy_argv(
    manifest:
        VN97ProductionRunManifest,
) -> list[str]:
    intake = dict(
        manifest.intake_options
    )
    speech = dict(
        manifest.speech_options
    )
    argv: list[str] = []

    mapping = (
        (
            "min_device_runs",
            "--device-evidence-min-runs",
        ),
        (
            "max_text_prefill_p95_ms",
            "--max-text-prefill-p95-ms",
        ),
        (
            "max_text_decode_p95_ms_per_token",
            "--max-text-decode-p95-ms-per-token",
        ),
        (
            "max_device_peak_pss_kib",
            "--max-device-peak-pss-kib",
        ),
        (
            "max_device_thermal_status",
            "--max-device-thermal-status",
        ),
        (
            "max_speech_prefill_p95_ms",
            "--max-speech-prefill-p95-ms",
        ),
        (
            "max_abs_battery_energy_counter_delta_nwh",
            "--max-abs-battery-energy-counter-delta-nwh",
        ),
    )
    for key, flag in mapping:
        value = intake.get(key)
        if value is not None:
            argv.extend(
                [
                    flag,
                    str(value),
                ]
            )
    if intake.get(
        "require_energy_counter"
    ) is True:
        argv.append(
            "--require-device-energy-counter"
        )

    for key, flag in (
        (
            "max_model_image_bytes",
            "--max-model-image-bytes",
        ),
        (
            "max_recurrent_state_bytes",
            "--max-recurrent-state-bytes",
        ),
    ):
        value = speech.get(key)
        if value is not None:
            argv.extend(
                [
                    flag,
                    str(value),
                ]
            )
    return argv


def build_release_argv(
    *,
    manifest:
        VN97ProductionRunManifest,
    repository_root: Path,
    candidate_dir: Path,
    readiness_inputs:
        VN97ReleaseReadinessInputs,
    validation:
        VN97FinalValidationConfig,
) -> list[str]:
    required = (
        readiness_inputs
        .publisher_private_key,
        readiness_inputs.key_id,
        readiness_inputs
        .capability_version,
        readiness_inputs
        .source_origin,
        readiness_inputs
        .source_license,
        readiness_inputs.output_dir,
        readiness_inputs
        .max_validation_loss,
    )
    if any(
        value is None
        for value in required
    ):
        raise VN97ProductionMaterializationError(
            "READY materialization inputs are incomplete"
        )

    argv = [
        "--repository-root",
        str(repository_root),
        "--release-candidate-dir",
        str(candidate_dir),
        "--private-key",
        str(
            readiness_inputs
            .publisher_private_key
        ),
        "--key-id",
        str(
            readiness_inputs.key_id
        ),
        "--capability-version",
        str(
            readiness_inputs
            .capability_version
        ),
        "--source-origin",
        str(
            readiness_inputs
            .source_origin
        ),
        "--source-license",
        str(
            readiness_inputs
            .source_license
        ),
        "--validation-format",
        validation.validation_format,
        "--validation-sequence-length",
        str(
            validation
            .validation_sequence_length
        ),
        "--validation-batch-size",
        str(
            validation
            .validation_batch_size
        ),
        "--validation-device",
        validation.validation_device,
        "--max-validation-loss",
        str(
            readiness_inputs
            .max_validation_loss
        ),
        "--min-validation-accuracy",
        str(
            validation
            .min_validation_accuracy
        ),
        "--min-validation-target-tokens",
        str(
            validation
            .min_validation_target_tokens
        ),
        "--output-dir",
        str(
            readiness_inputs.output_dir
        ),
        "--gradle",
        readiness_inputs.gradle,
    ]

    for path in (
        readiness_inputs
        .validation_inputs
    ):
        argv.extend(
            [
                "--validation-input",
                str(path),
            ]
        )
    if (
        validation.validation_stride
        is not None
    ):
        argv.extend(
            [
                "--validation-stride",
                str(
                    validation
                    .validation_stride
                ),
            ]
        )

    if (
        readiness_inputs
        .speech_validation_input
        is not None
    ):
        argv.extend(
            [
                "--speech-validation-input",
                str(
                    readiness_inputs
                    .speech_validation_input
                ),
                "--speech-validation-max-examples",
                str(
                    validation
                    .speech_validation_max_examples
                ),
                "--speech-max-frames",
                str(
                    validation
                    .speech_max_frames
                ),
                "--speech-max-target-tokens",
                str(
                    validation
                    .speech_max_target_tokens
                ),
                "--max-speech-validation-loss",
                str(
                    readiness_inputs
                    .max_speech_validation_loss
                ),
                "--min-speech-validation-accuracy",
                str(
                    validation
                    .min_speech_validation_accuracy
                ),
                "--min-speech-validation-target-tokens",
                str(
                    validation
                    .min_speech_validation_target_tokens
                ),
                "--min-speech-validation-examples",
                str(
                    validation
                    .min_speech_validation_examples
                ),
            ]
        )
    if (
        readiness_inputs
        .vision_validation_input
        is not None
    ):
        argv.extend(
            [
                "--vision-validation-input",
                str(
                    readiness_inputs
                    .vision_validation_input
                ),
                "--vision-validation-max-examples",
                str(
                    validation
                    .vision_validation_max_examples
                ),
                "--vision-max-patches",
                str(
                    validation
                    .vision_max_patches
                ),
                "--vision-max-target-tokens",
                str(
                    validation
                    .vision_max_target_tokens
                ),
                "--max-vision-validation-loss",
                str(
                    readiness_inputs
                    .max_vision_validation_loss
                ),
                "--min-vision-validation-accuracy",
                str(
                    validation
                    .min_vision_validation_accuracy
                ),
                "--min-vision-validation-target-tokens",
                str(
                    validation
                    .min_vision_validation_target_tokens
                ),
                "--min-vision-validation-examples",
                str(
                    validation
                    .min_vision_validation_examples
                ),
            ]
        )

    if readiness_inputs.apksigner:
        argv.extend(
            [
                "--apksigner",
                readiness_inputs.apksigner,
            ]
        )
    if readiness_inputs.aapt:
        argv.extend(
            [
                "--aapt",
                readiness_inputs.aapt,
            ]
        )
    argv.extend(
        _release_policy_argv(
            manifest
        )
    )
    return argv


ReleaseRunner = Callable[
    [list[str] | None],
    int,
]


def _atomic_create(
    path: Path,
    data: bytes,
) -> None:
    if (
        path.exists()
        or path.is_symlink()
    ):
        raise VN97ProductionMaterializationError(
            "final receipt output must not already exist"
        )
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    if path.parent.is_symlink():
        raise VN97ProductionMaterializationError(
            "final receipt parent must not be a symlink"
        )
    fd, temporary = tempfile.mkstemp(
        prefix=".vn97-final-",
        dir=path.parent,
    )
    try:
        with os.fdopen(
            fd,
            "wb",
            closefd=True,
        ) as output:
            output.write(data)
            output.flush()
            os.fsync(
                output.fileno()
            )
        os.replace(
            temporary,
            path,
        )
        if path.read_bytes() != data:
            raise VN97ProductionMaterializationError(
                "final receipt post-write verification failed"
            )
    finally:
        if os.path.exists(
            temporary
        ):
            os.unlink(
                temporary
            )


def materialize_final_release(
    *,
    manifest_path: Path,
    workspace_root: Path,
    repository_root: Path,
    readiness_inputs:
        VN97ReleaseReadinessInputs,
    validation:
        VN97FinalValidationConfig
        | None = None,
    receipt_path: Path | None = None,
    release_runner:
        ReleaseRunner | None = None,
    external_probe:
        ExternalProbe | None = None,
) -> VN97FinalReleaseReceipt:
    from .production_run_cli import (
        _verify_repository,
    )

    manifest = (
        load_production_run_manifest(
            manifest_path
        )
    )
    resolved = (
        verify_production_run_inputs(
            manifest,
            workspace_root=
                workspace_root,
        )
    )
    repository = (
        _verify_repository(
            repository_root,
            expected_commit=
                manifest.repository_commit,
        )
    )
    _verify_materialization_source(
        repository
    )

    closure = run_production_closure(
        manifest_path=
            manifest_path,
        workspace_root=
            workspace_root,
        repository_root=
            repository,
        inspect_only=True,
        release_inputs=
            readiness_inputs,
    )
    if closure.phase != (
        "READY_TO_RELEASE"
    ):
        raise VN97ProductionMaterializationBlocked(
            closure
        )

    candidate = (
        load_release_candidate_directory(
            resolved.intake_output_dir
            / "release-candidate"
        )
    )
    if (
        closure
        .release_candidate_manifest_sha256
        != candidate.manifest_sha256
    ):
        raise VN97ProductionMaterializationError(
            "READY closure VN97RC1 identity mismatch"
        )
    output_dir = (
        readiness_inputs.output_dir
    )
    if output_dir is None:
        raise VN97ProductionMaterializationError(
            "release output directory is required"
        )
    if (
        output_dir.exists()
        or output_dir.is_symlink()
    ):
        raise VN97ProductionMaterializationError(
            "release output directory must not already exist"
        )

    args = build_release_argv(
        manifest=manifest,
        repository_root=repository,
        candidate_dir=candidate.root,
        readiness_inputs=
            readiness_inputs,
        validation=(
            validation
            or VN97FinalValidationConfig()
        ),
    )
    if release_runner is None:
        from .production_release_cli import (
            main as production_release_main,
        )
        release_runner = (
            production_release_main
        )

    stdout = io.StringIO()
    with redirect_stdout(stdout):
        result = release_runner(args)
    if result != 0:
        raise VN97ProductionMaterializationError(
            "canonical production release returned non-zero"
        )
    emitted = stdout.getvalue().strip()
    try:
        emitted_attestation = (
            parse_apk_attestation(
                emitted.encode(
                    "utf-8"
                )
            )
        )
    except Exception as exc:
        raise VN97ProductionMaterializationError(
            "canonical production release did not emit canonical VN97APK1"
        ) from exc

    final = verify_materialized_release(
        manifest=manifest,
        closure=closure,
        candidate=candidate,
        output_dir=output_dir,
        apksigner=
            readiness_inputs.apksigner,
        aapt=readiness_inputs.aapt,
        external_probe=
            external_probe,
    )
    published_attestation = (
        parse_apk_attestation(
            (
                output_dir
                / "release-attestation.vn97apk1"
            ).read_bytes()
        )
    )
    if (
        emitted_attestation
        != published_attestation
    ):
        raise VN97ProductionMaterializationError(
            "emitted/published VN97APK1 mismatch"
        )

    if receipt_path is not None:
        _atomic_create(
            receipt_path,
            final.to_bytes(),
        )
        parsed = (
            parse_final_release_receipt(
                receipt_path.read_bytes()
            )
        )
        if parsed != final:
            raise VN97ProductionMaterializationError(
                "persisted VN97FINAL1 does not round-trip"
            )
    return final
