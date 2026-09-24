from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import stat
from typing import Any


SCHEMA = "VN97RC1"
MAX_MANIFEST_BYTES = 4 * 1024 * 1024


class VN97ReleaseCandidateError(RuntimeError):
    pass


def _require_sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(ch not in "0123456789abcdef" for ch in value)
    ):
        raise VN97ReleaseCandidateError(
            f"{label} must be lowercase SHA-256"
        )
    return value


def _require_nonempty_text(
    value: object,
    label: str,
    *,
    max_chars: int = 256,
) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > max_chars
        or any(ord(ch) < 0x20 for ch in value)
    ):
        raise VN97ReleaseCandidateError(
            f"{label} must be bounded non-empty text"
        )
    return value


def _require_positive_int(
    value: object,
    label: str,
) -> int:
    if type(value) is not int or value <= 0:
        raise VN97ReleaseCandidateError(
            f"{label} must be positive integer"
        )
    return value


def _require_nonnegative_number(
    value: object,
    label: str,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
    ):
        raise VN97ReleaseCandidateError(
            f"{label} must be numeric"
        )
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise VN97ReleaseCandidateError(
            f"{label} must be finite and non-negative"
        )
    return result


@dataclass(frozen=True)
class VN97ReleaseCandidateDeviceEvidence:
    evidence_sha256: str
    manufacturer: str
    model: str
    sdk_int: int
    abi: str
    runs: int
    text_prefill_p95_ms: float
    text_decode_p95_ms_per_token: float
    speech_prefill_p95_ms: float | None
    peak_pss_kib: int
    thermal_status_max: int
    battery_energy_counter_delta_nwh: int | None

    def __post_init__(self) -> None:
        _require_sha256(
            self.evidence_sha256,
            "device evidence SHA-256",
        )
        _require_nonempty_text(
            self.manufacturer,
            "device manufacturer",
        )
        _require_nonempty_text(
            self.model,
            "device model",
        )
        _require_nonempty_text(
            self.abi,
            "device ABI",
        )
        _require_positive_int(
            self.sdk_int,
            "device sdk_int",
        )
        if self.sdk_int < 26:
            raise VN97ReleaseCandidateError(
                "device evidence is below VN97 minSdk 26"
            )
        _require_positive_int(
            self.runs,
            "device evidence runs",
        )
        _require_nonnegative_number(
            self.text_prefill_p95_ms,
            "text prefill p95",
        )
        _require_nonnegative_number(
            self.text_decode_p95_ms_per_token,
            "text decode p95/token",
        )
        if self.speech_prefill_p95_ms is not None:
            _require_nonnegative_number(
                self.speech_prefill_p95_ms,
                "speech prefill p95",
            )
        _require_positive_int(
            self.peak_pss_kib,
            "peak PSS",
        )
        if (
            type(self.thermal_status_max) is not int
            or not 0 <= self.thermal_status_max <= 6
        ):
            raise VN97ReleaseCandidateError(
                "thermal status must be integer in [0, 6]"
            )
        if (
            self.battery_energy_counter_delta_nwh
            is not None
            and type(
                self.battery_energy_counter_delta_nwh
            )
            is not int
        ):
            raise VN97ReleaseCandidateError(
                "battery energy counter delta must be integer or null"
            )

    def canonical_object(self) -> dict[str, object]:
        return {
            "abi": self.abi,
            "battery_energy_counter_delta_nwh":
                self.battery_energy_counter_delta_nwh,
            "evidence_sha256": self.evidence_sha256,
            "manufacturer": self.manufacturer,
            "model": self.model,
            "peak_pss_kib": self.peak_pss_kib,
            "runs": self.runs,
            "sdk_int": self.sdk_int,
            "speech_prefill_p95_ms":
                self.speech_prefill_p95_ms,
            "text_decode_p95_ms_per_token":
                self.text_decode_p95_ms_per_token,
            "text_prefill_p95_ms":
                self.text_prefill_p95_ms,
            "thermal_status_max":
                self.thermal_status_max,
        }


@dataclass(frozen=True)
class VN97ReleaseCandidateManifest:
    selected_candidate_id: str
    checkpoint_sha256: str
    checkpoint_bytes: int
    tokenizer_sha256: str
    tokenizer_bytes: int
    production_campaign_report_sha256: str
    model_image_sha256: str
    tile_rows: int
    tile_cols: int
    speech_enabled: bool
    vision_enabled: bool
    speech_training_report_sha256: str | None
    vision_training_report_sha256: str | None
    device_evidence: tuple[
        VN97ReleaseCandidateDeviceEvidence,
        ...,
    ]

    def __post_init__(self) -> None:
        if (
            len(self.selected_candidate_id) != 16
            or any(
                ch not in "0123456789abcdef"
                for ch in self.selected_candidate_id
            )
        ):
            raise VN97ReleaseCandidateError(
                "selected candidate id must be 16 lowercase hex characters"
            )
        _require_sha256(
            self.checkpoint_sha256,
            "checkpoint SHA-256",
        )
        _require_positive_int(
            self.checkpoint_bytes,
            "checkpoint bytes",
        )
        _require_sha256(
            self.tokenizer_sha256,
            "tokenizer SHA-256",
        )
        _require_positive_int(
            self.tokenizer_bytes,
            "tokenizer bytes",
        )
        _require_sha256(
            self.production_campaign_report_sha256,
            "production campaign report SHA-256",
        )
        _require_sha256(
            self.model_image_sha256,
            "model image SHA-256",
        )
        for value, label in (
            (self.tile_rows, "tile_rows"),
            (self.tile_cols, "tile_cols"),
        ):
            if type(value) is not int or not 1 <= value <= 1024:
                raise VN97ReleaseCandidateError(
                    f"{label} must be integer in [1, 1024]"
                )
        if type(self.speech_enabled) is not bool:
            raise VN97ReleaseCandidateError(
                "speech_enabled must be boolean"
            )
        if type(self.vision_enabled) is not bool:
            raise VN97ReleaseCandidateError(
                "vision_enabled must be boolean"
            )
        if self.speech_enabled:
            if self.speech_training_report_sha256 is None:
                raise VN97ReleaseCandidateError(
                    "speech-enabled release candidate requires speech training report"
                )
            _require_sha256(
                self.speech_training_report_sha256,
                "speech training report SHA-256",
            )
        elif self.speech_training_report_sha256 is not None:
            raise VN97ReleaseCandidateError(
                "speech training report is not allowed without speech weights"
            )
        if self.vision_enabled:
            if self.vision_training_report_sha256 is None:
                raise VN97ReleaseCandidateError(
                    "vision-enabled release candidate requires vision training report"
                )
            _require_sha256(
                self.vision_training_report_sha256,
                "vision training report SHA-256",
            )
        elif self.vision_training_report_sha256 is not None:
            raise VN97ReleaseCandidateError(
                "vision training report is not allowed without vision weights"
            )
        if not self.device_evidence:
            raise VN97ReleaseCandidateError(
                "release candidate requires mobile device evidence"
            )
        digests = [
            item.evidence_sha256
            for item in self.device_evidence
        ]
        if len(set(digests)) != len(digests):
            raise VN97ReleaseCandidateError(
                "duplicate device evidence is not allowed"
            )

    def canonical_object(self) -> dict[str, object]:
        return {
            "checkpoint_bytes": self.checkpoint_bytes,
            "checkpoint_sha256": self.checkpoint_sha256,
            "deployment": {
                "tile_cols": self.tile_cols,
                "tile_rows": self.tile_rows,
            },
            "device_evidence": [
                item.canonical_object()
                for item in self.device_evidence
            ],
            "model_image_sha256":
                self.model_image_sha256,
            "production_campaign_report_sha256":
                self.production_campaign_report_sha256,
            "schema": SCHEMA,
            "selected_candidate_id":
                self.selected_candidate_id,
            "speech_enabled": self.speech_enabled,
            "speech_training_report_sha256":
                self.speech_training_report_sha256,
            "tokenizer_bytes": self.tokenizer_bytes,
            "tokenizer_sha256": self.tokenizer_sha256,
            "vision_enabled": self.vision_enabled,
            "vision_training_report_sha256":
                self.vision_training_report_sha256,
        }

    def to_bytes(self) -> bytes:
        return json.dumps(
            self.canonical_object(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")


def parse_release_candidate_manifest(
    data: bytes,
) -> VN97ReleaseCandidateManifest:
    if not 0 < len(data) <= MAX_MANIFEST_BYTES:
        raise VN97ReleaseCandidateError(
            "VN97RC1 byte size is outside bounds"
        )
    duplicates: list[str] = []

    def hook(
        pairs: list[tuple[str, Any]],
    ) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in pairs:
            if key in out:
                duplicates.append(key)
            out[key] = value
        return out

    try:
        text = data.decode(
            "utf-8",
            errors="strict",
        )
        root = json.loads(
            text,
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
        raise VN97ReleaseCandidateError(
            "VN97RC1 must be strict UTF-8 JSON"
        ) from exc

    if duplicates or not isinstance(root, dict):
        raise VN97ReleaseCandidateError(
            "VN97RC1 must be one object without duplicate keys"
        )
    canonical = json.dumps(
        root,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    if canonical != data:
        raise VN97ReleaseCandidateError(
            "VN97RC1 must use canonical JSON"
        )

    expected = {
        "checkpoint_bytes",
        "checkpoint_sha256",
        "deployment",
        "device_evidence",
        "model_image_sha256",
        "production_campaign_report_sha256",
        "schema",
        "selected_candidate_id",
        "speech_enabled",
        "speech_training_report_sha256",
        "tokenizer_bytes",
        "tokenizer_sha256",
        "vision_enabled",
        "vision_training_report_sha256",
    }
    if set(root) != expected or root["schema"] != SCHEMA:
        raise VN97ReleaseCandidateError(
            "VN97RC1 schema/keys mismatch"
        )
    deployment = root["deployment"]
    if (
        not isinstance(deployment, dict)
        or set(deployment)
        != {"tile_cols", "tile_rows"}
    ):
        raise VN97ReleaseCandidateError(
            "VN97RC1 deployment keys are invalid"
        )
    raw_evidence = root["device_evidence"]
    if (
        not isinstance(raw_evidence, list)
        or not raw_evidence
    ):
        raise VN97ReleaseCandidateError(
            "VN97RC1 device_evidence must be non-empty array"
        )
    evidence: list[
        VN97ReleaseCandidateDeviceEvidence
    ] = []
    evidence_keys = {
        "abi",
        "battery_energy_counter_delta_nwh",
        "evidence_sha256",
        "manufacturer",
        "model",
        "peak_pss_kib",
        "runs",
        "sdk_int",
        "speech_prefill_p95_ms",
        "text_decode_p95_ms_per_token",
        "text_prefill_p95_ms",
        "thermal_status_max",
    }
    for item in raw_evidence:
        if (
            not isinstance(item, dict)
            or set(item) != evidence_keys
        ):
            raise VN97ReleaseCandidateError(
                "VN97RC1 device evidence keys are invalid"
            )
        evidence.append(
            VN97ReleaseCandidateDeviceEvidence(
                evidence_sha256=
                    item["evidence_sha256"],
                manufacturer=item["manufacturer"],
                model=item["model"],
                sdk_int=item["sdk_int"],
                abi=item["abi"],
                runs=item["runs"],
                text_prefill_p95_ms=
                    item["text_prefill_p95_ms"],
                text_decode_p95_ms_per_token=
                    item[
                        "text_decode_p95_ms_per_token"
                    ],
                speech_prefill_p95_ms=
                    item["speech_prefill_p95_ms"],
                peak_pss_kib=
                    item["peak_pss_kib"],
                thermal_status_max=
                    item["thermal_status_max"],
                battery_energy_counter_delta_nwh=
                    item[
                        "battery_energy_counter_delta_nwh"
                    ],
            )
        )

    try:
        return VN97ReleaseCandidateManifest(
            selected_candidate_id=
                root["selected_candidate_id"],
            checkpoint_sha256=
                root["checkpoint_sha256"],
            checkpoint_bytes=
                root["checkpoint_bytes"],
            tokenizer_sha256=
                root["tokenizer_sha256"],
            tokenizer_bytes=
                root["tokenizer_bytes"],
            production_campaign_report_sha256=
                root[
                    "production_campaign_report_sha256"
                ],
            model_image_sha256=
                root["model_image_sha256"],
            tile_rows=deployment["tile_rows"],
            tile_cols=deployment["tile_cols"],
            speech_enabled=
                root["speech_enabled"],
            vision_enabled=
                root["vision_enabled"],
            speech_training_report_sha256=
                root[
                    "speech_training_report_sha256"
                ],
            vision_training_report_sha256=
                root[
                    "vision_training_report_sha256"
                ],
            device_evidence=tuple(evidence),
        )
    except (
        TypeError,
        ValueError,
        VN97ReleaseCandidateError,
    ) as exc:
        if isinstance(
            exc,
            VN97ReleaseCandidateError,
        ):
            raise
        raise VN97ReleaseCandidateError(
            str(exc)
        ) from exc


@dataclass(frozen=True)
class VN97LoadedReleaseCandidate:
    root: Path
    manifest: VN97ReleaseCandidateManifest
    manifest_sha256: str
    checkpoint_path: Path
    tokenizer_path: Path
    production_campaign_report_path: Path
    speech_training_report_path: Path | None
    vision_training_report_path: Path | None
    device_evidence_paths: tuple[Path, ...]


def _read_candidate_file(
    path: Path,
    *,
    max_bytes: int,
    label: str,
) -> bytes:
    flags = os.O_RDONLY | getattr(
        os,
        "O_NOFOLLOW",
        0,
    )
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise VN97ReleaseCandidateError(
            f"{label} could not be opened safely"
        ) from exc
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise VN97ReleaseCandidateError(
                f"{label} must be a regular file"
            )
        if not 0 < info.st_size <= max_bytes:
            raise VN97ReleaseCandidateError(
                f"{label} byte size is outside bounds"
            )
        out = bytearray()
        while len(out) < info.st_size:
            chunk = os.read(
                fd,
                min(
                    1024 * 1024,
                    info.st_size - len(out),
                ),
            )
            if not chunk:
                break
            out.extend(chunk)
        after = os.fstat(fd)
        if (
            len(out) != info.st_size
            or after.st_size != info.st_size
            or after.st_ino != info.st_ino
            or after.st_dev != info.st_dev
        ):
            raise VN97ReleaseCandidateError(
                f"{label} changed while being read"
            )
        return bytes(out)
    finally:
        os.close(fd)


def _require_candidate_file_identity(
    path: Path,
    *,
    expected_bytes: int | None,
    expected_sha256: str,
    max_bytes: int,
    label: str,
) -> None:
    flags = os.O_RDONLY | getattr(
        os,
        "O_NOFOLLOW",
        0,
    )
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise VN97ReleaseCandidateError(
            f"{label} could not be opened safely"
        ) from exc
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise VN97ReleaseCandidateError(
                f"{label} must be a regular file"
            )
        if not 0 < info.st_size <= max_bytes:
            raise VN97ReleaseCandidateError(
                f"{label} byte size is outside bounds"
            )
        if (
            expected_bytes is not None
            and info.st_size != expected_bytes
        ):
            raise VN97ReleaseCandidateError(
                f"{label} byte size does not match VN97RC1"
            )
        digest = hashlib.sha256()
        read_bytes = 0
        while read_bytes < info.st_size:
            chunk = os.read(
                fd,
                min(
                    1024 * 1024,
                    info.st_size - read_bytes,
                ),
            )
            if not chunk:
                break
            read_bytes += len(chunk)
            digest.update(chunk)
        after = os.fstat(fd)
        if (
            read_bytes != info.st_size
            or after.st_size != info.st_size
            or after.st_ino != info.st_ino
            or after.st_dev != info.st_dev
        ):
            raise VN97ReleaseCandidateError(
                f"{label} changed while being verified"
            )
        if digest.hexdigest() != expected_sha256:
            raise VN97ReleaseCandidateError(
                f"{label} SHA-256 does not match VN97RC1"
            )
    finally:
        os.close(fd)


def load_release_candidate_directory(
    path: str | os.PathLike[str],
) -> VN97LoadedReleaseCandidate:
    root = Path(path)
    try:
        info = os.lstat(root)
    except OSError as exc:
        raise VN97ReleaseCandidateError(
            "release candidate directory is unavailable"
        ) from exc
    if (
        stat.S_ISLNK(info.st_mode)
        or not stat.S_ISDIR(info.st_mode)
    ):
        raise VN97ReleaseCandidateError(
            "release candidate root must be a real directory"
        )

    manifest_path = root / "release-candidate.vn97rc1"
    manifest_bytes = _read_candidate_file(
        manifest_path,
        max_bytes=MAX_MANIFEST_BYTES,
        label="VN97RC1 manifest",
    )
    manifest = parse_release_candidate_manifest(
        manifest_bytes
    )
    manifest_sha256 = hashlib.sha256(
        manifest_bytes
    ).hexdigest()

    expected_top = {
        "device-evidence",
        "model.vn97ck1",
        "production-campaign-report.json",
        "release-candidate.vn97rc1",
        "tokenizer.vn97tk1",
    }
    if manifest.speech_enabled:
        expected_top.add(
            "speech-training-report.json"
        )
    if manifest.vision_enabled:
        expected_top.add(
            "vision-training-report.json"
        )
    actual_top = {
        entry.name
        for entry in root.iterdir()
    }
    if actual_top != expected_top:
        raise VN97ReleaseCandidateError(
            "release candidate directory entries do not match VN97RC1"
        )

    checkpoint_path = root / "model.vn97ck1"
    tokenizer_path = root / "tokenizer.vn97tk1"
    production_report_path = (
        root /
        "production-campaign-report.json"
    )
    _require_candidate_file_identity(
        checkpoint_path,
        expected_bytes=manifest.checkpoint_bytes,
        expected_sha256=manifest.checkpoint_sha256,
        max_bytes=2 * 1024 * 1024 * 1024,
        label="release candidate checkpoint",
    )
    _require_candidate_file_identity(
        tokenizer_path,
        expected_bytes=manifest.tokenizer_bytes,
        expected_sha256=manifest.tokenizer_sha256,
        max_bytes=64 * 1024 * 1024,
        label="release candidate tokenizer",
    )
    _require_candidate_file_identity(
        production_report_path,
        expected_bytes=None,
        expected_sha256=
            manifest.production_campaign_report_sha256,
        max_bytes=4 * 1024 * 1024,
        label="release candidate production report",
    )

    speech_path: Path | None = None
    if manifest.speech_enabled:
        speech_path = (
            root /
            "speech-training-report.json"
        )
        assert (
            manifest
            .speech_training_report_sha256
            is not None
        )
        _require_candidate_file_identity(
            speech_path,
            expected_bytes=None,
            expected_sha256=
                manifest
                .speech_training_report_sha256,
            max_bytes=4 * 1024 * 1024,
            label="release candidate speech report",
        )

    vision_path: Path | None = None
    if manifest.vision_enabled:
        vision_path = (
            root /
            "vision-training-report.json"
        )
        assert (
            manifest
            .vision_training_report_sha256
            is not None
        )
        _require_candidate_file_identity(
            vision_path,
            expected_bytes=None,
            expected_sha256=
                manifest
                .vision_training_report_sha256,
            max_bytes=4 * 1024 * 1024,
            label="release candidate vision report",
        )

    evidence_root = root / "device-evidence"
    try:
        evidence_info = os.lstat(
            evidence_root
        )
    except OSError as exc:
        raise VN97ReleaseCandidateError(
            "release candidate device-evidence directory is unavailable"
        ) from exc
    if (
        stat.S_ISLNK(evidence_info.st_mode)
        or not stat.S_ISDIR(
            evidence_info.st_mode
        )
    ):
        raise VN97ReleaseCandidateError(
            "release candidate device-evidence must be a real directory"
        )

    ordered = sorted(
        manifest.device_evidence,
        key=lambda item:
            item.evidence_sha256,
    )
    expected_names = [
        (
            f"{index:03d}-"
            f"{item.evidence_sha256}.json"
        )
        for index, item in enumerate(
            ordered,
            start=1,
        )
    ]
    actual_names = sorted(
        entry.name
        for entry in evidence_root.iterdir()
    )
    if actual_names != expected_names:
        raise VN97ReleaseCandidateError(
            "release candidate device-evidence entries do not match VN97RC1"
        )

    evidence_paths: list[Path] = []
    for name, item in zip(
        expected_names,
        ordered,
        strict=True,
    ):
        evidence_path = (
            evidence_root /
            name
        )
        _require_candidate_file_identity(
            evidence_path,
            expected_bytes=None,
            expected_sha256=
                item.evidence_sha256,
            max_bytes=1024 * 1024,
            label="release candidate device evidence",
        )
        evidence_paths.append(
            evidence_path
        )

    return VN97LoadedReleaseCandidate(
        root=root,
        manifest=manifest,
        manifest_sha256=
            manifest_sha256,
        checkpoint_path=
            checkpoint_path,
        tokenizer_path=
            tokenizer_path,
        production_campaign_report_path=
            production_report_path,
        speech_training_report_path=
            speech_path,
        vision_training_report_path=
            vision_path,
        device_evidence_paths=
            tuple(evidence_paths),
    )
