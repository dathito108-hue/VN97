from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any

from .device_evidence import (
    VN97DeviceEvidence,
    load_device_evidence,
)


SCHEMA = "VN97INTAKE1"
MAX_REPORT_BYTES = 4 * 1024 * 1024
MAX_CHECKPOINT_BYTES = 2 * 1024 * 1024 * 1024
MAX_TOKENIZER_BYTES = 64 * 1024 * 1024
MAX_DEVICE_EVIDENCE = 128


class VN97ProductionIntakeError(RuntimeError):
    pass


def _sha256_file(
    path: Path,
    *,
    max_bytes: int,
    label: str,
) -> tuple[str, int]:
    flags = os.O_RDONLY | getattr(
        os,
        "O_NOFOLLOW",
        0,
    )
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise VN97ProductionIntakeError(
            f"{label} could not be opened safely"
        ) from exc
    try:
        info = os.fstat(fd)
        if (
            not stat.S_ISREG(info.st_mode)
            or not 0 < info.st_size <= max_bytes
        ):
            raise VN97ProductionIntakeError(
                f"{label} byte size/type is invalid"
            )
        digest = hashlib.sha256()
        consumed = 0
        while consumed < info.st_size:
            chunk = os.read(
                fd,
                min(
                    1024 * 1024,
                    info.st_size - consumed,
                ),
            )
            if not chunk:
                break
            consumed += len(chunk)
            digest.update(chunk)
        after = os.fstat(fd)
        if (
            consumed != info.st_size
            or after.st_size != info.st_size
            or after.st_ino != info.st_ino
            or after.st_dev != info.st_dev
        ):
            raise VN97ProductionIntakeError(
                f"{label} changed while being hashed"
            )
        return digest.hexdigest(), int(info.st_size)
    finally:
        os.close(fd)


def _read_json_file(
    path: Path,
    *,
    max_bytes: int,
    label: str,
) -> tuple[dict[str, Any], bytes, str]:
    flags = os.O_RDONLY | getattr(
        os,
        "O_NOFOLLOW",
        0,
    )
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise VN97ProductionIntakeError(
            f"{label} could not be opened safely"
        ) from exc
    try:
        info = os.fstat(fd)
        if (
            not stat.S_ISREG(info.st_mode)
            or not 0 < info.st_size <= max_bytes
        ):
            raise VN97ProductionIntakeError(
                f"{label} byte size/type is invalid"
            )
        out = bytearray()
        while len(out) < info.st_size:
            chunk = os.read(
                fd,
                min(
                    256 * 1024,
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
            raise VN97ProductionIntakeError(
                f"{label} changed while being read"
            )
    finally:
        os.close(fd)

    data = bytes(out)
    duplicates: list[str] = []

    def hook(
        pairs: list[tuple[str, Any]],
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
            parse_constant=lambda raw: (
                _ for _ in ()
            ).throw(ValueError(raw)),
        )
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        ValueError,
    ) as exc:
        raise VN97ProductionIntakeError(
            f"{label} must be strict UTF-8 JSON"
        ) from exc
    if duplicates or not isinstance(root, dict):
        raise VN97ProductionIntakeError(
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
        raise VN97ProductionIntakeError(
            f"{label} must use canonical JSON"
        )
    return (
        root,
        data,
        hashlib.sha256(data).hexdigest(),
    )


def _require_sha256(
    value: object,
    label: str,
) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(
            ch not in "0123456789abcdef"
            for ch in value
        )
    ):
        raise VN97ProductionIntakeError(
            f"{label} must be lowercase SHA-256"
        )
    return value


def _require_candidate_id(
    value: object,
    label: str,
) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 16
        or any(
            ch not in "0123456789abcdef"
            for ch in value
        )
    ):
        raise VN97ProductionIntakeError(
            f"{label} must be 16 lowercase hex characters"
        )
    return value


def _real_directory(
    path: Path,
    *,
    label: str,
    expected_entries: set[str],
) -> Path:
    try:
        info = os.lstat(path)
    except OSError as exc:
        raise VN97ProductionIntakeError(
            f"{label} is unavailable"
        ) from exc
    if (
        stat.S_ISLNK(info.st_mode)
        or not stat.S_ISDIR(info.st_mode)
    ):
        raise VN97ProductionIntakeError(
            f"{label} must be a real directory"
        )
    root = path.resolve(strict=True)
    actual = {
        entry.name
        for entry in root.iterdir()
    }
    if actual != expected_entries:
        raise VN97ProductionIntakeError(
            f"{label} entries do not match canonical output layout"
        )
    return root


@dataclass(frozen=True)
class VN97LanguageCampaignSource:
    root: Path
    report_sha256: str
    selected_candidate_id: str
    selected_checkpoint_sha256: str
    tokenizer_sha256: str
    checkpoint_path: Path
    tokenizer_path: Path
    report_path: Path


@dataclass(frozen=True)
class VN97ProductionCampaignSource:
    root: Path
    report_sha256: str
    speech_training_report_sha256: str
    selected_candidate_id: str
    language_campaign_report_sha256: str
    language_checkpoint_sha256: str
    unified_checkpoint_sha256: str
    tokenizer_sha256: str
    model_image_sha256: str
    checkpoint_path: Path
    tokenizer_path: Path
    speech_training_report_path: Path
    report_path: Path


def inspect_language_campaign_directory(
    path: str | os.PathLike[str],
) -> VN97LanguageCampaignSource:
    root = _real_directory(
        Path(path),
        label="language campaign directory",
        expected_entries={
            "campaign-report.json",
            "model.vn97ck1",
            "tokenizer.vn97tk1",
        },
    )
    report_path = root / "campaign-report.json"
    report, _, report_sha256 = _read_json_file(
        report_path,
        max_bytes=MAX_REPORT_BYTES,
        label="language campaign report",
    )
    expected_root_keys = {
        "candidates",
        "criteria",
        "release_dataset_sha256",
        "release_evaluation",
        "schema",
        "selected_candidate_id",
        "selected_checkpoint_sha256",
        "tokenizer_sha256",
        "training_dataset_sha256",
        "validation_dataset_sha256",
    }
    if (
        set(report) != expected_root_keys
        or report["schema"] != "VN97CAMP2"
    ):
        raise VN97ProductionIntakeError(
            "language campaign report schema/keys mismatch"
        )
    selected_candidate_id = _require_candidate_id(
        report["selected_candidate_id"],
        "language selected candidate id",
    )
    selected_checkpoint_sha256 = _require_sha256(
        report["selected_checkpoint_sha256"],
        "language selected checkpoint SHA-256",
    )
    tokenizer_sha256 = _require_sha256(
        report["tokenizer_sha256"],
        "language tokenizer SHA-256",
    )
    for key in (
        "training_dataset_sha256",
        "validation_dataset_sha256",
        "release_dataset_sha256",
    ):
        _require_sha256(
            report[key],
            f"language {key}",
        )
    if (
        not isinstance(report["candidates"], list)
        or not report["candidates"]
    ):
        raise VN97ProductionIntakeError(
            "language campaign report must contain candidate observations"
        )
    selected_rows = [
        row
        for row in report["candidates"]
        if (
            isinstance(row, dict)
            and row.get("candidate_id")
            == selected_candidate_id
        )
    ]
    if len(selected_rows) != 1:
        raise VN97ProductionIntakeError(
            "language selected candidate is not uniquely represented"
        )

    checkpoint_path = root / "model.vn97ck1"
    actual_checkpoint_sha, _ = _sha256_file(
        checkpoint_path,
        max_bytes=MAX_CHECKPOINT_BYTES,
        label="language campaign checkpoint",
    )
    if (
        actual_checkpoint_sha
        != selected_checkpoint_sha256
    ):
        raise VN97ProductionIntakeError(
            "language campaign checkpoint SHA-256 does not match VN97CAMP2"
        )

    tokenizer_path = root / "tokenizer.vn97tk1"
    actual_tokenizer_sha, _ = _sha256_file(
        tokenizer_path,
        max_bytes=MAX_TOKENIZER_BYTES,
        label="language campaign tokenizer",
    )
    if actual_tokenizer_sha != tokenizer_sha256:
        raise VN97ProductionIntakeError(
            "language campaign tokenizer SHA-256 does not match VN97CAMP2"
        )

    return VN97LanguageCampaignSource(
        root=root,
        report_sha256=report_sha256,
        selected_candidate_id=
            selected_candidate_id,
        selected_checkpoint_sha256=
            selected_checkpoint_sha256,
        tokenizer_sha256=tokenizer_sha256,
        checkpoint_path=checkpoint_path,
        tokenizer_path=tokenizer_path,
        report_path=report_path,
    )


def inspect_production_campaign_directory(
    path: str | os.PathLike[str],
    *,
    language: VN97LanguageCampaignSource,
) -> VN97ProductionCampaignSource:
    root = _real_directory(
        Path(path),
        label="production campaign directory",
        expected_entries={
            "model.vn97ck1",
            "production-campaign-report.json",
            "speech-training-report.json",
            "tokenizer.vn97tk1",
        },
    )
    report_path = (
        root /
        "production-campaign-report.json"
    )
    report, _, report_sha256 = _read_json_file(
        report_path,
        max_bytes=MAX_REPORT_BYTES,
        label="production campaign report",
    )
    expected_root_keys = {
        "deployment",
        "language_campaign_report_sha256",
        "language_checkpoint_sha256",
        "mobile_budget",
        "mobile_footprint",
        "model_image_sha256",
        "schema",
        "selected_candidate_id",
        "speech_release",
        "speech_training_dataset_sha256",
        "speech_validation",
        "tokenizer_sha256",
        "unified_checkpoint_sha256",
    }
    if (
        set(report) != expected_root_keys
        or report["schema"] != "VN97PRODCAMP1"
    ):
        raise VN97ProductionIntakeError(
            "production campaign report schema/keys mismatch"
        )

    selected_candidate_id = _require_candidate_id(
        report["selected_candidate_id"],
        "production selected candidate id",
    )
    language_report_sha = _require_sha256(
        report[
            "language_campaign_report_sha256"
        ],
        "production language report SHA-256",
    )
    language_checkpoint_sha = _require_sha256(
        report["language_checkpoint_sha256"],
        "production language checkpoint SHA-256",
    )
    unified_checkpoint_sha = _require_sha256(
        report["unified_checkpoint_sha256"],
        "production unified checkpoint SHA-256",
    )
    tokenizer_sha = _require_sha256(
        report["tokenizer_sha256"],
        "production tokenizer SHA-256",
    )
    model_image_sha = _require_sha256(
        report["model_image_sha256"],
        "production model image SHA-256",
    )
    _require_sha256(
        report["speech_training_dataset_sha256"],
        "production speech training dataset SHA-256",
    )

    if (
        selected_candidate_id
        != language.selected_candidate_id
        or language_report_sha
        != language.report_sha256
        or language_checkpoint_sha
        != language.selected_checkpoint_sha256
        or tokenizer_sha
        != language.tokenizer_sha256
    ):
        raise VN97ProductionIntakeError(
            "VN97PRODCAMP1 does not chain to the supplied VN97CAMP2 winner"
        )

    checkpoint_path = root / "model.vn97ck1"
    actual_checkpoint_sha, _ = _sha256_file(
        checkpoint_path,
        max_bytes=MAX_CHECKPOINT_BYTES,
        label="production unified checkpoint",
    )
    if actual_checkpoint_sha != unified_checkpoint_sha:
        raise VN97ProductionIntakeError(
            "production checkpoint SHA-256 does not match VN97PRODCAMP1"
        )

    tokenizer_path = root / "tokenizer.vn97tk1"
    actual_tokenizer_sha, _ = _sha256_file(
        tokenizer_path,
        max_bytes=MAX_TOKENIZER_BYTES,
        label="production tokenizer",
    )
    if actual_tokenizer_sha != tokenizer_sha:
        raise VN97ProductionIntakeError(
            "production tokenizer SHA-256 does not match VN97PRODCAMP1"
        )

    speech_path = (
        root /
        "speech-training-report.json"
    )
    speech, _, speech_sha = _read_json_file(
        speech_path,
        max_bytes=MAX_REPORT_BYTES,
        label="speech training report",
    )
    speech_keys = {
        "base_checkpoint_sha256",
        "checkpoint_sha256",
        "dataset_sha256",
        "examples",
        "final_loss",
        "mean_loss",
        "schema",
        "steps",
        "target_tokens",
        "tokenizer_sha256",
        "training",
    }
    if (
        set(speech) != speech_keys
        or speech["schema"]
        != "VN97SPEECHTRAIN1"
    ):
        raise VN97ProductionIntakeError(
            "speech training report schema/keys mismatch"
        )
    if (
        _require_sha256(
            speech["base_checkpoint_sha256"],
            "speech base checkpoint SHA-256",
        )
        != language_checkpoint_sha
        or _require_sha256(
            speech["checkpoint_sha256"],
            "speech unified checkpoint SHA-256",
        )
        != unified_checkpoint_sha
        or _require_sha256(
            speech["tokenizer_sha256"],
            "speech tokenizer SHA-256",
        )
        != tokenizer_sha
        or _require_sha256(
            speech["dataset_sha256"],
            "speech dataset SHA-256",
        )
        != report[
            "speech_training_dataset_sha256"
        ]
    ):
        raise VN97ProductionIntakeError(
            "speech training provenance does not match VN97PRODCAMP1"
        )

    return VN97ProductionCampaignSource(
        root=root,
        report_sha256=report_sha256,
        speech_training_report_sha256=
            speech_sha,
        selected_candidate_id=
            selected_candidate_id,
        language_campaign_report_sha256=
            language_report_sha,
        language_checkpoint_sha256=
            language_checkpoint_sha,
        unified_checkpoint_sha256=
            unified_checkpoint_sha,
        tokenizer_sha256=tokenizer_sha,
        model_image_sha256=model_image_sha,
        checkpoint_path=checkpoint_path,
        tokenizer_path=tokenizer_path,
        speech_training_report_path=
            speech_path,
        report_path=report_path,
    )


def inspect_device_evidence_files(
    paths: list[Path],
    *,
    expected_model_image_sha256: str,
) -> tuple[VN97DeviceEvidence, ...]:
    if not 1 <= len(paths) <= MAX_DEVICE_EVIDENCE:
        raise VN97ProductionIntakeError(
            "device evidence count is outside bounds"
        )
    evidence: list[VN97DeviceEvidence] = []
    seen: set[str] = set()
    for path in paths:
        item = load_device_evidence(path)
        if (
            item.evidence_sha256 in seen
            or item.model_image_sha256
            != expected_model_image_sha256
        ):
            raise VN97ProductionIntakeError(
                "device evidence identity/model-image chain is invalid"
            )
        if item.sdk_int < 26:
            raise VN97ProductionIntakeError(
                "device evidence is below VN97 Android minSdk 26"
            )
        seen.add(item.evidence_sha256)
        evidence.append(item)
    evidence.sort(
        key=lambda item:
            item.evidence_sha256
    )
    return tuple(evidence)


@dataclass(frozen=True)
class VN97ProductionIntakeReport:
    selected_candidate_id: str
    language_campaign_report_sha256: str
    language_checkpoint_sha256: str
    production_campaign_report_sha256: str
    speech_training_report_sha256: str
    unified_checkpoint_sha256: str
    tokenizer_sha256: str
    model_image_sha256: str
    device_evidence_sha256: tuple[str, ...]
    distinct_device_profiles: int
    release_candidate_manifest_sha256: str

    def __post_init__(self) -> None:
        _require_candidate_id(
            self.selected_candidate_id,
            "intake selected candidate id",
        )
        for value, label in (
            (
                self.language_campaign_report_sha256,
                "language campaign report SHA-256",
            ),
            (
                self.language_checkpoint_sha256,
                "language checkpoint SHA-256",
            ),
            (
                self.production_campaign_report_sha256,
                "production campaign report SHA-256",
            ),
            (
                self.speech_training_report_sha256,
                "speech training report SHA-256",
            ),
            (
                self.unified_checkpoint_sha256,
                "unified checkpoint SHA-256",
            ),
            (
                self.tokenizer_sha256,
                "tokenizer SHA-256",
            ),
            (
                self.model_image_sha256,
                "model image SHA-256",
            ),
            (
                self.release_candidate_manifest_sha256,
                "release candidate manifest SHA-256",
            ),
        ):
            _require_sha256(value, label)
        if (
            not self.device_evidence_sha256
            or tuple(
                sorted(
                    set(
                        self.device_evidence_sha256
                    )
                )
            )
            != self.device_evidence_sha256
        ):
            raise VN97ProductionIntakeError(
                "device evidence SHA identities must be unique, non-empty and sorted"
            )
        for digest in self.device_evidence_sha256:
            _require_sha256(
                digest,
                "device evidence SHA-256",
            )
        if (
            type(self.distinct_device_profiles)
            is not int
            or not 1
            <= self.distinct_device_profiles
            <= len(
                self.device_evidence_sha256
            )
        ):
            raise VN97ProductionIntakeError(
                "distinct device profile count is invalid"
            )

    def canonical_object(self) -> dict[str, object]:
        return {
            "device_evidence_sha256":
                list(
                    self.device_evidence_sha256
                ),
            "distinct_device_profiles":
                self.distinct_device_profiles,
            "language_campaign_report_sha256":
                self.language_campaign_report_sha256,
            "language_checkpoint_sha256":
                self.language_checkpoint_sha256,
            "model_image_sha256":
                self.model_image_sha256,
            "production_campaign_report_sha256":
                self.production_campaign_report_sha256,
            "release_candidate_manifest_sha256":
                self.release_candidate_manifest_sha256,
            "schema": SCHEMA,
            "selected_candidate_id":
                self.selected_candidate_id,
            "speech_training_report_sha256":
                self.speech_training_report_sha256,
            "tokenizer_sha256":
                self.tokenizer_sha256,
            "unified_checkpoint_sha256":
                self.unified_checkpoint_sha256,
        }

    def to_bytes(self) -> bytes:
        return json.dumps(
            self.canonical_object(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")


def parse_production_intake_report(
    data: bytes,
) -> VN97ProductionIntakeReport:
    if not 0 < len(data) <= 256 * 1024:
        raise VN97ProductionIntakeError(
            "VN97INTAKE1 byte size is outside bounds"
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
        raise VN97ProductionIntakeError(
            "VN97INTAKE1 must be strict UTF-8 JSON"
        ) from exc
    if duplicates or not isinstance(root, dict):
        raise VN97ProductionIntakeError(
            "VN97INTAKE1 must be one object without duplicate keys"
        )
    canonical = json.dumps(
        root,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    if canonical != data:
        raise VN97ProductionIntakeError(
            "VN97INTAKE1 must use canonical JSON"
        )

    expected = {
        "device_evidence_sha256",
        "distinct_device_profiles",
        "language_campaign_report_sha256",
        "language_checkpoint_sha256",
        "model_image_sha256",
        "production_campaign_report_sha256",
        "release_candidate_manifest_sha256",
        "schema",
        "selected_candidate_id",
        "speech_training_report_sha256",
        "tokenizer_sha256",
        "unified_checkpoint_sha256",
    }
    if (
        set(root) != expected
        or root["schema"] != SCHEMA
    ):
        raise VN97ProductionIntakeError(
            "VN97INTAKE1 schema/keys mismatch"
        )
    raw_evidence = root[
        "device_evidence_sha256"
    ]
    if (
        not isinstance(raw_evidence, list)
        or not all(
            isinstance(item, str)
            for item in raw_evidence
        )
    ):
        raise VN97ProductionIntakeError(
            "VN97INTAKE1 device evidence list is invalid"
        )
    try:
        return VN97ProductionIntakeReport(
            selected_candidate_id=
                root["selected_candidate_id"],
            language_campaign_report_sha256=
                root[
                    "language_campaign_report_sha256"
                ],
            language_checkpoint_sha256=
                root[
                    "language_checkpoint_sha256"
                ],
            production_campaign_report_sha256=
                root[
                    "production_campaign_report_sha256"
                ],
            speech_training_report_sha256=
                root[
                    "speech_training_report_sha256"
                ],
            unified_checkpoint_sha256=
                root[
                    "unified_checkpoint_sha256"
                ],
            tokenizer_sha256=
                root["tokenizer_sha256"],
            model_image_sha256=
                root["model_image_sha256"],
            device_evidence_sha256=
                tuple(raw_evidence),
            distinct_device_profiles=
                root[
                    "distinct_device_profiles"
                ],
            release_candidate_manifest_sha256=
                root[
                    "release_candidate_manifest_sha256"
                ],
        )
    except (
        TypeError,
        ValueError,
        VN97ProductionIntakeError,
    ) as exc:
        if isinstance(
            exc,
            VN97ProductionIntakeError,
        ):
            raise
        raise VN97ProductionIntakeError(
            str(exc)
        ) from exc
