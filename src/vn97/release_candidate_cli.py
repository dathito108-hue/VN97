from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import stat
import sys
import tempfile

from .bootstrap_release_cli import (
    _load_speech_training_report,
    _load_vision_training_report,
)
from .deployment_checkpoint import (
    MAX_CHECKPOINT_BYTES,
    load_deployment_checkpoint_file,
)
from .device_evidence import (
    VN97DeviceEvidenceCriteria,
    load_device_evidence,
    require_device_evidence,
)
from .mobile_budget import (
    VN97MobileBudget,
    estimate_vn97_mobile_footprint,
)
from .model_image import build_model_image
from .release_candidate import (
    VN97ReleaseCandidateDeviceEvidence,
    VN97ReleaseCandidateManifest,
)
from .tokenizer import VN97TokenizerPackage


_MAX_REPORT_BYTES = 4 * 1024 * 1024
_MAX_TOKENIZER_BYTES = 64 * 1024 * 1024
_MAX_EVIDENCE_BYTES = 1024 * 1024


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _read_regular_file(
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
        raise ValueError(
            f"{label} could not be opened safely: {path}"
        ) from exc
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise ValueError(
                f"{label} must be a regular file"
            )
        if not 0 < info.st_size <= max_bytes:
            raise ValueError(
                f"{label} size is outside bounds"
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
            raise ValueError(
                f"{label} changed while being read"
            )
        return bytes(out)
    finally:
        os.close(fd)


def _strict_json_object(
    data: bytes,
    *,
    label: str,
) -> dict[str, object]:
    duplicates: list[str] = []

    def hook(
        pairs: list[tuple[str, object]],
    ) -> dict[str, object]:
        out: dict[str, object] = {}
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
        raise ValueError(
            f"{label} must be strict UTF-8 JSON"
        ) from exc
    if duplicates or not isinstance(root, dict):
        raise ValueError(
            f"{label} must be one object without duplicate keys"
        )
    if _canonical_json_bytes(root) != data:
        raise ValueError(
            f"{label} must use canonical JSON"
        )
    return root


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
        raise ValueError(
            f"{label} must be lowercase SHA-256"
        )
    return value


def _require_finite(
    value: object,
    label: str,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(
            value,
            (int, float),
        )
    ):
        raise ValueError(
            f"{label} must be numeric"
        )
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(
            f"{label} must be finite"
        )
    return result


def _require_quality_observation(
    raw: object,
    *,
    label: str,
) -> None:
    keys = {
        "dataset_sha256",
        "examples",
        "max_loss",
        "mean_loss",
        "min_target_tokens",
        "min_top1_accuracy",
        "target_tokens",
        "top1_accuracy",
    }
    if (
        not isinstance(raw, dict)
        or set(raw) != keys
    ):
        raise ValueError(
            f"{label} keys are invalid"
        )
    _require_sha256(
        raw["dataset_sha256"],
        f"{label} dataset_sha256",
    )
    for key in (
        "examples",
        "min_target_tokens",
        "target_tokens",
    ):
        value = raw[key]
        if type(value) is not int or value <= 0:
            raise ValueError(
                f"{label} {key} must be positive integer"
            )
    mean_loss = _require_finite(
        raw["mean_loss"],
        f"{label} mean_loss",
    )
    max_loss = _require_finite(
        raw["max_loss"],
        f"{label} max_loss",
    )
    top1 = _require_finite(
        raw["top1_accuracy"],
        f"{label} top1_accuracy",
    )
    minimum = _require_finite(
        raw["min_top1_accuracy"],
        f"{label} min_top1_accuracy",
    )
    if (
        max_loss <= 0.0
        or not 0.0 <= top1 <= 1.0
        or not 0.0 <= minimum <= 1.0
    ):
        raise ValueError(
            f"{label} thresholds/results are outside bounds"
        )
    if (
        mean_loss > max_loss
        or top1 < minimum
        or raw["target_tokens"]
        < raw["min_target_tokens"]
    ):
        raise ValueError(
            f"{label} does not satisfy its recorded release criteria"
        )


def _load_production_report(
    path: Path,
) -> tuple[
    dict[str, object],
    bytes,
    str,
]:
    data = _read_regular_file(
        path,
        max_bytes=_MAX_REPORT_BYTES,
        label="production campaign report",
    )
    root = _strict_json_object(
        data,
        label="production campaign report",
    )
    expected = {
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
        set(root) != expected
        or root["schema"] != "VN97PRODCAMP1"
    ):
        raise ValueError(
            "production campaign report schema/keys mismatch"
        )
    for key in (
        "language_campaign_report_sha256",
        "language_checkpoint_sha256",
        "model_image_sha256",
        "speech_training_dataset_sha256",
        "tokenizer_sha256",
        "unified_checkpoint_sha256",
    ):
        _require_sha256(
            root[key],
            f"production report {key}",
        )

    candidate_id = root["selected_candidate_id"]
    if (
        not isinstance(candidate_id, str)
        or len(candidate_id) != 16
        or any(
            ch not in "0123456789abcdef"
            for ch in candidate_id
        )
    ):
        raise ValueError(
            "production report selected_candidate_id is invalid"
        )

    deployment = root["deployment"]
    if (
        not isinstance(deployment, dict)
        or set(deployment)
        != {"tile_cols", "tile_rows"}
    ):
        raise ValueError(
            "production report deployment keys are invalid"
        )
    for key in ("tile_rows", "tile_cols"):
        value = deployment[key]
        if (
            type(value) is not int
            or not 1 <= value <= 256
        ):
            raise ValueError(
                f"production report {key} is invalid"
            )

    mobile_budget = root["mobile_budget"]
    if (
        not isinstance(mobile_budget, dict)
        or set(mobile_budget)
        != {
            "max_model_image_bytes",
            "max_recurrent_state_bytes",
        }
    ):
        raise ValueError(
            "production report mobile_budget keys are invalid"
        )
    VN97MobileBudget(
        max_model_image_bytes=
            mobile_budget[
                "max_model_image_bytes"
            ],
        max_recurrent_state_bytes=
            mobile_budget[
                "max_recurrent_state_bytes"
            ],
    )

    footprint = root["mobile_footprint"]
    footprint_keys = {
        "float_parameter_bytes",
        "model_image_bytes",
        "packed_ternary_bytes",
        "recurrent_state_bytes",
        "section_count",
        "tokenizer_bytes",
    }
    if (
        not isinstance(footprint, dict)
        or set(footprint) != footprint_keys
        or any(
            type(footprint[key]) is not int
            or footprint[key] < 0
            for key in footprint_keys
        )
        or footprint["model_image_bytes"] <= 0
        or footprint["recurrent_state_bytes"] <= 0
        or footprint["section_count"] <= 0
    ):
        raise ValueError(
            "production report mobile_footprint is invalid"
        )

    _require_quality_observation(
        root["speech_validation"],
        label="production speech validation",
    )
    _require_quality_observation(
        root["speech_release"],
        label="production speech release",
    )
    return (
        root,
        data,
        _sha256_bytes(data),
    )


def _copy_regular_file(
    source: Path,
    target: Path,
    *,
    max_bytes: int,
    expected_sha256: str,
    label: str,
) -> int:
    flags = os.O_RDONLY | getattr(
        os,
        "O_NOFOLLOW",
        0,
    )
    try:
        source_fd = os.open(
            source,
            flags,
        )
    except OSError as exc:
        raise ValueError(
            f"{label} could not be opened safely"
        ) from exc
    try:
        info = os.fstat(source_fd)
        if (
            not stat.S_ISREG(info.st_mode)
            or not 0 < info.st_size <= max_bytes
        ):
            raise ValueError(
                f"{label} source is outside byte/safety bounds"
            )
        target.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        temp_fd, temp_name = tempfile.mkstemp(
            prefix=".vn97rc-file-",
            dir=target.parent,
        )
        digest = hashlib.sha256()
        copied = 0
        try:
            with os.fdopen(
                temp_fd,
                "wb",
                closefd=True,
            ) as output:
                while copied < info.st_size:
                    chunk = os.read(
                        source_fd,
                        min(
                            1024 * 1024,
                            info.st_size - copied,
                        ),
                    )
                    if not chunk:
                        break
                    copied += len(chunk)
                    if copied > max_bytes:
                        raise ValueError(
                            f"{label} copy exceeds byte bound"
                        )
                    digest.update(chunk)
                    output.write(chunk)
                output.flush()
                os.fsync(output.fileno())

            after = os.fstat(source_fd)
            if (
                copied != info.st_size
                or after.st_size != info.st_size
                or after.st_ino != info.st_ino
                or after.st_dev != info.st_dev
            ):
                raise ValueError(
                    f"{label} changed while being copied"
                )
            if (
                digest.hexdigest()
                != expected_sha256
            ):
                raise ValueError(
                    f"{label} SHA-256 changed during assembly"
                )
            os.replace(
                temp_name,
                target,
            )
            return copied
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)
    finally:
        os.close(source_fd)


def _write_regular_file(
    target: Path,
    data: bytes,
) -> None:
    target.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    fd, temp_name = tempfile.mkstemp(
        prefix=".vn97rc-data-",
        dir=target.parent,
    )
    try:
        with os.fdopen(
            fd,
            "wb",
            closefd=True,
        ) as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
        os.replace(
            temp_name,
            target,
        )
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def _fsync_directory(
    path: Path,
) -> None:
    fd = os.open(
        path,
        os.O_RDONLY,
    )
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Qualify and assemble a self-contained VN97 production "
            "release candidate before M10N private-key signing."
        )
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
    )
    parser.add_argument(
        "--tokenizer",
        required=True,
    )
    parser.add_argument(
        "--production-campaign-report",
        required=True,
    )
    parser.add_argument(
        "--speech-training-report",
    )
    parser.add_argument(
        "--vision-training-report",
    )
    parser.add_argument(
        "--device-evidence",
        action="append",
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        required=True,
    )
    parser.add_argument(
        "--min-distinct-device-profiles",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--min-device-runs",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--max-text-prefill-p95-ms",
        type=float,
        default=10_000.0,
    )
    parser.add_argument(
        "--max-text-decode-p95-ms-per-token",
        type=float,
        default=2_000.0,
    )
    parser.add_argument(
        "--max-device-peak-pss-kib",
        type=int,
        default=2 * 1024 * 1024,
    )
    parser.add_argument(
        "--max-device-thermal-status",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--max-speech-prefill-p95-ms",
        type=float,
    )
    parser.add_argument(
        "--require-energy-counter",
        action="store_true",
    )
    parser.add_argument(
        "--max-abs-battery-energy-counter-delta-nwh",
        type=int,
    )
    return parser


def main(
    argv: list[str] | None = None,
) -> int:
    args = _parser().parse_args(argv)
    if args.min_distinct_device_profiles <= 0:
        raise ValueError(
            "min-distinct-device-profiles must be positive"
        )

    checkpoint_path = Path(args.checkpoint)
    tokenizer_path = Path(args.tokenizer)
    production_report_path = Path(
        args.production_campaign_report
    )

    production, _, production_sha256 = _load_production_report(
        production_report_path
    )
    deployment = production["deployment"]
    assert isinstance(deployment, dict)
    tile_rows = deployment["tile_rows"]
    tile_cols = deployment["tile_cols"]
    assert isinstance(tile_rows, int)
    assert isinstance(tile_cols, int)

    checkpoint = load_deployment_checkpoint_file(
        checkpoint_path
    )
    if (
        checkpoint.checkpoint_sha256
        != production["unified_checkpoint_sha256"]
    ):
        raise ValueError(
            "production checkpoint identity does not match VN97PRODCAMP1"
        )

    tokenizer_bytes = _read_regular_file(
        tokenizer_path,
        max_bytes=_MAX_TOKENIZER_BYTES,
        label="release tokenizer",
    )
    tokenizer_sha256 = _sha256_bytes(
        tokenizer_bytes
    )
    if (
        tokenizer_sha256
        != production["tokenizer_sha256"]
    ):
        raise ValueError(
            "production tokenizer identity does not match VN97PRODCAMP1"
        )
    tokenizer = VN97TokenizerPackage.from_bytes(
        tokenizer_bytes
    )
    if (
        tokenizer.vocab_size
        != checkpoint.config.vocab_size
    ):
        raise ValueError(
            "production tokenizer vocabulary does not match checkpoint"
        )

    preview = build_model_image(
        checkpoint.model,
        tokenizer=tokenizer,
        audio_adapter=checkpoint.audio_adapter,
        vision_adapter=checkpoint.vision_adapter,
        tile_rows=tile_rows,
        tile_cols=tile_cols,
    )
    model_image_sha256 = _sha256_bytes(
        preview.data
    )
    if (
        model_image_sha256
        != production["model_image_sha256"]
    ):
        raise ValueError(
            "reconstructed VN97MI1 identity does not match production campaign"
        )

    footprint = estimate_vn97_mobile_footprint(
        checkpoint.config,
        tokenizer_nbytes=len(tokenizer_bytes),
        audio_frame_size=(
            None
            if checkpoint.audio_adapter is None
            else int(
                checkpoint.audio_adapter
                .projection
                .in_features
            )
        ),
        vision_input_features=(
            None
            if checkpoint.vision_adapter is None
            else int(
                checkpoint.vision_adapter
                .projection
                .in_features
            )
        ),
        tile_rows=tile_rows,
        tile_cols=tile_cols,
        batch_size=1,
    )
    if (
        footprint.canonical_object()
        != production["mobile_footprint"]
    ):
        raise ValueError(
            "reconstructed mobile footprint does not match production campaign"
        )
    budget_raw = production["mobile_budget"]
    assert isinstance(budget_raw, dict)
    budget = VN97MobileBudget(
        max_model_image_bytes=budget_raw[
            "max_model_image_bytes"
        ],
        max_recurrent_state_bytes=budget_raw[
            "max_recurrent_state_bytes"
        ],
    )
    rejection = budget.rejection_status(
        footprint
    )
    if rejection is not None:
        raise ValueError(
            "production candidate no longer fits recorded mobile budget: "
            + rejection
        )

    speech_enabled = (
        checkpoint.audio_adapter is not None
    )
    vision_enabled = (
        checkpoint.vision_adapter is not None
    )

    speech_report_path: Path | None = None
    speech_report_sha256: str | None = None
    if speech_enabled:
        if not args.speech_training_report:
            raise ValueError(
                "speech-enabled checkpoint requires --speech-training-report"
            )
        speech_report_path = Path(
            args.speech_training_report
        )
        _, speech_report_sha256 = _load_speech_training_report(
            speech_report_path,
            checkpoint_sha256=checkpoint.checkpoint_sha256,
            tokenizer_sha256=tokenizer_sha256,
        )
    elif args.speech_training_report:
        raise ValueError(
            "speech training report supplied for checkpoint without speech weights"
        )

    vision_report_path: Path | None = None
    vision_report_sha256: str | None = None
    if vision_enabled:
        if not args.vision_training_report:
            raise ValueError(
                "vision-enabled checkpoint requires --vision-training-report"
            )
        vision_report_path = Path(
            args.vision_training_report
        )
        _, vision_report_sha256 = _load_vision_training_report(
            vision_report_path,
            checkpoint_sha256=checkpoint.checkpoint_sha256,
            tokenizer_sha256=tokenizer_sha256,
        )
    elif args.vision_training_report:
        raise ValueError(
            "vision training report supplied for checkpoint without vision weights"
        )

    criteria = VN97DeviceEvidenceCriteria(
        min_runs=args.min_device_runs,
        max_text_prefill_p95_ms=
            args.max_text_prefill_p95_ms,
        max_text_decode_p95_ms_per_token=
            args.max_text_decode_p95_ms_per_token,
        max_peak_pss_kib=
            args.max_device_peak_pss_kib,
        max_thermal_status=
            args.max_device_thermal_status,
        max_speech_prefill_p95_ms=
            args.max_speech_prefill_p95_ms,
        require_energy_counter=
            args.require_energy_counter,
        max_abs_battery_energy_counter_delta_nwh=
            args.max_abs_battery_energy_counter_delta_nwh,
    )

    evidence_entries: list[
        VN97ReleaseCandidateDeviceEvidence
    ] = []
    evidence_sources: list[
        tuple[Path, str]
    ] = []
    seen_digests: set[str] = set()
    profiles: set[
        tuple[str, str, int, str]
    ] = set()

    for raw_path in args.device_evidence:
        path = Path(raw_path)
        evidence = load_device_evidence(
            path
        )
        if (
            evidence.evidence_sha256
            in seen_digests
        ):
            raise ValueError(
                "duplicate mobile evidence file is not allowed"
            )
        require_device_evidence(
            evidence,
            criteria,
            expected_model_image_sha256=
                model_image_sha256,
            speech_enabled=speech_enabled,
        )
        if evidence.sdk_int < 26:
            raise ValueError(
                "mobile evidence device is below VN97 minSdk 26"
            )
        seen_digests.add(
            evidence.evidence_sha256
        )
        profiles.add(
            (
                evidence.manufacturer,
                evidence.model,
                evidence.sdk_int,
                evidence.abi,
            )
        )
        evidence_sources.append(
            (
                path,
                evidence.evidence_sha256,
            )
        )
        evidence_entries.append(
            VN97ReleaseCandidateDeviceEvidence(
                evidence_sha256=
                    evidence.evidence_sha256,
                manufacturer=
                    evidence.manufacturer,
                model=evidence.model,
                sdk_int=evidence.sdk_int,
                abi=evidence.abi,
                runs=evidence.runs,
                text_prefill_p95_ms=
                    evidence.text_prefill.p95_ms,
                text_decode_p95_ms_per_token=
                    evidence
                    .text_decode_per_token
                    .p95_ms,
                speech_prefill_p95_ms=(
                    None
                    if evidence.speech_prefill is None
                    else evidence.speech_prefill.p95_ms
                ),
                peak_pss_kib=
                    evidence.peak_pss_kib,
                thermal_status_max=
                    evidence.thermal_status_max,
                battery_energy_counter_delta_nwh=
                    evidence
                    .battery_energy_counter_delta_nwh,
            )
        )

    if (
        len(profiles)
        < args.min_distinct_device_profiles
    ):
        raise ValueError(
            "insufficient distinct mobile device profiles for release candidate"
        )

    evidence_entries.sort(
        key=lambda item: item.evidence_sha256
    )
    checkpoint_info = checkpoint_path.stat()
    if (
        not stat.S_ISREG(checkpoint_info.st_mode)
        or checkpoint_info.st_size <= 0
    ):
        raise ValueError(
            "release checkpoint must be a non-empty regular file"
        )
    manifest = VN97ReleaseCandidateManifest(
        selected_candidate_id=production[
            "selected_candidate_id"
        ],
        checkpoint_sha256=
            checkpoint.checkpoint_sha256,
        checkpoint_bytes=
            checkpoint_info.st_size,
        tokenizer_sha256=tokenizer_sha256,
        tokenizer_bytes=len(tokenizer_bytes),
        production_campaign_report_sha256=
            production_sha256,
        model_image_sha256=
            model_image_sha256,
        tile_rows=tile_rows,
        tile_cols=tile_cols,
        speech_enabled=speech_enabled,
        vision_enabled=vision_enabled,
        speech_training_report_sha256=
            speech_report_sha256,
        vision_training_report_sha256=
            vision_report_sha256,
        device_evidence=
            tuple(evidence_entries),
    )

    output = Path(args.output_dir)
    if output.exists() or output.is_symlink():
        raise ValueError(
            "release candidate output-dir must not already exist"
        )
    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    if output.parent.is_symlink():
        raise ValueError(
            "release candidate output parent must not be a symlink"
        )
    staging = Path(
        tempfile.mkdtemp(
            prefix=".vn97rc-",
            dir=output.parent,
        )
    )
    try:
        _copy_regular_file(
            checkpoint_path,
            staging / "model.vn97ck1",
            max_bytes=MAX_CHECKPOINT_BYTES,
            expected_sha256=
                checkpoint.checkpoint_sha256,
            label="release checkpoint",
        )
        _copy_regular_file(
            tokenizer_path,
            staging / "tokenizer.vn97tk1",
            max_bytes=_MAX_TOKENIZER_BYTES,
            expected_sha256=tokenizer_sha256,
            label="release tokenizer",
        )
        _copy_regular_file(
            production_report_path,
            staging /
                "production-campaign-report.json",
            max_bytes=_MAX_REPORT_BYTES,
            expected_sha256=production_sha256,
            label="production campaign report",
        )
        if (
            speech_report_path is not None
            and speech_report_sha256 is not None
        ):
            _copy_regular_file(
                speech_report_path,
                staging /
                    "speech-training-report.json",
                max_bytes=_MAX_REPORT_BYTES,
                expected_sha256=
                    speech_report_sha256,
                label="speech training report",
            )
        if (
            vision_report_path is not None
            and vision_report_sha256 is not None
        ):
            _copy_regular_file(
                vision_report_path,
                staging /
                    "vision-training-report.json",
                max_bytes=_MAX_REPORT_BYTES,
                expected_sha256=
                    vision_report_sha256,
                label="vision training report",
            )

        evidence_dir = staging / "device-evidence"
        evidence_dir.mkdir()
        for index, (
            source,
            digest,
        ) in enumerate(
            sorted(
                evidence_sources,
                key=lambda item: item[1],
            ),
            start=1,
        ):
            _copy_regular_file(
                source,
                evidence_dir /
                    (
                        f"{index:03d}-"
                        f"{digest}.json"
                    ),
                max_bytes=_MAX_EVIDENCE_BYTES,
                expected_sha256=digest,
                label="mobile device evidence",
            )

        _write_regular_file(
            staging /
                "release-candidate.vn97rc1",
            manifest.to_bytes(),
        )
        _fsync_directory(evidence_dir)
        _fsync_directory(staging)
        os.replace(staging, output)
        _fsync_directory(output.parent)
    except BaseException:
        shutil.rmtree(
            staging,
            ignore_errors=True,
        )
        raise

    print(
        "VN97RC1 "
        f"candidate={manifest.selected_candidate_id} "
        f"checkpoint_sha256={manifest.checkpoint_sha256} "
        f"model_image_sha256={manifest.model_image_sha256} "
        f"device_profiles={len(profiles)}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(
            main()
        )
    except Exception as exc:
        print(
            f"vn97-release-candidate: {exc}",
            file=sys.stderr,
        )
        raise
