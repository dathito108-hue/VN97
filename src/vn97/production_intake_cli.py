from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import stat
import sys
import tempfile

from .production_intake import (
    MAX_REPORT_BYTES,
    VN97ProductionIntakeReport,
    inspect_device_evidence_files,
    inspect_language_campaign_directory,
    inspect_production_campaign_directory,
)
from .release_candidate import (
    load_release_candidate_directory,
)


def _copy_bound_file(
    source: Path,
    target: Path,
    *,
    expected_sha256: str,
    max_bytes: int,
    label: str,
) -> None:
    import hashlib

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
        fd, temp_name = tempfile.mkstemp(
            prefix=".vn97-intake-source-",
            dir=target.parent,
        )
        digest = hashlib.sha256()
        copied = 0
        try:
            with os.fdopen(
                fd,
                "wb",
                closefd=True,
            ) as output:
                while copied < info.st_size:
                    chunk = os.read(
                        source_fd,
                        min(
                            256 * 1024,
                            info.st_size - copied,
                        ),
                    )
                    if not chunk:
                        break
                    copied += len(chunk)
                    digest.update(chunk)
                    output.write(chunk)
                output.flush()
                os.fsync(output.fileno())
            after = os.fstat(source_fd)
            if (
                copied != info.st_size
                or after.st_size
                != info.st_size
                or after.st_ino
                != info.st_ino
                or after.st_dev
                != info.st_dev
            ):
                raise ValueError(
                    f"{label} changed while being copied"
                )
            if (
                digest.hexdigest()
                != expected_sha256
            ):
                raise ValueError(
                    f"{label} SHA-256 changed during intake"
                )
            os.replace(
                temp_name,
                target,
            )
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)
    finally:
        os.close(source_fd)


def _write_atomic(
    target: Path,
    data: bytes,
) -> None:
    target.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    fd, temp_name = tempfile.mkstemp(
        prefix=".vn97-intake-report-",
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


def _fsync_directory(path: Path) -> None:
    fd = os.open(
        path,
        os.O_RDONLY |
        getattr(os, "O_DIRECTORY", 0),
    )
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify real VN97 campaign outputs and physical-device evidence, "
            "then assemble one M19B VN97RC1 intake bundle."
        )
    )
    parser.add_argument(
        "--language-campaign-dir",
        required=True,
    )
    parser.add_argument(
        "--production-campaign-dir",
        required=True,
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

    language = (
        inspect_language_campaign_directory(
            args.language_campaign_dir
        )
    )
    production = (
        inspect_production_campaign_directory(
            args.production_campaign_dir,
            language=language,
        )
    )
    evidence = inspect_device_evidence_files(
        [
            Path(value)
            for value in args.device_evidence
        ],
        expected_model_image_sha256=
            production.model_image_sha256,
    )
    profiles = {
        (
            item.manufacturer,
            item.model,
            item.sdk_int,
            item.abi,
        )
        for item in evidence
    }
    if (
        len(profiles)
        < args.min_distinct_device_profiles
    ):
        raise ValueError(
            "insufficient distinct physical-device profiles for production intake"
        )

    output = Path(args.output_dir)
    if output.exists() or output.is_symlink():
        raise ValueError(
            "production intake output-dir must not already exist"
        )
    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    try:
        parent_info = os.lstat(
            output.parent
        )
    except OSError as exc:
        raise ValueError(
            "production intake output parent is unavailable"
        ) from exc
    if (
        stat.S_ISLNK(parent_info.st_mode)
        or not stat.S_ISDIR(
            parent_info.st_mode
        )
    ):
        raise ValueError(
            "production intake output parent must be a real directory"
        )

    staging = Path(
        tempfile.mkdtemp(
            prefix=".vn97-production-intake-",
            dir=output.parent,
        )
    )
    try:
        candidate_output = (
            staging /
            "release-candidate"
        )

        from .release_candidate_cli import (
            main as release_candidate_main,
        )

        candidate_args = [
            "--checkpoint",
            str(
                production.checkpoint_path
            ),
            "--tokenizer",
            str(
                production.tokenizer_path
            ),
            "--production-campaign-report",
            str(
                production.report_path
            ),
            "--speech-training-report",
            str(
                production
                .speech_training_report_path
            ),
            "--output-dir",
            str(candidate_output),
            "--min-distinct-device-profiles",
            str(
                args
                .min_distinct_device_profiles
            ),
            "--min-device-runs",
            str(args.min_device_runs),
            "--max-text-prefill-p95-ms",
            str(
                args
                .max_text_prefill_p95_ms
            ),
            "--max-text-decode-p95-ms-per-token",
            str(
                args
                .max_text_decode_p95_ms_per_token
            ),
            "--max-device-peak-pss-kib",
            str(
                args
                .max_device_peak_pss_kib
            ),
            "--max-device-thermal-status",
            str(
                args
                .max_device_thermal_status
            ),
        ]
        if (
            args.max_speech_prefill_p95_ms
            is not None
        ):
            candidate_args.extend(
                [
                    "--max-speech-prefill-p95-ms",
                    str(
                        args
                        .max_speech_prefill_p95_ms
                    ),
                ]
            )
        if args.require_energy_counter:
            candidate_args.append(
                "--require-energy-counter"
            )
        if (
            args
            .max_abs_battery_energy_counter_delta_nwh
            is not None
        ):
            candidate_args.extend(
                [
                    "--max-abs-battery-energy-counter-delta-nwh",
                    str(
                        args
                        .max_abs_battery_energy_counter_delta_nwh
                    ),
                ]
            )
        for path in args.device_evidence:
            candidate_args.extend(
                [
                    "--device-evidence",
                    path,
                ]
            )

        result = release_candidate_main(
            candidate_args
        )
        if result != 0:
            raise RuntimeError(
                "M19B release candidate assembly failed"
            )

        candidate = (
            load_release_candidate_directory(
                candidate_output
            )
        )
        manifest = candidate.manifest
        expected_evidence = tuple(
            sorted(
                item.evidence_sha256
                for item in evidence
            )
        )
        actual_evidence = tuple(
            sorted(
                item.evidence_sha256
                for item in
                manifest.device_evidence
            )
        )
        if (
            manifest.selected_candidate_id
            != production.selected_candidate_id
            or manifest.checkpoint_sha256
            != production.unified_checkpoint_sha256
            or manifest.tokenizer_sha256
            != production.tokenizer_sha256
            or manifest.model_image_sha256
            != production.model_image_sha256
            or manifest
            .production_campaign_report_sha256
            != production.report_sha256
            or manifest
            .speech_training_report_sha256
            != production
            .speech_training_report_sha256
            or actual_evidence
            != expected_evidence
        ):
            raise RuntimeError(
                "M19B VN97RC1 identities do not match M19F intake chain"
            )

        intake = VN97ProductionIntakeReport(
            selected_candidate_id=
                production.selected_candidate_id,
            language_campaign_report_sha256=
                language.report_sha256,
            language_checkpoint_sha256=
                language
                .selected_checkpoint_sha256,
            production_campaign_report_sha256=
                production.report_sha256,
            speech_training_report_sha256=
                production
                .speech_training_report_sha256,
            unified_checkpoint_sha256=
                production
                .unified_checkpoint_sha256,
            tokenizer_sha256=
                production.tokenizer_sha256,
            model_image_sha256=
                production.model_image_sha256,
            device_evidence_sha256=
                expected_evidence,
            distinct_device_profiles=
                len(profiles),
            release_candidate_manifest_sha256=
                candidate.manifest_sha256,
        )
        _copy_bound_file(
            language.report_path,
            staging /
                "language-campaign-report.json",
            expected_sha256=
                language.report_sha256,
            max_bytes=MAX_REPORT_BYTES,
            label="language campaign report",
        )
        _write_atomic(
            staging /
                "production-intake.vn97intake1",
            intake.to_bytes(),
        )
        _fsync_directory(
            candidate_output /
            "device-evidence"
        )
        _fsync_directory(candidate_output)
        _fsync_directory(staging)
        os.replace(
            staging,
            output,
        )
        _fsync_directory(
            output.parent
        )
    except BaseException:
        shutil.rmtree(
            staging,
            ignore_errors=True,
        )
        raise

    print(
        "VN97INTAKE1 "
        f"candidate={production.selected_candidate_id} "
        f"model_image_sha256={production.model_image_sha256} "
        f"device_profiles={len(profiles)} "
        f"release_candidate_sha256={candidate.manifest_sha256}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(
            f"vn97-production-intake: {exc}",
            file=sys.stderr,
        )
        raise
