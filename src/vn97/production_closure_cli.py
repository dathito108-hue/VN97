from __future__ import annotations

import argparse
from pathlib import Path
import sys

from .device_evidence_campaign import (
    VN97PhysicalEvidenceConfig,
)
from .production_closure import (
    VN97ReleaseReadinessInputs,
    run_production_closure,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Resume one VN97RUN1 campaign through training, physical evidence, "
            "intake and VN97READY1 readiness without bypassing canonical gates."
        )
    )
    parser.add_argument(
        "--manifest",
        required=True,
    )
    parser.add_argument(
        "--workspace-root",
        help=(
            "VN97RUN1 workspace root; defaults to the manifest parent"
        ),
    )
    parser.add_argument(
        "--repository-root",
        default=".",
    )
    parser.add_argument(
        "--inspect-only",
        action="store_true",
        help=(
            "inspect canonical artifact state without running training, M19J or intake"
        ),
    )

    parser.add_argument(
        "--serial",
        action="append",
        default=[],
        help=(
            "explicit physical adb serial for M19J; repeat for multiple phones"
        ),
    )
    parser.add_argument(
        "--adb",
        default="adb",
    )
    parser.add_argument(
        "--gradle",
        default="gradle",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--measured-runs",
        type=int,
    )
    parser.add_argument(
        "--decode-tokens",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--speech-frames",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--evidence-timeout-seconds",
        type=int,
        default=300,
    )

    parser.add_argument(
        "--private-key",
    )
    parser.add_argument(
        "--key-id",
    )
    parser.add_argument(
        "--capability-version",
        type=int,
    )
    parser.add_argument(
        "--source-origin",
    )
    parser.add_argument(
        "--source-license",
    )
    parser.add_argument(
        "--validation-input",
        action="append",
        default=[],
        help=(
            "fresh held-out language validation input for M19E; repeat as needed"
        ),
    )
    parser.add_argument(
        "--speech-validation-input",
    )
    parser.add_argument(
        "--vision-validation-input",
    )
    parser.add_argument(
        "--max-validation-loss",
        type=float,
    )
    parser.add_argument(
        "--max-speech-validation-loss",
        type=float,
    )
    parser.add_argument(
        "--max-vision-validation-loss",
        type=float,
    )
    parser.add_argument(
        "--release-output-dir",
    )
    parser.add_argument(
        "--apksigner",
    )
    parser.add_argument(
        "--aapt",
    )
    parser.add_argument(
        "--closure-report",
        help=(
            "optional atomically replaced canonical VN97CLOSE1 status path"
        ),
    )
    parser.add_argument(
        "--readiness-report",
        help=(
            "optional atomically replaced canonical VN97READY1 path once intake exists"
        ),
    )
    return parser


def main(
    argv: list[str] | None = None,
) -> int:
    args = _parser().parse_args(
        argv
    )
    manifest_path = Path(
        args.manifest
    )
    workspace_root = (
        Path(
            args.workspace_root
        )
        if args.workspace_root
        is not None
        else manifest_path.parent
    )

    evidence_config = (
        None
        if args.measured_runs
        is None
        else VN97PhysicalEvidenceConfig(
            warmup_runs=
                args.warmup_runs,
            measured_runs=
                args.measured_runs,
            decode_tokens=
                args.decode_tokens,
            speech_frames=
                args.speech_frames,
            timeout_seconds=
                args
                .evidence_timeout_seconds,
        )
    )
    if (
        args.measured_runs is None
        and (
            args.warmup_runs != 1
            or args.decode_tokens != 16
            or args.speech_frames != 8
            or args.evidence_timeout_seconds
            != 300
        )
    ):
        raise ValueError(
            "custom M19J benchmark controls require explicit --measured-runs"
        )

    release_inputs = (
        VN97ReleaseReadinessInputs(
            publisher_private_key=(
                None
                if args.private_key
                is None
                else Path(
                    args.private_key
                )
            ),
            validation_inputs=tuple(
                Path(value)
                for value in
                    args.validation_input
            ),
            speech_validation_input=(
                None
                if args
                .speech_validation_input
                is None
                else Path(
                    args
                    .speech_validation_input
                )
            ),
            vision_validation_input=(
                None
                if args
                .vision_validation_input
                is None
                else Path(
                    args
                    .vision_validation_input
                )
            ),
            output_dir=(
                None
                if args
                .release_output_dir
                is None
                else Path(
                    args
                    .release_output_dir
                )
            ),
            key_id=args.key_id,
            capability_version=
                args.capability_version,
            source_origin=
                args.source_origin,
            source_license=
                args.source_license,
            max_validation_loss=
                args.max_validation_loss,
            max_speech_validation_loss=
                args
                .max_speech_validation_loss,
            max_vision_validation_loss=
                args
                .max_vision_validation_loss,
            gradle=args.gradle,
            apksigner=
                args.apksigner,
            aapt=args.aapt,
        )
    )

    report = run_production_closure(
        manifest_path=
            manifest_path,
        workspace_root=
            workspace_root,
        repository_root=
            Path(
                args.repository_root
            ),
        serials=tuple(
            args.serial
        ),
        inspect_only=
            args.inspect_only,
        evidence_config=
            evidence_config,
        release_inputs=
            release_inputs,
        closure_report_path=(
            None
            if args.closure_report
            is None
            else Path(
                args.closure_report
            )
        ),
        readiness_report_path=(
            None
            if args.readiness_report
            is None
            else Path(
                args.readiness_report
            )
        ),
        adb_executable=args.adb,
        gradle_executable=
            args.gradle,
    )
    print(
        report.to_bytes()
        .decode("utf-8")
    )
    return (
        0
        if report.phase
        == "READY_TO_RELEASE"
        else 2
    )


if __name__ == "__main__":
    try:
        raise SystemExit(
            main()
        )
    except Exception as exc:
        print(
            "vn97-production-close: "
            + str(exc),
            file=sys.stderr,
        )
        raise
