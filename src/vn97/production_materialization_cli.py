from __future__ import annotations

import argparse
from pathlib import Path
import sys

from .production_closure import (
    VN97ReleaseReadinessInputs,
)
from .production_materialization import (
    VN97FinalValidationConfig,
    VN97ProductionMaterializationBlocked,
    materialize_final_release,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize and independently verify one final VN97 turnkey production APK."
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
        "--validation-format",
        choices=("text", "chat"),
        default="chat",
    )
    parser.add_argument(
        "--validation-sequence-length",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--validation-stride",
        type=int,
    )
    parser.add_argument(
        "--validation-batch-size",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--validation-device",
        default="auto",
    )
    parser.add_argument(
        "--min-validation-accuracy",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--min-validation-target-tokens",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--speech-validation-max-examples",
        type=int,
        default=100_000,
    )
    parser.add_argument(
        "--speech-max-frames",
        type=int,
        default=1500,
    )
    parser.add_argument(
        "--speech-max-target-tokens",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--min-speech-validation-accuracy",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--min-speech-validation-target-tokens",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--min-speech-validation-examples",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--vision-validation-max-examples",
        type=int,
        default=100_000,
    )
    parser.add_argument(
        "--vision-max-patches",
        type=int,
        default=196,
    )
    parser.add_argument(
        "--vision-max-target-tokens",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--min-vision-validation-accuracy",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--min-vision-validation-target-tokens",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--min-vision-validation-examples",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--gradle",
        default="gradle",
    )
    parser.add_argument(
        "--apksigner",
    )
    parser.add_argument(
        "--aapt",
    )
    parser.add_argument(
        "--receipt",
        help=(
            "optional new path for canonical VN97FINAL1 final-release receipt"
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

    readiness_inputs = (
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
    validation = (
        VN97FinalValidationConfig(
            validation_format=
                args.validation_format,
            validation_sequence_length=
                args
                .validation_sequence_length,
            validation_stride=
                args.validation_stride,
            validation_batch_size=
                args
                .validation_batch_size,
            validation_device=
                args.validation_device,
            min_validation_accuracy=
                args
                .min_validation_accuracy,
            min_validation_target_tokens=
                args
                .min_validation_target_tokens,
            speech_validation_max_examples=
                args
                .speech_validation_max_examples,
            speech_max_frames=
                args.speech_max_frames,
            speech_max_target_tokens=
                args
                .speech_max_target_tokens,
            min_speech_validation_accuracy=
                args
                .min_speech_validation_accuracy,
            min_speech_validation_target_tokens=
                args
                .min_speech_validation_target_tokens,
            min_speech_validation_examples=
                args
                .min_speech_validation_examples,
            vision_validation_max_examples=
                args
                .vision_validation_max_examples,
            vision_max_patches=
                args.vision_max_patches,
            vision_max_target_tokens=
                args
                .vision_max_target_tokens,
            min_vision_validation_accuracy=
                args
                .min_vision_validation_accuracy,
            min_vision_validation_target_tokens=
                args
                .min_vision_validation_target_tokens,
            min_vision_validation_examples=
                args
                .min_vision_validation_examples,
        )
    )

    try:
        final = materialize_final_release(
            manifest_path=
                manifest_path,
            workspace_root=
                workspace_root,
            repository_root=
                Path(
                    args.repository_root
                ),
            readiness_inputs=
                readiness_inputs,
            validation=validation,
            receipt_path=(
                None
                if args.receipt
                is None
                else Path(
                    args.receipt
                )
            ),
        )
    except (
        VN97ProductionMaterializationBlocked
    ) as exc:
        print(
            exc.closure_report
            .to_bytes()
            .decode("utf-8")
        )
        return 2

    print(
        final.to_bytes()
        .decode("utf-8")
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(
            main()
        )
    except Exception as exc:
        print(
            "vn97-production-materialize: "
            + str(exc),
            file=sys.stderr,
        )
        raise
