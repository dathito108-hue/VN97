from __future__ import annotations

import argparse
from pathlib import Path
import sys

from .device_evidence_campaign import (
    VN97PhysicalEvidenceConfig,
    _production_policy,
    collect_physical_device_evidence,
)
from .production_run_manifest import (
    load_production_run_manifest,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Collect real VN97MOBEVID1 records from explicit physical Android devices "
            "using the VN97RUN1-bound debug evidence harness."
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
        "--serial",
        action="append",
        required=True,
        help=(
            "explicit physical adb serial; repeat for multiple devices"
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
        "--timeout-seconds",
        type=int,
        default=300,
    )
    return parser


def main(
    argv: list[str] | None = None,
) -> int:
    args = _parser().parse_args(argv)
    manifest_path = Path(
        args.manifest
    )
    manifest = (
        load_production_run_manifest(
            manifest_path
        )
    )
    policy = _production_policy(
        manifest
    )
    measured_runs = (
        max(
            5,
            policy.criteria.min_runs,
        )
        if args.measured_runs
        is None
        else args.measured_runs
    )
    config = VN97PhysicalEvidenceConfig(
        warmup_runs=args.warmup_runs,
        measured_runs=measured_runs,
        decode_tokens=args.decode_tokens,
        speech_frames=args.speech_frames,
        timeout_seconds=
            args.timeout_seconds,
    )
    workspace_root = (
        Path(args.workspace_root)
        if args.workspace_root
        is not None
        else manifest_path.parent
    )
    records = (
        collect_physical_device_evidence(
            manifest_path=
                manifest_path,
            workspace_root=
                workspace_root,
            repository_root=
                Path(
                    args.repository_root
                ),
            serials=
                tuple(args.serial),
            config=config,
            adb_executable=args.adb,
            gradle_executable=
                args.gradle,
        )
    )
    profiles = {
        (
            item.evidence.manufacturer,
            item.evidence.model,
            item.evidence.sdk_int,
            item.evidence.abi,
        )
        for item in records
    }
    print(
        "M19J physical evidence campaign PASS "
        f"records={len(records)} "
        f"distinct_profiles={len(profiles)} "
        f"model_image_sha256="
        f"{records[0].evidence.model_image_sha256}"
    )
    for item in records:
        evidence = item.evidence
        print(
            "VN97MOBEVID1 "
            f"sha256={evidence.evidence_sha256} "
            f"device={evidence.manufacturer}/{evidence.model} "
            f"sdk={evidence.sdk_int} "
            f"abi={evidence.abi} "
            f"runs={evidence.runs}"
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(
            f"vn97-device-evidence-campaign: {exc}",
            file=sys.stderr,
        )
        raise
