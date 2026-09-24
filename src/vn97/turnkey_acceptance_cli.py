from __future__ import annotations

import argparse
from pathlib import Path
import sys

from .turnkey_acceptance import (
    run_clean_device_acceptance,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run M19M clean-device acceptance against one exact materialized "
            "VN97 production APK on one explicit physical Android device."
        )
    )
    parser.add_argument(
        "--final-receipt",
        required=True,
        help="canonical VN97FINAL1 receipt",
    )
    parser.add_argument(
        "--release-dir",
        required=True,
        help="canonical four-file M19L release directory",
    )
    parser.add_argument(
        "--repository-root",
        default=".",
    )
    parser.add_argument(
        "--serial",
        required=True,
        help="explicit physical adb serial",
    )
    parser.add_argument(
        "--adb",
        default="adb",
    )
    parser.add_argument(
        "--output",
        help="optional new path for immutable VN97ACCEPT1 receipt",
    )
    parser.add_argument(
        "--keep-installed",
        action="store_true",
        help=(
            "leave the accepted production APK installed after success; "
            "default is uninstall cleanup"
        ),
    )
    parser.add_argument(
        "--launch-timeout-seconds",
        type=int,
        default=120,
    )
    parser.add_argument(
        "--reboot-timeout-seconds",
        type=int,
        default=240,
    )
    parser.add_argument(
        "--selftest-timeout-seconds",
        type=int,
        default=60,
    )
    return parser


def main(
    argv: list[str] | None = None,
) -> int:
    args = _parser().parse_args(argv)
    for value, label in (
        (
            args.launch_timeout_seconds,
            "launch timeout",
        ),
        (
            args.reboot_timeout_seconds,
            "reboot timeout",
        ),
        (
            args.selftest_timeout_seconds,
            "self-test timeout",
        ),
    ):
        if (
            type(value) is not int
            or not 10 <= value <= 1800
        ):
            raise ValueError(
                f"{label} must be in [10, 1800] seconds"
            )

    receipt = run_clean_device_acceptance(
        final_receipt_path=
            Path(args.final_receipt),
        release_dir=
            Path(args.release_dir),
        repository_root=
            Path(args.repository_root),
        serial=args.serial,
        output_path=(
            None
            if args.output is None
            else Path(args.output)
        ),
        adb_executable=args.adb,
        keep_installed=
            args.keep_installed,
        launch_timeout_seconds=
            args.launch_timeout_seconds,
        reboot_timeout_seconds=
            args.reboot_timeout_seconds,
        selftest_timeout_seconds=
            args.selftest_timeout_seconds,
    )
    print(
        receipt.to_bytes()
        .decode("utf-8")
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(
            "vn97-turnkey-accept: "
            + str(exc),
            file=sys.stderr,
        )
        raise
