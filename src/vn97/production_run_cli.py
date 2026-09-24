from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile

from .device_evidence import (
    load_device_evidence,
)
from .production_intake import (
    inspect_language_campaign_directory,
    inspect_production_campaign_directory,
    parse_production_intake_report,
)
from .production_run_manifest import (
    VN97ProductionRunManifest,
    VN97ProductionRunReceipt,
    load_production_run_manifest,
    options_to_argv,
    verify_production_run_inputs,
)
from .release_candidate import (
    load_release_candidate_directory,
)



def _sha256_file(path: Path) -> str:
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


def _real_directory(
    path: Path,
    *,
    label: str,
) -> Path:
    try:
        info = os.lstat(path)
    except OSError as exc:
        raise ValueError(
            f"{label} is unavailable"
        ) from exc
    if (
        stat.S_ISLNK(info.st_mode)
        or not stat.S_ISDIR(info.st_mode)
    ):
        raise ValueError(
            f"{label} must be a real directory"
        )
    return path.resolve(strict=True)


def _verify_repository(
    repository_root: Path,
    *,
    expected_commit: str,
) -> Path:
    root = _real_directory(
        repository_root,
        label="VN97 repository root",
    )
    if not (
        root /
        "src/vn97/campaign_cli.py"
    ).is_file():
        raise ValueError(
            "repository root is not the canonical VN97 source checkout"
        )
    try:
        commit = subprocess.run(
            [
                "git",
                "rev-parse",
                "HEAD",
            ],
            cwd=root,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout.strip().lower()
        dirty = subprocess.run(
            [
                "git",
                "status",
                "--porcelain",
                "--untracked-files=no",
            ],
            cwd=root,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout
    except (
        OSError,
        subprocess.CalledProcessError,
    ) as exc:
        raise ValueError(
            "VN97 repository Git identity could not be verified"
        ) from exc
    if commit != expected_commit:
        raise ValueError(
            "VN97RUN1 repository_commit does not match checkout HEAD"
        )
    if dirty:
        raise ValueError(
            "VN97 production run requires a clean tracked Git worktree"
        )

    current_runner = Path(__file__).resolve()
    current_manifest_module = sys.modules[
        VN97ProductionRunManifest.__module__
    ]
    current_manifest_file = Path(
        current_manifest_module.__file__
    ).resolve()
    bound_runner = (
        root /
        "src/vn97/production_run_cli.py"
    )
    bound_manifest = (
        root /
        "src/vn97/production_run_manifest.py"
    )
    if (
        _sha256_file(current_runner)
        != _sha256_file(bound_runner)
        or _sha256_file(
            current_manifest_file
        )
        != _sha256_file(
            bound_manifest
        )
    ):
        raise ValueError(
            "running M19G code does not match the repository checkout bound by VN97RUN1"
        )
    return root


def _environment_ready(
    manifest: VN97ProductionRunManifest,
) -> bool:
    try:
        torch_version = (
            importlib.metadata.version(
                "torch"
            )
        )
    except (
        importlib.metadata.PackageNotFoundError,
        ValueError,
    ):
        return False
    return (
        platform.python_version()
        == manifest.python_version
        and torch_version
        == manifest.torch_version
        and platform.system()
        == manifest.platform_system
        and platform.machine()
        == manifest.platform_machine
    )


def _stage_environment(
    repository_root: Path,
) -> dict[str, str]:
    env = dict(os.environ)
    source = str(
        repository_root /
        "src"
    )
    current = env.get(
        "PYTHONPATH"
    )
    env["PYTHONPATH"] = (
        source
        if not current
        else source
        + os.pathsep
        + current
    )
    env["PYTHONHASHSEED"] = "0"
    env.setdefault(
        "CUBLAS_WORKSPACE_CONFIG",
        ":4096:8",
    )
    return env


def _run_module(
    repository_root: Path,
    module: str,
    argv: list[str],
) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            module,
            *argv,
        ],
        cwd=repository_root,
        env=_stage_environment(
            repository_root
        ),
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"{module} failed with exit code {result.returncode}"
        )


def _require_absent(
    path: Path,
    *,
    label: str,
) -> None:
    if (
        path.exists()
        or path.is_symlink()
    ):
        raise ValueError(
            f"{label} must not already exist"
        )


def _evidence_paths(
    directory: Path,
) -> tuple[Path, ...]:
    try:
        info = os.lstat(directory)
    except OSError as exc:
        raise ValueError(
            "physical-device evidence directory is unavailable"
        ) from exc
    if (
        stat.S_ISLNK(info.st_mode)
        or not stat.S_ISDIR(info.st_mode)
    ):
        raise ValueError(
            "physical-device evidence directory must be a real directory"
        )
    paths: list[Path] = []
    for entry in sorted(
        directory.iterdir(),
        key=lambda item:
            item.name,
    ):
        try:
            entry_info = os.lstat(
                entry
            )
        except OSError as exc:
            raise ValueError(
                "physical-device evidence entry became unavailable"
            ) from exc
        if (
            stat.S_ISLNK(
                entry_info.st_mode
            )
            or not stat.S_ISREG(
                entry_info.st_mode
            )
            or entry.suffix != ".json"
        ):
            raise ValueError(
                "physical-device evidence directory may contain only regular .json files"
            )
        load_device_evidence(
            entry
        )
        paths.append(
            entry.resolve(strict=True)
        )
    if not paths:
        raise ValueError(
            "physical-device evidence directory contains no VN97MOBEVID1 JSON files"
        )
    return tuple(paths)


def _language_argv(
    manifest: VN97ProductionRunManifest,
    resolved,
) -> list[str]:
    argv: list[str] = []
    for path in (
        resolved.language_training
    ):
        argv.extend(
            ["--input", str(path)]
        )
    for path in (
        resolved.language_validation
    ):
        argv.extend(
            [
                "--validation-input",
                str(path),
            ]
        )
    for path in (
        resolved.language_release
    ):
        argv.extend(
            [
                "--release-input",
                str(path),
            ]
        )
    argv.extend(
        [
            "--campaign",
            str(
                resolved
                .campaign_definition
            ),
            "--output-dir",
            str(
                resolved
                .language_output_dir
            ),
        ]
    )
    argv.extend(
        options_to_argv(
            manifest.language_options
        )
    )
    return argv


def _speech_argv(
    manifest: VN97ProductionRunManifest,
    resolved,
) -> list[str]:
    argv = [
        "--language-campaign-dir",
        str(
            resolved
            .language_output_dir
        ),
        "--speech-input",
        str(
            resolved
            .speech_training_manifest
        ),
        "--speech-validation-input",
        str(
            resolved
            .speech_validation_manifest
        ),
        "--speech-release-input",
        str(
            resolved
            .speech_release_manifest
        ),
        "--output-dir",
        str(
            resolved
            .production_output_dir
        ),
    ]
    argv.extend(
        options_to_argv(
            manifest.speech_options
        )
    )
    return argv


def _intake_argv(
    manifest: VN97ProductionRunManifest,
    resolved,
    evidence_paths: tuple[
        Path,
        ...
    ],
) -> list[str]:
    argv = [
        "--language-campaign-dir",
        str(
            resolved
            .language_output_dir
        ),
        "--production-campaign-dir",
        str(
            resolved
            .production_output_dir
        ),
        "--output-dir",
        str(
            resolved
            .intake_output_dir
        ),
    ]
    for path in evidence_paths:
        argv.extend(
            [
                "--device-evidence",
                str(path),
            ]
        )
    argv.extend(
        options_to_argv(
            manifest.intake_options
        )
    )
    return argv


def _canonical_receipt(
    *,
    manifest: VN97ProductionRunManifest,
    stage: str,
    repository_commit: str,
    device_evidence_ready: bool,
    environment_ready: bool,
    language_report_sha256: str | None,
    production_report_sha256: str | None,
    intake_report_sha256: str | None,
    release_candidate_manifest_sha256: str | None,
) -> bytes:
    return VN97ProductionRunReceipt(
        manifest_sha256=
            manifest.manifest_sha256,
        repository_commit=
            repository_commit,
        stage=stage,
        environment_ready=
            environment_ready,
        device_evidence_ready=
            device_evidence_ready,
        language_campaign_report_sha256=
            language_report_sha256,
        production_campaign_report_sha256=
            production_report_sha256,
        intake_report_sha256=
            intake_report_sha256,
        release_candidate_manifest_sha256=
            release_candidate_manifest_sha256,
    ).to_bytes()


def _write_receipt(
    path: Path,
    data: bytes,
) -> None:
    if (
        path.exists()
        or path.is_symlink()
    ):
        raise ValueError(
            "run receipt output must not already exist"
        )
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    if path.parent.is_symlink():
        raise ValueError(
            "run receipt parent must not be a symlink"
        )
    fd, temporary = tempfile.mkstemp(
        prefix=".vn97-run-receipt-",
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
    finally:
        if os.path.exists(
            temporary
        ):
            os.unlink(
                temporary
            )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify or execute one canonical VN97RUN1 production campaign."
        )
    )
    parser.add_argument(
        "--manifest",
        required=True,
    )
    parser.add_argument(
        "--workspace-root",
        help=(
            "workspace containing all VN97RUN1 relative paths; "
            "defaults to the manifest parent"
        ),
    )
    parser.add_argument(
        "--repository-root",
        default=".",
    )
    parser.add_argument(
        "--stage",
        choices=(
            "verify",
            "language",
            "production",
            "train",
            "intake",
            "all",
        ),
        default="verify",
    )
    parser.add_argument(
        "--receipt",
        help=(
            "optional new path for deterministic VN97RUNEXEC1 sidecar"
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
    manifest = (
        load_production_run_manifest(
            manifest_path
        )
    )
    workspace_root = (
        Path(args.workspace_root)
        if args.workspace_root
        is not None
        else manifest_path.parent
    )
    resolved = (
        verify_production_run_inputs(
            manifest,
            workspace_root=
                workspace_root,
        )
    )
    repository_root = (
        _verify_repository(
            Path(
                args.repository_root
            ),
            expected_commit=
                manifest
                .repository_commit,
        )
    )
    environment_ready = (
        _environment_ready(
            manifest
        )
    )

    evidence_ready = False
    try:
        _evidence_paths(
            resolved
            .device_evidence_dir
        )
        evidence_ready = True
    except ValueError:
        evidence_ready = False

    if args.stage == "verify":
        receipt = (
            _canonical_receipt(
                manifest=manifest,
                stage="verify",
                repository_commit=
                    manifest
                    .repository_commit,
                device_evidence_ready=
                    evidence_ready,
                environment_ready=
                    environment_ready,
                language_report_sha256=None,
                production_report_sha256=None,
                intake_report_sha256=None,
                release_candidate_manifest_sha256=None,
            )
        )
        if args.receipt:
            _write_receipt(
                Path(args.receipt),
                receipt,
            )
        print(
            receipt.decode("utf-8")
        )
        return 0

    if args.stage != "verify" and not environment_ready:
        raise ValueError(
            "VN97RUN1 Python/Torch/platform environment does not match the current execution environment"
        )
    if args.stage == "all" and not evidence_ready:
        raise ValueError(
            "stage all requires physical-device evidence to be present before training starts"
        )

    if args.stage in {
        "language",
        "train",
        "all",
    }:
        _require_absent(
            resolved
            .language_output_dir,
            label=
                "language campaign output",
        )
        _require_absent(
            resolved
            .production_output_dir,
            label=
                "production campaign output",
        )
        _require_absent(
            resolved
            .intake_output_dir,
            label=
                "production intake output",
        )
        _run_module(
            repository_root,
            "vn97.campaign_cli",
            _language_argv(
                manifest,
                resolved,
            ),
        )

    language = (
        inspect_language_campaign_directory(
            resolved
            .language_output_dir
        )
    )

    if args.stage in {
        "production",
        "train",
        "all",
    }:
        if args.stage == "production":
            _require_absent(
                resolved
                .production_output_dir,
                label=
                    "production campaign output",
            )
            _require_absent(
                resolved
                .intake_output_dir,
                label=
                    "production intake output",
            )
        _run_module(
            repository_root,
            "vn97.production_campaign_cli",
            _speech_argv(
                manifest,
                resolved,
            ),
        )

    production = None
    if args.stage in {
        "production",
        "train",
        "intake",
        "all",
    }:
        production = (
            inspect_production_campaign_directory(
                resolved
                .production_output_dir,
                language=language,
            )
        )

    intake_report_sha = None
    release_candidate_sha = None
    if args.stage in {
        "intake",
        "all",
    }:
        _require_absent(
            resolved
            .intake_output_dir,
            label=
                "production intake output",
        )
        evidence = _evidence_paths(
            resolved
            .device_evidence_dir
        )
        _run_module(
            repository_root,
            "vn97.production_intake_cli",
            _intake_argv(
                manifest,
                resolved,
                evidence,
            ),
        )
        intake_path = (
            resolved
            .intake_output_dir
            /
            "production-intake.vn97intake1"
        )
        intake_bytes = (
            intake_path.read_bytes()
        )
        intake = (
            parse_production_intake_report(
                intake_bytes
            )
        )
        candidate = (
            load_release_candidate_directory(
                resolved
                .intake_output_dir
                /
                "release-candidate"
            )
        )
        if (
            intake
            .release_candidate_manifest_sha256
            != candidate
            .manifest_sha256
        ):
            raise RuntimeError(
                "VN97RUN1 intake receipt/candidate identity mismatch"
            )
        intake_report_sha = (
            hashlib.sha256(
                intake_bytes
            ).hexdigest()
        )
        release_candidate_sha = (
            candidate.manifest_sha256
        )

    receipt = _canonical_receipt(
        manifest=manifest,
        stage=args.stage,
        repository_commit=
            manifest.repository_commit,
        device_evidence_ready=
            evidence_ready,
        environment_ready=
            environment_ready,
        language_report_sha256=
            language.report_sha256,
        production_report_sha256=(
            None
            if production is None
            else production
                .report_sha256
        ),
        intake_report_sha256=
            intake_report_sha,
        release_candidate_manifest_sha256=
            release_candidate_sha,
    )
    if args.receipt:
        _write_receipt(
            Path(args.receipt),
            receipt,
        )
    print(
        receipt.decode("utf-8")
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(
            f"vn97-production-run: {exc}",
            file=sys.stderr,
        )
        raise
