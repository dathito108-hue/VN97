from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile

from .capability_package import parse_capability_package
from .capability_trust import (
    Ed25519Verifier,
    parse_signature_envelope,
)
from .release_attestation import (
    VN97ApkAttestation,
    VN97ApkAttestationError,
    parse_aapt_badging,
    parse_apksigner_certificate_sha256,
    sha256_bytes,
    sha256_file,
    verify_apk_payload,
)
from .release_candidate import (
    load_release_candidate_directory,
)
from .production_readiness import (
    evaluate_production_readiness,
)


_REQUIRED_ANDROID_SIGNING_ENV = (
    "VN97_RELEASE_KEYSTORE",
    "VN97_RELEASE_STORE_PASSWORD",
    "VN97_RELEASE_KEY_ALIAS",
    "VN97_RELEASE_KEY_PASSWORD",
)


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


def _regular_file(
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
        or not stat.S_ISREG(info.st_mode)
        or info.st_size <= 0
    ):
        raise ValueError(
            f"{label} must be a non-empty regular file"
        )
    return path.resolve(strict=True)


def _find_build_tool(
    name: str,
    *,
    explicit: str | None,
) -> Path:
    if explicit is not None:
        candidate = Path(explicit)
        _regular_file(
            candidate,
            label=name,
        )
        if not os.access(candidate, os.X_OK):
            raise ValueError(
                f"{name} is not executable"
            )
        return candidate.resolve(strict=True)

    direct = shutil.which(name)
    if direct:
        return Path(direct).resolve(strict=True)

    sdk_raw = (
        os.environ.get("ANDROID_SDK_ROOT")
        or os.environ.get("ANDROID_HOME")
    )
    if not sdk_raw:
        raise ValueError(
            f"{name} is unavailable and Android SDK root is unset"
        )
    build_tools = (
        Path(sdk_raw) /
        "build-tools"
    )
    if not build_tools.is_dir():
        raise ValueError(
            "Android build-tools directory is unavailable"
        )
    candidates = sorted(
        (
            entry / name
            for entry in build_tools.iterdir()
            if entry.is_dir()
            and (entry / name).is_file()
        ),
        key=lambda path: path.parent.name,
        reverse=True,
    )
    if not candidates:
        raise ValueError(
            f"{name} was not found in Android build-tools"
        )
    chosen = candidates[0]
    if not os.access(chosen, os.X_OK):
        raise ValueError(
            f"{name} is not executable"
        )
    return chosen.resolve(strict=True)


def _preflight_android_signing(
    repository_root: Path,
) -> Path:
    missing = [
        name
        for name in _REQUIRED_ANDROID_SIGNING_ENV
        if not os.environ.get(name)
    ]
    if missing:
        raise ValueError(
            "missing Android release signing environment: "
            + ", ".join(missing)
        )
    keystore = _regular_file(
        Path(
            os.environ[
                "VN97_RELEASE_KEYSTORE"
            ]
        ),
        label="Android release keystore",
    )
    if (
        keystore == repository_root
        or repository_root in keystore.parents
    ):
        raise ValueError(
            "Android release keystore must stay outside repository"
        )
    return keystore


def _preflight_source_bootstrap_slot(
    repository_root: Path,
) -> None:
    slot = (
        repository_root /
        "android/app/src/main/assets/vn97-bootstrap"
    )
    slot = _real_directory(
        slot,
        label="source bootstrap slot",
    )
    names = {
        entry.name
        for entry in slot.iterdir()
    }
    if names != {"README.txt"}:
        raise ValueError(
            "source bootstrap slot must contain only README.txt for M19D"
        )


def _run_checked(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
    label: str,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"{label} failed with exit code {result.returncode}"
        )
    return result


def _bootstrap_args(
    args: argparse.Namespace,
    *,
    assets_dir: Path,
) -> list[str]:
    values = [
        "--release-candidate-dir",
        str(Path(args.release_candidate_dir)),
        "--private-key",
        str(Path(args.private_key)),
        "--key-id",
        args.key_id,
        "--capability-version",
        str(args.capability_version),
        "--source-origin",
        args.source_origin,
        "--source-license",
        args.source_license,
        "--assets-dir",
        str(assets_dir),
        "--validation-format",
        args.validation_format,
        "--validation-sequence-length",
        str(args.validation_sequence_length),
        "--validation-batch-size",
        str(args.validation_batch_size),
        "--validation-device",
        args.validation_device,
        "--max-validation-loss",
        str(args.max_validation_loss),
        "--min-validation-accuracy",
        str(args.min_validation_accuracy),
        "--min-validation-target-tokens",
        str(args.min_validation_target_tokens),
        "--device-evidence-min-runs",
        str(args.device_evidence_min_runs),
        "--max-text-prefill-p95-ms",
        str(args.max_text_prefill_p95_ms),
        "--max-text-decode-p95-ms-per-token",
        str(args.max_text_decode_p95_ms_per_token),
        "--max-device-peak-pss-kib",
        str(args.max_device_peak_pss_kib),
        "--max-device-thermal-status",
        str(args.max_device_thermal_status),
        "--max-speech-prefill-p95-ms",
        str(args.max_speech_prefill_p95_ms),
        "--max-model-image-bytes",
        str(args.max_model_image_bytes),
        "--max-recurrent-state-bytes",
        str(args.max_recurrent_state_bytes),
    ]
    for path in (args.validation_input or []):
        values.extend(
            [
                "--validation-input",
                path,
            ]
        )
    if args.validation_stride is not None:
        values.extend(
            [
                "--validation-stride",
                str(args.validation_stride),
            ]
        )
    if args.require_device_energy_counter:
        values.append(
            "--require-device-energy-counter"
        )
    if (
        args.max_abs_battery_energy_counter_delta_nwh
        is not None
    ):
        values.extend(
            [
                "--max-abs-battery-energy-counter-delta-nwh",
                str(
                    args
                    .max_abs_battery_energy_counter_delta_nwh
                ),
            ]
        )

    if args.speech_validation_input is not None:
        values.extend(
            [
                "--speech-validation-input",
                args.speech_validation_input,
                "--speech-validation-max-examples",
                str(args.speech_validation_max_examples),
                "--speech-max-frames",
                str(args.speech_max_frames),
                "--speech-max-target-tokens",
                str(args.speech_max_target_tokens),
                "--max-speech-validation-loss",
                str(args.max_speech_validation_loss),
                "--min-speech-validation-accuracy",
                str(args.min_speech_validation_accuracy),
                "--min-speech-validation-target-tokens",
                str(args.min_speech_validation_target_tokens),
                "--min-speech-validation-examples",
                str(args.min_speech_validation_examples),
            ]
        )
    if args.vision_validation_input is not None:
        values.extend(
            [
                "--vision-validation-input",
                args.vision_validation_input,
                "--vision-validation-max-examples",
                str(args.vision_validation_max_examples),
                "--vision-max-patches",
                str(args.vision_max_patches),
                "--vision-max-target-tokens",
                str(args.vision_max_target_tokens),
                "--max-vision-validation-loss",
                str(args.max_vision_validation_loss),
                "--min-vision-validation-accuracy",
                str(args.min_vision_validation_accuracy),
                "--min-vision-validation-target-tokens",
                str(args.min_vision_validation_target_tokens),
                "--min-vision-validation-examples",
                str(args.min_vision_validation_examples),
            ]
        )
    return values


def _verify_bootstrap_assets(
    assets_dir: Path,
    *,
    candidate_manifest_sha256: str,
) -> tuple[bytes, bytes, bytes]:
    package = _regular_file(
        assets_dir /
        "model.vn97cap1",
        label="signed VN97CAP1",
    ).read_bytes()
    signature = _regular_file(
        assets_dir /
        "model.vn97sig1",
        label="VN97SIG1",
    ).read_bytes()
    publisher = _regular_file(
        assets_dir /
        "publisher.ed25519",
        label="publisher Ed25519 key",
    ).read_bytes()
    if len(publisher) != 32:
        raise ValueError(
            "publisher Ed25519 key must be exactly 32 bytes"
        )

    parsed = parse_capability_package(
        package
    )
    if (
        parsed.manifest.source.source_sha256
        != candidate_manifest_sha256
    ):
        raise ValueError(
            "signed VN97CAP1 source identity does not match VN97RC1"
        )
    envelope = parse_signature_envelope(
        signature
    )
    if (
        envelope.package_sha256
        != parsed.package_sha256
        or envelope.capability_id
        != parsed.manifest.capability_id
        or envelope.capability_version
        != parsed.manifest.capability_version
    ):
        raise ValueError(
            "VN97SIG1 claims do not match signed package"
        )
    if not Ed25519Verifier().verify(
        public_key=publisher,
        message=envelope.signing_message(),
        signature=envelope.signature,
    ):
        raise ValueError(
            "VN97SIG1 Ed25519 verification failed"
        )
    return (
        package,
        signature,
        publisher,
    )


def _publish_release(
    output_dir: Path,
    *,
    apk_path: Path,
    bootstrap_report_bytes: bytes,
    attestation_bytes: bytes,
    readiness_report_bytes: bytes,
) -> None:
    if output_dir.exists() or output_dir.is_symlink():
        raise ValueError(
            "release output directory must not already exist"
        )
    output_dir.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    staging = Path(
        tempfile.mkdtemp(
            prefix=".vn97-release-",
            dir=output_dir.parent,
        )
    )
    try:
        shutil.copyfile(
            apk_path,
            staging / "VN97-production.apk",
        )
        (
            staging /
            "bootstrap-release.vn97bootrel6.json"
        ).write_bytes(
            bootstrap_report_bytes
        )
        (
            staging /
            "release-attestation.vn97apk1"
        ).write_bytes(
            attestation_bytes
        )
        (
            staging /
            "production-readiness.vn97ready1"
        ).write_bytes(
            readiness_report_bytes
        )
        for path in staging.iterdir():
            with path.open("rb") as stream:
                os.fsync(stream.fileno())
        directory_fd = os.open(
            staging,
            os.O_RDONLY |
            getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        os.replace(
            staging,
            output_dir,
        )
        parent_fd = os.open(
            output_dir.parent,
            os.O_RDONLY |
            getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    finally:
        if staging.exists():
            shutil.rmtree(
                staging,
                ignore_errors=True,
            )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build one candidate-bound, signed, attested VN97 production APK."
        )
    )
    parser.add_argument(
        "--repository-root",
        default=".",
    )
    parser.add_argument(
        "--release-candidate-dir",
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
        "--max-validation-loss",
        type=float,
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
        "--speech-validation-input",
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
        "--max-speech-validation-loss",
        type=float,
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
        "--vision-validation-input",
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
        "--max-vision-validation-loss",
        type=float,
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
        "--device-evidence-min-runs",
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
        default=15_000.0,
    )
    parser.add_argument(
        "--require-device-energy-counter",
        action="store_true",
    )
    parser.add_argument(
        "--max-abs-battery-energy-counter-delta-nwh",
        type=int,
    )
    parser.add_argument(
        "--max-model-image-bytes",
        type=int,
        default=512 * 1024 * 1024,
    )
    parser.add_argument(
        "--max-recurrent-state-bytes",
        type=int,
        default=512 * 1024 * 1024,
    )
    parser.add_argument(
        "--output-dir",
    )
    parser.add_argument(
        "--gradle",
        default="gradle",
    )
    parser.add_argument("--apksigner")
    parser.add_argument("--aapt")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help=(
            "emit canonical VN97READY1 readiness and stop before signing/build"
        ),
    )
    parser.add_argument(
        "--readiness-report",
        help=(
            "optional path to write canonical VN97READY1 during preflight"
        ),
    )
    return parser


def main(
    argv: list[str] | None = None,
) -> int:
    args = _parser().parse_args(argv)

    if (
        args.readiness_report is not None
        and not args.preflight_only
    ):
        raise ValueError(
            "--readiness-report is only valid with --preflight-only; "
            "normal releases already publish production-readiness.vn97ready1"
        )

    repository_root_path = Path(
        args.repository_root
    )
    readiness = evaluate_production_readiness(
        repository_root=repository_root_path,
        release_candidate_dir=(
            None
            if args.release_candidate_dir is None
            else Path(args.release_candidate_dir)
        ),
        publisher_private_key=(
            None
            if args.private_key is None
            else Path(args.private_key)
        ),
        validation_inputs=tuple(
            Path(path)
            for path in (
                args.validation_input
                or []
            )
        ),
        speech_validation_input=(
            None
            if args.speech_validation_input is None
            else Path(
                args.speech_validation_input
            )
        ),
        vision_validation_input=(
            None
            if args.vision_validation_input is None
            else Path(
                args.vision_validation_input
            )
        ),
        output_dir=(
            None
            if args.output_dir is None
            else Path(args.output_dir)
        ),
        key_id=args.key_id,
        capability_version=
            args.capability_version,
        source_origin=args.source_origin,
        source_license=args.source_license,
        max_validation_loss=
            args.max_validation_loss,
        max_speech_validation_loss=
            args.max_speech_validation_loss,
        max_vision_validation_loss=
            args.max_vision_validation_loss,
        gradle=args.gradle,
        apksigner=args.apksigner,
        aapt=args.aapt,
    )
    readiness_bytes = readiness.to_bytes()

    if args.readiness_report is not None:
        report_path = Path(
            args.readiness_report
        )
        if (
            report_path.exists()
            or report_path.is_symlink()
        ):
            raise ValueError(
                "readiness report output must not already exist"
            )
        report_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        fd, temporary_report = (
            tempfile.mkstemp(
                prefix=".vn97ready-",
                dir=report_path.parent,
            )
        )
        try:
            with os.fdopen(
                fd,
                "wb",
                closefd=True,
            ) as output:
                output.write(
                    readiness_bytes
                )
                output.flush()
                os.fsync(
                    output.fileno()
                )
            os.replace(
                temporary_report,
                report_path,
            )
        finally:
            if os.path.exists(
                temporary_report
            ):
                os.unlink(
                    temporary_report
                )

    if args.preflight_only:
        print(
            readiness_bytes.decode(
                "utf-8"
            )
        )
        return (
            0
            if readiness.ready
            else 2
        )

    if not readiness.ready:
        codes = ",".join(
            blocker.code
            for blocker in
                readiness.blockers
        )
        raise RuntimeError(
            "VN97 production release is BLOCKED by VN97READY1: "
            + codes
        )

    repository_root = _real_directory(
        repository_root_path,
        label="VN97 repository root",
    )
    assert (
        args.release_candidate_dir
        is not None
    )
    assert args.private_key is not None
    assert args.key_id is not None
    assert (
        args.capability_version
        is not None
    )
    assert args.source_origin is not None
    assert args.source_license is not None
    assert args.validation_input
    assert (
        args.max_validation_loss
        is not None
    )
    assert args.output_dir is not None

    candidate = load_release_candidate_directory(
        args.release_candidate_dir
    )
    output_dir = Path(args.output_dir)

    _preflight_source_bootstrap_slot(
        repository_root
    )
    _preflight_android_signing(
        repository_root
    )
    gradle = shutil.which(args.gradle)
    if gradle is None:
        if (
            os.path.sep in args.gradle
            and os.access(
                args.gradle,
                os.X_OK,
            )
        ):
            gradle = str(
                Path(args.gradle)
                .resolve(strict=True)
            )
        else:
            raise ValueError(
                "Gradle executable is unavailable"
            )
    apksigner = _find_build_tool(
        "apksigner",
        explicit=args.apksigner,
    )
    aapt = _find_build_tool(
        "aapt",
        explicit=args.aapt,
    )

    from .bootstrap_release_cli import (
        main as bootstrap_release_main,
    )

    with tempfile.TemporaryDirectory(
        prefix="vn97-m19d-assets-"
    ) as temporary:
        asset_root = Path(temporary)
        bootstrap_dir = (
            asset_root /
            "vn97-bootstrap"
        )
        bootstrap_dir.mkdir()

        stdout = io.StringIO()
        with redirect_stdout(stdout):
            result = bootstrap_release_main(
                _bootstrap_args(
                    args,
                    assets_dir=bootstrap_dir,
                )
            )
        if result != 0:
            raise RuntimeError(
                "M10N bootstrap release failed"
            )
        raw_report = stdout.getvalue().strip()
        try:
            report = json.loads(raw_report)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "M10N did not emit valid release JSON"
            ) from exc
        canonical_report = json.dumps(
            report,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        if canonical_report != raw_report:
            raise RuntimeError(
                "M10N release report is not canonical JSON"
            )
        if report.get("schema") != "VN97BOOTREL6":
            raise RuntimeError(
                "M10N release report schema mismatch"
            )
        release_info = report.get(
            "release_candidate"
        )
        if not isinstance(
            release_info,
            dict,
        ):
            raise RuntimeError(
                "M10N release report lacks candidate binding"
            )
        if (
            release_info.get(
                "manifest_sha256"
            )
            != candidate.manifest_sha256
            or report.get(
                "signed_source_sha256"
            )
            != candidate.manifest_sha256
        ):
            raise RuntimeError(
                "M10N release report does not bind VN97RC1"
            )

        package, signature, publisher = (
            _verify_bootstrap_assets(
                bootstrap_dir,
                candidate_manifest_sha256=
                    candidate.manifest_sha256,
            )
        )
        if (
            report.get("package_sha256")
            != sha256_bytes(package)
            or report.get(
                "publisher_public_key_sha256"
            )
            != sha256_bytes(publisher)
        ):
            raise RuntimeError(
                "M10N report identities differ from signed assets"
            )

        build_env = dict(os.environ)
        build_env[
            "VN97_RELEASE_ASSET_ROOT"
        ] = str(asset_root)
        _run_checked(
            [
                gradle,
                "-p",
                "android",
                "--no-daemon",
                ":app:clean",
                ":app:assembleRelease",
            ],
            cwd=repository_root,
            env=build_env,
            label="VN97 release APK build",
        )

        apk_path = (
            repository_root /
            "android/app/build/outputs/apk/release/app-release.apk"
        )
        _regular_file(
            apk_path,
            label="signed release APK",
        )

        signer_result = _run_checked(
            [
                str(apksigner),
                "verify",
                "--verbose",
                "--print-certs",
                str(apk_path),
            ],
            cwd=repository_root,
            label="APK signature verification",
        )
        signer_certificates = (
            parse_apksigner_certificate_sha256(
                signer_result.stdout
                + "\n"
                + signer_result.stderr
            )
        )

        aapt_result = _run_checked(
            [
                str(aapt),
                "dump",
                "badging",
                str(apk_path),
            ],
            cwd=repository_root,
            label="APK package inspection",
        )
        (
            application_id,
            version_code,
            version_name,
        ) = parse_aapt_badging(
            aapt_result.stdout
        )
        if application_id != "ai.vn97.app":
            raise VN97ApkAttestationError(
                "release APK application id is not ai.vn97.app"
            )

        _, release_manifest_bytes = (
            verify_apk_payload(
                apk_path,
                expected_application_id=
                    application_id,
                expected_version_code=
                    version_code,
                expected_version_name=
                    version_name,
                expected_package=package,
                expected_signature=signature,
                expected_publisher=publisher,
            )
        )

        report_bytes = (
            canonical_report
            .encode("utf-8")
        )
        attestation = VN97ApkAttestation(
            application_id=application_id,
            version_code=version_code,
            version_name=version_name,
            apk_bytes=
                apk_path.stat().st_size,
            apk_sha256=
                sha256_file(apk_path),
            signer_certificate_sha256=
                signer_certificates,
            release_candidate_manifest_sha256=
                candidate.manifest_sha256,
            bootstrap_release_report_sha256=
                sha256_bytes(report_bytes),
            bootstrap_package_sha256=
                sha256_bytes(package),
            bootstrap_signature_sha256=
                sha256_bytes(signature),
            bootstrap_publisher_sha256=
                sha256_bytes(publisher),
            release_manifest_sha256=
                sha256_bytes(
                    release_manifest_bytes
                ),
        )
        _publish_release(
            output_dir,
            apk_path=apk_path,
            bootstrap_report_bytes=
                report_bytes,
            attestation_bytes=
                attestation.to_bytes(),
            readiness_report_bytes=
                readiness_bytes,
        )

    print(
        json.dumps(
            attestation.canonical_object(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(
            f"vn97-production-release: {exc}",
            file=sys.stderr,
        )
        raise
