from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import tempfile
import time
from typing import Callable, Sequence

from .device_evidence import (
    VN97DeviceEvidence,
    VN97DeviceEvidenceCriteria,
    load_device_evidence,
    parse_device_evidence,
    require_device_evidence,
)
from .production_intake import (
    inspect_language_campaign_directory,
    inspect_production_campaign_directory,
)
from .production_run_manifest import (
    VN97ProductionRunManifest,
    load_production_run_manifest,
    options_to_argv,
    verify_production_run_inputs,
)


ANDROID_PACKAGE = "ai.vn97.app"
ANDROID_ACTIVITY = "ai.vn97.app/.VN97MainActivity"
ANDROID_ACTION = (
    "ai.vn97.app.action.M19J_COLLECT_MOBILE_EVIDENCE"
)
EXTRA_WARMUP_RUNS = (
    "ai.vn97.app.extra.M19J_WARMUP_RUNS"
)
EXTRA_MEASURED_RUNS = (
    "ai.vn97.app.extra.M19J_MEASURED_RUNS"
)
EXTRA_DECODE_TOKENS = (
    "ai.vn97.app.extra.M19J_DECODE_TOKENS"
)
EXTRA_SPEECH_FRAMES = (
    "ai.vn97.app.extra.M19J_SPEECH_FRAMES"
)
PRIVATE_STAGE = "files/m19j-evidence"
EVIDENCE_FILE = (
    PRIVATE_STAGE +
    "/vn97-mobile-evidence.json"
)
ERROR_FILE = (
    PRIVATE_STAGE +
    "/vn97-mobile-evidence.error.txt"
)


class VN97PhysicalEvidenceCampaignError(
    RuntimeError
):
    pass


@dataclass(frozen=True)
class VN97PhysicalEvidenceConfig:
    warmup_runs: int = 1
    measured_runs: int = 5
    decode_tokens: int = 16
    speech_frames: int = 8
    timeout_seconds: int = 300

    def __post_init__(self) -> None:
        if not 0 <= self.warmup_runs <= 10:
            raise ValueError(
                "warmup_runs must be in [0, 10]"
            )
        if not 3 <= self.measured_runs <= 100:
            raise ValueError(
                "measured_runs must be in [3, 100]"
            )
        if not 1 <= self.decode_tokens <= 256:
            raise ValueError(
                "decode_tokens must be in [1, 256]"
            )
        if not 1 <= self.speech_frames <= 256:
            raise ValueError(
                "speech_frames must be in [1, 256]"
            )
        if not 30 <= self.timeout_seconds <= 3600:
            raise ValueError(
                "timeout_seconds must be in [30, 3600]"
            )


@dataclass(frozen=True)
class VN97EvidenceAssets:
    capability_package: bytes
    signature_envelope: bytes
    publisher_public_key: bytes
    model_image_sha256: str

    def __post_init__(self) -> None:
        if not self.capability_package:
            raise ValueError(
                "capability_package must not be empty"
            )
        if not self.signature_envelope:
            raise ValueError(
                "signature_envelope must not be empty"
            )
        if len(self.publisher_public_key) != 32:
            raise ValueError(
                "publisher_public_key must be 32 bytes"
            )
        _require_sha256(
            self.model_image_sha256,
            label="model image SHA-256",
        )


@dataclass(frozen=True)
class VN97PhysicalEvidencePolicy:
    min_distinct_device_profiles: int
    criteria: VN97DeviceEvidenceCriteria

    def __post_init__(self) -> None:
        if (
            type(
                self.min_distinct_device_profiles
            )
            is not int
            or self.min_distinct_device_profiles <= 0
        ):
            raise ValueError(
                "min_distinct_device_profiles must be positive"
            )


@dataclass(frozen=True)
class VN97CollectedEvidence:
    canonical_bytes: bytes
    evidence: VN97DeviceEvidence

    def __post_init__(self) -> None:
        if not self.canonical_bytes:
            raise ValueError(
                "canonical evidence bytes must not be empty"
            )
        if (
            hashlib.sha256(
                self.canonical_bytes
            ).hexdigest()
            != self.evidence.evidence_sha256
        ):
            raise ValueError(
                "evidence bytes/hash mismatch"
            )


CommandExecutor = Callable[
    [list[str], bytes | None, float | None],
    subprocess.CompletedProcess[bytes],
]


def _default_executor(
    command: list[str],
    input_bytes: bytes | None,
    timeout: float | None,
) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        command,
        input=input_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=timeout,
    )


class VN97AdbClient:
    def __init__(
        self,
        executable: str = "adb",
        *,
        executor: CommandExecutor | None = None,
    ) -> None:
        if not executable:
            raise ValueError(
                "adb executable must not be empty"
            )
        self.executable = executable
        self._executor = (
            executor or _default_executor
        )

    def run(
        self,
        serial: str,
        args: Sequence[str],
        *,
        input_bytes: bytes | None = None,
        timeout: float | None = 60.0,
        check: bool = True,
    ) -> subprocess.CompletedProcess[bytes]:
        _validate_serial(serial)
        command = [
            self.executable,
            "-s",
            serial,
            *args,
        ]
        try:
            result = self._executor(
                command,
                input_bytes,
                timeout,
            )
        except (
            OSError,
            subprocess.TimeoutExpired,
        ) as exc:
            raise VN97PhysicalEvidenceCampaignError(
                "ADB command could not complete"
            ) from exc
        if check and result.returncode != 0:
            detail = (
                result.stderr
                .decode(
                    "utf-8",
                    errors="replace",
                )
                .strip()
                .replace("\n", " ")
            )[:512]
            raise VN97PhysicalEvidenceCampaignError(
                "ADB command failed"
                + (
                    f": {detail}"
                    if detail
                    else ""
                )
            )
        return result

    def text(
        self,
        serial: str,
        args: Sequence[str],
        *,
        timeout: float | None = 60.0,
        check: bool = True,
    ) -> str:
        result = self.run(
            serial,
            args,
            timeout=timeout,
            check=check,
        )
        return result.stdout.decode(
            "utf-8",
            errors="strict",
        ).strip()

    def app_installed(
        self,
        serial: str,
    ) -> bool:
        output = self.text(
            serial,
            [
                "shell",
                "pm",
                "path",
                ANDROID_PACKAGE,
            ],
            check=False,
        )
        return any(
            line.startswith("package:")
            for line in output.splitlines()
        )

    def write_private_file(
        self,
        serial: str,
        relative_path: str,
        data: bytes,
    ) -> None:
        if (
            not relative_path
            or relative_path.startswith("/")
            or ".." in Path(
                relative_path
            ).parts
        ):
            raise ValueError(
                "private ADB path is unsafe"
            )
        self.run(
            serial,
            [
                "shell",
                "run-as",
                ANDROID_PACKAGE,
                "mkdir",
                "-p",
                PRIVATE_STAGE,
            ],
        )
        self.run(
            serial,
            [
                "shell",
                "run-as",
                ANDROID_PACKAGE,
                "sh",
                "-c",
                "cat > " +
                    relative_path,
            ],
            input_bytes=data,
            timeout=120.0,
        )

    def read_private_file_or_none(
        self,
        serial: str,
        relative_path: str,
    ) -> bytes | None:
        result = self.run(
            serial,
            [
                "exec-out",
                "run-as",
                ANDROID_PACKAGE,
                "cat",
                relative_path,
            ],
            check=False,
        )
        if result.returncode != 0:
            return None
        return bytes(result.stdout)


def _validate_serial(
    serial: str,
) -> None:
    if (
        not isinstance(serial, str)
        or not serial
        or len(serial) > 256
        or any(
            ch.isspace()
            or ord(ch) < 0x21
            for ch in serial
        )
    ):
        raise ValueError(
            "ADB serial is invalid"
        )


def _require_sha256(
    value: str,
    *,
    label: str,
) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(
            ch not in
            "0123456789abcdef"
            for ch in value
        )
    ):
        raise ValueError(
            f"{label} must be lowercase SHA-256"
        )
    return value


def _regular_file(
    path: Path,
    *,
    label: str,
    max_bytes: int,
) -> Path:
    try:
        info = os.lstat(path)
    except OSError as exc:
        raise VN97PhysicalEvidenceCampaignError(
            f"{label} is unavailable"
        ) from exc
    if (
        stat.S_ISLNK(info.st_mode)
        or not stat.S_ISREG(info.st_mode)
        or not 0 < info.st_size <= max_bytes
    ):
        raise VN97PhysicalEvidenceCampaignError(
            f"{label} must be a bounded regular non-symlink file"
        )
    return path.resolve(strict=True)


def _read_regular_bytes(
    path: Path,
    *,
    label: str,
    max_bytes: int,
) -> bytes:
    target = _regular_file(
        path,
        label=label,
        max_bytes=max_bytes,
    )
    flags = os.O_RDONLY | getattr(
        os,
        "O_NOFOLLOW",
        0,
    )
    fd = os.open(target, flags)
    try:
        info = os.fstat(fd)
        output = bytearray()
        while len(output) < info.st_size:
            chunk = os.read(
                fd,
                min(
                    1024 * 1024,
                    info.st_size - len(output),
                ),
            )
            if not chunk:
                break
            output.extend(chunk)
        after = os.fstat(fd)
        if (
            len(output) != info.st_size
            or after.st_size != info.st_size
            or after.st_dev != info.st_dev
            or after.st_ino != info.st_ino
        ):
            raise VN97PhysicalEvidenceCampaignError(
                f"{label} changed while being read"
            )
        return bytes(output)
    finally:
        os.close(fd)


def _sha256_file(
    path: Path,
) -> str:
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


def _verify_repository(
    repository_root: Path,
    *,
    expected_commit: str,
) -> Path:
    try:
        info = os.lstat(
            repository_root
        )
    except OSError as exc:
        raise VN97PhysicalEvidenceCampaignError(
            "VN97 repository root is unavailable"
        ) from exc
    if (
        stat.S_ISLNK(info.st_mode)
        or not stat.S_ISDIR(info.st_mode)
    ):
        raise VN97PhysicalEvidenceCampaignError(
            "VN97 repository root must be a real directory"
        )
    root = repository_root.resolve(
        strict=True
    )
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
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
        raise VN97PhysicalEvidenceCampaignError(
            "VN97 repository Git identity could not be verified"
        ) from exc
    if commit != expected_commit:
        raise VN97PhysicalEvidenceCampaignError(
            "VN97RUN1 repository_commit does not match M19J checkout HEAD"
        )
    if dirty:
        raise VN97PhysicalEvidenceCampaignError(
            "M19J requires a clean tracked Git worktree"
        )

    current_root = Path(
        __file__
    ).resolve().parent
    bindings = (
        "device_evidence_campaign.py",
        "device_evidence.py",
        "production_intake.py",
        "production_run_manifest.py",
        "bootstrap_bundle.py",
    )
    for name in bindings:
        current = (
            current_root /
            name
        )
        bound = (
            root /
            "src/vn97" /
            name
        )
        if (
            not current.is_file()
            or not bound.is_file()
            or _sha256_file(current)
            != _sha256_file(bound)
        ):
            raise VN97PhysicalEvidenceCampaignError(
                "running M19J source does not match the VN97RUN1-bound checkout"
            )
    activity = (
        root /
        "android/app/src/main/java/ai/vn97/app/VN97MainActivity.kt"
    )
    if not activity.is_file():
        raise VN97PhysicalEvidenceCampaignError(
            "M19J Android evidence action source is missing"
        )
    return root


def _build_bound_debug_apk(
    repository_root: Path,
    *,
    gradle_executable: str,
) -> Path:
    if not gradle_executable:
        raise ValueError(
            "Gradle executable must not be empty"
        )
    try:
        result = subprocess.run(
            [
                gradle_executable,
                "-p",
                "android",
                "--no-daemon",
                "--stacktrace",
                ":app:assembleDebug",
            ],
            cwd=repository_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=45 * 60,
        )
    except (
        OSError,
        subprocess.TimeoutExpired,
    ) as exc:
        raise VN97PhysicalEvidenceCampaignError(
            "bound M19J debug APK build could not complete"
        ) from exc
    if result.returncode != 0:
        detail = result.stderr.decode(
            "utf-8",
            errors="replace",
        )[-4000:]
        raise VN97PhysicalEvidenceCampaignError(
            "bound M19J debug APK build failed: "
            + detail.replace("\n", " ")
        )
    apk = (
        repository_root /
        "android/app/build/outputs/apk/debug/app-debug.apk"
    )
    return _regular_file(
        apk,
        label="bound M19J debug APK",
        max_bytes=2 * 1024 * 1024 * 1024,
    )


def _deployment_tiles(
    production_report: Path,
) -> tuple[int, int]:
    data = _read_regular_bytes(
        production_report,
        label="VN97PRODCAMP1 report",
        max_bytes=4 * 1024 * 1024,
    )
    try:
        value = json.loads(
            data.decode(
                "utf-8",
                errors="strict",
            )
        )
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
    ) as exc:
        raise VN97PhysicalEvidenceCampaignError(
            "VN97PRODCAMP1 report could not be parsed"
        ) from exc
    deployment = (
        value.get("deployment")
        if isinstance(value, dict)
        else None
    )
    if (
        not isinstance(deployment, dict)
        or set(deployment)
        != {"tile_cols", "tile_rows"}
        or type(
            deployment["tile_rows"]
        ) is not int
        or type(
            deployment["tile_cols"]
        ) is not int
        or not 0
        < deployment["tile_rows"]
        <= 256
        or not 0
        < deployment["tile_cols"]
        <= 256
    ):
        raise VN97PhysicalEvidenceCampaignError(
            "VN97PRODCAMP1 deployment geometry is invalid"
        )
    return (
        deployment["tile_rows"],
        deployment["tile_cols"],
    )


def _production_policy(
    manifest: VN97ProductionRunManifest,
) -> VN97PhysicalEvidencePolicy:
    from .production_intake_cli import (
        _parser as intake_parser,
    )

    argv = [
        "--language-campaign-dir",
        "language",
        "--production-campaign-dir",
        "production",
        "--device-evidence",
        "evidence.json",
        "--output-dir",
        "intake",
        *options_to_argv(
            manifest.intake_options
        ),
    ]
    args = intake_parser().parse_args(argv)
    criteria = VN97DeviceEvidenceCriteria(
        min_runs=args.min_device_runs,
        max_text_prefill_p95_ms=
            args.max_text_prefill_p95_ms,
        max_text_decode_p95_ms_per_token=
            args
            .max_text_decode_p95_ms_per_token,
        max_peak_pss_kib=
            args.max_device_peak_pss_kib,
        max_thermal_status=
            args.max_device_thermal_status,
        max_speech_prefill_p95_ms=
            args.max_speech_prefill_p95_ms,
        require_energy_counter=
            args.require_energy_counter,
        max_abs_battery_energy_counter_delta_nwh=
            args
            .max_abs_battery_energy_counter_delta_nwh,
    )
    return VN97PhysicalEvidencePolicy(
        min_distinct_device_profiles=
            args
            .min_distinct_device_profiles,
        criteria=criteria,
    )


def _ephemeral_assets(
    *,
    production,
    tile_rows: int,
    tile_cols: int,
) -> tuple[
    VN97EvidenceAssets,
    bool,
]:
    # Heavy Torch/cryptography imports stay out of module import and host-only
    # structural tests. The actual campaign environment is already M19H-bound.
    try:
        from cryptography.hazmat.primitives import (
            serialization,
        )
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PrivateKey,
        )
    except ImportError as exc:
        raise VN97PhysicalEvidenceCampaignError(
            "M19J evidence harness requires the cryptography package"
        ) from exc

    from .bootstrap_bundle import (
        Ed25519PrivateKeySigner,
        build_bootstrap_bundle,
    )
    from .capability_package import (
        CapabilitySource,
    )
    from .deployment_checkpoint import (
        load_deployment_checkpoint_file,
    )
    from .tokenizer import (
        VN97TokenizerPackage,
    )

    loaded = load_deployment_checkpoint_file(
        production.checkpoint_path
    )
    tokenizer_bytes = _read_regular_bytes(
        production.tokenizer_path,
        label="production tokenizer",
        max_bytes=64 * 1024 * 1024,
    )
    tokenizer = (
        VN97TokenizerPackage.from_bytes(
            tokenizer_bytes
        )
    )
    if (
        tokenizer.vocab_size
        != loaded.config.vocab_size
    ):
        raise VN97PhysicalEvidenceCampaignError(
            "production tokenizer/checkpoint vocabulary mismatch"
        )

    private_key = (
        Ed25519PrivateKey.generate()
    )
    raw_private = private_key.private_bytes(
        encoding=
            serialization.Encoding.Raw,
        format=
            serialization.PrivateFormat.Raw,
        encryption_algorithm=
            serialization.NoEncryption(),
    )
    signer = Ed25519PrivateKeySigner(
        "m19j-evidence",
        raw_private,
    )
    bundle = build_bootstrap_bundle(
        loaded.model,
        tokenizer=tokenizer,
        audio_adapter=
            loaded.audio_adapter,
        vision_adapter=
            loaded.vision_adapter,
        source=CapabilitySource(
            origin=
                "vn97:m19j:physical-evidence",
            source_sha256=
                production.report_sha256,
            license="evidence-only",
        ),
        capability_version=1,
        signer=signer,
        tile_rows=tile_rows,
        tile_cols=tile_cols,
    )
    if (
        bundle.model_image_sha256
        != production.model_image_sha256
    ):
        raise VN97PhysicalEvidenceCampaignError(
            "M19J evidence bundle VN97MI1 identity does not match VN97PRODCAMP1"
        )
    return (
        VN97EvidenceAssets(
            capability_package=
                bundle.capability_package,
            signature_envelope=
                bundle.signature_envelope,
            publisher_public_key=
                bundle.publisher_public_key,
            model_image_sha256=
                bundle.model_image_sha256,
        ),
        loaded.audio_adapter is not None,
    )


def _ensure_physical_device(
    adb: VN97AdbClient,
    serial: str,
) -> None:
    state = adb.text(
        serial,
        ["get-state"],
    )
    if state != "device":
        raise VN97PhysicalEvidenceCampaignError(
            "ADB target is not in device state"
        )
    for prop in (
        "ro.kernel.qemu",
        "ro.boot.qemu",
    ):
        value = adb.text(
            serial,
            [
                "shell",
                "getprop",
                prop,
            ],
            check=False,
        )
        if value == "1":
            raise VN97PhysicalEvidenceCampaignError(
                "M19J rejects Android emulator evidence"
            )


def _collect_one(
    *,
    adb: VN97AdbClient,
    serial: str,
    apk: Path,
    assets: VN97EvidenceAssets,
    config: VN97PhysicalEvidenceConfig,
    criteria: VN97DeviceEvidenceCriteria,
    speech_enabled: bool,
) -> VN97CollectedEvidence:
    _ensure_physical_device(
        adb,
        serial,
    )
    if adb.app_installed(serial):
        raise VN97PhysicalEvidenceCampaignError(
            "ai.vn97.app is already installed; M19J refuses to touch an existing user/app state"
        )

    installed = False
    try:
        install = adb.run(
            serial,
            [
                "install",
                "--no-streaming",
                str(apk),
            ],
            timeout=300.0,
        )
        if (
            b"Success"
            not in install.stdout
            and b"Success"
            not in install.stderr
        ):
            raise VN97PhysicalEvidenceCampaignError(
                "ADB install did not report Success"
            )
        installed = True

        adb.write_private_file(
            serial,
            PRIVATE_STAGE +
                "/model.vn97cap1",
            assets.capability_package,
        )
        adb.write_private_file(
            serial,
            PRIVATE_STAGE +
                "/model.vn97sig1",
            assets.signature_envelope,
        )
        adb.write_private_file(
            serial,
            PRIVATE_STAGE +
                "/publisher.ed25519",
            assets.publisher_public_key,
        )

        adb.run(
            serial,
            [
                "shell",
                "am",
                "start",
                "-W",
                "-n",
                ANDROID_ACTIVITY,
                "-a",
                ANDROID_ACTION,
                "--ei",
                EXTRA_WARMUP_RUNS,
                str(config.warmup_runs),
                "--ei",
                EXTRA_MEASURED_RUNS,
                str(config.measured_runs),
                "--ei",
                EXTRA_DECODE_TOKENS,
                str(config.decode_tokens),
                "--ei",
                EXTRA_SPEECH_FRAMES,
                str(config.speech_frames),
            ],
            timeout=60.0,
        )

        deadline = (
            time.monotonic()
            + config.timeout_seconds
        )
        while time.monotonic() < deadline:
            raw = (
                adb.read_private_file_or_none(
                    serial,
                    EVIDENCE_FILE,
                )
            )
            if raw:
                evidence = (
                    parse_device_evidence(
                        raw
                    )
                )
                if (
                    evidence.runs
                    != config.measured_runs
                ):
                    raise VN97PhysicalEvidenceCampaignError(
                        "device evidence run count does not match requested measured runs"
                    )
                require_device_evidence(
                    evidence,
                    criteria,
                    expected_model_image_sha256=
                        assets
                        .model_image_sha256,
                    speech_enabled=
                        speech_enabled,
                )
                return VN97CollectedEvidence(
                    canonical_bytes=raw,
                    evidence=evidence,
                )

            error = (
                adb.read_private_file_or_none(
                    serial,
                    ERROR_FILE,
                )
            )
            if error:
                detail = (
                    error.decode(
                        "utf-8",
                        errors="replace",
                    )
                    .strip()
                    .replace("\n", " ")
                )[:1024]
                raise VN97PhysicalEvidenceCampaignError(
                    "Android M19J evidence action failed"
                    + (
                        f": {detail}"
                        if detail
                        else ""
                    )
                )
            time.sleep(1.0)

        raise VN97PhysicalEvidenceCampaignError(
            "timed out waiting for physical-device evidence"
        )
    finally:
        if installed:
            adb.run(
                serial,
                [
                    "uninstall",
                    ANDROID_PACKAGE,
                ],
                timeout=120.0,
                check=False,
            )


def _require_empty_evidence_directory(
    path: Path,
) -> Path:
    try:
        info = os.lstat(path)
    except OSError as exc:
        raise VN97PhysicalEvidenceCampaignError(
            "VN97RUN1 device-evidence directory is unavailable"
        ) from exc
    if (
        stat.S_ISLNK(info.st_mode)
        or not stat.S_ISDIR(info.st_mode)
    ):
        raise VN97PhysicalEvidenceCampaignError(
            "device-evidence must be a real directory"
        )
    root = path.resolve(strict=True)
    if any(root.iterdir()):
        raise VN97PhysicalEvidenceCampaignError(
            "M19J requires an empty device-evidence directory"
        )
    return root


def _publish_evidence_atomic(
    evidence_dir: Path,
    records: Sequence[
        VN97CollectedEvidence
    ],
) -> None:
    if not records:
        raise ValueError(
            "at least one evidence record is required"
        )
    root = _require_empty_evidence_directory(
        evidence_dir
    )
    parent = root.parent
    staging = Path(
        tempfile.mkdtemp(
            prefix=".m19j-device-evidence-",
            dir=parent,
        )
    )
    try:
        seen: set[str] = set()
        for record in records:
            digest = (
                record
                .evidence
                .evidence_sha256
            )
            if digest in seen:
                raise VN97PhysicalEvidenceCampaignError(
                    "duplicate VN97MOBEVID1 evidence identity"
                )
            seen.add(digest)
            target = (
                staging /
                (
                    digest[:24]
                    + ".json"
                )
            )
            target.write_bytes(
                record.canonical_bytes
            )
            with target.open(
                "rb"
            ) as stream:
                os.fsync(
                    stream.fileno()
                )
            loaded = load_device_evidence(
                target
            )
            if (
                loaded.evidence_sha256
                != digest
            ):
                raise VN97PhysicalEvidenceCampaignError(
                    "staged evidence post-write identity mismatch"
                )

        directory_fd = os.open(
            staging,
            os.O_RDONLY |
            getattr(
                os,
                "O_DIRECTORY",
                0,
            ),
        )
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)

        # The bootstrap workspace owns an intentionally empty evidence slot.
        # Replace it only after every requested device passed all gates.
        os.rmdir(root)
        os.replace(
            staging,
            root,
        )
        parent_fd = os.open(
            parent,
            os.O_RDONLY |
            getattr(
                os,
                "O_DIRECTORY",
                0,
            ),
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


def collect_physical_device_evidence(
    *,
    manifest_path: Path,
    workspace_root: Path,
    repository_root: Path,
    serials: Sequence[str],
    config: VN97PhysicalEvidenceConfig | None = None,
    adb_executable: str = "adb",
    gradle_executable: str = "gradle",
    adb_client: VN97AdbClient | None = None,
    prebuilt_apk_for_test: Path | None = None,
) -> tuple[VN97CollectedEvidence, ...]:
    if not serials:
        raise ValueError(
            "at least one explicit ADB serial is required"
        )
    normalized = tuple(serials)
    for serial in normalized:
        _validate_serial(serial)
    if len(set(normalized)) != len(normalized):
        raise ValueError(
            "ADB serials must be unique"
        )

    manifest = (
        load_production_run_manifest(
            manifest_path
        )
    )
    resolved = (
        verify_production_run_inputs(
            manifest,
            workspace_root=
                workspace_root,
        )
    )
    repository = _verify_repository(
        repository_root,
        expected_commit=
            manifest.repository_commit,
    )
    if (
        resolved.intake_output_dir.exists()
        or resolved
        .intake_output_dir
        .is_symlink()
    ):
        raise VN97PhysicalEvidenceCampaignError(
            "production intake already exists; M19J evidence must be collected before intake"
        )
    _require_empty_evidence_directory(
        resolved.device_evidence_dir
    )

    language = (
        inspect_language_campaign_directory(
            resolved
            .language_output_dir
        )
    )
    production = (
        inspect_production_campaign_directory(
            resolved
            .production_output_dir,
            language=language,
        )
    )
    policy = _production_policy(
        manifest
    )
    chosen = config or (
        VN97PhysicalEvidenceConfig(
            measured_runs=max(
                5,
                policy.criteria.min_runs,
            )
        )
    )
    if (
        chosen.measured_runs
        < policy.criteria.min_runs
    ):
        raise VN97PhysicalEvidenceCampaignError(
            "requested measured runs are below VN97RUN1 intake policy"
        )

    apk = (
        _regular_file(
            prebuilt_apk_for_test,
            label="M19J test APK",
            max_bytes=
                2 * 1024 * 1024 * 1024,
        )
        if prebuilt_apk_for_test
        is not None
        else _build_bound_debug_apk(
            repository,
            gradle_executable=
                gradle_executable,
        )
    )
    tile_rows, tile_cols = (
        _deployment_tiles(
            production.report_path
        )
    )
    assets, speech_enabled = (
        _ephemeral_assets(
            production=production,
            tile_rows=tile_rows,
            tile_cols=tile_cols,
        )
    )
    if not speech_enabled:
        raise VN97PhysicalEvidenceCampaignError(
            "M19J production campaign is expected to contain speech weights"
        )

    adb = (
        adb_client
        or VN97AdbClient(
            adb_executable
        )
    )
    collected: list[
        VN97CollectedEvidence
    ] = []
    for serial in normalized:
        collected.append(
            _collect_one(
                adb=adb,
                serial=serial,
                apk=apk,
                assets=assets,
                config=chosen,
                criteria=
                    policy.criteria,
                speech_enabled=
                    speech_enabled,
            )
        )

    profiles = {
        (
            item.evidence.manufacturer,
            item.evidence.model,
            item.evidence.sdk_int,
            item.evidence.abi,
        )
        for item in collected
    }
    if (
        len(profiles)
        < policy
        .min_distinct_device_profiles
    ):
        raise VN97PhysicalEvidenceCampaignError(
            "collected devices do not satisfy VN97RUN1 distinct physical-device profile requirement"
        )

    _publish_evidence_atomic(
        resolved.device_evidence_dir,
        collected,
    )
    return tuple(collected)
