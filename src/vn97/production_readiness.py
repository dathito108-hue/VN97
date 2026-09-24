from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
from typing import Callable, Mapping, Any

from .release_candidate import (
    VN97LoadedReleaseCandidate,
    load_release_candidate_directory,
)


SCHEMA = "VN97READY1"
MAX_REPORT_BYTES = 256 * 1024
APPLICATION_ID = "ai.vn97.app"
_REQUIRED_ANDROID_ENV = (
    "VN97_RELEASE_KEYSTORE",
    "VN97_RELEASE_STORE_PASSWORD",
    "VN97_RELEASE_KEY_ALIAS",
    "VN97_RELEASE_KEY_PASSWORD",
)
_CHECK_NAMES = (
    "aapt",
    "android_keystore",
    "android_signing_environment",
    "apksigner",
    "candidate",
    "gradle",
    "language_validation",
    "output_directory",
    "publisher_private_key",
    "python_dependencies",
    "quality_thresholds",
    "release_identity",
    "release_metadata",
    "repository",
    "source_bootstrap_slot",
    "speech_validation",
    "vision_validation",
)
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_VERSION_NAME = re.compile(
    r"^1\.0\.0(?:-rc[0-9]+)?$"
)


class VN97ProductionReadinessError(RuntimeError):
    pass


@dataclass(frozen=True, order=True)
class VN97ReadinessBlocker:
    code: str
    domain: str
    message: str

    def __post_init__(self) -> None:
        for value, label in (
            (self.code, "blocker code"),
            (self.domain, "blocker domain"),
            (self.message, "blocker message"),
        ):
            if (
                not isinstance(value, str)
                or not value
                or len(value) > 512
                or any(ord(ch) < 0x20 for ch in value)
            ):
                raise VN97ProductionReadinessError(
                    f"{label} is invalid"
                )

    def canonical_object(self) -> dict[str, str]:
        return {
            "code": self.code,
            "domain": self.domain,
            "message": self.message,
        }


@dataclass(frozen=True)
class VN97ReadinessCandidate:
    manifest_sha256: str
    selected_candidate_id: str
    checkpoint_sha256: str
    tokenizer_sha256: str
    model_image_sha256: str
    device_evidence_count: int
    distinct_device_profiles: int
    speech_enabled: bool
    vision_enabled: bool

    def __post_init__(self) -> None:
        for value, label in (
            (self.manifest_sha256, "candidate manifest SHA-256"),
            (self.checkpoint_sha256, "checkpoint SHA-256"),
            (self.tokenizer_sha256, "tokenizer SHA-256"),
            (self.model_image_sha256, "model image SHA-256"),
        ):
            if (
                not isinstance(value, str)
                or _HEX64.fullmatch(value) is None
            ):
                raise VN97ProductionReadinessError(
                    f"{label} is invalid"
                )
        if (
            not isinstance(self.selected_candidate_id, str)
            or len(self.selected_candidate_id) != 16
            or any(
                ch not in "0123456789abcdef"
                for ch in self.selected_candidate_id
            )
        ):
            raise VN97ProductionReadinessError(
                "selected candidate id is invalid"
            )
        if (
            type(self.device_evidence_count) is not int
            or self.device_evidence_count <= 0
            or type(self.distinct_device_profiles) is not int
            or self.distinct_device_profiles <= 0
            or self.distinct_device_profiles
            > self.device_evidence_count
        ):
            raise VN97ProductionReadinessError(
                "candidate device evidence counters are invalid"
            )
        if (
            type(self.speech_enabled) is not bool
            or type(self.vision_enabled) is not bool
        ):
            raise VN97ProductionReadinessError(
                "candidate modality flags are invalid"
            )

    def canonical_object(self) -> dict[str, object]:
        return {
            "checkpoint_sha256": self.checkpoint_sha256,
            "device_evidence_count": self.device_evidence_count,
            "distinct_device_profiles": self.distinct_device_profiles,
            "manifest_sha256": self.manifest_sha256,
            "model_image_sha256": self.model_image_sha256,
            "selected_candidate_id": self.selected_candidate_id,
            "speech_enabled": self.speech_enabled,
            "tokenizer_sha256": self.tokenizer_sha256,
            "vision_enabled": self.vision_enabled,
        }


@dataclass(frozen=True)
class VN97ProductionReadinessReport:
    status: str
    repository_commit: str | None
    application_id: str
    version_code: int | None
    version_name: str | None
    candidate: VN97ReadinessCandidate | None
    checks: tuple[tuple[str, bool], ...]
    blockers: tuple[VN97ReadinessBlocker, ...]

    def __post_init__(self) -> None:
        if self.status not in {"READY", "BLOCKED"}:
            raise VN97ProductionReadinessError(
                "readiness status is invalid"
            )
        if (
            self.repository_commit is not None
            and (
                not isinstance(self.repository_commit, str)
                or _HEX40.fullmatch(
                    self.repository_commit
                )
                is None
            )
        ):
            raise VN97ProductionReadinessError(
                "repository commit is invalid"
            )
        if self.application_id != APPLICATION_ID:
            raise VN97ProductionReadinessError(
                "readiness application id is invalid"
            )
        if (
            self.version_code is not None
            and (
                type(self.version_code) is not int
                or self.version_code <= 0
            )
        ):
            raise VN97ProductionReadinessError(
                "readiness version code is invalid"
            )
        if (
            self.version_name is not None
            and (
                not isinstance(self.version_name, str)
                or not self.version_name
            )
        ):
            raise VN97ProductionReadinessError(
                "readiness version name is invalid"
            )
        expected_checks = tuple(
            (name, dict(self.checks).get(name))
            for name in _CHECK_NAMES
        )
        if (
            len(self.checks) != len(_CHECK_NAMES)
            or any(
                type(value) is not bool
                for _, value in self.checks
            )
            or self.checks != expected_checks
        ):
            raise VN97ProductionReadinessError(
                "readiness checks are invalid or unsorted"
            )
        if tuple(sorted(self.blockers)) != self.blockers:
            raise VN97ProductionReadinessError(
                "readiness blockers must be sorted"
            )
        ready = (
            all(value for _, value in self.checks)
            and not self.blockers
        )
        if (self.status == "READY") != ready:
            raise VN97ProductionReadinessError(
                "readiness status does not match checks/blockers"
            )

    @property
    def ready(self) -> bool:
        return self.status == "READY"

    def canonical_object(self) -> dict[str, object]:
        return {
            "application_id": self.application_id,
            "blockers": [
                item.canonical_object()
                for item in self.blockers
            ],
            "candidate": (
                None
                if self.candidate is None
                else self.candidate.canonical_object()
            ),
            "checks": {
                name: value
                for name, value in self.checks
            },
            "repository_commit": self.repository_commit,
            "schema": SCHEMA,
            "status": self.status,
            "version_code": self.version_code,
            "version_name": self.version_name,
        }

    def to_bytes(self) -> bytes:
        return json.dumps(
            self.canonical_object(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")


def _block(
    blockers: list[VN97ReadinessBlocker],
    checks: dict[str, bool],
    check: str,
    code: str,
    domain: str,
    message: str,
) -> None:
    checks[check] = False
    blockers.append(
        VN97ReadinessBlocker(
            code=code,
            domain=domain,
            message=message,
        )
    )


def _safe_regular_file(
    path: Path,
    *,
    max_bytes: int | None = None,
) -> bool:
    try:
        info = os.lstat(path)
    except OSError:
        return False
    if (
        stat.S_ISLNK(info.st_mode)
        or not stat.S_ISREG(info.st_mode)
        or info.st_size <= 0
    ):
        return False
    if (
        max_bytes is not None
        and info.st_size > max_bytes
    ):
        return False
    return True


def _safe_real_directory(path: Path) -> bool:
    try:
        info = os.lstat(path)
    except OSError:
        return False
    return (
        stat.S_ISDIR(info.st_mode)
        and not stat.S_ISLNK(info.st_mode)
    )


def _is_inside(
    path: Path,
    root: Path,
) -> bool:
    try:
        resolved = path.resolve(strict=True)
        root_resolved = root.resolve(strict=True)
    except OSError:
        return False
    return (
        resolved == root_resolved
        or root_resolved in resolved.parents
    )


def _read_publisher_private_key_shape(
    path: Path,
) -> bool:
    if not _safe_regular_file(
        path,
        max_bytes=256,
    ):
        return False
    flags = os.O_RDONLY | getattr(
        os,
        "O_NOFOLLOW",
        0,
    )
    try:
        fd = os.open(path, flags)
    except OSError:
        return False
    try:
        info = os.fstat(fd)
        out = bytearray()
        while len(out) < info.st_size:
            chunk = os.read(
                fd,
                min(
                    257 - len(out),
                    info.st_size - len(out),
                ),
            )
            if not chunk:
                break
            out.extend(chunk)
        data = bytes(out)
        after = os.fstat(fd)
    finally:
        os.close(fd)
    if (
        after.st_ino != info.st_ino
        or after.st_dev != info.st_dev
        or after.st_size != info.st_size
        or len(data) != info.st_size
    ):
        return False
    if len(data) == 32:
        return True
    if len(data) != 64:
        return False
    try:
        text = data.decode(
            "ascii",
            errors="strict",
        )
    except UnicodeDecodeError:
        return False
    return all(
        ch in "0123456789abcdef"
        for ch in text
    )


def _tool_path(
    name: str,
    *,
    explicit: str | None,
    environment: Mapping[str, str],
    which: Callable[[str], str | None],
) -> Path | None:
    if explicit:
        path = Path(explicit)
        if (
            _safe_regular_file(path)
            and os.access(path, os.X_OK)
        ):
            return path.resolve(strict=True)
        return None

    direct = which(name)
    if direct:
        path = Path(direct)
        if (
            _safe_regular_file(path)
            and os.access(path, os.X_OK)
        ):
            return path.resolve(strict=True)

    sdk_raw = (
        environment.get("ANDROID_SDK_ROOT")
        or environment.get("ANDROID_HOME")
    )
    if not sdk_raw:
        return None
    build_tools = (
        Path(sdk_raw) /
        "build-tools"
    )
    if not _safe_real_directory(build_tools):
        return None
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
    for path in candidates:
        if (
            _safe_regular_file(path)
            and os.access(path, os.X_OK)
        ):
            return path.resolve(strict=True)
    return None


def _git_probe(
    repository_root: Path,
) -> tuple[str | None, bool]:
    try:
        commit = subprocess.run(
            [
                "git",
                "rev-parse",
                "HEAD",
            ],
            cwd=repository_root,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        ).stdout.strip().lower()
        dirty = subprocess.run(
            [
                "git",
                "status",
                "--porcelain",
                "--untracked-files=no",
            ],
            cwd=repository_root,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        ).stdout
    except (
        OSError,
        subprocess.CalledProcessError,
    ):
        return None, False
    return (
        commit if _HEX40.fullmatch(commit) else None,
        dirty == "",
    )


def _parse_release_identity(
    build_gradle: Path,
) -> tuple[int | None, str | None]:
    if not _safe_regular_file(
        build_gradle,
        max_bytes=2 * 1024 * 1024,
    ):
        return None, None
    try:
        text = build_gradle.read_text(
            encoding="utf-8",
            errors="strict",
        )
    except (
        OSError,
        UnicodeDecodeError,
    ):
        return None, None
    code = re.search(
        r"(?m)^val vn97VersionCode = ([0-9]+)$",
        text,
    )
    name = re.search(
        r'(?m)^val vn97VersionName = "([^"]+)"$',
        text,
    )
    if code is None or name is None:
        return None, None
    try:
        return int(code.group(1)), name.group(1)
    except ValueError:
        return None, None


def _candidate_summary(
    loaded: VN97LoadedReleaseCandidate,
) -> VN97ReadinessCandidate:
    manifest = loaded.manifest
    profiles = {
        (
            item.manufacturer,
            item.model,
            item.sdk_int,
            item.abi,
        )
        for item in manifest.device_evidence
    }
    return VN97ReadinessCandidate(
        manifest_sha256=
            loaded.manifest_sha256,
        selected_candidate_id=
            manifest.selected_candidate_id,
        checkpoint_sha256=
            manifest.checkpoint_sha256,
        tokenizer_sha256=
            manifest.tokenizer_sha256,
        model_image_sha256=
            manifest.model_image_sha256,
        device_evidence_count=
            len(manifest.device_evidence),
        distinct_device_profiles=
            len(profiles),
        speech_enabled=
            manifest.speech_enabled,
        vision_enabled=
            manifest.vision_enabled,
    )


def evaluate_production_readiness(
    *,
    repository_root: Path,
    release_candidate_dir: Path | None,
    publisher_private_key: Path | None,
    validation_inputs: tuple[Path, ...],
    speech_validation_input: Path | None,
    vision_validation_input: Path | None,
    output_dir: Path | None,
    key_id: str | None,
    capability_version: int | None,
    source_origin: str | None,
    source_license: str | None,
    max_validation_loss: float | None,
    max_speech_validation_loss: float | None,
    max_vision_validation_loss: float | None,
    gradle: str,
    apksigner: str | None,
    aapt: str | None,
    environment: Mapping[str, str] | None = None,
    which: Callable[[str], str | None] = shutil.which,
    module_available: Callable[[str], bool] | None = None,
) -> VN97ProductionReadinessReport:
    env = (
        dict(os.environ)
        if environment is None
        else dict(environment)
    )
    if module_available is None:
        module_available = (
            lambda name:
                importlib.util.find_spec(name)
                is not None
        )

    checks = {
        name: True
        for name in _CHECK_NAMES
    }
    blockers: list[
        VN97ReadinessBlocker
    ] = []
    candidate: VN97ReadinessCandidate | None = None
    loaded: VN97LoadedReleaseCandidate | None = None
    repository_commit: str | None = None
    version_code: int | None = None
    version_name: str | None = None

    if not _safe_real_directory(
        repository_root
    ):
        _block(
            blockers,
            checks,
            "repository",
            "repository.invalid",
            "repository",
            "VN97 repository root is missing, unsafe, or not a real directory.",
        )
    else:
        expected = (
            repository_root /
            "android/app/build.gradle.kts"
        )
        if not _safe_regular_file(expected):
            _block(
                blockers,
                checks,
                "repository",
                "repository.not_vn97",
                "repository",
                "Repository root does not contain the canonical VN97 Android application.",
            )
        repository_commit, clean = (
            _git_probe(repository_root)
        )
        if (
            repository_commit is None
            or not clean
        ):
            _block(
                blockers,
                checks,
                "repository",
                "repository.git_not_clean",
                "repository",
                "Production release requires a clean Git worktree with a resolvable HEAD commit.",
            )

        version_code, version_name = (
            _parse_release_identity(expected)
        )
        if (
            version_code is None
            or version_code < 190100
            or version_name is None
            or _VERSION_NAME.fullmatch(
                version_name
            )
            is None
        ):
            _block(
                blockers,
                checks,
                "release_identity",
                "release_identity.invalid",
                "android",
                "Android application/version identity is outside the M19 production contract.",
            )

        slot = (
            repository_root /
            "android/app/src/main/assets/vn97-bootstrap"
        )
        if not _safe_real_directory(slot):
            _block(
                blockers,
                checks,
                "source_bootstrap_slot",
                "source_bootstrap_slot.invalid",
                "android",
                "Source bootstrap slot is missing or unsafe.",
            )
        else:
            names = {
                entry.name
                for entry in slot.iterdir()
            }
            if names != {"README.txt"}:
                _block(
                    blockers,
                    checks,
                    "source_bootstrap_slot",
                    "source_bootstrap_slot.not_empty",
                    "android",
                    "Source bootstrap slot must contain only README.txt; production assets must stay externally staged.",
                )

    if release_candidate_dir is None:
        _block(
            blockers,
            checks,
            "candidate",
            "candidate.missing",
            "intelligence",
            "A complete M19B VN97RC1 release candidate directory is required.",
        )
    else:
        try:
            loaded = (
                load_release_candidate_directory(
                    release_candidate_dir
                )
            )
            candidate = _candidate_summary(
                loaded
            )
        except Exception:
            _block(
                blockers,
                checks,
                "candidate",
                "candidate.invalid",
                "intelligence",
                "VN97RC1 release candidate failed canonical directory/identity verification.",
            )

    metadata_values = (
        key_id,
        source_origin,
        source_license,
    )
    if (
        any(
            not isinstance(value, str)
            or not value
            or len(value) > 256
            or any(ord(ch) < 0x20 for ch in value)
            for value in metadata_values
        )
        or type(capability_version) is not int
        or not 1 <= capability_version <= 0xffffffff
    ):
        _block(
            blockers,
            checks,
            "release_metadata",
            "release_metadata.invalid",
            "release",
            "Publisher key id, capability version, source origin and source license must be valid production metadata.",
        )

    language_loss_valid = (
        isinstance(max_validation_loss, (int, float))
        and not isinstance(max_validation_loss, bool)
        and math.isfinite(float(max_validation_loss))
        and float(max_validation_loss) > 0.0
    )
    speech_loss_valid = (
        max_speech_validation_loss is None
        or (
            isinstance(
                max_speech_validation_loss,
                (int, float),
            )
            and not isinstance(
                max_speech_validation_loss,
                bool,
            )
            and math.isfinite(
                float(
                    max_speech_validation_loss
                )
            )
            and float(
                max_speech_validation_loss
            ) > 0.0
        )
    )
    vision_loss_valid = (
        max_vision_validation_loss is None
        or (
            isinstance(
                max_vision_validation_loss,
                (int, float),
            )
            and not isinstance(
                max_vision_validation_loss,
                bool,
            )
            and math.isfinite(
                float(
                    max_vision_validation_loss
                )
            )
            and float(
                max_vision_validation_loss
            ) > 0.0
        )
    )
    if (
        not language_loss_valid
        or (
            loaded is not None
            and loaded.manifest.speech_enabled
            and not (
                speech_loss_valid
                and max_speech_validation_loss
                is not None
            )
        )
        or (
            loaded is not None
            and loaded.manifest.vision_enabled
            and not (
                vision_loss_valid
                and max_vision_validation_loss
                is not None
            )
        )
        or not speech_loss_valid
        or not vision_loss_valid
    ):
        _block(
            blockers,
            checks,
            "quality_thresholds",
            "quality_thresholds.invalid",
            "quality",
            "Fresh language and enabled-modality maximum validation-loss thresholds must be finite positive values.",
        )

    if publisher_private_key is None:
        _block(
            blockers,
            checks,
            "publisher_private_key",
            "publisher_private_key.missing",
            "signing",
            "M10N Ed25519 publisher private key is required.",
        )
    else:
        if not _read_publisher_private_key_shape(
            publisher_private_key
        ):
            _block(
                blockers,
                checks,
                "publisher_private_key",
                "publisher_private_key.invalid",
                "signing",
                "Publisher private key must be a safe 32-byte raw Ed25519 seed or 64 lowercase hex characters.",
            )
        elif (
            _safe_real_directory(repository_root)
            and _is_inside(
                publisher_private_key,
                repository_root,
            )
        ):
            _block(
                blockers,
                checks,
                "publisher_private_key",
                "publisher_private_key.inside_repository",
                "signing",
                "Publisher private key must stay outside the repository.",
            )

    if not validation_inputs:
        _block(
            blockers,
            checks,
            "language_validation",
            "language_validation.missing",
            "quality",
            "At least one fresh held-out language validation input is required.",
        )
    elif any(
        not _safe_regular_file(path)
        for path in validation_inputs
    ):
        _block(
            blockers,
            checks,
            "language_validation",
            "language_validation.invalid",
            "quality",
            "Every held-out language validation input must be a safe non-empty regular file.",
        )

    if loaded is not None:
        if loaded.manifest.speech_enabled:
            if (
                speech_validation_input is None
                or not _safe_regular_file(
                    speech_validation_input
                )
            ):
                _block(
                    blockers,
                    checks,
                    "speech_validation",
                    "speech_validation.missing",
                    "quality",
                    "Speech-enabled VN97RC1 requires a fresh held-out speech validation input.",
                )
        elif speech_validation_input is not None:
            _block(
                blockers,
                checks,
                "speech_validation",
                "speech_validation.unexpected",
                "quality",
                "Speech validation input is not allowed for a VN97RC1 without speech weights.",
            )

        if loaded.manifest.vision_enabled:
            if (
                vision_validation_input is None
                or not _safe_regular_file(
                    vision_validation_input
                )
            ):
                _block(
                    blockers,
                    checks,
                    "vision_validation",
                    "vision_validation.missing",
                    "quality",
                    "Vision-enabled VN97RC1 requires a fresh held-out vision validation input.",
                )
        elif vision_validation_input is not None:
            _block(
                blockers,
                checks,
                "vision_validation",
                "vision_validation.unexpected",
                "quality",
                "Vision validation input is not allowed for a VN97RC1 without vision weights.",
            )

    missing_modules = [
        name
        for name in ("torch", "cryptography")
        if not module_available(name)
    ]
    if missing_modules:
        _block(
            blockers,
            checks,
            "python_dependencies",
            "python_dependencies.missing",
            "toolchain",
            "Production release requires installed torch and cryptography runtime dependencies.",
        )

    missing_env = [
        name
        for name in _REQUIRED_ANDROID_ENV
        if not env.get(name)
    ]
    if missing_env:
        _block(
            blockers,
            checks,
            "android_signing_environment",
            "android_signing_environment.missing",
            "signing",
            "Android release signing environment is incomplete.",
        )

    keystore_raw = env.get(
        "VN97_RELEASE_KEYSTORE"
    )
    if not keystore_raw:
        _block(
            blockers,
            checks,
            "android_keystore",
            "android_keystore.missing",
            "signing",
            "Android release keystore is required.",
        )
    else:
        keystore = Path(keystore_raw)
        if not _safe_regular_file(
            keystore
        ):
            _block(
                blockers,
                checks,
                "android_keystore",
                "android_keystore.invalid",
                "signing",
                "Android release keystore must be a safe non-empty regular file.",
            )
        elif (
            _safe_real_directory(repository_root)
            and _is_inside(
                keystore,
                repository_root,
            )
        ):
            _block(
                blockers,
                checks,
                "android_keystore",
                "android_keystore.inside_repository",
                "signing",
                "Android release keystore must stay outside the repository.",
            )

    gradle_path = _tool_path(
        "gradle",
        explicit=(
            gradle
            if os.path.sep in gradle
            or (
                os.path.altsep is not None
                and os.path.altsep in gradle
            )
            else None
        ),
        environment=env,
        which=which,
    )
    if gradle_path is None and not (
        os.path.sep in gradle
        or (
            os.path.altsep is not None
            and os.path.altsep in gradle
        )
    ):
        direct = which(gradle)
        if direct is not None:
            candidate_path = Path(direct)
            if (
                _safe_regular_file(
                    candidate_path
                )
                and os.access(
                    candidate_path,
                    os.X_OK,
                )
            ):
                gradle_path = (
                    candidate_path
                    .resolve(strict=True)
                )
    if gradle_path is None:
        _block(
            blockers,
            checks,
            "gradle",
            "gradle.missing",
            "toolchain",
            "Gradle executable is unavailable.",
        )

    if _tool_path(
        "apksigner",
        explicit=apksigner,
        environment=env,
        which=which,
    ) is None:
        _block(
            blockers,
            checks,
            "apksigner",
            "apksigner.missing",
            "toolchain",
            "Android apksigner executable is unavailable.",
        )

    if _tool_path(
        "aapt",
        explicit=aapt,
        environment=env,
        which=which,
    ) is None:
        _block(
            blockers,
            checks,
            "aapt",
            "aapt.missing",
            "toolchain",
            "Android aapt executable is unavailable.",
        )

    if output_dir is None:
        _block(
            blockers,
            checks,
            "output_directory",
            "output_directory.missing",
            "output",
            "Production release output directory is required.",
        )
    elif (
        output_dir.exists()
        or output_dir.is_symlink()
    ):
        _block(
            blockers,
            checks,
            "output_directory",
            "output_directory.exists",
            "output",
            "Production release output directory must not already exist.",
        )
    else:
        parent = output_dir.parent
        while (
            not parent.exists()
            and parent != parent.parent
        ):
            parent = parent.parent
        if not _safe_real_directory(parent):
            _block(
                blockers,
                checks,
                "output_directory",
                "output_directory.parent_invalid",
                "output",
                "Production release output path does not have a safe real ancestor directory.",
            )

    ordered_checks = tuple(
        (name, checks[name])
        for name in _CHECK_NAMES
    )
    ordered_blockers = tuple(
        sorted(blockers)
    )
    status = (
        "READY"
        if (
            all(
                value
                for _, value in ordered_checks
            )
            and not ordered_blockers
        )
        else "BLOCKED"
    )
    return VN97ProductionReadinessReport(
        status=status,
        repository_commit=
            repository_commit,
        application_id=APPLICATION_ID,
        version_code=version_code,
        version_name=version_name,
        candidate=candidate,
        checks=ordered_checks,
        blockers=ordered_blockers,
    )


def parse_production_readiness_report(
    data: bytes,
) -> VN97ProductionReadinessReport:
    if not 0 < len(data) <= MAX_REPORT_BYTES:
        raise VN97ProductionReadinessError(
            "VN97READY1 byte size is outside bounds"
        )
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
        raise VN97ProductionReadinessError(
            "VN97READY1 must be strict UTF-8 JSON"
        ) from exc
    if duplicates or not isinstance(root, dict):
        raise VN97ProductionReadinessError(
            "VN97READY1 must be one object without duplicate keys"
        )
    canonical = json.dumps(
        root,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    if canonical != data:
        raise VN97ProductionReadinessError(
            "VN97READY1 must use canonical JSON"
        )
    expected = {
        "application_id",
        "blockers",
        "candidate",
        "checks",
        "repository_commit",
        "schema",
        "status",
        "version_code",
        "version_name",
    }
    if set(root) != expected or root["schema"] != SCHEMA:
        raise VN97ProductionReadinessError(
            "VN97READY1 schema/keys mismatch"
        )

    raw_checks = root["checks"]
    if (
        not isinstance(raw_checks, dict)
        or tuple(sorted(raw_checks)) != _CHECK_NAMES
    ):
        raise VN97ProductionReadinessError(
            "VN97READY1 check keys mismatch"
        )
    checks = tuple(
        (name, raw_checks[name])
        for name in _CHECK_NAMES
    )

    raw_blockers = root["blockers"]
    if not isinstance(raw_blockers, list):
        raise VN97ProductionReadinessError(
            "VN97READY1 blockers must be an array"
        )
    blockers: list[VN97ReadinessBlocker] = []
    for item in raw_blockers:
        if (
            not isinstance(item, dict)
            or set(item)
            != {"code", "domain", "message"}
        ):
            raise VN97ProductionReadinessError(
                "VN97READY1 blocker keys mismatch"
            )
        blockers.append(
            VN97ReadinessBlocker(
                code=item["code"],
                domain=item["domain"],
                message=item["message"],
            )
        )

    raw_candidate = root["candidate"]
    candidate: VN97ReadinessCandidate | None
    if raw_candidate is None:
        candidate = None
    else:
        candidate_keys = {
            "checkpoint_sha256",
            "device_evidence_count",
            "distinct_device_profiles",
            "manifest_sha256",
            "model_image_sha256",
            "selected_candidate_id",
            "speech_enabled",
            "tokenizer_sha256",
            "vision_enabled",
        }
        if (
            not isinstance(raw_candidate, dict)
            or set(raw_candidate)
            != candidate_keys
        ):
            raise VN97ProductionReadinessError(
                "VN97READY1 candidate keys mismatch"
            )
        candidate = VN97ReadinessCandidate(
            manifest_sha256=
                raw_candidate["manifest_sha256"],
            selected_candidate_id=
                raw_candidate["selected_candidate_id"],
            checkpoint_sha256=
                raw_candidate["checkpoint_sha256"],
            tokenizer_sha256=
                raw_candidate["tokenizer_sha256"],
            model_image_sha256=
                raw_candidate["model_image_sha256"],
            device_evidence_count=
                raw_candidate["device_evidence_count"],
            distinct_device_profiles=
                raw_candidate["distinct_device_profiles"],
            speech_enabled=
                raw_candidate["speech_enabled"],
            vision_enabled=
                raw_candidate["vision_enabled"],
        )

    try:
        return VN97ProductionReadinessReport(
            status=root["status"],
            repository_commit=
                root["repository_commit"],
            application_id=
                root["application_id"],
            version_code=
                root["version_code"],
            version_name=
                root["version_name"],
            candidate=candidate,
            checks=checks,
            blockers=tuple(blockers),
        )
    except (
        TypeError,
        ValueError,
        VN97ProductionReadinessError,
    ) as exc:
        if isinstance(
            exc,
            VN97ProductionReadinessError,
        ):
            raise
        raise VN97ProductionReadinessError(
            str(exc)
        ) from exc
