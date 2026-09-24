from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import stat
import subprocess
import tempfile
from typing import Any, Mapping

from .production_run_manifest import (
    MAX_BOUND_FILE_BYTES,
    VN97ProductionRunManifest,
    VN97RunFile,
    VN97SpeechRunInput,
    verify_production_run_inputs,
)


_BOOTSTRAP_DIRS = (
    "config",
    "data/train",
    "data/validation",
    "data/release",
    "speech/train",
    "speech/validation",
    "speech/release",
    "device-evidence",
    "out",
)
_LANGUAGE_EXTENSIONS = {".jsonl", ".txt"}
_MAX_OPTIONS_BYTES = 256 * 1024
_MAX_CAMPAIGN_BYTES = 256 * 1024
_MAX_SPEECH_MANIFEST_BYTES = 64 * 1024 * 1024

_LANGUAGE_OPTIONS_PATH = "config/language-options.json"
_SPEECH_OPTIONS_PATH = "config/speech-options.json"
_INTAKE_OPTIONS_PATH = "config/intake-options.json"
_CAMPAIGN_PATH = "config/campaign.json"

_LANGUAGE_DIRS = {
    "training": "data/train",
    "validation": "data/validation",
    "release": "data/release",
}
_SPEECH_MANIFESTS = {
    "training": "speech/train/manifest.jsonl",
    "validation": "speech/validation/manifest.jsonl",
    "release": "speech/release/manifest.jsonl",
}


class VN97ProductionRunSealerError(RuntimeError):
    pass


@dataclass(frozen=True)
class VN97ProductionEnvironment:
    python_version: str
    torch_version: str
    platform_system: str
    platform_machine: str

    def __post_init__(self) -> None:
        for value, label in (
            (self.python_version, "Python version"),
            (self.torch_version, "Torch version"),
            (self.platform_system, "platform system"),
            (self.platform_machine, "platform machine"),
        ):
            if (
                not isinstance(value, str)
                or not value
                or len(value) > 128
                or any(ord(ch) < 0x20 for ch in value)
            ):
                raise VN97ProductionRunSealerError(
                    f"{label} is invalid"
                )


def probe_production_environment() -> VN97ProductionEnvironment:
    try:
        torch_version = importlib.metadata.version("torch")
    except importlib.metadata.PackageNotFoundError as exc:
        raise VN97ProductionRunSealerError(
            "Torch package is not installed; production environment cannot be sealed"
        ) from exc
    except ValueError as exc:
        raise VN97ProductionRunSealerError(
            "Torch package version could not be determined"
        ) from exc
    return VN97ProductionEnvironment(
        python_version=platform.python_version(),
        torch_version=torch_version,
        platform_system=platform.system(),
        platform_machine=platform.machine(),
    )


def _real_directory(path: Path, *, label: str) -> Path:
    try:
        info = os.lstat(path)
    except OSError as exc:
        raise VN97ProductionRunSealerError(
            f"{label} is unavailable"
        ) from exc
    if (
        stat.S_ISLNK(info.st_mode)
        or not stat.S_ISDIR(info.st_mode)
    ):
        raise VN97ProductionRunSealerError(
            f"{label} must be a real directory"
        )
    return path.resolve(strict=True)


def _strict_json_object(
    path: Path,
    *,
    max_bytes: int,
    label: str,
) -> dict[str, Any]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise VN97ProductionRunSealerError(
            f"{label} could not be opened safely"
        ) from exc
    try:
        info = os.fstat(fd)
        if (
            not stat.S_ISREG(info.st_mode)
            or not 0 < info.st_size <= max_bytes
        ):
            raise VN97ProductionRunSealerError(
                f"{label} byte size/type is invalid"
            )
        out = bytearray()
        while len(out) < info.st_size:
            chunk = os.read(
                fd,
                min(256 * 1024, info.st_size - len(out)),
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
            raise VN97ProductionRunSealerError(
                f"{label} changed while being read"
            )
    finally:
        os.close(fd)

    duplicates: list[str] = []

    def hook(
        pairs: list[tuple[str, Any]],
    ) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                duplicates.append(key)
            value[key] = item
        return value

    try:
        root = json.loads(
            bytes(out).decode("utf-8", errors="strict"),
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
        raise VN97ProductionRunSealerError(
            f"{label} must be strict UTF-8 JSON"
        ) from exc
    if duplicates or not isinstance(root, dict):
        raise VN97ProductionRunSealerError(
            f"{label} must be one object without duplicate keys"
        )
    return root


def _measure_regular_file(
    path: Path,
    *,
    workspace_root: Path,
    label: str,
) -> VN97RunFile:
    try:
        root = workspace_root.resolve(strict=True)
        parent = path.parent.resolve(strict=True)
        relative = path.relative_to(root).as_posix()
        parent.relative_to(root)
    except (
        OSError,
        ValueError,
    ) as exc:
        raise VN97ProductionRunSealerError(
            f"{label} escapes or is unavailable in workspace"
        ) from exc

    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise VN97ProductionRunSealerError(
            f"{label} could not be opened safely"
        ) from exc
    try:
        info = os.fstat(fd)
        if (
            not stat.S_ISREG(info.st_mode)
            or not 0 < info.st_size <= MAX_BOUND_FILE_BYTES
        ):
            raise VN97ProductionRunSealerError(
                f"{label} byte size/type is invalid"
            )
        digest = hashlib.sha256()
        consumed = 0
        while consumed < info.st_size:
            chunk = os.read(
                fd,
                min(1024 * 1024, info.st_size - consumed),
            )
            if not chunk:
                break
            consumed += len(chunk)
            digest.update(chunk)
        after = os.fstat(fd)
        if (
            consumed != info.st_size
            or after.st_size != info.st_size
            or after.st_ino != info.st_ino
            or after.st_dev != info.st_dev
        ):
            raise VN97ProductionRunSealerError(
                f"{label} changed while being hashed"
            )
        return VN97RunFile(
            path=relative,
            bytes=int(info.st_size),
            sha256=digest.hexdigest(),
        )
    finally:
        os.close(fd)


def _scan_language_split(
    workspace_root: Path,
    relative_dir: str,
    *,
    label: str,
) -> tuple[VN97RunFile, ...]:
    directory = _real_directory(
        workspace_root / relative_dir,
        label=f"{label} directory",
    )
    files: list[Path] = []
    for path in directory.rglob("*"):
        try:
            info = os.lstat(path)
        except OSError as exc:
            raise VN97ProductionRunSealerError(
                f"{label} entry became unavailable"
            ) from exc
        if stat.S_ISDIR(info.st_mode):
            continue
        if (
            stat.S_ISLNK(info.st_mode)
            or not stat.S_ISREG(info.st_mode)
            or path.suffix.lower() not in _LANGUAGE_EXTENSIONS
        ):
            raise VN97ProductionRunSealerError(
                f"{label} may contain only regular .jsonl/.txt files"
            )
        files.append(path)
    if not files:
        raise VN97ProductionRunSealerError(
            f"{label} contains no language dataset files"
        )
    result = tuple(
        sorted(
            (
                _measure_regular_file(
                    path,
                    workspace_root=workspace_root,
                    label=f"{label} file",
                )
                for path in files
            ),
            key=lambda item: item.path,
        )
    )
    if len(result) > 256:
        raise VN97ProductionRunSealerError(
            f"{label} exceeds VN97RUN1 file-count bound"
        )
    return result


def _speech_references(
    manifest_path: Path,
    *,
    workspace_root: Path,
    label: str,
) -> tuple[Path, ...]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(manifest_path, flags)
    except OSError as exc:
        raise VN97ProductionRunSealerError(
            f"{label} manifest could not be opened safely"
        ) from exc
    try:
        info = os.fstat(fd)
        if (
            not stat.S_ISREG(info.st_mode)
            or not 0 < info.st_size <= _MAX_SPEECH_MANIFEST_BYTES
        ):
            raise VN97ProductionRunSealerError(
                f"{label} manifest byte size/type is invalid"
            )
        out = bytearray()
        while len(out) < info.st_size:
            chunk = os.read(
                fd,
                min(256 * 1024, info.st_size - len(out)),
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
            raise VN97ProductionRunSealerError(
                f"{label} manifest changed while being read"
            )
    finally:
        os.close(fd)

    try:
        text = bytes(out).decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise VN97ProductionRunSealerError(
            f"{label} manifest must be strict UTF-8"
        ) from exc

    workspace = workspace_root.resolve(strict=True)
    manifest_parent = manifest_path.parent.resolve(strict=True)
    references: set[Path] = set()
    count = 0
    for line_number, line in enumerate(
        text.splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        duplicates: list[str] = []

        def hook(
            pairs: list[tuple[str, Any]],
        ) -> dict[str, Any]:
            value: dict[str, Any] = {}
            for key, item in pairs:
                if key in value:
                    duplicates.append(key)
                value[key] = item
            return value

        try:
            item = json.loads(
                line,
                object_pairs_hook=hook,
                parse_constant=lambda raw: (
                    _ for _ in ()
                ).throw(ValueError(raw)),
            )
        except (
            json.JSONDecodeError,
            ValueError,
        ) as exc:
            raise VN97ProductionRunSealerError(
                f"{label} manifest has invalid JSONL at line {line_number}"
            ) from exc
        if (
            duplicates
            or not isinstance(item, dict)
            or set(item) != {"audio", "text"}
            or not isinstance(item["audio"], str)
            or not item["audio"]
            or not isinstance(item["text"], str)
            or not item["text"].strip()
        ):
            raise VN97ProductionRunSealerError(
                f"{label} speech record is invalid at line {line_number}"
            )
        relative = Path(item["audio"])
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or "." in relative.parts
            or "\\" in item["audio"]
        ):
            raise VN97ProductionRunSealerError(
                f"{label} audio reference must be canonical relative path"
            )
        target = manifest_parent / relative
        try:
            resolved = target.resolve(strict=True)
            resolved.relative_to(workspace)
        except (
            OSError,
            ValueError,
        ) as exc:
            raise VN97ProductionRunSealerError(
                f"{label} audio reference escapes/unavailable in workspace"
            ) from exc
        if target.is_symlink():
            raise VN97ProductionRunSealerError(
                f"{label} audio reference must not be a symlink"
            )
        references.add(resolved)
        count += 1
        if count > 100_000:
            raise VN97ProductionRunSealerError(
                f"{label} speech example count exceeds bound"
            )
    if not references:
        raise VN97ProductionRunSealerError(
            f"{label} speech manifest contains no records"
        )
    return tuple(
        sorted(
            references,
            key=lambda path:
                path.relative_to(workspace).as_posix(),
        )
    )


def _seal_speech_input(
    workspace_root: Path,
    manifest_relative: str,
    *,
    label: str,
) -> VN97SpeechRunInput:
    manifest_path = workspace_root / manifest_relative
    references = _speech_references(
        manifest_path,
        workspace_root=workspace_root,
        label=label,
    )
    return VN97SpeechRunInput(
        manifest=_measure_regular_file(
            manifest_path,
            workspace_root=workspace_root,
            label=f"{label} manifest",
        ),
        audio=tuple(
            _measure_regular_file(
                path,
                workspace_root=workspace_root,
                label=f"{label} audio",
            )
            for path in references
        ),
    )


def _repository_commit(repository_root: Path) -> str:
    root = _real_directory(
        repository_root,
        label="VN97 repository root",
    )
    if not (
        root / "src/vn97/production_run_cli.py"
    ).is_file():
        raise VN97ProductionRunSealerError(
            "repository root is not a canonical VN97 source checkout"
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
        raise VN97ProductionRunSealerError(
            "repository Git identity could not be verified"
        ) from exc
    if (
        len(commit) != 40
        or any(
            ch not in "0123456789abcdef"
            for ch in commit
        )
    ):
        raise VN97ProductionRunSealerError(
            "repository HEAD is not a lowercase 40-hex commit"
        )
    if dirty:
        raise VN97ProductionRunSealerError(
            "production manifest sealing requires a clean tracked Git worktree"
        )
    return commit


def _option_object(
    workspace_root: Path,
    relative: str,
    *,
    label: str,
) -> Mapping[str, object]:
    return _strict_json_object(
        workspace_root / relative,
        max_bytes=_MAX_OPTIONS_BYTES,
        label=label,
    )


def build_production_run_manifest(
    *,
    workspace_root: Path,
    repository_root: Path,
    environment: VN97ProductionEnvironment,
) -> VN97ProductionRunManifest:
    workspace = _real_directory(
        workspace_root,
        label="production workspace root",
    )
    campaign_path = workspace / _CAMPAIGN_PATH
    _strict_json_object(
        campaign_path,
        max_bytes=_MAX_CAMPAIGN_BYTES,
        label="VN97CAMPDEF1 campaign definition",
    )

    language_options = _option_object(
        workspace,
        _LANGUAGE_OPTIONS_PATH,
        label="language options",
    )
    speech_options = _option_object(
        workspace,
        _SPEECH_OPTIONS_PATH,
        label="speech options",
    )
    intake_options = _option_object(
        workspace,
        _INTAKE_OPTIONS_PATH,
        label="intake options",
    )

    manifest = VN97ProductionRunManifest(
        repository_commit=_repository_commit(
            repository_root
        ),
        python_version=environment.python_version,
        torch_version=environment.torch_version,
        platform_system=environment.platform_system,
        platform_machine=environment.platform_machine,
        campaign_definition=_measure_regular_file(
            campaign_path,
            workspace_root=workspace,
            label="campaign definition",
        ),
        language_training=_scan_language_split(
            workspace,
            _LANGUAGE_DIRS["training"],
            label="language training",
        ),
        language_validation=_scan_language_split(
            workspace,
            _LANGUAGE_DIRS["validation"],
            label="language validation",
        ),
        language_release=_scan_language_split(
            workspace,
            _LANGUAGE_DIRS["release"],
            label="language release",
        ),
        speech_training=_seal_speech_input(
            workspace,
            _SPEECH_MANIFESTS["training"],
            label="speech training",
        ),
        speech_validation=_seal_speech_input(
            workspace,
            _SPEECH_MANIFESTS["validation"],
            label="speech validation",
        ),
        speech_release=_seal_speech_input(
            workspace,
            _SPEECH_MANIFESTS["release"],
            label="speech release",
        ),
        device_evidence_dir="device-evidence",
        language_options=tuple(
            sorted(language_options.items())
        ),
        speech_options=tuple(
            sorted(speech_options.items())
        ),
        intake_options=tuple(
            sorted(intake_options.items())
        ),
        language_output_dir="out/language",
        production_output_dir="out/production",
        intake_output_dir="out/intake",
    )
    verify_production_run_inputs(
        manifest,
        workspace_root=workspace,
    )
    return manifest


def write_manifest_atomic(
    path: Path,
    manifest: VN97ProductionRunManifest,
) -> None:
    data = manifest.to_bytes()
    if path.exists() or path.is_symlink():
        raise VN97ProductionRunSealerError(
            "VN97RUN1 output must not already exist"
        )
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    if path.parent.is_symlink():
        raise VN97ProductionRunSealerError(
            "VN97RUN1 output parent must not be a symlink"
        )
    fd, temporary = tempfile.mkstemp(
        prefix=".vn97run1-",
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
            os.fsync(output.fileno())
        os.replace(
            temporary,
            path,
        )
        if path.read_bytes() != data:
            raise VN97ProductionRunSealerError(
                "VN97RUN1 post-write verification failed"
            )
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def bootstrap_workspace(workspace_root: Path) -> None:
    root = workspace_root
    if root.exists() or root.is_symlink():
        if root.is_symlink() or not root.is_dir():
            raise VN97ProductionRunSealerError(
                "bootstrap target must be absent or an empty real directory"
            )
        if any(root.iterdir()):
            raise VN97ProductionRunSealerError(
                "bootstrap target directory must be empty"
            )
    else:
        root.mkdir(parents=True)

    for relative in _BOOTSTRAP_DIRS:
        (root / relative).mkdir(
            parents=True,
            exist_ok=False,
        )

    readme = """VN97 production workspace (M19H)

This skeleton contains NO production data and is intentionally not sealable yet.

Required before vn97-production-seal seal:
- config/campaign.json
- config/language-options.json
- config/speech-options.json
- config/intake-options.json
- one or more .jsonl/.txt files in data/train, data/validation and data/release
- speech/train/manifest.jsonl plus referenced audio
- speech/validation/manifest.jsonl plus referenced audio
- speech/release/manifest.jsonl plus referenced audio

device-evidence/ remains empty until after vn97-production-run --stage train
produces the exact model image and real physical-phone evidence is collected.

out/ is only a parent. The runner creates out/language, out/production and
out/intake; do not pre-create those directories.
"""
    (root / "PRODUCTION_WORKSPACE.md").write_text(
        readme,
        encoding="utf-8",
    )

    examples = {
        "config/language-options.json.example": {
            "device": "CHANGE_ME",
            "max_parameters": 0,
            "max_validation_loss": 0.0,
        },
        "config/speech-options.json.example": {
            "device": "CHANGE_ME",
            "max_speech_validation_loss": 0.0,
        },
        "config/intake-options.json.example": {
            "min_device_runs": 5,
            "min_distinct_device_profiles": 1,
        },
        "config/campaign.json.example": {
            "candidates": [],
            "schema": "VN97CAMPDEF1",
        },
    }
    for relative, value in examples.items():
        (root / relative).write_text(
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            ) + "\n",
            encoding="utf-8",
        )
