from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import stat
from typing import Any, Mapping


SCHEMA = "VN97RUN1"
MAX_MANIFEST_BYTES = 1024 * 1024
MAX_BOUND_FILE_BYTES = 2 * 1024 * 1024 * 1024
_MAX_LANGUAGE_FILES = 256
_MAX_SPEECH_AUDIO_FILES = 100_000
_HEX40 = set("0123456789abcdef")
_HEX64 = set("0123456789abcdef")


class VN97ProductionRunManifestError(RuntimeError):
    pass


def _require_hex(
    value: object,
    *,
    length: int,
    label: str,
) -> str:
    if (
        not isinstance(value, str)
        or len(value) != length
        or any(ch not in _HEX64 for ch in value)
    ):
        raise VN97ProductionRunManifestError(
            f"{label} must be {length} lowercase hex characters"
        )
    return value


def _relative_path(
    value: object,
    *,
    label: str,
) -> str:
    if (
        not isinstance(value, str)
        or not value
        or "\\" in value
    ):
        raise VN97ProductionRunManifestError(
            f"{label} must be a non-empty POSIX relative path"
        )
    path = Path(value)
    if (
        path.is_absolute()
        or ".." in path.parts
        or "." in path.parts
        or value.startswith("/")
        or value.endswith("/")
    ):
        raise VN97ProductionRunManifestError(
            f"{label} must stay inside the production workspace"
        )
    normalized = path.as_posix()
    if normalized != value:
        raise VN97ProductionRunManifestError(
            f"{label} must use canonical POSIX path spelling"
        )
    return value


def _safe_regular_sha256(
    path: Path,
    *,
    expected_bytes: int,
    expected_sha256: str,
    label: str,
) -> None:
    flags = os.O_RDONLY | getattr(
        os,
        "O_NOFOLLOW",
        0,
    )
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise VN97ProductionRunManifestError(
            f"{label} could not be opened safely"
        ) from exc
    try:
        info = os.fstat(fd)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_size != expected_bytes
            or not 0 < info.st_size <= MAX_BOUND_FILE_BYTES
        ):
            raise VN97ProductionRunManifestError(
                f"{label} byte size/type does not match VN97RUN1"
            )
        digest = hashlib.sha256()
        consumed = 0
        while consumed < info.st_size:
            chunk = os.read(
                fd,
                min(
                    1024 * 1024,
                    info.st_size - consumed,
                ),
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
            raise VN97ProductionRunManifestError(
                f"{label} changed while being verified"
            )
        if digest.hexdigest() != expected_sha256:
            raise VN97ProductionRunManifestError(
                f"{label} SHA-256 does not match VN97RUN1"
            )
    finally:
        os.close(fd)


def _read_bound_bytes(
    path: Path,
    *,
    max_bytes: int,
    label: str,
) -> bytes:
    flags = os.O_RDONLY | getattr(
        os,
        "O_NOFOLLOW",
        0,
    )
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise VN97ProductionRunManifestError(
            f"{label} could not be opened safely"
        ) from exc
    try:
        info = os.fstat(fd)
        if (
            not stat.S_ISREG(info.st_mode)
            or not 0 < info.st_size <= max_bytes
        ):
            raise VN97ProductionRunManifestError(
                f"{label} byte size/type is invalid"
            )
        out = bytearray()
        while len(out) < info.st_size:
            chunk = os.read(
                fd,
                min(
                    256 * 1024,
                    info.st_size - len(out),
                ),
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
            raise VN97ProductionRunManifestError(
                f"{label} changed while being read"
            )
        return bytes(out)
    finally:
        os.close(fd)


@dataclass(frozen=True)
class VN97RunFile:
    path: str
    bytes: int
    sha256: str

    def __post_init__(self) -> None:
        _relative_path(
            self.path,
            label="bound file path",
        )
        if (
            type(self.bytes) is not int
            or not 0 < self.bytes <= MAX_BOUND_FILE_BYTES
        ):
            raise VN97ProductionRunManifestError(
                "bound file byte count is invalid"
            )
        _require_hex(
            self.sha256,
            length=64,
            label="bound file SHA-256",
        )

    def canonical_object(self) -> dict[str, object]:
        return {
            "bytes": self.bytes,
            "path": self.path,
            "sha256": self.sha256,
        }

    def resolve(
        self,
        workspace_root: Path,
    ) -> Path:
        target = (
            workspace_root /
            Path(self.path)
        )
        try:
            root = workspace_root.resolve(
                strict=True
            )
            parent = target.parent.resolve(
                strict=True
            )
        except OSError as exc:
            raise VN97ProductionRunManifestError(
                f"bound file parent is unavailable: {self.path}"
            ) from exc
        try:
            parent.relative_to(root)
        except ValueError as exc:
            raise VN97ProductionRunManifestError(
                f"bound file escapes workspace: {self.path}"
            ) from exc
        _safe_regular_sha256(
            target,
            expected_bytes=self.bytes,
            expected_sha256=self.sha256,
            label=f"bound file {self.path}",
        )
        return target.resolve(strict=True)


@dataclass(frozen=True)
class VN97SpeechRunInput:
    manifest: VN97RunFile
    audio: tuple[VN97RunFile, ...]

    def __post_init__(self) -> None:
        if (
            not self.audio
            or len(self.audio)
            > _MAX_SPEECH_AUDIO_FILES
        ):
            raise VN97ProductionRunManifestError(
                "speech input audio file count is outside bounds"
            )
        paths = tuple(
            item.path
            for item in self.audio
        )
        if tuple(sorted(set(paths))) != paths:
            raise VN97ProductionRunManifestError(
                "speech audio file paths must be unique and sorted"
            )

    def canonical_object(self) -> dict[str, object]:
        return {
            "audio": [
                item.canonical_object()
                for item in self.audio
            ],
            "manifest":
                self.manifest.canonical_object(),
        }


_LANGUAGE_ALLOWED = {
    "batch_size",
    "deployment_tile_cols",
    "deployment_tile_rows",
    "device",
    "epochs",
    "format",
    "learned_tokens",
    "max_examples",
    "max_grad_norm",
    "max_input_bytes",
    "max_model_image_bytes",
    "max_parameters",
    "max_recurrent_state_bytes",
    "max_validation_loss",
    "max_windows",
    "min_pair_count",
    "min_validation_accuracy",
    "min_validation_target_tokens",
    "release_batch_size",
    "release_format",
    "release_max_examples",
    "release_max_input_bytes",
    "release_max_validation_loss",
    "release_min_validation_accuracy",
    "release_max_validation_target_tokens",
    "release_max_windows",
    "sequence_length",
    "stride",
    "validation_batch_size",
    "validation_format",
    "validation_max_examples",
    "validation_max_input_bytes",
    "validation_max_windows",
    "weight_decay",
}
_LANGUAGE_REQUIRED = {
    "device",
    "max_parameters",
    "max_validation_loss",
}
_SPEECH_ALLOWED = {
    "device",
    "max_model_image_bytes",
    "max_recurrent_state_bytes",
    "max_speech_validation_loss",
    "min_speech_validation_accuracy",
    "min_speech_validation_examples",
    "min_speech_validation_target_tokens",
    "release_max_speech_loss",
    "release_min_speech_accuracy",
    "release_min_speech_examples",
    "release_min_speech_target_tokens",
    "speech_epochs",
    "speech_learning_rate",
    "speech_max_examples",
    "speech_max_frames",
    "speech_max_grad_norm",
    "speech_max_target_tokens",
    "speech_seed",
    "speech_weight_decay",
    "tile_cols",
    "tile_rows",
}
_SPEECH_REQUIRED = {
    "device",
    "max_speech_validation_loss",
}
_INTAKE_ALLOWED = {
    "max_abs_battery_energy_counter_delta_nwh",
    "max_device_peak_pss_kib",
    "max_device_thermal_status",
    "max_speech_prefill_p95_ms",
    "max_text_decode_p95_ms_per_token",
    "max_text_prefill_p95_ms",
    "min_device_runs",
    "min_distinct_device_profiles",
    "require_energy_counter",
}

_INT_KEYS = {
    "batch_size",
    "deployment_tile_cols",
    "deployment_tile_rows",
    "epochs",
    "learned_tokens",
    "max_examples",
    "max_input_bytes",
    "max_model_image_bytes",
    "max_parameters",
    "max_recurrent_state_bytes",
    "max_windows",
    "min_pair_count",
    "min_validation_target_tokens",
    "release_batch_size",
    "release_max_examples",
    "release_max_input_bytes",
    "release_max_validation_target_tokens",
    "release_max_windows",
    "sequence_length",
    "stride",
    "validation_batch_size",
    "validation_max_examples",
    "validation_max_input_bytes",
    "validation_max_windows",
    "min_speech_validation_examples",
    "min_speech_validation_target_tokens",
    "release_min_speech_examples",
    "release_min_speech_target_tokens",
    "speech_epochs",
    "speech_max_examples",
    "speech_max_frames",
    "speech_max_target_tokens",
    "speech_seed",
    "tile_cols",
    "tile_rows",
    "max_abs_battery_energy_counter_delta_nwh",
    "max_device_peak_pss_kib",
    "max_device_thermal_status",
    "min_device_runs",
    "min_distinct_device_profiles",
}
_FLOAT_KEYS = {
    "max_grad_norm",
    "max_validation_loss",
    "min_validation_accuracy",
    "release_max_validation_loss",
    "release_min_validation_accuracy",
    "weight_decay",
    "max_speech_validation_loss",
    "min_speech_validation_accuracy",
    "release_max_speech_loss",
    "release_min_speech_accuracy",
    "speech_learning_rate",
    "speech_max_grad_norm",
    "speech_weight_decay",
    "max_speech_prefill_p95_ms",
    "max_text_decode_p95_ms_per_token",
    "max_text_prefill_p95_ms",
}
_OPTIONAL_KEYS = {
    "release_format",
    "release_max_validation_loss",
    "release_min_validation_accuracy",
    "release_max_validation_target_tokens",
    "stride",
    "validation_format",
    "release_max_speech_loss",
    "release_min_speech_accuracy",
    "release_min_speech_examples",
    "release_min_speech_target_tokens",
    "max_abs_battery_energy_counter_delta_nwh",
    "max_speech_prefill_p95_ms",
}
_ACCURACY_KEYS = {
    "min_validation_accuracy",
    "release_min_validation_accuracy",
    "min_speech_validation_accuracy",
    "release_min_speech_accuracy",
}
_NONNEG_FLOAT_KEYS = {
    "min_validation_accuracy",
    "release_min_validation_accuracy",
    "min_speech_validation_accuracy",
    "release_min_speech_accuracy",
    "speech_weight_decay",
    "weight_decay",
}
_NONNEG_INT_KEYS = {
    "speech_seed",
    "max_device_thermal_status",
}
_STRING_KEYS = {
    "device",
    "format",
    "release_format",
    "validation_format",
}
_BOOL_KEYS = {
    "require_energy_counter",
}


def _validate_options(
    raw: object,
    *,
    allowed: set[str],
    required: set[str],
    label: str,
) -> tuple[tuple[str, object], ...]:
    if not isinstance(raw, dict):
        raise VN97ProductionRunManifestError(
            f"{label} options must be an object"
        )
    keys = set(raw)
    if (
        not required.issubset(keys)
        or not keys.issubset(allowed)
    ):
        raise VN97ProductionRunManifestError(
            f"{label} option keys are invalid"
        )
    result: list[
        tuple[str, object]
    ] = []
    for key in sorted(keys):
        value = raw[key]
        if value is None:
            if key not in _OPTIONAL_KEYS:
                raise VN97ProductionRunManifestError(
                    f"{label} option {key} may not be null"
                )
            result.append(
                (key, None)
            )
            continue
        if key in _BOOL_KEYS:
            if type(value) is not bool:
                raise VN97ProductionRunManifestError(
                    f"{label} option {key} must be boolean"
                )
        elif key in _INT_KEYS:
            if type(value) is not int:
                raise VN97ProductionRunManifestError(
                    f"{label} option {key} must be integer"
                )
            if key in _NONNEG_INT_KEYS:
                if value < 0:
                    raise VN97ProductionRunManifestError(
                        f"{label} option {key} must be non-negative"
                    )
            elif value <= 0:
                raise VN97ProductionRunManifestError(
                    f"{label} option {key} must be positive"
                )
            if (
                key in {
                    "deployment_tile_cols",
                    "deployment_tile_rows",
                    "tile_cols",
                    "tile_rows",
                }
                and value > 256
            ):
                raise VN97ProductionRunManifestError(
                    f"{label} option {key} exceeds VN97T2 tile bound"
                )
            if (
                key == "max_device_thermal_status"
                and value > 6
            ):
                raise VN97ProductionRunManifestError(
                    "max_device_thermal_status must be in [0, 6]"
                )
        elif key in _FLOAT_KEYS:
            if (
                isinstance(value, bool)
                or not isinstance(
                    value,
                    (int, float),
                )
            ):
                raise VN97ProductionRunManifestError(
                    f"{label} option {key} must be numeric"
                )
            numeric = float(value)
            if not math.isfinite(numeric):
                raise VN97ProductionRunManifestError(
                    f"{label} option {key} must be finite"
                )
            if key in _NONNEG_FLOAT_KEYS:
                if numeric < 0.0:
                    raise VN97ProductionRunManifestError(
                        f"{label} option {key} must be non-negative"
                    )
            elif numeric <= 0.0:
                raise VN97ProductionRunManifestError(
                    f"{label} option {key} must be positive"
                )
            if (
                key in _ACCURACY_KEYS
                and numeric > 1.0
            ):
                raise VN97ProductionRunManifestError(
                    f"{label} option {key} must be <= 1"
                )
        elif key in _STRING_KEYS:
            if not isinstance(value, str) or not value:
                raise VN97ProductionRunManifestError(
                    f"{label} option {key} must be non-empty text"
                )
            if key == "device" and value == "auto":
                raise VN97ProductionRunManifestError(
                    f"{label} production device must be explicit, not auto"
                )
            if (
                key in {
                    "format",
                    "validation_format",
                    "release_format",
                }
                and value not in {"text", "chat"}
            ):
                raise VN97ProductionRunManifestError(
                    f"{label} option {key} must be text or chat"
                )
        else:
            raise VN97ProductionRunManifestError(
                f"{label} option {key} has no manifest type contract"
            )
        result.append(
            (key, value)
        )
    return tuple(result)


def _options_object(
    options: tuple[
        tuple[str, object],
        ...
    ],
) -> dict[str, object]:
    return {
        key: value
        for key, value in options
    }


@dataclass(frozen=True)
class VN97ProductionRunManifest:
    repository_commit: str
    campaign_definition: VN97RunFile
    language_training: tuple[VN97RunFile, ...]
    language_validation: tuple[VN97RunFile, ...]
    language_release: tuple[VN97RunFile, ...]
    speech_training: VN97SpeechRunInput
    speech_validation: VN97SpeechRunInput
    speech_release: VN97SpeechRunInput
    device_evidence_dir: str
    language_options: tuple[
        tuple[str, object],
        ...
    ]
    speech_options: tuple[
        tuple[str, object],
        ...
    ]
    intake_options: tuple[
        tuple[str, object],
        ...
    ]
    language_output_dir: str
    production_output_dir: str
    intake_output_dir: str

    def __post_init__(self) -> None:
        _require_hex(
            self.repository_commit,
            length=40,
            label="repository commit",
        )
        for group, label in (
            (
                self.language_training,
                "language training",
            ),
            (
                self.language_validation,
                "language validation",
            ),
            (
                self.language_release,
                "language release",
            ),
        ):
            if (
                not group
                or len(group)
                > _MAX_LANGUAGE_FILES
            ):
                raise VN97ProductionRunManifestError(
                    f"{label} file count is outside bounds"
                )
            paths = tuple(
                item.path
                for item in group
            )
            if tuple(sorted(set(paths))) != paths:
                raise VN97ProductionRunManifestError(
                    f"{label} paths must be unique and sorted"
                )

        all_language_paths = [
            item.path
            for group in (
                self.language_training,
                self.language_validation,
                self.language_release,
            )
            for item in group
        ]
        if len(set(all_language_paths)) != len(
            all_language_paths
        ):
            raise VN97ProductionRunManifestError(
                "language train/validation/release paths must be distinct"
            )

        _relative_path(
            self.device_evidence_dir,
            label="device evidence directory",
        )
        outputs = (
            self.language_output_dir,
            self.production_output_dir,
            self.intake_output_dir,
        )
        for value, label in zip(
            outputs,
            (
                "language output directory",
                "production output directory",
                "intake output directory",
            ),
            strict=True,
        ):
            _relative_path(
                value,
                label=label,
            )
        if len(set(outputs)) != 3:
            raise VN97ProductionRunManifestError(
                "production output directories must be distinct"
            )
        output_paths = [
            Path(value)
            for value in outputs
        ]
        for index, left in enumerate(
            output_paths
        ):
            for right in output_paths[
                index + 1:
            ]:
                if (
                    left in right.parents
                    or right in left.parents
                ):
                    raise VN97ProductionRunManifestError(
                        "production output directories must not be nested"
                    )

    def canonical_object(self) -> dict[str, object]:
        return {
            "inputs": {
                "campaign_definition":
                    self.campaign_definition
                    .canonical_object(),
                "language_release": [
                    item.canonical_object()
                    for item in
                        self.language_release
                ],
                "language_training": [
                    item.canonical_object()
                    for item in
                        self.language_training
                ],
                "language_validation": [
                    item.canonical_object()
                    for item in
                        self.language_validation
                ],
                "speech_release":
                    self.speech_release
                    .canonical_object(),
                "speech_training":
                    self.speech_training
                    .canonical_object(),
                "speech_validation":
                    self.speech_validation
                    .canonical_object(),
            },
            "intake": {
                "device_evidence_dir":
                    self.device_evidence_dir,
                "options":
                    _options_object(
                        self.intake_options
                    ),
            },
            "language": {
                "options":
                    _options_object(
                        self.language_options
                    ),
            },
            "outputs": {
                "intake":
                    self.intake_output_dir,
                "language":
                    self.language_output_dir,
                "production":
                    self.production_output_dir,
            },
            "repository_commit":
                self.repository_commit,
            "schema": SCHEMA,
            "speech": {
                "options":
                    _options_object(
                        self.speech_options
                    ),
            },
        }

    def to_bytes(self) -> bytes:
        return json.dumps(
            self.canonical_object(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")

    @property
    def manifest_sha256(self) -> str:
        return hashlib.sha256(
            self.to_bytes()
        ).hexdigest()


def _file_from_raw(
    raw: object,
    *,
    label: str,
) -> VN97RunFile:
    if (
        not isinstance(raw, dict)
        or set(raw)
        != {"bytes", "path", "sha256"}
    ):
        raise VN97ProductionRunManifestError(
            f"{label} file identity keys are invalid"
        )
    try:
        return VN97RunFile(
            path=raw["path"],
            bytes=raw["bytes"],
            sha256=raw["sha256"],
        )
    except (
        TypeError,
        ValueError,
        VN97ProductionRunManifestError,
    ) as exc:
        if isinstance(
            exc,
            VN97ProductionRunManifestError,
        ):
            raise
        raise VN97ProductionRunManifestError(
            str(exc)
        ) from exc


def _speech_from_raw(
    raw: object,
    *,
    label: str,
) -> VN97SpeechRunInput:
    if (
        not isinstance(raw, dict)
        or set(raw)
        != {"audio", "manifest"}
        or not isinstance(
            raw["audio"],
            list,
        )
    ):
        raise VN97ProductionRunManifestError(
            f"{label} speech input keys are invalid"
        )
    return VN97SpeechRunInput(
        manifest=_file_from_raw(
            raw["manifest"],
            label=f"{label} manifest",
        ),
        audio=tuple(
            _file_from_raw(
                item,
                label=f"{label} audio",
            )
            for item in raw["audio"]
        ),
    )


def parse_production_run_manifest(
    data: bytes,
) -> VN97ProductionRunManifest:
    if not 0 < len(data) <= MAX_MANIFEST_BYTES:
        raise VN97ProductionRunManifestError(
            "VN97RUN1 byte size is outside bounds"
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
        raise VN97ProductionRunManifestError(
            "VN97RUN1 must be strict UTF-8 JSON"
        ) from exc
    if duplicates or not isinstance(root, dict):
        raise VN97ProductionRunManifestError(
            "VN97RUN1 must be one object without duplicate keys"
        )
    canonical = json.dumps(
        root,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    if canonical != data:
        raise VN97ProductionRunManifestError(
            "VN97RUN1 must use canonical JSON"
        )
    if (
        set(root)
        != {
            "inputs",
            "intake",
            "language",
            "outputs",
            "repository_commit",
            "schema",
            "speech",
        }
        or root["schema"] != SCHEMA
    ):
        raise VN97ProductionRunManifestError(
            "VN97RUN1 schema/keys mismatch"
        )

    inputs = root["inputs"]
    if (
        not isinstance(inputs, dict)
        or set(inputs)
        != {
            "campaign_definition",
            "language_release",
            "language_training",
            "language_validation",
            "speech_release",
            "speech_training",
            "speech_validation",
        }
    ):
        raise VN97ProductionRunManifestError(
            "VN97RUN1 input keys mismatch"
        )
    for key in (
        "language_training",
        "language_validation",
        "language_release",
    ):
        if not isinstance(
            inputs[key],
            list,
        ):
            raise VN97ProductionRunManifestError(
                f"VN97RUN1 {key} must be an array"
            )

    language = root["language"]
    speech = root["speech"]
    intake = root["intake"]
    outputs = root["outputs"]
    if (
        not isinstance(language, dict)
        or set(language)
        != {"options"}
        or not isinstance(speech, dict)
        or set(speech)
        != {"options"}
        or not isinstance(intake, dict)
        or set(intake)
        != {
            "device_evidence_dir",
            "options",
        }
        or not isinstance(outputs, dict)
        or set(outputs)
        != {
            "intake",
            "language",
            "production",
        }
    ):
        raise VN97ProductionRunManifestError(
            "VN97RUN1 stage/output keys mismatch"
        )

    return VN97ProductionRunManifest(
        repository_commit=_require_hex(
            root["repository_commit"],
            length=40,
            label="repository commit",
        ),
        campaign_definition=_file_from_raw(
            inputs[
                "campaign_definition"
            ],
            label="campaign definition",
        ),
        language_training=tuple(
            _file_from_raw(
                item,
                label="language training",
            )
            for item in
                inputs["language_training"]
        ),
        language_validation=tuple(
            _file_from_raw(
                item,
                label="language validation",
            )
            for item in
                inputs[
                    "language_validation"
                ]
        ),
        language_release=tuple(
            _file_from_raw(
                item,
                label="language release",
            )
            for item in
                inputs["language_release"]
        ),
        speech_training=_speech_from_raw(
            inputs["speech_training"],
            label="speech training",
        ),
        speech_validation=_speech_from_raw(
            inputs[
                "speech_validation"
            ],
            label="speech validation",
        ),
        speech_release=_speech_from_raw(
            inputs["speech_release"],
            label="speech release",
        ),
        device_evidence_dir=
            _relative_path(
                intake[
                    "device_evidence_dir"
                ],
                label=
                    "device evidence directory",
            ),
        language_options=
            _validate_options(
                language["options"],
                allowed=
                    _LANGUAGE_ALLOWED,
                required=
                    _LANGUAGE_REQUIRED,
                label="language",
            ),
        speech_options=
            _validate_options(
                speech["options"],
                allowed=
                    _SPEECH_ALLOWED,
                required=
                    _SPEECH_REQUIRED,
                label="speech",
            ),
        intake_options=
            _validate_options(
                intake["options"],
                allowed=
                    _INTAKE_ALLOWED,
                required=set(),
                label="intake",
            ),
        language_output_dir=
            _relative_path(
                outputs["language"],
                label=
                    "language output directory",
            ),
        production_output_dir=
            _relative_path(
                outputs["production"],
                label=
                    "production output directory",
            ),
        intake_output_dir=
            _relative_path(
                outputs["intake"],
                label=
                    "intake output directory",
            ),
    )


def load_production_run_manifest(
    path: str | os.PathLike[str],
) -> VN97ProductionRunManifest:
    target = Path(path)
    data = _read_bound_bytes(
        target,
        max_bytes=MAX_MANIFEST_BYTES,
        label="VN97RUN1 manifest",
    )
    return parse_production_run_manifest(
        data
    )


def _parse_campaign_definition(
    data: bytes,
) -> None:
    duplicates: list[str] = []

    def hook(
        pairs: list[tuple[str, Any]],
    ) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in pairs:
            if key in out:
                duplicates.append(key)
            out[key] = value
        return out

    try:
        root = json.loads(
            data.decode(
                "utf-8",
                errors="strict",
            ),
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
        raise VN97ProductionRunManifestError(
            "VN97CAMPDEF1 must be strict UTF-8 JSON"
        ) from exc
    if (
        duplicates
        or not isinstance(root, dict)
        or set(root)
        != {"candidates", "schema"}
        or root["schema"]
        != "VN97CAMPDEF1"
        or not isinstance(
            root["candidates"],
            list,
        )
        or not 1
        <= len(root["candidates"])
        <= 64
    ):
        raise VN97ProductionRunManifestError(
            "campaign definition schema/candidate count is invalid"
        )
    seen: set[str] = set()
    for item in root["candidates"]:
        if (
            not isinstance(item, dict)
            or set(item)
            != {
                "d_model",
                "d_state",
                "embedding_rank",
                "learning_rate",
                "n_layers",
                "seed",
            }
        ):
            raise VN97ProductionRunManifestError(
                "campaign candidate keys are invalid"
            )
        for key in (
            "d_model",
            "d_state",
            "n_layers",
        ):
            if (
                type(item[key]) is not int
                or item[key] <= 0
            ):
                raise VN97ProductionRunManifestError(
                    f"campaign candidate {key} must be positive integer"
                )
        if (
            type(item["seed"]) is not int
            or item["seed"] < 0
        ):
            raise VN97ProductionRunManifestError(
                "campaign candidate seed must be non-negative integer"
            )
        rank = item[
            "embedding_rank"
        ]
        if (
            rank is not None
            and (
                type(rank) is not int
                or rank <= 0
            )
        ):
            raise VN97ProductionRunManifestError(
                "campaign candidate embedding_rank must be positive integer or null"
            )
        lr = item["learning_rate"]
        if (
            isinstance(lr, bool)
            or not isinstance(
                lr,
                (int, float),
            )
            or not math.isfinite(
                float(lr)
            )
            or float(lr) <= 0.0
        ):
            raise VN97ProductionRunManifestError(
                "campaign candidate learning_rate must be finite positive"
            )
        identity = json.dumps(
            item,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        if identity in seen:
            raise VN97ProductionRunManifestError(
                "campaign definition contains duplicate candidates"
            )
        seen.add(identity)


def _speech_references(
    manifest_path: Path,
    data: bytes,
    *,
    workspace_root: Path,
) -> tuple[str, ...]:
    try:
        text = data.decode(
            "utf-8",
            errors="strict",
        )
    except UnicodeDecodeError as exc:
        raise VN97ProductionRunManifestError(
            "speech manifest must be strict UTF-8"
        ) from exc

    try:
        workspace = workspace_root.resolve(
            strict=True
        )
        manifest_parent = (
            manifest_path.parent.resolve(
                strict=True
            )
        )
    except OSError as exc:
        raise VN97ProductionRunManifestError(
            "speech manifest workspace is unavailable"
        ) from exc

    refs: set[str] = set()
    for line_number, line in enumerate(
        text.splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        duplicates: list[str] = []

        def hook(
            pairs: list[
                tuple[str, Any]
            ],
        ) -> dict[str, Any]:
            out: dict[str, Any] = {}
            for key, value in pairs:
                if key in out:
                    duplicates.append(key)
                out[key] = value
            return out

        try:
            value = json.loads(
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
            raise VN97ProductionRunManifestError(
                f"invalid speech JSONL at line {line_number}"
            ) from exc
        if (
            duplicates
            or not isinstance(
                value,
                dict,
            )
            or set(value)
            != {"audio", "text"}
            or not isinstance(
                value["audio"],
                str,
            )
            or not value["audio"]
            or not isinstance(
                value["text"],
                str,
            )
            or not value[
                "text"
            ].strip()
        ):
            raise VN97ProductionRunManifestError(
                f"invalid speech record at line {line_number}"
            )
        rel = Path(value["audio"])
        if (
            rel.is_absolute()
            or ".." in rel.parts
            or "." in rel.parts
            or "\\" in value["audio"]
        ):
            raise VN97ProductionRunManifestError(
                "speech audio references must be canonical relative paths"
            )
        absolute = (
            manifest_parent /
            rel
        )
        try:
            resolved_parent = (
                absolute.parent.resolve(
                    strict=True
                )
            )
            resolved_parent.relative_to(
                workspace
            )
        except (
            OSError,
            ValueError,
        ) as exc:
            raise VN97ProductionRunManifestError(
                "speech audio reference escapes/unavailable in workspace"
            ) from exc
        workspace_rel = (
            absolute.relative_to(
                workspace
            ).as_posix()
        )
        refs.add(workspace_rel)
    if not refs:
        raise VN97ProductionRunManifestError(
            "speech manifest contains no records"
        )
    return tuple(sorted(refs))


@dataclass(frozen=True)
class VN97ResolvedProductionRun:
    workspace_root: Path
    campaign_definition: Path
    language_training: tuple[
        Path,
        ...
    ]
    language_validation: tuple[
        Path,
        ...
    ]
    language_release: tuple[
        Path,
        ...
    ]
    speech_training_manifest: Path
    speech_validation_manifest: Path
    speech_release_manifest: Path
    device_evidence_dir: Path
    language_output_dir: Path
    production_output_dir: Path
    intake_output_dir: Path


def verify_production_run_inputs(
    manifest: VN97ProductionRunManifest,
    *,
    workspace_root: Path,
) -> VN97ResolvedProductionRun:
    try:
        info = os.lstat(
            workspace_root
        )
    except OSError as exc:
        raise VN97ProductionRunManifestError(
            "production workspace root is unavailable"
        ) from exc
    if (
        stat.S_ISLNK(info.st_mode)
        or not stat.S_ISDIR(info.st_mode)
    ):
        raise VN97ProductionRunManifestError(
            "production workspace root must be a real directory"
        )
    workspace = workspace_root.resolve(
        strict=True
    )

    campaign = (
        manifest.campaign_definition
        .resolve(workspace)
    )
    _parse_campaign_definition(
        _read_bound_bytes(
            campaign,
            max_bytes=256 * 1024,
            label=
                "campaign definition",
        )
    )

    language_groups: list[
        tuple[Path, ...]
    ] = []
    identities: set[
        tuple[int, int]
    ] = set()
    for group, label in (
        (
            manifest.language_training,
            "language training",
        ),
        (
            manifest.language_validation,
            "language validation",
        ),
        (
            manifest.language_release,
            "language release",
        ),
    ):
        resolved: list[Path] = []
        for item in group:
            path = item.resolve(
                workspace
            )
            info = os.stat(
                path,
                follow_symlinks=False,
            )
            identity = (
                int(info.st_dev),
                int(info.st_ino),
            )
            if identity in identities:
                raise VN97ProductionRunManifestError(
                    "language train/validation/release inputs reuse one physical file"
                )
            identities.add(identity)
            resolved.append(path)
        language_groups.append(
            tuple(resolved)
        )

    speech_manifests: list[
        Path
    ] = []
    for spec, label in (
        (
            manifest.speech_training,
            "speech training",
        ),
        (
            manifest.speech_validation,
            "speech validation",
        ),
        (
            manifest.speech_release,
            "speech release",
        ),
    ):
        manifest_path = (
            spec.manifest.resolve(
                workspace
            )
        )
        manifest_data = (
            _read_bound_bytes(
                manifest_path,
                max_bytes=
                    64 * 1024 * 1024,
                label=
                    f"{label} manifest",
            )
        )
        refs = _speech_references(
            manifest_path,
            manifest_data,
            workspace_root=workspace,
        )
        expected = tuple(
            item.path
            for item in spec.audio
        )
        if refs != expected:
            raise VN97ProductionRunManifestError(
                f"{label} bound audio set does not exactly match manifest references"
            )
        for item in spec.audio:
            item.resolve(
                workspace
            )
        speech_manifests.append(
            manifest_path
        )

    def resolve_relative_dir(
        relative: str,
    ) -> Path:
        target = workspace / relative
        try:
            target.parent.resolve(
                strict=True
            ).relative_to(
                workspace
            )
        except (
            OSError,
            ValueError,
        ) as exc:
            raise VN97ProductionRunManifestError(
                f"run path parent escapes/unavailable: {relative}"
            ) from exc
        return target

    return VN97ResolvedProductionRun(
        workspace_root=workspace,
        campaign_definition=campaign,
        language_training=
            language_groups[0],
        language_validation=
            language_groups[1],
        language_release=
            language_groups[2],
        speech_training_manifest=
            speech_manifests[0],
        speech_validation_manifest=
            speech_manifests[1],
        speech_release_manifest=
            speech_manifests[2],
        device_evidence_dir=
            resolve_relative_dir(
                manifest
                .device_evidence_dir
            ),
        language_output_dir=
            resolve_relative_dir(
                manifest
                .language_output_dir
            ),
        production_output_dir=
            resolve_relative_dir(
                manifest
                .production_output_dir
            ),
        intake_output_dir=
            resolve_relative_dir(
                manifest
                .intake_output_dir
            ),
    )


def options_to_argv(
    options: tuple[
        tuple[str, object],
        ...
    ],
) -> list[str]:
    argv: list[str] = []
    for key, value in options:
        if value is None:
            continue
        flag = "--" + key.replace(
            "_",
            "-",
        )
        if isinstance(value, bool):
            if value:
                argv.append(flag)
            continue
        argv.extend(
            [
                flag,
                str(value),
            ]
        )
    return argv
