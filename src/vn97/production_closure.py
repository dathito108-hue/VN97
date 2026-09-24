from __future__ import annotations

from contextlib import redirect_stdout
from dataclasses import dataclass
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import tempfile
from typing import Callable, Sequence, Any

from .device_evidence import (
    require_device_evidence,
)
from .device_evidence_campaign import (
    VN97PhysicalEvidenceConfig,
    _production_policy,
    collect_physical_device_evidence,
)
from .production_intake import (
    inspect_device_evidence_files,
    inspect_language_campaign_directory,
    inspect_production_campaign_directory,
    parse_production_intake_report,
)
from .production_readiness import (
    VN97ProductionReadinessReport,
    evaluate_production_readiness,
    parse_production_readiness_report,
)
from .production_run_manifest import (
    VN97ProductionRunManifest,
    load_production_run_manifest,
    verify_production_run_inputs,
)
from .release_candidate import (
    VN97LoadedReleaseCandidate,
    load_release_candidate_directory,
)


SCHEMA = "VN97CLOSE1"
MAX_REPORT_BYTES = 256 * 1024

PHASES = {
    "BLOCKED_ENVIRONMENT",
    "NEEDS_TRAINING",
    "NEEDS_PHYSICAL_EVIDENCE",
    "NEEDS_INTAKE",
    "NEEDS_RELEASE_INPUTS",
    "NEEDS_SIGNING",
    "READINESS_BLOCKED",
    "READY_TO_RELEASE",
}

_SIGNING_BLOCKER_CODES = {
    "publisher_private_key.missing",
    "publisher_private_key.invalid",
    "publisher_private_key.inside_repository",
    "android_signing_environment.missing",
    "android_keystore.missing",
    "android_keystore.invalid",
    "android_keystore.inside_repository",
}

_RELEASE_INPUT_BLOCKER_CODES = {
    "candidate.missing",
    "candidate.invalid",
    "release_metadata.invalid",
    "quality_thresholds.invalid",
    "language_validation.missing",
    "language_validation.invalid",
    "speech_validation.missing",
    "speech_validation.invalid",
    "speech_validation.unexpected",
    "vision_validation.missing",
    "vision_validation.invalid",
    "vision_validation.unexpected",
    "output_directory.missing",
    "output_directory.exists",
    "output_directory.parent_invalid",
}


class VN97ProductionClosureError(
    RuntimeError
):
    pass


def _require_sha256(
    value: str | None,
    *,
    label: str,
    allow_none: bool = True,
) -> str | None:
    if value is None and allow_none:
        return None
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(
            ch not in "0123456789abcdef"
            for ch in value
        )
    ):
        raise VN97ProductionClosureError(
            f"{label} must be lowercase SHA-256"
        )
    return value


def _require_commit(
    value: str,
) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(
            ch not in "0123456789abcdef"
            for ch in value
        )
    ):
        raise VN97ProductionClosureError(
            "repository commit must be lowercase 40-hex"
        )


@dataclass(frozen=True)
class VN97ProductionClosureReport:
    manifest_sha256: str
    repository_commit: str
    phase: str
    environment_ready: bool
    language_campaign_report_sha256: str | None
    production_campaign_report_sha256: str | None
    device_evidence_sha256: tuple[str, ...]
    intake_report_sha256: str | None
    release_candidate_manifest_sha256: str | None
    readiness_report_sha256: str | None
    readiness_status: str | None
    readiness_blocker_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_sha256(
            self.manifest_sha256,
            label="manifest SHA-256",
            allow_none=False,
        )
        _require_commit(
            self.repository_commit
        )
        if self.phase not in PHASES:
            raise VN97ProductionClosureError(
                "closure phase is invalid"
            )
        if (
            type(self.environment_ready)
            is not bool
        ):
            raise VN97ProductionClosureError(
                "environment_ready must be boolean"
            )
        for value, label in (
            (
                self.language_campaign_report_sha256,
                "language campaign report SHA-256",
            ),
            (
                self.production_campaign_report_sha256,
                "production campaign report SHA-256",
            ),
            (
                self.intake_report_sha256,
                "intake report SHA-256",
            ),
            (
                self.release_candidate_manifest_sha256,
                "release candidate manifest SHA-256",
            ),
            (
                self.readiness_report_sha256,
                "readiness report SHA-256",
            ),
        ):
            _require_sha256(
                value,
                label=label,
            )

        if (
            tuple(
                sorted(
                    set(
                        self.device_evidence_sha256
                    )
                )
            )
            != self.device_evidence_sha256
        ):
            raise VN97ProductionClosureError(
                "device evidence SHA list must be unique and sorted"
            )
        for value in (
            self.device_evidence_sha256
        ):
            _require_sha256(
                value,
                label="device evidence SHA-256",
                allow_none=False,
            )

        if (
            self.readiness_status
            not in {None, "READY", "BLOCKED"}
        ):
            raise VN97ProductionClosureError(
                "readiness_status is invalid"
            )
        if (
            tuple(
                sorted(
                    set(
                        self.readiness_blocker_codes
                    )
                )
            )
            != self.readiness_blocker_codes
        ):
            raise VN97ProductionClosureError(
                "readiness blocker codes must be unique and sorted"
            )
        for code in (
            self.readiness_blocker_codes
        ):
            if (
                not isinstance(code, str)
                or not code
                or len(code) > 256
                or any(
                    ord(ch) < 0x20
                    for ch in code
                )
            ):
                raise VN97ProductionClosureError(
                    "readiness blocker code is invalid"
                )

        if (
            self.readiness_report_sha256
            is None
        ) != (
            self.readiness_status is None
        ):
            raise VN97ProductionClosureError(
                "readiness report/status presence mismatch"
            )
        if (
            self.readiness_status == "READY"
            and self.readiness_blocker_codes
        ):
            raise VN97ProductionClosureError(
                "READY closure cannot retain readiness blockers"
            )
        if (
            self.readiness_status == "BLOCKED"
            and not self.readiness_blocker_codes
        ):
            raise VN97ProductionClosureError(
                "BLOCKED readiness requires blocker codes"
            )
        if (
            self.phase == "READY_TO_RELEASE"
            and self.readiness_status != "READY"
        ):
            raise VN97ProductionClosureError(
                "READY_TO_RELEASE requires READY VN97READY1"
            )
        if (
            self.phase
            in {
                "NEEDS_RELEASE_INPUTS",
                "NEEDS_SIGNING",
                "READINESS_BLOCKED",
            }
            and self.readiness_status
            != "BLOCKED"
        ):
            raise VN97ProductionClosureError(
                "readiness dependency phases require BLOCKED VN97READY1"
            )
        if (
            self.phase
            in {
                "NEEDS_RELEASE_INPUTS",
                "NEEDS_SIGNING",
                "READINESS_BLOCKED",
                "READY_TO_RELEASE",
            }
            and (
                self.language_campaign_report_sha256
                is None
                or self.production_campaign_report_sha256
                is None
                or not self.device_evidence_sha256
                or self.intake_report_sha256
                is None
                or self.release_candidate_manifest_sha256
                is None
                or self.readiness_report_sha256
                is None
            )
        ):
            raise VN97ProductionClosureError(
                "post-intake closure phase lacks complete artifact/readiness chain"
            )
        if (
            self.phase == "NEEDS_INTAKE"
            and (
                self.language_campaign_report_sha256
                is None
                or self.production_campaign_report_sha256
                is None
                or not self.device_evidence_sha256
                or self.intake_report_sha256
                is not None
                or self.release_candidate_manifest_sha256
                is not None
                or self.readiness_report_sha256
                is not None
            )
        ):
            raise VN97ProductionClosureError(
                "NEEDS_INTAKE artifact chain is inconsistent"
            )
        if (
            self.phase
            == "NEEDS_PHYSICAL_EVIDENCE"
            and (
                self.language_campaign_report_sha256
                is None
                or self.production_campaign_report_sha256
                is None
                or self.device_evidence_sha256
                or self.intake_report_sha256
                is not None
                or self.release_candidate_manifest_sha256
                is not None
                or self.readiness_report_sha256
                is not None
            )
        ):
            raise VN97ProductionClosureError(
                "NEEDS_PHYSICAL_EVIDENCE artifact chain is inconsistent"
            )

    def canonical_object(
        self,
    ) -> dict[str, object]:
        return {
            "device_evidence_sha256":
                list(
                    self
                    .device_evidence_sha256
                ),
            "environment_ready":
                self.environment_ready,
            "intake_report_sha256":
                self.intake_report_sha256,
            "language_campaign_report_sha256":
                self
                .language_campaign_report_sha256,
            "manifest_sha256":
                self.manifest_sha256,
            "phase":
                self.phase,
            "production_campaign_report_sha256":
                self
                .production_campaign_report_sha256,
            "readiness_blocker_codes":
                list(
                    self
                    .readiness_blocker_codes
                ),
            "readiness_report_sha256":
                self
                .readiness_report_sha256,
            "readiness_status":
                self.readiness_status,
            "release_candidate_manifest_sha256":
                self
                .release_candidate_manifest_sha256,
            "repository_commit":
                self.repository_commit,
            "schema": SCHEMA,
        }

    def to_bytes(self) -> bytes:
        return json.dumps(
            self.canonical_object(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")


def parse_production_closure_report(
    data: bytes,
) -> VN97ProductionClosureReport:
    if not 0 < len(data) <= MAX_REPORT_BYTES:
        raise VN97ProductionClosureError(
            "VN97CLOSE1 byte size is outside bounds"
        )
    duplicates: list[str] = []

    def hook(
        pairs: list[
            tuple[str, Any]
        ],
    ) -> dict[str, Any]:
        output: dict[
            str,
            Any,
        ] = {}
        for key, value in pairs:
            if key in output:
                duplicates.append(key)
            output[key] = value
        return output

    try:
        text = data.decode(
            "utf-8",
            errors="strict",
        )
        root = json.loads(
            text,
            object_pairs_hook=hook,
            parse_constant=lambda raw:
                (_ for _ in ())
                .throw(
                    ValueError(raw)
                ),
        )
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        ValueError,
    ) as exc:
        raise VN97ProductionClosureError(
            "VN97CLOSE1 must be strict UTF-8 JSON"
        ) from exc

    if duplicates or not isinstance(
        root,
        dict,
    ):
        raise VN97ProductionClosureError(
            "VN97CLOSE1 must be one object without duplicate keys"
        )
    canonical = json.dumps(
        root,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    if canonical != data:
        raise VN97ProductionClosureError(
            "VN97CLOSE1 must use canonical JSON"
        )

    expected = {
        "device_evidence_sha256",
        "environment_ready",
        "intake_report_sha256",
        "language_campaign_report_sha256",
        "manifest_sha256",
        "phase",
        "production_campaign_report_sha256",
        "readiness_blocker_codes",
        "readiness_report_sha256",
        "readiness_status",
        "release_candidate_manifest_sha256",
        "repository_commit",
        "schema",
    }
    if (
        set(root) != expected
        or root["schema"] != SCHEMA
    ):
        raise VN97ProductionClosureError(
            "VN97CLOSE1 schema/keys mismatch"
        )
    if not isinstance(
        root[
            "device_evidence_sha256"
        ],
        list,
    ):
        raise VN97ProductionClosureError(
            "VN97CLOSE1 device evidence list is invalid"
        )
    if not isinstance(
        root[
            "readiness_blocker_codes"
        ],
        list,
    ):
        raise VN97ProductionClosureError(
            "VN97CLOSE1 readiness blocker list is invalid"
        )
    try:
        return VN97ProductionClosureReport(
            manifest_sha256=
                root[
                    "manifest_sha256"
                ],
            repository_commit=
                root[
                    "repository_commit"
                ],
            phase=root["phase"],
            environment_ready=
                root[
                    "environment_ready"
                ],
            language_campaign_report_sha256=
                root[
                    "language_campaign_report_sha256"
                ],
            production_campaign_report_sha256=
                root[
                    "production_campaign_report_sha256"
                ],
            device_evidence_sha256=
                tuple(
                    root[
                        "device_evidence_sha256"
                    ]
                ),
            intake_report_sha256=
                root[
                    "intake_report_sha256"
                ],
            release_candidate_manifest_sha256=
                root[
                    "release_candidate_manifest_sha256"
                ],
            readiness_report_sha256=
                root[
                    "readiness_report_sha256"
                ],
            readiness_status=
                root[
                    "readiness_status"
                ],
            readiness_blocker_codes=
                tuple(
                    root[
                        "readiness_blocker_codes"
                    ]
                ),
        )
    except (
        TypeError,
        ValueError,
        VN97ProductionClosureError,
    ) as exc:
        if isinstance(
            exc,
            VN97ProductionClosureError,
        ):
            raise
        raise VN97ProductionClosureError(
            str(exc)
        ) from exc


@dataclass(frozen=True)
class VN97ReleaseReadinessInputs:
    publisher_private_key: Path | None = None
    validation_inputs: tuple[Path, ...] = tuple()
    speech_validation_input: Path | None = None
    vision_validation_input: Path | None = None
    output_dir: Path | None = None
    key_id: str | None = None
    capability_version: int | None = None
    source_origin: str | None = None
    source_license: str | None = None
    max_validation_loss: float | None = None
    max_speech_validation_loss: float | None = None
    max_vision_validation_loss: float | None = None
    gradle: str = "gradle"
    apksigner: str | None = None
    aapt: str | None = None


@dataclass(frozen=True)
class VN97ClosureObservation:
    language_ready: bool
    production_ready: bool
    evidence_ready: bool
    intake_ready: bool
    readiness: VN97ProductionReadinessReport | None

    def __post_init__(
        self,
    ) -> None:
        for value in (
            self.language_ready,
            self.production_ready,
            self.evidence_ready,
            self.intake_ready,
        ):
            if type(value) is not bool:
                raise ValueError(
                    "closure readiness flags must be boolean"
                )
        if (
            self.production_ready
            and not self.language_ready
        ):
            raise ValueError(
                "production cannot be ready without language"
            )
        if (
            self.intake_ready
            and not (
                self.language_ready
                and self.production_ready
                and self.evidence_ready
            )
        ):
            raise ValueError(
                "intake cannot be ready before language/production/evidence"
            )


def closure_phase(
    observation:
        VN97ClosureObservation,
) -> str:
    if not observation.language_ready:
        return "NEEDS_TRAINING"
    if not observation.production_ready:
        return "NEEDS_TRAINING"
    if not observation.evidence_ready:
        return (
            "NEEDS_PHYSICAL_EVIDENCE"
        )
    if not observation.intake_ready:
        return "NEEDS_INTAKE"
    readiness = observation.readiness
    if readiness is None:
        return "NEEDS_RELEASE_INPUTS"
    if readiness.ready:
        return "READY_TO_RELEASE"

    codes = {
        blocker.code
        for blocker in
            readiness.blockers
    }
    if (
        codes
        and codes
        .issubset(
            _SIGNING_BLOCKER_CODES
        )
    ):
        return "NEEDS_SIGNING"
    if (
        codes
        & _RELEASE_INPUT_BLOCKER_CODES
    ):
        return "NEEDS_RELEASE_INPUTS"
    if (
        codes
        & _SIGNING_BLOCKER_CODES
    ):
        return "NEEDS_SIGNING"
    return "READINESS_BLOCKED"


StageRunner = Callable[
    [list[str]],
    int,
]
EvidenceRunner = Callable[..., object]
ReadinessEvaluator = Callable[
    ...,
    VN97ProductionReadinessReport,
]


def _sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()
    with path.open(
        "rb"
    ) as stream:
        while True:
            chunk = stream.read(
                1024 * 1024
            )
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _verify_closure_source(
    repository_root: Path,
) -> None:
    current = Path(
        __file__
    ).resolve()
    bound = (
        repository_root
        / "src/vn97/production_closure.py"
    )
    if (
        not bound.is_file()
        or _sha256_file(
            current
        )
        != _sha256_file(
            bound
        )
    ):
        raise VN97ProductionClosureError(
            "running M19K code does not match VN97RUN1-bound checkout"
        )


def _path_exists(
    path: Path,
) -> bool:
    return (
        path.exists()
        or path.is_symlink()
    )


def _evidence_paths(
    directory: Path,
) -> tuple[Path, ...]:
    from .production_run_cli import (
        _evidence_paths
        as canonical_evidence_paths,
    )

    return canonical_evidence_paths(
        directory
    )


def _invoke_stage(
    runner: StageRunner,
    argv: list[str],
) -> None:
    captured = io.StringIO()
    with redirect_stdout(
        captured
    ):
        result = runner(argv)
    if result != 0:
        raise VN97ProductionClosureError(
            "canonical production stage returned non-zero"
        )


def _inspect_campaigns(
    *,
    resolved,
) -> tuple[
    object | None,
    object | None,
]:
    language_exists = _path_exists(
        resolved.language_output_dir
    )
    production_exists = _path_exists(
        resolved.production_output_dir
    )
    if (
        production_exists
        and not language_exists
    ):
        raise VN97ProductionClosureError(
            "production output exists without language output"
        )
    language = (
        inspect_language_campaign_directory(
            resolved.language_output_dir
        )
        if language_exists
        else None
    )
    production = (
        inspect_production_campaign_directory(
            resolved.production_output_dir,
            language=language,
        )
        if production_exists
        else None
    )
    return language, production


def _inspect_evidence(
    directory: Path,
    *,
    manifest:
        VN97ProductionRunManifest,
    production,
) -> tuple[
    tuple[Path, ...],
    tuple[str, ...],
]:
    try:
        paths = _evidence_paths(
            directory
        )
    except ValueError:
        return tuple(), tuple()

    if production is None:
        raise VN97ProductionClosureError(
            "physical-device evidence exists before canonical production output"
        )

    evidence = (
        inspect_device_evidence_files(
            list(paths),
            expected_model_image_sha256=
                production
                .model_image_sha256,
        )
    )
    policy = _production_policy(
        manifest
    )
    profiles = set()
    for item in evidence:
        require_device_evidence(
            item,
            policy.criteria,
            expected_model_image_sha256=
                production
                .model_image_sha256,
            speech_enabled=True,
        )
        profiles.add(
            (
                item.manufacturer,
                item.model,
                item.sdk_int,
                item.abi,
            )
        )
    if (
        len(profiles)
        < policy
        .min_distinct_device_profiles
    ):
        raise VN97ProductionClosureError(
            "physical-device evidence does not satisfy VN97RUN1 distinct-profile policy"
        )

    return (
        paths,
        tuple(
            item.evidence_sha256
            for item in evidence
        ),
    )


def _inspect_intake(
    *,
    resolved,
    language,
    production,
    evidence_sha256:
        tuple[str, ...],
) -> tuple[
    object | None,
    VN97LoadedReleaseCandidate | None,
    str | None,
]:
    if not _path_exists(
        resolved.intake_output_dir
    ):
        return None, None, None
    if (
        language is None
        or production is None
        or not evidence_sha256
    ):
        raise VN97ProductionClosureError(
            "intake output exists before prerequisite chain is complete"
        )

    intake_path = (
        resolved.intake_output_dir
        / "production-intake.vn97intake1"
    )
    try:
        data = intake_path.read_bytes()
    except OSError as exc:
        raise VN97ProductionClosureError(
            "VN97INTAKE1 could not be read"
        ) from exc
    intake = (
        parse_production_intake_report(
            data
        )
    )
    candidate = (
        load_release_candidate_directory(
            resolved.intake_output_dir
            / "release-candidate"
        )
    )
    if (
        intake
        .language_campaign_report_sha256
        != language.report_sha256
        or intake
        .production_campaign_report_sha256
        != production.report_sha256
        or tuple(
            intake.device_evidence_sha256
        )
        != evidence_sha256
        or intake
        .release_candidate_manifest_sha256
        != candidate.manifest_sha256
        or candidate.manifest
        .model_image_sha256
        != production.model_image_sha256
    ):
        raise VN97ProductionClosureError(
            "existing intake/candidate does not match canonical production chain"
        )
    return (
        intake,
        candidate,
        hashlib.sha256(
            data
        ).hexdigest(),
    )


def _readiness_report(
    *,
    repository_root: Path,
    candidate:
        VN97LoadedReleaseCandidate,
    inputs:
        VN97ReleaseReadinessInputs,
    evaluator:
        ReadinessEvaluator,
) -> VN97ProductionReadinessReport:
    return evaluator(
        repository_root=
            repository_root,
        release_candidate_dir=
            candidate.root,
        publisher_private_key=
            inputs
            .publisher_private_key,
        validation_inputs=
            inputs.validation_inputs,
        speech_validation_input=
            inputs
            .speech_validation_input,
        vision_validation_input=
            inputs
            .vision_validation_input,
        output_dir=
            inputs.output_dir,
        key_id=inputs.key_id,
        capability_version=
            inputs.capability_version,
        source_origin=
            inputs.source_origin,
        source_license=
            inputs.source_license,
        max_validation_loss=
            inputs
            .max_validation_loss,
        max_speech_validation_loss=
            inputs
            .max_speech_validation_loss,
        max_vision_validation_loss=
            inputs
            .max_vision_validation_loss,
        gradle=inputs.gradle,
        apksigner=
            inputs.apksigner,
        aapt=inputs.aapt,
    )


def _report(
    *,
    manifest:
        VN97ProductionRunManifest,
    phase: str,
    environment_ready: bool,
    language,
    production,
    evidence_sha256:
        tuple[str, ...],
    intake_report_sha256:
        str | None,
    candidate:
        VN97LoadedReleaseCandidate
        | None,
    readiness:
        VN97ProductionReadinessReport
        | None,
) -> VN97ProductionClosureReport:
    readiness_bytes = (
        None
        if readiness is None
        else readiness.to_bytes()
    )
    return VN97ProductionClosureReport(
        manifest_sha256=
            manifest.manifest_sha256,
        repository_commit=
            manifest.repository_commit,
        phase=phase,
        environment_ready=
            environment_ready,
        language_campaign_report_sha256=(
            None
            if language is None
            else language.report_sha256
        ),
        production_campaign_report_sha256=(
            None
            if production is None
            else production.report_sha256
        ),
        device_evidence_sha256=
            evidence_sha256,
        intake_report_sha256=
            intake_report_sha256,
        release_candidate_manifest_sha256=(
            None
            if candidate is None
            else candidate
                .manifest_sha256
        ),
        readiness_report_sha256=(
            None
            if readiness_bytes is None
            else hashlib.sha256(
                readiness_bytes
            ).hexdigest()
        ),
        readiness_status=(
            None
            if readiness is None
            else readiness.status
        ),
        readiness_blocker_codes=(
            tuple()
            if readiness is None
            else tuple(
                sorted(
                    blocker.code
                    for blocker in
                        readiness.blockers
                )
            )
        ),
    )


def _atomic_write_replace(
    path: Path,
    data: bytes,
) -> None:
    if path.is_symlink():
        raise VN97ProductionClosureError(
            "report target must not be a symlink"
        )
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    if path.parent.is_symlink():
        raise VN97ProductionClosureError(
            "report parent must not be a symlink"
        )
    fd, temporary = (
        tempfile.mkstemp(
            prefix=".vn97-m19k-",
            dir=path.parent,
        )
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
        if path.read_bytes() != data:
            raise VN97ProductionClosureError(
                "report post-write verification failed"
            )
    finally:
        if os.path.exists(
            temporary
        ):
            os.unlink(
                temporary
            )


def run_production_closure(
    *,
    manifest_path: Path,
    workspace_root: Path,
    repository_root: Path,
    serials:
        Sequence[str] = tuple(),
    inspect_only: bool = False,
    evidence_config:
        VN97PhysicalEvidenceConfig
        | None = None,
    release_inputs:
        VN97ReleaseReadinessInputs
        | None = None,
    closure_report_path:
        Path | None = None,
    readiness_report_path:
        Path | None = None,
    adb_executable: str = "adb",
    gradle_executable: str = "gradle",
    stage_runner:
        StageRunner | None = None,
    evidence_runner:
        EvidenceRunner | None = None,
    readiness_evaluator:
        ReadinessEvaluator | None = None,
) -> VN97ProductionClosureReport:
    from .production_run_cli import (
        _environment_ready,
        _verify_repository,
        main as production_run_main,
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
    repository = (
        _verify_repository(
            repository_root,
            expected_commit=
                manifest.repository_commit,
        )
    )
    _verify_closure_source(
        repository
    )
    environment_ready = (
        _environment_ready(
            manifest
        )
    )

    runner = (
        stage_runner
        or production_run_main
    )
    evidence_call = (
        evidence_runner
        or collect_physical_device_evidence
    )
    readiness_call = (
        readiness_evaluator
        or evaluate_production_readiness
    )

    language, production = (
        _inspect_campaigns(
            resolved=resolved
        )
    )
    _, evidence_sha = (
        _inspect_evidence(
            resolved.device_evidence_dir,
            manifest=manifest,
            production=production,
        )
    )
    (
        intake,
        candidate,
        intake_sha,
    ) = _inspect_intake(
        resolved=resolved,
        language=language,
        production=production,
        evidence_sha256=
            evidence_sha,
    )

    if not environment_ready:
        report = _report(
            manifest=manifest,
            phase="BLOCKED_ENVIRONMENT",
            environment_ready=False,
            language=language,
            production=production,
            evidence_sha256=
                evidence_sha,
            intake_report_sha256=
                intake_sha,
            candidate=candidate,
            readiness=None,
        )
        if closure_report_path:
            _atomic_write_replace(
                closure_report_path,
                report.to_bytes(),
            )
        return report

    stage_base = [
        "--manifest",
        str(manifest_path),
        "--workspace-root",
        str(workspace_root),
        "--repository-root",
        str(repository),
    ]

    if language is None:
        if inspect_only:
            report = _report(
                manifest=manifest,
                phase="NEEDS_TRAINING",
                environment_ready=True,
                language=None,
                production=None,
                evidence_sha256=
                    evidence_sha,
                intake_report_sha256=None,
                candidate=None,
                readiness=None,
            )
            if closure_report_path:
                _atomic_write_replace(
                    closure_report_path,
                    report.to_bytes(),
                )
            return report
        _invoke_stage(
            runner,
            [
                *stage_base,
                "--stage",
                "train",
            ],
        )
        language, production = (
            _inspect_campaigns(
                resolved=resolved
            )
        )

    elif production is None:
        if inspect_only:
            report = _report(
                manifest=manifest,
                phase="NEEDS_TRAINING",
                environment_ready=True,
                language=language,
                production=None,
                evidence_sha256=
                    evidence_sha,
                intake_report_sha256=None,
                candidate=None,
                readiness=None,
            )
            if closure_report_path:
                _atomic_write_replace(
                    closure_report_path,
                    report.to_bytes(),
                )
            return report
        _invoke_stage(
            runner,
            [
                *stage_base,
                "--stage",
                "production",
            ],
        )
        language, production = (
            _inspect_campaigns(
                resolved=resolved
            )
        )

    if (
        language is None
        or production is None
    ):
        raise VN97ProductionClosureError(
            "canonical training stage did not produce complete campaign chain"
        )

    _, evidence_sha = (
        _inspect_evidence(
            resolved.device_evidence_dir,
            manifest=manifest,
            production=production,
        )
    )
    if not evidence_sha:
        if (
            inspect_only
            or not serials
        ):
            report = _report(
                manifest=manifest,
                phase=
                    "NEEDS_PHYSICAL_EVIDENCE",
                environment_ready=True,
                language=language,
                production=production,
                evidence_sha256=tuple(),
                intake_report_sha256=None,
                candidate=None,
                readiness=None,
            )
            if closure_report_path:
                _atomic_write_replace(
                    closure_report_path,
                    report.to_bytes(),
                )
            return report

        evidence_call(
            manifest_path=
                manifest_path,
            workspace_root=
                workspace_root,
            repository_root=
                repository,
            serials=tuple(
                serials
            ),
            config=evidence_config,
            adb_executable=
                adb_executable,
            gradle_executable=
                gradle_executable,
        )
        _, evidence_sha = (
            _inspect_evidence(
                resolved
                .device_evidence_dir,
                manifest=manifest,
                production=production,
            )
        )
        if not evidence_sha:
            raise VN97ProductionClosureError(
                "M19J completed without physical evidence"
            )

    (
        intake,
        candidate,
        intake_sha,
    ) = _inspect_intake(
        resolved=resolved,
        language=language,
        production=production,
        evidence_sha256=
            evidence_sha,
    )
    if intake is None:
        if inspect_only:
            report = _report(
                manifest=manifest,
                phase="NEEDS_INTAKE",
                environment_ready=True,
                language=language,
                production=production,
                evidence_sha256=
                    evidence_sha,
                intake_report_sha256=None,
                candidate=None,
                readiness=None,
            )
            if closure_report_path:
                _atomic_write_replace(
                    closure_report_path,
                    report.to_bytes(),
                )
            return report
        _invoke_stage(
            runner,
            [
                *stage_base,
                "--stage",
                "intake",
            ],
        )
        (
            intake,
            candidate,
            intake_sha,
        ) = _inspect_intake(
            resolved=resolved,
            language=language,
            production=production,
            evidence_sha256=
                evidence_sha,
        )

    if (
        intake is None
        or candidate is None
        or intake_sha is None
    ):
        raise VN97ProductionClosureError(
            "canonical intake stage did not produce VN97INTAKE1/VN97RC1"
        )

    readiness_inputs = (
        release_inputs
        or VN97ReleaseReadinessInputs()
    )
    readiness = _readiness_report(
        repository_root=
            repository,
        candidate=candidate,
        inputs=readiness_inputs,
        evaluator=
            readiness_call,
    )
    phase = closure_phase(
        VN97ClosureObservation(
            language_ready=True,
            production_ready=True,
            evidence_ready=True,
            intake_ready=True,
            readiness=readiness,
        )
    )
    report = _report(
        manifest=manifest,
        phase=phase,
        environment_ready=True,
        language=language,
        production=production,
        evidence_sha256=
            evidence_sha,
        intake_report_sha256=
            intake_sha,
        candidate=candidate,
        readiness=readiness,
    )

    if readiness_report_path:
        readiness_bytes = (
            readiness.to_bytes()
        )
        _atomic_write_replace(
            readiness_report_path,
            readiness_bytes,
        )
        parsed = (
            parse_production_readiness_report(
                readiness_report_path
                .read_bytes()
            )
        )
        if parsed != readiness:
            raise VN97ProductionClosureError(
                "persisted VN97READY1 does not round-trip"
            )

    if closure_report_path:
        _atomic_write_replace(
            closure_report_path,
            report.to_bytes(),
        )
        parsed = (
            parse_production_closure_report(
                closure_report_path
                .read_bytes()
            )
        )
        if parsed != report:
            raise VN97ProductionClosureError(
                "persisted VN97CLOSE1 does not round-trip"
            )
    return report
