from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from .deployment_checkpoint import (
    VN97LoadedDeploymentCheckpoint,
    load_deployment_checkpoint_file,
)
from .tokenizer import (
    VN97TokenizerPackage,
)
from .training_cli import (
    _read_bounded_regular_file,
)


_P4C_FILES = {
    "SHA256SUMS",
    "model.vn97ck1",
    "model.vn97mi1",
    "p4c-report.json",
    "tokenizer.vn97tk1",
}
_P4C_HASHED_FILES = (
    _P4C_FILES
    - {"SHA256SUMS"}
)
_MAX_MODEL_BYTES = (
    512 * 1024 * 1024
)
_MAX_JSON_BYTES = (
    16 * 1024 * 1024
)


class VN97P4ArtifactError(RuntimeError):
    pass


def _sha256(data: bytes) -> str:
    return hashlib.sha256(
        data
    ).hexdigest()


def _canonical_json(
    value: object,
) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _read(
    path: Path,
    *,
    max_bytes: int,
) -> bytes:
    try:
        return _read_bounded_regular_file(
            path,
            max_bytes=max_bytes,
        )
    except (OSError, ValueError) as exc:
        raise VN97P4ArtifactError(
            f"could not safely read {path.name}"
        ) from exc


def _require_sha256(
    value: object,
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
        raise VN97P4ArtifactError(
            f"{label} must be lowercase SHA-256"
        )
    return value


def _parse_sums(
    data: bytes,
) -> dict[str, str]:
    try:
        text = data.decode(
            "ascii",
            errors="strict",
        )
    except UnicodeDecodeError as exc:
        raise VN97P4ArtifactError(
            "P4C SHA256SUMS must be ASCII"
        ) from exc
    if not text.endswith("\n"):
        raise VN97P4ArtifactError(
            "P4C SHA256SUMS must end with newline"
        )

    result: dict[str, str] = {}
    for line in text.splitlines():
        if (
            len(line) < 67
            or line[64:66] != "  "
        ):
            raise VN97P4ArtifactError(
                "P4C SHA256SUMS line is malformed"
            )
        digest = _require_sha256(
            line[:64],
            label="P4C SHA256SUMS digest",
        )
        name = line[66:]
        if (
            not name
            or "/" in name
            or "\\" in name
            or name in result
        ):
            raise VN97P4ArtifactError(
                "P4C SHA256SUMS filename is invalid"
            )
        result[name] = digest

    if set(result) != _P4C_HASHED_FILES:
        raise VN97P4ArtifactError(
            "P4C SHA256SUMS file set mismatch"
        )
    return result


@dataclass(frozen=True)
class VN97P4CArtifact:
    root: Path
    report_sha256: str
    checkpoint_sha256: str
    tokenizer_sha256: str
    model_image_sha256: str
    parent_p3_checkpoint_sha256: str
    parent_p3_corpus_manifest_id: str
    parent_p3_corpus_manifest_sha256: str
    checkpoint: VN97LoadedDeploymentCheckpoint
    tokenizer_package: VN97TokenizerPackage
    report: dict[str, object]


def verify_p4c_artifact(
    root: Path,
) -> VN97P4CArtifact:
    if root.is_symlink():
        raise VN97P4ArtifactError(
            "P4C root must not be a symlink"
        )
    try:
        resolved = root.resolve(
            strict=True
        )
    except FileNotFoundError as exc:
        raise VN97P4ArtifactError(
            "P4C root does not exist"
        ) from exc
    if not resolved.is_dir():
        raise VN97P4ArtifactError(
            "P4C root must be a directory"
        )
    names = {
        item.name
        for item
        in resolved.iterdir()
    }
    if names != _P4C_FILES:
        raise VN97P4ArtifactError(
            "P4C artifact file set mismatch"
        )
    if any(
        (resolved / name).is_symlink()
        for name in names
    ):
        raise VN97P4ArtifactError(
            "P4C artifact must not contain symlinks"
        )

    sums = _parse_sums(
        _read(
            resolved / "SHA256SUMS",
            max_bytes=64 * 1024,
        )
    )
    raw: dict[str, bytes] = {}
    for name in sorted(
        _P4C_HASHED_FILES
    ):
        limit = (
            _MAX_MODEL_BYTES
            if name in {
                "model.vn97ck1",
                "model.vn97mi1",
            }
            else _MAX_JSON_BYTES
        )
        data = _read(
            resolved / name,
            max_bytes=limit,
        )
        if _sha256(data) != sums[name]:
            raise VN97P4ArtifactError(
                f"P4C SHA256SUMS mismatch: {name}"
            )
        raw[name] = data

    try:
        text = raw[
            "p4c-report.json"
        ].decode(
            "utf-8",
            errors="strict",
        )
        body = (
            text[:-1]
            if text.endswith("\n")
            else text
        )
        report = json.loads(
            body,
        )
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
    ) as exc:
        raise VN97P4ArtifactError(
            "P4C report must be UTF-8 JSON"
        ) from exc

    if (
        not isinstance(report, dict)
        or report.get("schema")
        != "VN97P4C1"
        or report.get("status")
        != "ELIGIBLE"
    ):
        raise VN97P4ArtifactError(
            "P4C report schema/status mismatch"
        )
    canonical = (
        _canonical_json(report)
        + (
            b"\n"
            if text.endswith("\n")
            else b""
        )
    )
    if canonical != raw[
        "p4c-report.json"
    ]:
        raise VN97P4ArtifactError(
            "P4C report must use canonical JSON"
        )

    checkpoint_sha = (
        _require_sha256(
            report.get(
                "output_checkpoint_sha256"
            ),
            label="P4C checkpoint SHA-256",
        )
    )
    tokenizer_sha = (
        _require_sha256(
            report.get(
                "tokenizer_sha256"
            ),
            label="P4C tokenizer SHA-256",
        )
    )
    model_image_sha = (
        _require_sha256(
            report.get(
                "model_image_sha256"
            ),
            label="P4C model image SHA-256",
        )
    )

    if (
        _sha256(
            raw["tokenizer.vn97tk1"]
        )
        != tokenizer_sha
        or _sha256(
            raw["model.vn97mi1"]
        )
        != model_image_sha
    ):
        raise VN97P4ArtifactError(
            "P4C report output identity mismatch"
        )

    checkpoint = (
        load_deployment_checkpoint_file(
            resolved / "model.vn97ck1"
        )
    )
    if (
        checkpoint.checkpoint_sha256
        != checkpoint_sha
    ):
        raise VN97P4ArtifactError(
            "P4C checkpoint identity mismatch"
        )

    try:
        tokenizer_package = (
            VN97TokenizerPackage.from_bytes(
                raw["tokenizer.vn97tk1"]
            )
        )
    except (TypeError, ValueError) as exc:
        raise VN97P4ArtifactError(
            "P4C tokenizer package is invalid"
        ) from exc
    if (
        tokenizer_package.vocab_size
        != checkpoint.config.vocab_size
    ):
        raise VN97P4ArtifactError(
            "P4C checkpoint/tokenizer vocabulary mismatch"
        )

    parent = report.get(
        "parent_p3"
    )
    if not isinstance(
        parent,
        dict,
    ):
        raise VN97P4ArtifactError(
            "P4C parent_p3 is invalid"
        )
    parent_checkpoint = _require_sha256(
        parent.get(
            "checkpoint_sha256"
        ),
        label="P4C parent P3 checkpoint SHA-256",
    )
    parent_manifest_id = _require_sha256(
        parent.get(
            "corpus_manifest_id"
        ),
        label="P4C parent P3 corpus manifest ID",
    )
    parent_manifest_sha = _require_sha256(
        parent.get(
            "corpus_manifest_sha256"
        ),
        label="P4C parent P3 corpus manifest SHA-256",
    )

    return VN97P4CArtifact(
        root=resolved,
        report_sha256=_sha256(
            raw["p4c-report.json"]
        ),
        checkpoint_sha256=
            checkpoint_sha,
        tokenizer_sha256=
            tokenizer_sha,
        model_image_sha256=
            model_image_sha,
        parent_p3_checkpoint_sha256=
            parent_checkpoint,
        parent_p3_corpus_manifest_id=
            parent_manifest_id,
        parent_p3_corpus_manifest_sha256=
            parent_manifest_sha,
        checkpoint=checkpoint,
        tokenizer_package=
            tokenizer_package,
        report=report,
    )


_P4D_FILES = {
    "SHA256SUMS",
    "model.vn97ck1",
    "model.vn97mi1",
    "p4d-report.json",
    "tokenizer.vn97tk1",
}
_P4D_HASHED_FILES = (
    _P4D_FILES
    - {"SHA256SUMS"}
)
_P4D_STATUSES = {
    "ELIGIBLE",
    "REJECTED_VALIDATION",
    "REJECTED_RETENTION",
    "REJECTED_GENERALIZATION",
}


def _parse_p4d_sums(
    data: bytes,
) -> dict[str, str]:
    try:
        text = data.decode(
            "ascii",
            errors="strict",
        )
    except UnicodeDecodeError as exc:
        raise VN97P4ArtifactError(
            "P4D SHA256SUMS must be ASCII"
        ) from exc
    if not text.endswith("\n"):
        raise VN97P4ArtifactError(
            "P4D SHA256SUMS must end with newline"
        )

    result: dict[str, str] = {}
    for line in text.splitlines():
        if (
            len(line) < 67
            or line[64:66] != "  "
        ):
            raise VN97P4ArtifactError(
                "P4D SHA256SUMS line is malformed"
            )
        digest = _require_sha256(
            line[:64],
            label="P4D SHA256SUMS digest",
        )
        name = line[66:]
        if (
            not name
            or "/" in name
            or "\\" in name
            or name in result
        ):
            raise VN97P4ArtifactError(
                "P4D SHA256SUMS filename is invalid"
            )
        result[name] = digest

    if set(result) != _P4D_HASHED_FILES:
        raise VN97P4ArtifactError(
            "P4D SHA256SUMS file set mismatch"
        )
    return result


@dataclass(frozen=True)
class VN97P4DArtifact:
    root: Path
    report_sha256: str
    checkpoint_sha256: str
    tokenizer_sha256: str
    model_image_sha256: str
    status: str
    checkpoint: VN97LoadedDeploymentCheckpoint
    tokenizer_package: VN97TokenizerPackage
    report: dict[str, object]


def verify_p4d_artifact(
    root: Path,
) -> VN97P4DArtifact:
    if root.is_symlink():
        raise VN97P4ArtifactError(
            "P4D root must not be a symlink"
        )
    try:
        resolved = root.resolve(
            strict=True
        )
    except FileNotFoundError as exc:
        raise VN97P4ArtifactError(
            "P4D root does not exist"
        ) from exc
    if not resolved.is_dir():
        raise VN97P4ArtifactError(
            "P4D root must be a directory"
        )

    names = {
        item.name
        for item in resolved.iterdir()
    }
    if names != _P4D_FILES:
        raise VN97P4ArtifactError(
            "P4D artifact file set mismatch"
        )
    if any(
        (resolved / name).is_symlink()
        for name in names
    ):
        raise VN97P4ArtifactError(
            "P4D artifact must not contain symlinks"
        )

    sums = _parse_p4d_sums(
        _read(
            resolved / "SHA256SUMS",
            max_bytes=64 * 1024,
        )
    )
    raw: dict[str, bytes] = {}
    for name in sorted(
        _P4D_HASHED_FILES
    ):
        limit = (
            _MAX_MODEL_BYTES
            if name in {
                "model.vn97ck1",
                "model.vn97mi1",
            }
            else _MAX_JSON_BYTES
        )
        data = _read(
            resolved / name,
            max_bytes=limit,
        )
        if _sha256(data) != sums[name]:
            raise VN97P4ArtifactError(
                f"P4D SHA256SUMS mismatch: {name}"
            )
        raw[name] = data

    try:
        text = raw[
            "p4d-report.json"
        ].decode(
            "utf-8",
            errors="strict",
        )
        body = (
            text[:-1]
            if text.endswith("\n")
            else text
        )
        report = json.loads(body)
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
    ) as exc:
        raise VN97P4ArtifactError(
            "P4D report must be UTF-8 JSON"
        ) from exc

    status = report.get("status") if isinstance(
        report,
        dict,
    ) else None
    if (
        not isinstance(report, dict)
        or report.get("schema")
        != "VN97P4D1"
        or status not in _P4D_STATUSES
    ):
        raise VN97P4ArtifactError(
            "P4D report schema/status mismatch"
        )
    canonical = (
        _canonical_json(report)
        + (
            b"\n"
            if text.endswith("\n")
            else b""
        )
    )
    if canonical != raw[
        "p4d-report.json"
    ]:
        raise VN97P4ArtifactError(
            "P4D report must use canonical JSON"
        )

    checkpoint_sha = _require_sha256(
        report.get(
            "output_checkpoint_sha256"
        ),
        label="P4D checkpoint SHA-256",
    )
    tokenizer_sha = _require_sha256(
        report.get(
            "tokenizer_sha256"
        ),
        label="P4D tokenizer SHA-256",
    )
    model_image_sha = _require_sha256(
        report.get(
            "model_image_sha256"
        ),
        label="P4D model image SHA-256",
    )

    if (
        _sha256(
            raw["tokenizer.vn97tk1"]
        )
        != tokenizer_sha
        or _sha256(
            raw["model.vn97mi1"]
        )
        != model_image_sha
    ):
        raise VN97P4ArtifactError(
            "P4D report output identity mismatch"
        )

    checkpoint = (
        load_deployment_checkpoint_file(
            resolved / "model.vn97ck1"
        )
    )
    if (
        checkpoint.checkpoint_sha256
        != checkpoint_sha
    ):
        raise VN97P4ArtifactError(
            "P4D checkpoint identity mismatch"
        )

    try:
        tokenizer_package = (
            VN97TokenizerPackage.from_bytes(
                raw["tokenizer.vn97tk1"]
            )
        )
    except (TypeError, ValueError) as exc:
        raise VN97P4ArtifactError(
            "P4D tokenizer package is invalid"
        ) from exc
    if (
        tokenizer_package.vocab_size
        != checkpoint.config.vocab_size
    ):
        raise VN97P4ArtifactError(
            "P4D checkpoint/tokenizer vocabulary mismatch"
        )

    return VN97P4DArtifact(
        root=resolved,
        report_sha256=_sha256(
            raw["p4d-report.json"]
        ),
        checkpoint_sha256=
            checkpoint_sha,
        tokenizer_sha256=
            tokenizer_sha,
        model_image_sha256=
            model_image_sha,
        status=str(status),
        checkpoint=checkpoint,
        tokenizer_package=
            tokenizer_package,
        report=report,
    )


_P4E_C_FILES = {
    "SHA256SUMS",
    "model.vn97ck1",
    "model.vn97mi1",
    "p4e-report.json",
    "tokenizer.vn97tk1",
}
_P4E_C_HASHED_FILES = (
    _P4E_C_FILES
    - {"SHA256SUMS"}
)
_P4E_C_STATUSES = {
    "ELIGIBLE",
    "REJECTED_VALIDATION",
    "REJECTED_RETENTION",
    "REJECTED_RAW_GENERALIZATION",
    "REJECTED_BOUNDARY_REGRESSION",
}


def _parse_p4e_c_sums(
    data: bytes,
) -> dict[str, str]:
    try:
        text = data.decode(
            "ascii",
            errors="strict",
        )
    except UnicodeDecodeError as exc:
        raise VN97P4ArtifactError(
            "P4E-C SHA256SUMS must be ASCII"
        ) from exc
    if not text.endswith("\n"):
        raise VN97P4ArtifactError(
            "P4E-C SHA256SUMS must end with newline"
        )

    result: dict[str, str] = {}
    for line in text.splitlines():
        if (
            len(line) < 67
            or line[64:66] != "  "
        ):
            raise VN97P4ArtifactError(
                "P4E-C SHA256SUMS line is malformed"
            )
        digest = _require_sha256(
            line[:64],
            label="P4E-C SHA256SUMS digest",
        )
        name = line[66:]
        if (
            not name
            or "/" in name
            or "\\" in name
            or name in result
        ):
            raise VN97P4ArtifactError(
                "P4E-C SHA256SUMS filename is invalid"
            )
        result[name] = digest

    if set(result) != _P4E_C_HASHED_FILES:
        raise VN97P4ArtifactError(
            "P4E-C SHA256SUMS file set mismatch"
        )
    return result


@dataclass(frozen=True)
class VN97P4ECArtifact:
    root: Path
    report_sha256: str
    checkpoint_sha256: str
    tokenizer_sha256: str
    model_image_sha256: str
    status: str
    checkpoint: VN97LoadedDeploymentCheckpoint
    tokenizer_package: VN97TokenizerPackage
    report: dict[str, object]


def verify_p4e_c_artifact(
    root: Path,
) -> VN97P4ECArtifact:
    if root.is_symlink():
        raise VN97P4ArtifactError(
            "P4E-C root must not be a symlink"
        )
    try:
        resolved = root.resolve(
            strict=True
        )
    except FileNotFoundError as exc:
        raise VN97P4ArtifactError(
            "P4E-C root does not exist"
        ) from exc
    if not resolved.is_dir():
        raise VN97P4ArtifactError(
            "P4E-C root must be a directory"
        )

    names = {
        item.name
        for item in resolved.iterdir()
    }
    if names != _P4E_C_FILES:
        raise VN97P4ArtifactError(
            "P4E-C artifact file set mismatch"
        )
    if any(
        (resolved / name).is_symlink()
        for name in names
    ):
        raise VN97P4ArtifactError(
            "P4E-C artifact must not contain symlinks"
        )

    sums = _parse_p4e_c_sums(
        _read(
            resolved / "SHA256SUMS",
            max_bytes=64 * 1024,
        )
    )
    raw: dict[str, bytes] = {}
    for name in sorted(
        _P4E_C_HASHED_FILES
    ):
        limit = (
            _MAX_MODEL_BYTES
            if name in {
                "model.vn97ck1",
                "model.vn97mi1",
            }
            else _MAX_JSON_BYTES
        )
        data = _read(
            resolved / name,
            max_bytes=limit,
        )
        if _sha256(data) != sums[name]:
            raise VN97P4ArtifactError(
                f"P4E-C SHA256SUMS mismatch: {name}"
            )
        raw[name] = data

    try:
        text = raw[
            "p4e-report.json"
        ].decode(
            "utf-8",
            errors="strict",
        )
        body = (
            text[:-1]
            if text.endswith("\n")
            else text
        )
        report = json.loads(body)
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
    ) as exc:
        raise VN97P4ArtifactError(
            "P4E-C report must be UTF-8 JSON"
        ) from exc

    status = report.get("status") if isinstance(
        report,
        dict,
    ) else None
    if (
        not isinstance(report, dict)
        or report.get("schema")
        != "VN97P4EC1"
        or status not in _P4E_C_STATUSES
    ):
        raise VN97P4ArtifactError(
            "P4E-C report schema/status mismatch"
        )

    canonical = (
        _canonical_json(report)
        + (
            b"\n"
            if text.endswith("\n")
            else b""
        )
    )
    if canonical != raw[
        "p4e-report.json"
    ]:
        raise VN97P4ArtifactError(
            "P4E-C report must use canonical JSON"
        )

    checkpoint_sha = _require_sha256(
        report.get(
            "output_checkpoint_sha256"
        ),
        label="P4E-C checkpoint SHA-256",
    )
    tokenizer_sha = _require_sha256(
        report.get(
            "tokenizer_sha256"
        ),
        label="P4E-C tokenizer SHA-256",
    )
    model_image_sha = _require_sha256(
        report.get(
            "model_image_sha256"
        ),
        label="P4E-C model image SHA-256",
    )

    if (
        _sha256(
            raw["tokenizer.vn97tk1"]
        )
        != tokenizer_sha
        or _sha256(
            raw["model.vn97mi1"]
        )
        != model_image_sha
    ):
        raise VN97P4ArtifactError(
            "P4E-C report output identity mismatch"
        )

    checkpoint = (
        load_deployment_checkpoint_file(
            resolved / "model.vn97ck1"
        )
    )
    if (
        checkpoint.checkpoint_sha256
        != checkpoint_sha
    ):
        raise VN97P4ArtifactError(
            "P4E-C checkpoint identity mismatch"
        )

    try:
        tokenizer_package = (
            VN97TokenizerPackage.from_bytes(
                raw["tokenizer.vn97tk1"]
            )
        )
    except (TypeError, ValueError) as exc:
        raise VN97P4ArtifactError(
            "P4E-C tokenizer package is invalid"
        ) from exc
    if (
        tokenizer_package.vocab_size
        != checkpoint.config.vocab_size
    ):
        raise VN97P4ArtifactError(
            "P4E-C checkpoint/tokenizer vocabulary mismatch"
        )

    return VN97P4ECArtifact(
        root=resolved,
        report_sha256=_sha256(
            raw["p4e-report.json"]
        ),
        checkpoint_sha256=
            checkpoint_sha,
        tokenizer_sha256=
            tokenizer_sha,
        model_image_sha256=
            model_image_sha,
        status=str(status),
        checkpoint=checkpoint,
        tokenizer_package=
            tokenizer_package,
        report=report,
    )
