from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import stat


class VN97P2PilotHandoffError(RuntimeError):
    pass


_REQUIRED_CORPUS_ROOT = {"corpus", "evidence", "SHA256SUMS"}
_REQUIRED_CORPUS_FILES = {
    "corpus.vn97corpus1.json",
    "training.jsonl",
    "validation.jsonl",
    "release.jsonl",
}
_REQUIRED_EVIDENCE_FILES = {
    "source-fetch.vn97p2fetch1.json",
    "p2-source-summary.vn97p2src1.json",
    "corpus-definition.vn97corpusdef1.json",
    "p2-corpus-bundle.vn97p2bundle1.json",
}
_REQUIRED_PILOT_FILES = {
    "tokenizer.vn97tk1",
    "model.vn97ck1",
    "campaign-report.json",
    "model.vn97mi1",
    "pilot.vn97pilot1.json",
}
_MAX_JSON_BYTES = 4 * 1024 * 1024
_MAX_FILE_BYTES = 128 * 1024 * 1024


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path, *, max_bytes: int = _MAX_FILE_BYTES) -> tuple[str, int]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise VN97P2PilotHandoffError(
            f"could not open file safely: {path}"
        ) from exc
    try:
        info = os.fstat(fd)
        if (
            not stat.S_ISREG(info.st_mode)
            or not 0 < info.st_size <= max_bytes
        ):
            raise VN97P2PilotHandoffError(
                f"file size/type is invalid: {path}"
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
            or after.st_dev != info.st_dev
            or after.st_ino != info.st_ino
        ):
            raise VN97P2PilotHandoffError(
                f"file changed while hashing: {path}"
            )
        return digest.hexdigest(), consumed
    finally:
        os.close(fd)


def _read_json(path: Path, *, label: str) -> tuple[dict[str, object], bytes]:
    if path.is_symlink():
        raise VN97P2PilotHandoffError(
            f"{label} must not be a symlink"
        )
    data = path.read_bytes()
    if not 0 < len(data) <= _MAX_JSON_BYTES:
        raise VN97P2PilotHandoffError(
            f"{label} size is outside bounds"
        )
    duplicates: list[str] = []

    def hook(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                duplicates.append(key)
            result[key] = value
        return result

    try:
        text = data.decode("utf-8", errors="strict")
        body = text[:-1] if text.endswith("\n") else text
        value = json.loads(
            body,
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
        raise VN97P2PilotHandoffError(
            f"{label} must be strict UTF-8 JSON"
        ) from exc
    if duplicates or not isinstance(value, dict):
        raise VN97P2PilotHandoffError(
            f"{label} must be one JSON object without duplicate keys"
        )
    canonical = _canonical_json(value).decode("utf-8")
    expected = canonical + ("\n" if text.endswith("\n") else "")
    if expected != text:
        raise VN97P2PilotHandoffError(
            f"{label} must use canonical JSON"
        )
    return value, data


def _require_sha256(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(ch not in "0123456789abcdef" for ch in value)
    ):
        raise VN97P2PilotHandoffError(
            f"{label} must be lowercase SHA-256"
        )
    return value


def _require_commit(value: str, *, label: str) -> str:
    if (
        len(value) != 40
        or any(ch not in "0123456789abcdef" for ch in value)
    ):
        raise VN97P2PilotHandoffError(
            f"{label} must be 40 lowercase hex"
        )
    return value


def _verify_sha256sums(root: Path) -> None:
    sums_path = root / "SHA256SUMS"
    lines = sums_path.read_text(
        encoding="utf-8",
        errors="strict",
    ).splitlines()
    expected_paths = {
        *(f"corpus/{name}" for name in _REQUIRED_CORPUS_FILES),
        *(f"evidence/{name}" for name in _REQUIRED_EVIDENCE_FILES),
    }
    seen: set[str] = set()
    for line in lines:
        if not line:
            continue
        if len(line) < 67 or line[64:66] not in {"  ", " *"}:
            raise VN97P2PilotHandoffError(
                "SHA256SUMS line format is invalid"
            )
        digest = _require_sha256(
            line[:64],
            label="SHA256SUMS digest",
        )
        relative = line[66:]
        if (
            not relative
            or relative.startswith("/")
            or ".." in Path(relative).parts
            or relative in seen
        ):
            raise VN97P2PilotHandoffError(
                "SHA256SUMS path is invalid or duplicated"
            )
        seen.add(relative)
        if relative not in expected_paths:
            raise VN97P2PilotHandoffError(
                f"SHA256SUMS contains unexpected path: {relative}"
            )
        actual, _ = _sha256_file(root / relative)
        if actual != digest:
            raise VN97P2PilotHandoffError(
                f"SHA256SUMS mismatch: {relative}"
            )
    if seen != expected_paths:
        raise VN97P2PilotHandoffError(
            "SHA256SUMS does not cover exact corpus artifact set"
        )


@dataclass(frozen=True)
class VN97VerifiedP2CorpusArtifact:
    corpus_commit: str
    bundle_id: str
    bundle_sha256: str
    corpus_manifest_id: str
    corpus_manifest_sha256: str


def verify_corpus_artifact(
    root: Path,
    *,
    expected_corpus_commit: str,
) -> VN97VerifiedP2CorpusArtifact:
    expected_corpus_commit = _require_commit(
        expected_corpus_commit,
        label="expected corpus commit",
    )
    if root.is_symlink():
        raise VN97P2PilotHandoffError(
            "corpus artifact root must not be a symlink"
        )
    resolved = root.resolve(strict=True)
    if not resolved.is_dir():
        raise VN97P2PilotHandoffError(
            "corpus artifact root must be a directory"
        )
    if {item.name for item in resolved.iterdir()} != _REQUIRED_CORPUS_ROOT:
        raise VN97P2PilotHandoffError(
            "corpus artifact root entry set is invalid"
        )
    corpus_dir = resolved / "corpus"
    evidence_dir = resolved / "evidence"
    if (
        corpus_dir.is_symlink()
        or evidence_dir.is_symlink()
        or not corpus_dir.is_dir()
        or not evidence_dir.is_dir()
    ):
        raise VN97P2PilotHandoffError(
            "corpus/evidence directories must be real directories"
        )
    if {item.name for item in corpus_dir.iterdir()} != _REQUIRED_CORPUS_FILES:
        raise VN97P2PilotHandoffError(
            "corpus artifact file set is invalid"
        )
    if {item.name for item in evidence_dir.iterdir()} != _REQUIRED_EVIDENCE_FILES:
        raise VN97P2PilotHandoffError(
            "corpus evidence file set is invalid"
        )
    _verify_sha256sums(resolved)

    bundle, bundle_bytes = _read_json(
        evidence_dir / "p2-corpus-bundle.vn97p2bundle1.json",
        label="VN97P2BUNDLE1",
    )
    if bundle.get("schema") != "VN97P2BUNDLE1":
        raise VN97P2PilotHandoffError(
            "corpus bundle schema must be VN97P2BUNDLE1"
        )
    if bundle.get("repository_commit") != expected_corpus_commit:
        raise VN97P2PilotHandoffError(
            "corpus bundle repository commit mismatch"
        )
    bundle_id = _require_sha256(
        bundle.get("bundle_id"),
        label="corpus bundle ID",
    )
    body = dict(bundle)
    body.pop("bundle_id", None)
    expected_bundle_id = _sha256_bytes(
        b"VN97P2BUNDLE1\0" + _canonical_json(body)
    )
    if bundle_id != expected_bundle_id:
        raise VN97P2PilotHandoffError(
            "corpus bundle identity mismatch"
        )

    manifest, manifest_bytes = _read_json(
        corpus_dir / "corpus.vn97corpus1.json",
        label="VN97CORPUS1",
    )
    if manifest.get("schema") != "VN97CORPUS1":
        raise VN97P2PilotHandoffError(
            "corpus manifest schema must be VN97CORPUS1"
        )
    manifest_id = _require_sha256(
        manifest.get("manifest_id"),
        label="corpus manifest ID",
    )
    manifest_sha256 = _sha256_bytes(manifest_bytes)
    if bundle.get("corpus_manifest_id") != manifest_id:
        raise VN97P2PilotHandoffError(
            "bundle/corpus manifest ID mismatch"
        )
    if bundle.get("corpus_manifest_sha256") != manifest_sha256:
        raise VN97P2PilotHandoffError(
            "bundle/corpus manifest SHA-256 mismatch"
        )

    return VN97VerifiedP2CorpusArtifact(
        corpus_commit=expected_corpus_commit,
        bundle_id=bundle_id,
        bundle_sha256=_sha256_bytes(bundle_bytes),
        corpus_manifest_id=manifest_id,
        corpus_manifest_sha256=manifest_sha256,
    )


@dataclass(frozen=True)
class VN97P2PilotRunReceipt:
    training_commit: str
    corpus_commit: str
    corpus_run_id: str
    training_run_id: str
    corpus_bundle_id: str
    corpus_bundle_sha256: str
    corpus_manifest_id: str
    pilot_receipt_sha256: str
    campaign_report_sha256: str
    checkpoint_sha256: str
    tokenizer_sha256: str
    model_image_sha256: str
    profile_sha256: str
    device: str

    def canonical_object(self) -> dict[str, object]:
        body = {
            "campaign_report_sha256": self.campaign_report_sha256,
            "checkpoint_sha256": self.checkpoint_sha256,
            "corpus_bundle_id": self.corpus_bundle_id,
            "corpus_bundle_sha256": self.corpus_bundle_sha256,
            "corpus_commit": self.corpus_commit,
            "corpus_manifest_id": self.corpus_manifest_id,
            "corpus_run_id": self.corpus_run_id,
            "device": self.device,
            "model_image_sha256": self.model_image_sha256,
            "pilot_receipt_sha256": self.pilot_receipt_sha256,
            "profile_sha256": self.profile_sha256,
            "schema": "VN97P2RUN1",
            "tokenizer_sha256": self.tokenizer_sha256,
            "training_commit": self.training_commit,
            "training_run_id": self.training_run_id,
        }
        body["run_identity"] = _sha256_bytes(
            b"VN97P2RUN1\0" + _canonical_json(body)
        )
        return body

    def to_bytes(self) -> bytes:
        return _canonical_json(self.canonical_object()) + b"\n"


def build_pilot_run_receipt(
    *,
    corpus: VN97VerifiedP2CorpusArtifact,
    pilot_output: Path,
    training_commit: str,
    corpus_run_id: str,
    training_run_id: str,
) -> VN97P2PilotRunReceipt:
    training_commit = _require_commit(
        training_commit,
        label="training commit",
    )
    if (
        not corpus_run_id.isdigit()
        or not training_run_id.isdigit()
    ):
        raise VN97P2PilotHandoffError(
            "GitHub run IDs must be decimal strings"
        )
    if pilot_output.is_symlink():
        raise VN97P2PilotHandoffError(
            "pilot output must not be a symlink"
        )
    root = pilot_output.resolve(strict=True)
    if {item.name for item in root.iterdir()} != _REQUIRED_PILOT_FILES:
        raise VN97P2PilotHandoffError(
            "pilot output file set is invalid"
        )

    pilot, pilot_bytes = _read_json(
        root / "pilot.vn97pilot1.json",
        label="VN97PILOT1",
    )
    if pilot.get("schema") != "VN97PILOT1":
        raise VN97P2PilotHandoffError(
            "pilot receipt schema must be VN97PILOT1"
        )
    if pilot.get("corpus_manifest_id") != corpus.corpus_manifest_id:
        raise VN97P2PilotHandoffError(
            "pilot/corpus manifest identity mismatch"
        )
    if pilot.get("device") != "cpu":
        raise VN97P2PilotHandoffError(
            "GitHub P2 pilot workflow requires cpu device evidence"
        )

    campaign_sha, _ = _sha256_file(
        root / "campaign-report.json",
    )
    checkpoint_sha, _ = _sha256_file(
        root / "model.vn97ck1",
    )
    tokenizer_sha, _ = _sha256_file(
        root / "tokenizer.vn97tk1",
    )
    image_sha, _ = _sha256_file(
        root / "model.vn97mi1",
    )

    if pilot.get("campaign_report_sha256") != campaign_sha:
        raise VN97P2PilotHandoffError(
            "pilot campaign report SHA-256 mismatch"
        )
    if pilot.get("checkpoint_sha256") != checkpoint_sha:
        raise VN97P2PilotHandoffError(
            "pilot checkpoint SHA-256 mismatch"
        )
    if pilot.get("tokenizer_sha256") != tokenizer_sha:
        raise VN97P2PilotHandoffError(
            "pilot tokenizer SHA-256 mismatch"
        )
    if pilot.get("model_image_sha256") != image_sha:
        raise VN97P2PilotHandoffError(
            "pilot model-image SHA-256 mismatch"
        )

    return VN97P2PilotRunReceipt(
        training_commit=training_commit,
        corpus_commit=corpus.corpus_commit,
        corpus_run_id=corpus_run_id,
        training_run_id=training_run_id,
        corpus_bundle_id=corpus.bundle_id,
        corpus_bundle_sha256=corpus.bundle_sha256,
        corpus_manifest_id=corpus.corpus_manifest_id,
        pilot_receipt_sha256=_sha256_bytes(pilot_bytes),
        campaign_report_sha256=campaign_sha,
        checkpoint_sha256=checkpoint_sha,
        tokenizer_sha256=tokenizer_sha,
        model_image_sha256=image_sha,
        profile_sha256=_require_sha256(
            pilot.get("profile_sha256"),
            label="pilot profile SHA-256",
        ),
        device="cpu",
    )
