from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
from typing import Sequence

from ..training_cli import _load_records
from .config import r2_mobile_1b_config
from .data_bridge import load_vn97tk1
from .pilot_contract import chat_record_digest
from .production_contract import (
    R2ProductionTrainingRecipe,
    assert_r2_production_scale,
)
from .production_launcher import build_production_manifest


R2D5_CAMPAIGN_SCHEMA = "VN97R2D5CAMPAIGN1"
R2D5_PREFLIGHT_RECEIPT_SCHEMA = "VN97R2D5PREFLIGHT1"
R2D4_PREFLIGHT_SCHEMA = "VN97R2D4PREFLIGHT1"
R2D5_CORPUS_PROFILE = "vn97-production-intelligence-v1"
R2D5_SPLITS = ("training", "validation", "release")
R2D5_MAX_SPLIT_BYTES = 256 * 1024 * 1024


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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _require_sha256(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(ch not in "0123456789abcdef" for ch in value)
    ):
        raise ValueError(f"{label} must be lowercase SHA-256")
    return value


def _require_commit(value: str) -> str:
    if (
        len(value) != 40
        or any(ch not in "0123456789abcdef" for ch in value)
    ):
        raise ValueError("repository_commit must be 40 lowercase hex")
    return value


def _read_regular(
    path: Path,
    *,
    max_bytes: int,
    label: str,
) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise ValueError(f"{label} could not be opened safely") from exc
    try:
        info = os.fstat(fd)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_size <= 0
            or info.st_size > max_bytes
        ):
            raise ValueError(f"{label} size/type is invalid")
        out = bytearray()
        while len(out) < info.st_size:
            chunk = os.read(
                fd,
                min(1024 * 1024, info.st_size - len(out)),
            )
            if not chunk:
                break
            out.extend(chunk)
        after = os.fstat(fd)
        if (
            len(out) != info.st_size
            or after.st_size != info.st_size
            or after.st_dev != info.st_dev
            or after.st_ino != info.st_ino
        ):
            raise ValueError(f"{label} changed while being read")
        return bytes(out)
    finally:
        os.close(fd)


def _strict_json_object(data: bytes, *, label: str) -> dict[str, object]:
    duplicates: list[str] = []

    def hook(pairs):
        out: dict[str, object] = {}
        for key, value in pairs:
            if key in out:
                duplicates.append(key)
            out[key] = value
        return out

    try:
        value = json.loads(
            data.decode("utf-8", errors="strict"),
            object_pairs_hook=hook,
            parse_constant=lambda raw: (
                _ for _ in ()
            ).throw(ValueError(raw)),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"{label} must be strict UTF-8 JSON") from exc
    if duplicates or not isinstance(value, dict):
        raise ValueError(
            f"{label} must be one object without duplicate keys"
        )
    return value


@dataclass(frozen=True)
class R2D5CorpusEvidence:
    profile_id: str
    manifest_id: str
    manifest_sha256: str
    split_identity: dict[str, dict[str, object]]
    source_count: int
    source_licenses: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "profile_id": self.profile_id,
            "manifest_id": self.manifest_id,
            "manifest_sha256": self.manifest_sha256,
            "split_identity": self.split_identity,
            "source_count": self.source_count,
            "source_licenses": list(self.source_licenses),
        }


def verify_r2d5_corpus(
    corpus_dir: Path,
    *,
    max_split_bytes: int = R2D5_MAX_SPLIT_BYTES,
) -> R2D5CorpusEvidence:
    if max_split_bytes <= 0:
        raise ValueError("max_split_bytes must be positive")
    if corpus_dir.is_symlink():
        raise ValueError("R2-D5 corpus-dir must not be a symlink")
    root = corpus_dir.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("R2-D5 corpus-dir must be a directory")

    manifest_path = root / "corpus.vn97corpus1.json"
    manifest_bytes = _read_regular(
        manifest_path,
        max_bytes=4 * 1024 * 1024,
        label="VN97CORPUS1 manifest",
    )
    manifest = _strict_json_object(
        manifest_bytes,
        label="VN97CORPUS1 manifest",
    )
    if manifest.get("schema") != "VN97CORPUS1":
        raise ValueError("R2-D5 requires VN97CORPUS1")
    if manifest.get("profile_id") != R2D5_CORPUS_PROFILE:
        raise ValueError("R2-D5 corpus profile mismatch")

    manifest_id = _require_sha256(
        manifest.get("manifest_id"),
        label="VN97CORPUS1 manifest_id",
    )
    identity = dict(manifest)
    identity.pop("manifest_id", None)
    expected_manifest_id = hashlib.sha256(
        b"VN97CORPUS1\0" + _canonical_json(identity)
    ).hexdigest()
    if manifest_id != expected_manifest_id:
        raise ValueError("VN97CORPUS1 manifest identity mismatch")

    sources = manifest.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("VN97CORPUS1 source list is invalid")
    licenses: set[str] = set()
    source_ids: set[str] = set()
    for source in sources:
        if not isinstance(source, dict):
            raise ValueError("VN97CORPUS1 source entry is invalid")
        if source.get("license_approved") is not True:
            raise ValueError("R2-D5 requires approved source licenses")
        source_id = source.get("source_id")
        license_name = source.get("license")
        if (
            not isinstance(source_id, str)
            or not source_id
            or source_id in source_ids
        ):
            raise ValueError("VN97CORPUS1 source identity is invalid")
        if not isinstance(license_name, str) or not license_name:
            raise ValueError("VN97CORPUS1 source license is invalid")
        source_ids.add(source_id)
        licenses.add(license_name)

    splits = manifest.get("splits")
    if (
        not isinstance(splits, dict)
        or set(splits) != set(R2D5_SPLITS)
    ):
        raise ValueError("VN97CORPUS1 split map is invalid")

    all_digests: dict[str, str] = {}
    split_identity: dict[str, dict[str, object]] = {}
    for split in R2D5_SPLITS:
        raw_identity = splits.get(split)
        if not isinstance(raw_identity, dict):
            raise ValueError(f"{split} split identity is invalid")
        if raw_identity.get("mode") != "chat":
            raise ValueError("R2-D5 requires chat-mode corpus splits")
        expected_sha = _require_sha256(
            raw_identity.get("sha256"),
            label=f"{split} split SHA-256",
        )
        expected_bytes = raw_identity.get("bytes")
        expected_records = raw_identity.get("records")
        if (
            not isinstance(expected_bytes, int)
            or not isinstance(expected_records, int)
            or expected_bytes <= 0
            or expected_records <= 0
        ):
            raise ValueError(f"{split} split bounds are invalid")

        filename = {
            "training": "training.jsonl",
            "validation": "validation.jsonl",
            "release": "release.jsonl",
        }[split]
        path = root / filename
        data = _read_regular(
            path,
            max_bytes=max_split_bytes,
            label=f"{split} split",
        )
        if len(data) != expected_bytes:
            raise ValueError(f"{split} split byte identity mismatch")
        actual_sha = _sha256_bytes(data)
        if actual_sha != expected_sha:
            raise ValueError(f"{split} split SHA-256 mismatch")

        records, _ = _load_records(
            [path],
            mode="chat",
            max_input_bytes=max_split_bytes,
            max_examples=max(expected_records, 1),
        )
        if len(records) != expected_records:
            raise ValueError(f"{split} split record-count mismatch")

        digests = [chat_record_digest(record) for record in records]
        if len(digests) != len(set(digests)):
            raise ValueError(f"{split} split contains duplicate records")
        for digest in digests:
            prior = all_digests.get(digest)
            if prior is not None:
                raise ValueError(
                    f"R2-D5 corpus leakage detected: {prior}/{split}"
                )
            all_digests[digest] = split

        split_identity[split] = {
            "bytes": expected_bytes,
            "mode": "chat",
            "records": expected_records,
            "sha256": expected_sha,
        }

    return R2D5CorpusEvidence(
        profile_id=R2D5_CORPUS_PROFILE,
        manifest_id=manifest_id,
        manifest_sha256=_sha256_bytes(manifest_bytes),
        split_identity=split_identity,
        source_count=len(sources),
        source_licenses=tuple(sorted(licenses)),
    )


def _copy_verified(source: Path, target: Path, expected_sha256: str) -> None:
    if target.exists() or target.is_symlink():
        raise ValueError(f"R2-D5 target already exists: {target.name}")
    shutil.copyfile(source, target)
    if _sha256_file(target) != expected_sha256:
        target.unlink(missing_ok=True)
        raise IOError(f"R2-D5 copied file hash mismatch: {target.name}")


def _campaign_identity(body: dict[str, object]) -> str:
    return hashlib.sha256(
        b"VN97R2D5CAMPAIGN1\0" + _canonical_json(body)
    ).hexdigest()


def _render_preflight_script(
    *,
    repository_commit: str,
    task_families: Sequence[str],
    recipe: R2ProductionTrainingRecipe,
    safety_fraction: float,
) -> str:
    task_args = " ".join(
        f"--task-family {json.dumps(item)}"
        for item in task_families
    )
    return f"""#!/usr/bin/env bash
set -euo pipefail

PACKAGE_DIR="$(cd "$(dirname "$0")" && pwd)"
VN97_REPO="${{VN97_REPO:-/kaggle/working/VN97}}"
R2D5_WORK_DIR="${{R2D5_WORK_DIR:-/kaggle/working/r2d5-work}}"
R2D5_OUTPUT_DIR="${{R2D5_OUTPUT_DIR:-/kaggle/working/r2d5-output}}"

cd "$VN97_REPO"
ACTUAL_COMMIT="$(git rev-parse HEAD)"
if [ "$ACTUAL_COMMIT" != "{repository_commit}" ]; then
  echo "R2-D5 repository commit mismatch: expected {repository_commit}, got $ACTUAL_COMMIT" >&2
  exit 2
fi

python -m pip install -e .

vn97-r2-production \
  --mode preflight \
  --stage dense_pretrain \
  --tokenizer "$PACKAGE_DIR/tokenizer.vn97tk1" \
  --train-jsonl "$PACKAGE_DIR/training.jsonl" \
  --validation-jsonl "$PACKAGE_DIR/validation.jsonl" \
  {task_args} \
  --work-dir "$R2D5_WORK_DIR" \
  --output-dir "$R2D5_OUTPUT_DIR" \
  --sequence-length {recipe.sequence_length} \
  --micro-batch-size {recipe.micro_batch_size} \
  --gradient-accumulation-steps {recipe.gradient_accumulation_steps} \
  --precision {recipe.precision} \
  --device cuda \
  --safety-fraction {safety_fraction:.6f} \
  --preflight-bundle "$R2D5_OUTPUT_DIR/r2d4-preflight.json"
"""


def build_r2d5_preflight_package(
    *,
    corpus_dir: Path,
    tokenizer_path: Path,
    repository_commit: str,
    output_dir: Path,
    task_families: Sequence[str],
    recipe: R2ProductionTrainingRecipe,
    safety_fraction: float,
    max_split_bytes: int = R2D5_MAX_SPLIT_BYTES,
) -> dict[str, object]:
    repository_commit = _require_commit(repository_commit)
    if not 0.0 < safety_fraction <= 1.0:
        raise ValueError("safety_fraction must be in (0, 1]")
    families = tuple(sorted(set(task_families)))
    if not families or any(not item for item in families):
        raise ValueError("R2-D5 requires non-empty task families")
    if recipe.quantization_used:
        raise ValueError("R2-D5 preflight forbids quantization")

    corpus = verify_r2d5_corpus(
        corpus_dir,
        max_split_bytes=max_split_bytes,
    )
    tokenizer_resolved = tokenizer_path.resolve(strict=True)
    tokenizer_sha = _sha256_file(tokenizer_resolved)
    tokenizer = load_vn97tk1(tokenizer_resolved)
    config = r2_mobile_1b_config(tokenizer.vocab_size)
    parameter_count = assert_r2_production_scale(config)

    root = corpus_dir.resolve(strict=True)
    d4_manifest = build_production_manifest(
        stage="dense_pretrain",
        tokenizer_path=tokenizer_resolved,
        train_path=root / "training.jsonl",
        validation_path=root / "validation.jsonl",
        train_records=int(
            corpus.split_identity["training"]["records"]
        ),
        validation_records=int(
            corpus.split_identity["validation"]["records"]
        ),
        task_families=families,
        parent_checkpoint_sha256=None,
    )

    if output_dir.exists():
        if (
            output_dir.is_symlink()
            or not output_dir.is_dir()
            or any(output_dir.iterdir())
        ):
            raise ValueError("R2-D5 output-dir must be new or empty")
    else:
        output_dir.mkdir(parents=True, exist_ok=False)
    output = output_dir.resolve(strict=True)

    package_files = {
        "corpus.vn97corpus1.json": corpus.manifest_sha256,
        "training.jsonl": str(
            corpus.split_identity["training"]["sha256"]
        ),
        "validation.jsonl": str(
            corpus.split_identity["validation"]["sha256"]
        ),
        "release.jsonl": str(
            corpus.split_identity["release"]["sha256"]
        ),
        "tokenizer.vn97tk1": tokenizer_sha,
    }
    for filename, expected_sha in package_files.items():
        source = (
            tokenizer_resolved
            if filename == "tokenizer.vn97tk1"
            else root / filename
        )
        _copy_verified(source, output / filename, expected_sha)

    script = _render_preflight_script(
        repository_commit=repository_commit,
        task_families=families,
        recipe=recipe,
        safety_fraction=safety_fraction,
    )
    script_bytes = script.encode("utf-8")

    body: dict[str, object] = {
        "architecture_fingerprint": config.fingerprint(),
        "corpus": corpus.as_dict(),
        "d4_manifest_identity": d4_manifest.identity(),
        "model_parameter_count": parameter_count,
        "preflight_script_sha256": _sha256_bytes(script_bytes),
        "purpose": "measured_cuda_preflight",
        "recipe": asdict(recipe),
        "recipe_fingerprint": recipe.fingerprint(),
        "release_used_for_training": False,
        "repository_commit": repository_commit,
        "safety_fraction": safety_fraction,
        "schema": R2D5_CAMPAIGN_SCHEMA,
        "task_families": list(families),
        "tokenizer_sha256": tokenizer_sha,
        "training_allowed": False,
    }
    campaign_id = _campaign_identity(body)
    payload = dict(body)
    payload["campaign_id"] = campaign_id
    (output / "r2d5-campaign.json").write_bytes(
        _canonical_json(payload) + b"\n"
    )

    script_path = output / "run_t4_preflight.sh"
    script_path.write_bytes(script_bytes)
    script_path.chmod(0o755)

    return payload


def verify_r2d5_package(package_dir: Path) -> dict[str, object]:
    if package_dir.is_symlink():
        raise ValueError("R2-D5 package-dir must not be a symlink")
    root = package_dir.resolve(strict=True)
    campaign_data = _read_regular(
        root / "r2d5-campaign.json",
        max_bytes=1024 * 1024,
        label="R2-D5 campaign",
    )
    payload = _strict_json_object(
        campaign_data,
        label="R2-D5 campaign",
    )
    if payload.get("schema") != R2D5_CAMPAIGN_SCHEMA:
        raise ValueError("R2-D5 campaign schema mismatch")
    campaign_id = _require_sha256(
        payload.get("campaign_id"),
        label="R2-D5 campaign_id",
    )
    body = dict(payload)
    body.pop("campaign_id", None)
    if campaign_id != _campaign_identity(body):
        raise ValueError("R2-D5 campaign identity mismatch")
    if payload.get("training_allowed") is not False:
        raise ValueError("R2-D5 package must remain preflight-only")
    if payload.get("release_used_for_training") is not False:
        raise ValueError("R2-D5 release split must remain held out")

    corpus_raw = payload.get("corpus")
    if not isinstance(corpus_raw, dict):
        raise ValueError("R2-D5 corpus evidence is missing")
    expected_manifest_sha = _require_sha256(
        corpus_raw.get("manifest_sha256"),
        label="R2-D5 corpus manifest SHA-256",
    )
    if _sha256_file(root / "corpus.vn97corpus1.json") != expected_manifest_sha:
        raise ValueError("R2-D5 packaged corpus manifest hash mismatch")
    split_identity = corpus_raw.get("split_identity")
    if not isinstance(split_identity, dict):
        raise ValueError("R2-D5 split identity is missing")
    for split in R2D5_SPLITS:
        entry = split_identity.get(split)
        if not isinstance(entry, dict):
            raise ValueError("R2-D5 split identity entry is invalid")
        expected = _require_sha256(
            entry.get("sha256"),
            label=f"R2-D5 {split} SHA-256",
        )
        if _sha256_file(root / f"{split}.jsonl") != expected:
            raise ValueError(f"R2-D5 {split} package hash mismatch")

    script_sha = _require_sha256(
        payload.get("preflight_script_sha256"),
        label="R2-D5 preflight script SHA-256",
    )
    if _sha256_file(root / "run_t4_preflight.sh") != script_sha:
        raise ValueError("R2-D5 preflight script hash mismatch")

    tokenizer_sha = _require_sha256(
        payload.get("tokenizer_sha256"),
        label="R2-D5 tokenizer SHA-256",
    )
    tokenizer_path = root / "tokenizer.vn97tk1"
    if _sha256_file(tokenizer_path) != tokenizer_sha:
        raise ValueError("R2-D5 tokenizer package hash mismatch")
    tokenizer = load_vn97tk1(tokenizer_path)
    config = r2_mobile_1b_config(tokenizer.vocab_size)
    if payload.get("architecture_fingerprint") != config.fingerprint():
        raise ValueError("R2-D5 architecture fingerprint mismatch")
    if payload.get("model_parameter_count") != config.estimated_parameter_count():
        raise ValueError("R2-D5 model parameter count mismatch")

    recipe_raw = payload.get("recipe")
    if not isinstance(recipe_raw, dict):
        raise ValueError("R2-D5 recipe is missing")
    recipe = R2ProductionTrainingRecipe(**recipe_raw)
    if payload.get("recipe_fingerprint") != recipe.fingerprint():
        raise ValueError("R2-D5 recipe fingerprint mismatch")

    return payload


def seal_r2d5_preflight(
    *,
    package_dir: Path,
    preflight_bundle_path: Path,
    output_path: Path,
) -> dict[str, object]:
    campaign = verify_r2d5_package(package_dir)
    data = _read_regular(
        preflight_bundle_path,
        max_bytes=1024 * 1024,
        label="R2-D4 preflight bundle",
    )
    bundle = _strict_json_object(
        data,
        label="R2-D4 preflight bundle",
    )
    if bundle.get("schema") != R2D4_PREFLIGHT_SCHEMA:
        raise ValueError("R2-D5 requires VN97R2D4PREFLIGHT1 evidence")
    if bundle.get("promotion_allowed") is not False:
        raise ValueError("R2-D4 evidence must remain no-promotion")
    if bundle.get("manifest_identity") != campaign.get("d4_manifest_identity"):
        raise ValueError("R2-D5 preflight manifest mismatch")
    if bundle.get("recipe_fingerprint") != campaign.get("recipe_fingerprint"):
        raise ValueError("R2-D5 preflight recipe mismatch")

    memory = bundle.get("memory_evidence")
    if not isinstance(memory, dict):
        raise ValueError("R2-D5 measured memory evidence is missing")
    if memory.get("passed") is not True:
        raise ValueError("R2-D5 measured memory preflight did not pass")
    if (
        memory.get("architecture_fingerprint")
        != campaign.get("architecture_fingerprint")
    ):
        raise ValueError("R2-D5 preflight architecture mismatch")
    if (
        memory.get("recipe_fingerprint")
        != campaign.get("recipe_fingerprint")
    ):
        raise ValueError("R2-D5 measured recipe mismatch")

    body = {
        "architecture_fingerprint": campaign["architecture_fingerprint"],
        "campaign_id": campaign["campaign_id"],
        "device_name": memory.get("device_name"),
        "free_device_bytes_before": memory.get(
            "free_device_bytes_before"
        ),
        "peak_allocated_bytes": memory.get("peak_allocated_bytes"),
        "peak_reserved_bytes": memory.get("peak_reserved_bytes"),
        "preflight_bundle_sha256": _sha256_bytes(data),
        "recipe_fingerprint": campaign["recipe_fingerprint"],
        "safety_fraction": memory.get("safety_fraction"),
        "schema": R2D5_PREFLIGHT_RECEIPT_SCHEMA,
        "training_allowed": False,
    }
    receipt_id = hashlib.sha256(
        b"VN97R2D5PREFLIGHT1\0" + _canonical_json(body)
    ).hexdigest()
    receipt = dict(body)
    receipt["receipt_id"] = receipt_id

    if output_path.exists() or output_path.is_symlink():
        raise ValueError("R2-D5 receipt output must not already exist")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(_canonical_json(receipt) + b"\n")
    return receipt
