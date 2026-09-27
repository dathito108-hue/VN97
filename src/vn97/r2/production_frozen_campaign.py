from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import shutil
from typing import Mapping

from ..training_cli import _atomic_write
from .config import r2_mobile_1b_config
from .data_bridge import load_vn97tk1
from .production_campaign import (
    R2D5_CAMPAIGN_SCHEMA,
    R2D5_PREFLIGHT_RECEIPT_SCHEMA,
    verify_r2d5_package,
)
from .production_contract import (
    R2ProductionTrainingRecipe,
    assert_production_training_contract,
)
from .production_curriculum import load_r2d8_plan
from .production_streaming import load_r2d5_memory_receipt
from .production_training import R2ProductionTrainerConfig
from .production_virtual_corpus import (
    projection_for_stage,
    verify_r2d12_view,
)


R2D13_CAMPAIGN_SCHEMA = "VN97R2D13CAMPAIGN1"
R2D13_READY_SCHEMA = "VN97R2D13READY1"
R2D13_SUPPORTED_STAGE = "dense_pretrain"
R2D13_PREFLIGHT_FILES = (
    "corpus.vn97corpus1.json",
    "training.jsonl",
    "validation.jsonl",
    "release.jsonl",
    "tokenizer.vn97tk1",
    "r2d5-campaign.json",
    "run_t4_preflight.sh",
)
R2D13_VIEW_FILES = (
    "r2d12-view.json",
    "r2d12-mounts.json",
    "tokenizer.vn97tk1",
    "r2d11-registry.json",
)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"R2-D13 file must be regular: {path}")
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


def _require_commit(value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(ch not in "0123456789abcdef" for ch in value)
    ):
        raise ValueError(
            "R2-D13 repository_commit must be 40 lowercase hex"
        )
    return value


def _strict_json(path: Path, *, label: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular non-symlink file")
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
            path.read_text(encoding="utf-8", errors="strict"),
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
        raise ValueError(f"{label} must be strict UTF-8 JSON") from exc
    if duplicates or not isinstance(value, dict):
        raise ValueError(
            f"{label} must be one object without duplicate keys"
        )
    return value


def _campaign_identity(body: Mapping[str, object]) -> str:
    return _sha256_bytes(
        b"VN97R2D13CAMPAIGN1\0"
        + _canonical_json(dict(body))
    )


def _ready_identity(body: Mapping[str, object]) -> str:
    return _sha256_bytes(
        b"VN97R2D13READY1\0"
        + _canonical_json(dict(body))
    )


def _copy_regular(source: Path, target: Path) -> str:
    if source.is_symlink() or not source.is_file():
        raise ValueError(
            f"R2-D13 source must be a regular file: {source}"
        )
    if target.exists() or target.is_symlink():
        raise ValueError(
            f"R2-D13 target must not already exist: {target}"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    expected = _sha256_file(source)
    shutil.copyfile(source, target)
    actual = _sha256_file(target)
    if actual != expected:
        target.unlink(missing_ok=True)
        raise IOError("R2-D13 copied file identity mismatch")
    return actual


def _copy_view(view_dir: Path, target: Path) -> None:
    source = view_dir.resolve(strict=True)
    target.mkdir(parents=True, exist_ok=False)
    for filename in R2D13_VIEW_FILES:
        _copy_regular(source / filename, target / filename)

    ledger_source = source / "r2d11-ledger"
    if ledger_source.is_symlink() or not ledger_source.is_dir():
        raise ValueError("R2-D13 D12 ledger directory is invalid")
    ledger_target = target / "r2d11-ledger"
    ledger_target.mkdir()
    ledger_files = sorted(ledger_source.glob("*.json"))
    if not ledger_files:
        raise ValueError("R2-D13 D12 ledger chain is empty")
    for path in ledger_files:
        _copy_regular(path, ledger_target / path.name)


def _copy_preflight(package_dir: Path, target: Path) -> None:
    source = package_dir.resolve(strict=True)
    target.mkdir(parents=True, exist_ok=False)
    for filename in R2D13_PREFLIGHT_FILES:
        _copy_regular(source / filename, target / filename)
    (target / "run_t4_preflight.sh").chmod(0o755)


def _trainer_config(
    *,
    plan: Mapping[str, object],
    learning_rate: float,
    weight_decay: float,
    max_grad_norm: float,
    checkpoint_every_optimizer_steps: int,
) -> R2ProductionTrainerConfig:
    return R2ProductionTrainerConfig(
        epochs=int(plan["epochs"]),
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        max_grad_norm=max_grad_norm,
        seed=int(plan["seed"]),
        checkpoint_every_optimizer_steps=(
            checkpoint_every_optimizer_steps
        ),
        require_measured_cuda_preflight=True,
    )


def _render_preflight_wrapper(
    *,
    repository_commit: str,
) -> str:
    return f"""#!/usr/bin/env bash
set -euo pipefail

PACKAGE_DIR="$(cd "$(dirname "$0")" && pwd)"
VN97_REPO="DOLLAR_LBRACEVN97_REPO:-/kaggle/working/VN97}"
VN97_WORKSPACE_ROOT="DOLLAR_LBRACEVN97_WORKSPACE_ROOT:-/kaggle/working}"

cd "$VN97_REPO"
ACTUAL_COMMIT="$(git rev-parse HEAD)"
if [ "$ACTUAL_COMMIT" != "{repository_commit}" ]; then
  echo "R2-D13 repository commit mismatch: expected {repository_commit}, got $ACTUAL_COMMIT" >&2
  exit 2
fi

python -m pip install -e .

vn97-r2-production-campaign gate-preflight \
  --package-dir "$PACKAGE_DIR" \
  --workspace-root "$VN97_WORKSPACE_ROOT"

exec "$PACKAGE_DIR/preflight/run_t4_preflight.sh"
""".replace("DOLLAR_LBRACE", "$"+"{")


def _render_seal_wrapper(
    *,
    repository_commit: str,
) -> str:
    return f"""#!/usr/bin/env bash
set -euo pipefail

PACKAGE_DIR="$(cd "$(dirname "$0")" && pwd)"
VN97_REPO="DOLLAR_LBRACEVN97_REPO:-/kaggle/working/VN97}"
VN97_WORKSPACE_ROOT="DOLLAR_LBRACEVN97_WORKSPACE_ROOT:-/kaggle/working}"
R2D5_OUTPUT_DIR="DOLLAR_LBRACER2D5_OUTPUT_DIR:-/kaggle/working/r2d5-output}"

cd "$VN97_REPO"
ACTUAL_COMMIT="$(git rev-parse HEAD)"
if [ "$ACTUAL_COMMIT" != "{repository_commit}" ]; then
  echo "R2-D13 repository commit mismatch: expected {repository_commit}, got $ACTUAL_COMMIT" >&2
  exit 2
fi
python -m pip install -e .

vn97-r2-preflight-package seal-preflight \
  --package-dir "$PACKAGE_DIR/preflight" \
  --preflight-bundle "$R2D5_OUTPUT_DIR/r2d4-preflight.json" \
  --output "$PACKAGE_DIR/preflight-receipt.json"

vn97-r2-production-campaign seal-ready \
  --package-dir "$PACKAGE_DIR" \
  --workspace-root "$VN97_WORKSPACE_ROOT" \
  --preflight-receipt "$PACKAGE_DIR/preflight-receipt.json" \
  --output "$PACKAGE_DIR/r2d13-ready.json"
""".replace("DOLLAR_LBRACE", "$"+"{")


def _render_train_wrapper(
    *,
    repository_commit: str,
    recipe: R2ProductionTrainingRecipe,
    trainer: R2ProductionTrainerConfig,
    max_run_seconds: float,
) -> str:
    return f"""#!/usr/bin/env bash
set -euo pipefail

PACKAGE_DIR="$(cd "$(dirname "$0")" && pwd)"
VN97_REPO="DOLLAR_LBRACEVN97_REPO:-/kaggle/working/VN97}"
VN97_WORKSPACE_ROOT="DOLLAR_LBRACEVN97_WORKSPACE_ROOT:-/kaggle/working}"
R2D13_WORK_DIR="DOLLAR_LBRACER2D13_WORK_DIR:-/kaggle/working/r2d13-work}"
R2D13_OUTPUT_DIR="DOLLAR_LBRACER2D13_OUTPUT_DIR:-/kaggle/working/r2d13-output}"

cd "$VN97_REPO"
ACTUAL_COMMIT="$(git rev-parse HEAD)"
if [ "$ACTUAL_COMMIT" != "{repository_commit}" ]; then
  echo "R2-D13 repository commit mismatch: expected {repository_commit}, got $ACTUAL_COMMIT" >&2
  exit 2
fi
python -m pip install -e .

vn97-r2-production-campaign verify-ready \
  --package-dir "$PACKAGE_DIR" \
  --workspace-root "$VN97_WORKSPACE_ROOT" \
  --ready-receipt "$PACKAGE_DIR/r2d13-ready.json" \
  --preflight-receipt "$PACKAGE_DIR/preflight-receipt.json"

vn97-r2-stream-train \
  --virtual-view "$PACKAGE_DIR/view" \
  --workspace-root "$VN97_WORKSPACE_ROOT" \
  --curriculum-plan "$PACKAGE_DIR/curriculum.json" \
  --preflight-receipt "$PACKAGE_DIR/preflight-receipt.json" \
  --work-dir "$R2D13_WORK_DIR" \
  --output-dir "$R2D13_OUTPUT_DIR" \
  --sequence-length {recipe.sequence_length} \
  --micro-batch-size {recipe.micro_batch_size} \
  --gradient-accumulation-steps {recipe.gradient_accumulation_steps} \
  --precision {recipe.precision} \
  --epochs {trainer.epochs} \
  --learning-rate {trainer.learning_rate:.17g} \
  --weight-decay {trainer.weight_decay:.17g} \
  --max-grad-norm {trainer.max_grad_norm:.17g} \
  --checkpoint-every {trainer.checkpoint_every_optimizer_steps} \
  --max-run-seconds {max_run_seconds:.6f} \
  --seed {trainer.seed} \
  --device cuda
""".replace("DOLLAR_LBRACE", "$"+"{")


def build_r2d13_campaign(
    *,
    view_dir: Path,
    workspace_root: Path,
    curriculum_plan_path: Path,
    preflight_package_dir: Path,
    repository_commit: str,
    output_dir: Path,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.01,
    max_grad_norm: float = 1.0,
    checkpoint_every_optimizer_steps: int = 25,
    max_run_seconds: float = 3000.0,
) -> dict[str, object]:
    repository_commit = _require_commit(repository_commit)
    if not math.isfinite(max_run_seconds) or max_run_seconds <= 0.0:
        raise ValueError("R2-D13 max_run_seconds must be positive")

    verified_view = verify_r2d12_view(
        view_dir,
        workspace_root=workspace_root,
    )
    view = verified_view["view"]
    full_index = verified_view["index"]
    raw_plan = _strict_json(
        curriculum_plan_path,
        label="R2-D13 curriculum plan",
    )
    weights = raw_plan.get("family_weights")
    if not isinstance(weights, dict) or not weights:
        raise ValueError("R2-D13 curriculum family weights are missing")
    stage = str(raw_plan.get("stage"))
    if stage != R2D13_SUPPORTED_STAGE:
        raise ValueError(
            "R2-D13 execution currently supports dense_pretrain only"
        )
    projected_index = projection_for_stage(
        full_index,
        stage=stage,
        family_weights=weights,
    )
    plan = load_r2d8_plan(
        curriculum_plan_path,
        projected_index,
    )

    preflight = verify_r2d5_package(preflight_package_dir)
    if preflight.get("schema") != R2D5_CAMPAIGN_SCHEMA:
        raise ValueError("R2-D13 preflight campaign schema mismatch")
    if preflight.get("repository_commit") != repository_commit:
        raise ValueError(
            "R2-D13 D5 preflight package repository commit mismatch"
        )
    if preflight.get("architecture_fingerprint") != projected_index.get(
        "architecture_fingerprint"
    ):
        raise ValueError(
            "R2-D13 D5 preflight architecture does not match D12"
        )
    if preflight.get("tokenizer_sha256") != projected_index.get(
        "tokenizer_sha256"
    ):
        raise ValueError(
            "R2-D13 D5 preflight tokenizer does not match D12"
        )

    recipe_raw = preflight.get("recipe")
    if not isinstance(recipe_raw, dict):
        raise ValueError("R2-D13 D5 preflight recipe is missing")
    recipe = R2ProductionTrainingRecipe(**recipe_raw)
    tokenizer = load_vn97tk1(
        verified_view["tokenizer_path"]
    )
    config = r2_mobile_1b_config(tokenizer.vocab_size)
    if config.fingerprint() != projected_index.get(
        "architecture_fingerprint"
    ):
        raise ValueError("R2-D13 projected architecture mismatch")
    assert_production_training_contract(config, recipe)
    if int(projected_index.get("sequence_length", -1)) != (
        recipe.sequence_length
    ):
        raise ValueError(
            "R2-D13 recipe sequence length differs from projected corpus"
        )
    if set(preflight.get("task_families", [])) != set(weights):
        raise ValueError(
            "R2-D13 D5 preflight task families differ from curriculum"
        )

    trainer = _trainer_config(
        plan=plan,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        max_grad_norm=max_grad_norm,
        checkpoint_every_optimizer_steps=(
            checkpoint_every_optimizer_steps
        ),
    )

    scale = projected_index.get("scale")
    if not isinstance(scale, dict):
        raise ValueError("R2-D13 projected scale evidence is missing")
    scale_floor_passed = scale.get("scale_floor_passed") is True
    preflight_allowed = bool(scale_floor_passed)

    if output_dir.exists():
        if (
            output_dir.is_symlink()
            or not output_dir.is_dir()
            or any(output_dir.iterdir())
        ):
            raise ValueError("R2-D13 output-dir must be new or empty")
    else:
        output_dir.mkdir(parents=True, exist_ok=False)
    output = output_dir.resolve(strict=True)

    _copy_view(view_dir, output / "view")
    _copy_regular(
        curriculum_plan_path.resolve(strict=True),
        output / "curriculum.json",
    )
    _copy_preflight(
        preflight_package_dir,
        output / "preflight",
    )

    preflight_script = _render_preflight_wrapper(
        repository_commit=repository_commit,
    ).encode("utf-8")
    seal_script = _render_seal_wrapper(
        repository_commit=repository_commit,
    ).encode("utf-8")
    train_script = _render_train_wrapper(
        repository_commit=repository_commit,
        recipe=recipe,
        trainer=trainer,
        max_run_seconds=max_run_seconds,
    ).encode("utf-8")

    for filename, data in (
        ("run_t4_preflight.sh", preflight_script),
        ("seal_t4_preflight.sh", seal_script),
        ("run_t4_train.sh", train_script),
    ):
        path = output / filename
        _atomic_write(path, data)
        path.chmod(0o755)

    body: dict[str, object] = {
        "schema": R2D13_CAMPAIGN_SCHEMA,
        "purpose": "frozen_dense_pretrain_campaign",
        "repository_commit": repository_commit,
        "stage": stage,
        "virtual_view_id": _require_sha256(
            view.get("view_id"),
            label="R2-D13 D12 view ID",
        ),
        "registry_generation": int(view["registry_generation"]),
        "projected_index_id": _require_sha256(
            projected_index.get("index_id"),
            label="R2-D13 projected index ID",
        ),
        "curriculum_plan_id": _require_sha256(
            plan.get("plan_id"),
            label="R2-D13 curriculum plan ID",
        ),
        "preflight_campaign_id": _require_sha256(
            preflight.get("campaign_id"),
            label="R2-D13 D5 campaign ID",
        ),
        "architecture_fingerprint": _require_sha256(
            projected_index.get("architecture_fingerprint"),
            label="R2-D13 architecture fingerprint",
        ),
        "tokenizer_sha256": _require_sha256(
            projected_index.get("tokenizer_sha256"),
            label="R2-D13 tokenizer SHA-256",
        ),
        "recipe": asdict(recipe),
        "recipe_fingerprint": recipe.fingerprint(),
        "trainer": asdict(trainer),
        "max_run_seconds": max_run_seconds,
        "scale": dict(scale),
        "scale_floor_passed": scale_floor_passed,
        "gpu_preflight_allowed": preflight_allowed,
        "training_launch_allowed": False,
        "release_held_out": True,
        "scripts": {
            "run_t4_preflight.sh": _sha256_bytes(preflight_script),
            "seal_t4_preflight.sh": _sha256_bytes(seal_script),
            "run_t4_train.sh": _sha256_bytes(train_script),
        },
    }
    campaign = dict(body)
    campaign["campaign_id"] = _campaign_identity(body)
    _atomic_write(
        output / "r2d13-campaign.json",
        _canonical_json(campaign) + b"\n",
    )

    verify_r2d13_campaign(
        output,
        workspace_root=workspace_root,
    )
    return campaign


def _load_campaign(package_dir: Path) -> dict[str, object]:
    payload = _strict_json(
        package_dir / "r2d13-campaign.json",
        label="R2-D13 campaign",
    )
    if payload.get("schema") != R2D13_CAMPAIGN_SCHEMA:
        raise ValueError("R2-D13 campaign schema mismatch")
    campaign_id = _require_sha256(
        payload.get("campaign_id"),
        label="R2-D13 campaign ID",
    )
    body = dict(payload)
    body.pop("campaign_id", None)
    if campaign_id != _campaign_identity(body):
        raise ValueError("R2-D13 campaign identity mismatch")
    return payload


def verify_r2d13_campaign(
    package_dir: Path,
    *,
    workspace_root: Path,
) -> dict[str, object]:
    if package_dir.is_symlink():
        raise ValueError("R2-D13 package-dir must not be a symlink")
    root = package_dir.resolve(strict=True)
    campaign = _load_campaign(root)
    _require_commit(campaign.get("repository_commit"))
    if campaign.get("purpose") != "frozen_dense_pretrain_campaign":
        raise ValueError("R2-D13 campaign purpose mismatch")
    if campaign.get("stage") != R2D13_SUPPORTED_STAGE:
        raise ValueError("R2-D13 campaign stage mismatch")
    if campaign.get("training_launch_allowed") is not False:
        raise ValueError(
            "R2-D13 base campaign must remain training-launch disabled"
        )
    if campaign.get("release_held_out") is not True:
        raise ValueError("R2-D13 release holdout is not locked")

    verified_view = verify_r2d12_view(
        root / "view",
        workspace_root=workspace_root,
    )
    view = verified_view["view"]
    if view.get("view_id") != campaign.get("virtual_view_id"):
        raise ValueError("R2-D13 D12 view identity mismatch")
    if int(view.get("registry_generation", -1)) != int(
        campaign.get("registry_generation", -2)
    ):
        raise ValueError("R2-D13 registry generation mismatch")

    raw_plan = _strict_json(
        root / "curriculum.json",
        label="R2-D13 curriculum plan",
    )
    weights = raw_plan.get("family_weights")
    if not isinstance(weights, dict) or not weights:
        raise ValueError("R2-D13 curriculum weights are missing")
    projected = projection_for_stage(
        verified_view["index"],
        stage=str(raw_plan.get("stage")),
        family_weights=weights,
    )
    if projected.get("index_id") != campaign.get("projected_index_id"):
        raise ValueError("R2-D13 projected corpus identity mismatch")
    plan = load_r2d8_plan(
        root / "curriculum.json",
        projected,
    )
    if plan.get("plan_id") != campaign.get("curriculum_plan_id"):
        raise ValueError("R2-D13 curriculum identity mismatch")

    preflight = verify_r2d5_package(root / "preflight")
    if preflight.get("campaign_id") != campaign.get(
        "preflight_campaign_id"
    ):
        raise ValueError("R2-D13 D5 preflight campaign mismatch")
    if preflight.get("repository_commit") != campaign.get(
        "repository_commit"
    ):
        raise ValueError("R2-D13 repository commit binding mismatch")
    if preflight.get("architecture_fingerprint") != campaign.get(
        "architecture_fingerprint"
    ):
        raise ValueError("R2-D13 preflight architecture mismatch")
    if preflight.get("tokenizer_sha256") != campaign.get(
        "tokenizer_sha256"
    ):
        raise ValueError("R2-D13 preflight tokenizer mismatch")

    recipe_raw = campaign.get("recipe")
    if not isinstance(recipe_raw, dict):
        raise ValueError("R2-D13 recipe is missing")
    recipe = R2ProductionTrainingRecipe(**recipe_raw)
    if (
        recipe.fingerprint() != campaign.get("recipe_fingerprint")
        or preflight.get("recipe_fingerprint")
        != campaign.get("recipe_fingerprint")
    ):
        raise ValueError("R2-D13 recipe fingerprint mismatch")
    if recipe.sequence_length != int(
        projected.get("sequence_length", -1)
    ):
        raise ValueError("R2-D13 recipe/corpus sequence mismatch")
    if set(preflight.get("task_families", [])) != set(weights):
        raise ValueError(
            "R2-D13 preflight/curriculum task-family mismatch"
        )

    trainer_raw = campaign.get("trainer")
    if not isinstance(trainer_raw, dict):
        raise ValueError("R2-D13 trainer config is missing")
    trainer = R2ProductionTrainerConfig(**trainer_raw)
    if (
        trainer.epochs != int(plan.get("epochs", -1))
        or trainer.seed != int(plan.get("seed", -1))
        or trainer.require_measured_cuda_preflight is not True
    ):
        raise ValueError("R2-D13 trainer/curriculum binding mismatch")

    max_run_seconds = float(campaign.get("max_run_seconds", float("nan")))
    if not math.isfinite(max_run_seconds) or max_run_seconds <= 0.0:
        raise ValueError("R2-D13 max_run_seconds is invalid")

    scale = projected.get("scale")
    if not isinstance(scale, dict) or scale != campaign.get("scale"):
        raise ValueError("R2-D13 stage scale evidence mismatch")
    passed = scale.get("scale_floor_passed") is True
    if campaign.get("scale_floor_passed") is not passed:
        raise ValueError("R2-D13 scale-floor decision mismatch")
    if campaign.get("gpu_preflight_allowed") is not passed:
        raise ValueError("R2-D13 GPU preflight gate mismatch")

    scripts = campaign.get("scripts")
    if not isinstance(scripts, dict) or set(scripts) != {
        "run_t4_preflight.sh",
        "seal_t4_preflight.sh",
        "run_t4_train.sh",
    }:
        raise ValueError("R2-D13 script evidence is invalid")
    for filename, expected in scripts.items():
        expected_sha = _require_sha256(
            expected,
            label=f"R2-D13 {filename} SHA-256",
        )
        if _sha256_file(root / filename) != expected_sha:
            raise ValueError(f"R2-D13 script hash mismatch: {filename}")

    return {
        "campaign": campaign,
        "view": verified_view,
        "projected_index": projected,
        "curriculum": plan,
        "preflight": preflight,
        "recipe": recipe,
        "trainer": trainer,
    }


def assert_r2d13_preflight_allowed(
    package_dir: Path,
    *,
    workspace_root: Path,
) -> dict[str, object]:
    verified = verify_r2d13_campaign(
        package_dir,
        workspace_root=workspace_root,
    )
    campaign = verified["campaign"]
    if campaign.get("gpu_preflight_allowed") is not True:
        scale = campaign.get("scale")
        ratio = (
            float(scale.get("tokens_per_parameter", 0.0))
            if isinstance(scale, dict)
            else 0.0
        )
        raise RuntimeError(
            "R2-D13 refuses GPU preflight below the stage scale floor: "
            f"tokens_per_parameter={ratio:.8f}"
        )
    return verified


def _load_receipt(path: Path) -> dict[str, object]:
    payload = _strict_json(
        path,
        label="R2-D13 preflight receipt",
    )
    if payload.get("schema") != R2D5_PREFLIGHT_RECEIPT_SCHEMA:
        raise ValueError("R2-D13 requires a D5 preflight receipt")
    return payload


def seal_r2d13_ready(
    *,
    package_dir: Path,
    workspace_root: Path,
    preflight_receipt_path: Path,
    output_path: Path,
) -> dict[str, object]:
    verified = assert_r2d13_preflight_allowed(
        package_dir,
        workspace_root=workspace_root,
    )
    campaign = verified["campaign"]
    recipe = verified["recipe"]
    assert isinstance(campaign, dict)
    assert isinstance(recipe, R2ProductionTrainingRecipe)

    receipt_raw = _load_receipt(preflight_receipt_path)
    evidence = load_r2d5_memory_receipt(
        preflight_receipt_path,
        architecture_fingerprint=str(
            campaign["architecture_fingerprint"]
        ),
        recipe=recipe,
    )
    if receipt_raw.get("campaign_id") != campaign.get(
        "preflight_campaign_id"
    ):
        raise ValueError(
            "R2-D13 preflight receipt belongs to another D5 campaign"
        )
    if evidence.passed is not True:
        raise ValueError("R2-D13 measured preflight did not pass")

    body = {
        "schema": R2D13_READY_SCHEMA,
        "campaign_id": campaign["campaign_id"],
        "preflight_receipt_id": _require_sha256(
            receipt_raw.get("receipt_id"),
            label="R2-D13 D5 receipt ID",
        ),
        "preflight_campaign_id": campaign["preflight_campaign_id"],
        "repository_commit": campaign["repository_commit"],
        "virtual_view_id": campaign["virtual_view_id"],
        "projected_index_id": campaign["projected_index_id"],
        "curriculum_plan_id": campaign["curriculum_plan_id"],
        "recipe_fingerprint": campaign["recipe_fingerprint"],
        "device_name": evidence.device_name,
        "peak_reserved_bytes": evidence.peak_reserved_bytes,
        "scale_floor_passed": True,
        "training_allowed": True,
    }
    ready = dict(body)
    ready["ready_id"] = _ready_identity(body)
    if output_path.exists() or output_path.is_symlink():
        raise ValueError(
            "R2-D13 ready receipt output must not already exist"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(
        output_path,
        _canonical_json(ready) + b"\n",
    )
    return ready


def verify_r2d13_ready(
    *,
    package_dir: Path,
    workspace_root: Path,
    ready_receipt_path: Path,
    preflight_receipt_path: Path,
) -> dict[str, object]:
    verified = assert_r2d13_preflight_allowed(
        package_dir,
        workspace_root=workspace_root,
    )
    campaign = verified["campaign"]
    recipe = verified["recipe"]
    assert isinstance(campaign, dict)
    assert isinstance(recipe, R2ProductionTrainingRecipe)

    ready = _strict_json(
        ready_receipt_path,
        label="R2-D13 ready receipt",
    )
    if ready.get("schema") != R2D13_READY_SCHEMA:
        raise ValueError("R2-D13 ready receipt schema mismatch")
    ready_id = _require_sha256(
        ready.get("ready_id"),
        label="R2-D13 ready ID",
    )
    body = dict(ready)
    body.pop("ready_id", None)
    if ready_id != _ready_identity(body):
        raise ValueError("R2-D13 ready receipt identity mismatch")
    if ready.get("training_allowed") is not True:
        raise ValueError("R2-D13 ready receipt does not allow training")

    preflight_raw = _load_receipt(preflight_receipt_path)
    evidence = load_r2d5_memory_receipt(
        preflight_receipt_path,
        architecture_fingerprint=str(
            campaign["architecture_fingerprint"]
        ),
        recipe=recipe,
    )
    expected = {
        "campaign_id": campaign["campaign_id"],
        "preflight_receipt_id": preflight_raw["receipt_id"],
        "preflight_campaign_id": campaign["preflight_campaign_id"],
        "repository_commit": campaign["repository_commit"],
        "virtual_view_id": campaign["virtual_view_id"],
        "projected_index_id": campaign["projected_index_id"],
        "curriculum_plan_id": campaign["curriculum_plan_id"],
        "recipe_fingerprint": campaign["recipe_fingerprint"],
        "device_name": evidence.device_name,
        "peak_reserved_bytes": evidence.peak_reserved_bytes,
    }
    for key, value in expected.items():
        if ready.get(key) != value:
            raise ValueError(
                f"R2-D13 ready receipt binding mismatch: {key}"
            )
    if ready.get("scale_floor_passed") is not True:
        raise ValueError("R2-D13 ready scale gate is not passed")
    return {
        **verified,
        "ready": ready,
        "preflight_receipt": preflight_raw,
    }
