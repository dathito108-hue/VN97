from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys
import time

import torch

from .campaign import VN97CampaignCandidate
from .campaign_cli import _estimate_parameter_count
from .config import VN97Config
from .deployment_checkpoint import (
    build_deployment_checkpoint,
    load_deployment_checkpoint,
)
from .evaluation import (
    VN97ReleaseCriteria,
    VN97ReleaseQualityError,
    require_release_quality,
)
from .mobile_budget import (
    VN97MobileBudget,
    estimate_vn97_mobile_footprint,
)
from .model import VN97LanguageCore
from .p3_kaggle import (
    VN97P3CandidateResult,
    VN97P3KaggleError,
)
from .p3_language_campaign import (
    BATCH_SIZE,
    CANDIDATES,
    EPOCHS,
    MAX_MODEL_IMAGE_BYTES,
    MAX_PARAMETERS,
    MAX_RECURRENT_STATE_BYTES,
    MAX_TRAIN_WINDOWS,
    MAX_VALIDATION_LOSS,
    MIN_TARGET_TOKENS,
    MIN_VALIDATION_ACCURACY,
    SEQUENCE_LENGTH,
    TILE_COLS,
    TILE_ROWS,
    profile_sha256,
)
from .p3_tensor_cache import (
    load_cache_manifest,
    load_tensor_bundle,
    train_vn97_from_tensors,
    evaluate_vn97_from_tensors,
)
from .tokenizer import (
    VN97TokenizerPackage,
)
from .training import VN97TrainingConfig
from .training_cli import (
    _atomic_write,
    _read_bounded_regular_file,
)


_MAX_TOKENIZER_BYTES = 128 * 1024 * 1024


def _candidate(index: int) -> VN97CampaignCandidate:
    if not 0 <= index < len(CANDIDATES):
        raise VN97P3KaggleError(
            f"candidate index must be in [0, {len(CANDIDATES)-1}]"
        )
    raw = CANDIDATES[index]
    return VN97CampaignCandidate(
        d_model=int(raw["d_model"]),
        n_layers=int(raw["n_layers"]),
        d_state=int(raw["d_state"]),
        embedding_rank=(
            None
            if raw["embedding_rank"] is None
            else int(raw["embedding_rank"])
        ),
        seed=int(raw["seed"]),
        learning_rate=float(raw["learning_rate"]),
    )


def _cache_split(
    manifest: dict[str, object],
    name: str,
) -> dict[str, object]:
    value = manifest.get(name)
    if (
        not isinstance(value, dict)
        or set(value)
        != {
            "cache_sha256",
            "dataset_sha256",
            "target_tokens",
            "windows",
        }
        or type(value.get("windows")) is not int
        or type(value.get("target_tokens")) is not int
        or value["windows"] <= 0
        or value["target_tokens"] <= 0
    ):
        raise VN97P3KaggleError(
            f"P3 cache {name} contract is invalid"
        )
    for key in (
        "cache_sha256",
        "dataset_sha256",
    ):
        digest = value.get(key)
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(
                ch not in "0123456789abcdef"
                for ch in digest
            )
        ):
            raise VN97P3KaggleError(
                f"P3 cache {name} {key} is invalid"
            )
    return value


def _require_sha256(
    value: object,
    *,
    label: str,
) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(ch not in "0123456789abcdef" for ch in value)
    ):
        raise VN97P3KaggleError(
            f"{label} must be lowercase SHA-256"
        )
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train exactly one frozen P3 candidate from the shared "
            "VN97P3CACHE1 tensor cache on one CUDA device."
        )
    )
    parser.add_argument(
        "--cache-dir",
        required=True,
    )
    parser.add_argument(
        "--candidate-index",
        required=True,
        type=int,
    )
    parser.add_argument(
        "--output-dir",
        required=True,
    )
    parser.add_argument(
        "--device",
        required=True,
    )
    parser.add_argument(
        "--cpu-prefetch-workers",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--micro-batch-size",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--progress-interval-steps",
        type=int,
        default=50,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not 0 <= args.cpu_prefetch_workers <= 8:
        raise VN97P3KaggleError(
            "cpu-prefetch-workers must be in [0, 8]"
        )
    if args.progress_interval_steps <= 0:
        raise VN97P3KaggleError(
            "progress-interval-steps must be positive"
        )
    if (
        args.micro_batch_size <= 0
        or args.micro_batch_size > BATCH_SIZE
        or BATCH_SIZE % args.micro_batch_size != 0
    ):
        raise VN97P3KaggleError(
            "micro-batch-size must be a positive divisor of logical P3 batch size"
        )
    if not args.device.startswith("cuda"):
        raise VN97P3KaggleError(
            "cached P3 candidate training requires CUDA"
        )
    if not torch.cuda.is_available():
        raise VN97P3KaggleError(
            "CUDA is not available on this host"
        )

    device = torch.device(args.device)
    if (
        device.index is not None
        and not 0
        <= device.index
        < torch.cuda.device_count()
    ):
        raise VN97P3KaggleError(
            "requested CUDA device is unavailable"
        )
    if device.index is not None:
        torch.cuda.set_device(device.index)

    candidate = _candidate(
        args.candidate_index
    )
    cache_dir = Path(
        args.cache_dir
    ).resolve(strict=True)
    manifest = load_cache_manifest(
        cache_dir
    )
    if manifest.get("profile_sha256") != profile_sha256():
        raise VN97P3KaggleError(
            "P3 cache profile identity mismatch"
        )
    if manifest.get("sequence_length") != SEQUENCE_LENGTH:
        raise VN97P3KaggleError(
            "P3 cache sequence length mismatch"
        )

    corpus_manifest_id = _require_sha256(
        manifest.get("corpus_manifest_id"),
        label="corpus manifest ID",
    )
    corpus_manifest_sha256 = _require_sha256(
        manifest.get("corpus_manifest_sha256"),
        label="corpus manifest SHA-256",
    )
    release_split_sha256 = _require_sha256(
        manifest.get("release_split_sha256"),
        label="release split SHA-256",
    )
    tokenizer_sha256 = _require_sha256(
        manifest.get("tokenizer_sha256"),
        label="tokenizer SHA-256",
    )

    train_spec = _cache_split(
        manifest,
        "training",
    )
    validation_spec = _cache_split(
        manifest,
        "validation",
    )

    tokenizer_bytes = _read_bounded_regular_file(
        cache_dir / "tokenizer.vn97tk1",
        max_bytes=_MAX_TOKENIZER_BYTES,
    )
    if (
        hashlib.sha256(tokenizer_bytes).hexdigest()
        != tokenizer_sha256
    ):
        raise VN97P3KaggleError(
            "P3 cache tokenizer SHA-256 mismatch"
        )
    tokenizer_package = VN97TokenizerPackage.from_bytes(
        tokenizer_bytes
    )

    train_inputs, train_labels = load_tensor_bundle(
        cache_dir / "training-windows.pt",
        expected_sha256=str(
            train_spec["cache_sha256"]
        ),
        expected_windows=int(
            train_spec["windows"]
        ),
        expected_sequence_length=SEQUENCE_LENGTH,
        expected_target_tokens=int(
            train_spec["target_tokens"]
        ),
    )
    validation_inputs, validation_labels = load_tensor_bundle(
        cache_dir / "validation-windows.pt",
        expected_sha256=str(
            validation_spec["cache_sha256"]
        ),
        expected_windows=int(
            validation_spec["windows"]
        ),
        expected_sequence_length=SEQUENCE_LENGTH,
        expected_target_tokens=int(
            validation_spec["target_tokens"]
        ),
    )

    output = Path(args.output_dir)
    if output.exists():
        if (
            output.is_symlink()
            or not output.is_dir()
            or any(output.iterdir())
        ):
            raise VN97P3KaggleError(
                "candidate output-dir must be new or empty"
            )
    else:
        output.mkdir(
            parents=True,
            exist_ok=False,
        )
    output = output.resolve(strict=True)

    model_config = VN97Config(
        vocab_size=tokenizer_package.vocab_size,
        d_model=candidate.d_model,
        n_layers=candidate.n_layers,
        d_state=candidate.d_state,
        embedding_rank=candidate.embedding_rank,
    )
    parameter_count = _estimate_parameter_count(
        vocab_size=tokenizer_package.vocab_size,
        candidate=candidate,
    )
    if parameter_count > MAX_PARAMETERS:
        raise VN97P3KaggleError(
            "frozen P3 candidate exceeds parameter budget"
        )

    footprint = estimate_vn97_mobile_footprint(
        model_config,
        tokenizer_nbytes=len(tokenizer_bytes),
        tile_rows=TILE_ROWS,
        tile_cols=TILE_COLS,
    )
    rejection = VN97MobileBudget(
        max_model_image_bytes=MAX_MODEL_IMAGE_BYTES,
        max_recurrent_state_bytes=MAX_RECURRENT_STATE_BYTES,
    ).rejection_status(footprint)
    if rejection is not None:
        raise VN97P3KaggleError(
            "frozen P3 candidate failed mobile admission: "
            f"{rejection}"
        )

    torch.manual_seed(candidate.seed)
    torch.cuda.manual_seed_all(candidate.seed)
    model = VN97LanguageCore(model_config)
    actual_parameter_count = sum(
        int(parameter.numel())
        for parameter in model.parameters()
    )
    if actual_parameter_count != parameter_count:
        raise VN97P3KaggleError(
            "candidate parameter count changed at materialization"
        )

    training_config = VN97TrainingConfig(
        sequence_length=SEQUENCE_LENGTH,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        learning_rate=candidate.learning_rate,
        weight_decay=0.01,
        max_grad_norm=1.0,
        seed=candidate.seed,
        shuffle=True,
        max_windows=MAX_TRAIN_WINDOWS,
    )

    started = time.time()
    print(
        "P3 GPU TRAIN "
        f"candidate={args.candidate_index} "
        f"id={candidate.candidate_id} "
        f"device={args.device} "
        f"windows={train_inputs.shape[0]} "
        f"cpu_prefetch_workers={args.cpu_prefetch_workers} "
        f"micro_batch_size={args.micro_batch_size}",
        flush=True,
    )
    training = train_vn97_from_tensors(
        model,
        train_inputs,
        train_labels,
        training_config,
        device=args.device,
        cpu_prefetch_workers=
            args.cpu_prefetch_workers,
        micro_batch_size=
            args.micro_batch_size,
        progress_label=
            f"candidate={args.candidate_index}",
        progress_interval_steps=
            args.progress_interval_steps,
    )
    evaluation = evaluate_vn97_from_tensors(
        model,
        validation_inputs,
        validation_labels,
        batch_size=BATCH_SIZE,
        device=args.device,
        cpu_prefetch_workers=
            args.cpu_prefetch_workers,
        micro_batch_size=
            args.micro_batch_size,
    )

    criteria = VN97ReleaseCriteria(
        max_validation_loss=MAX_VALIDATION_LOSS,
        min_top1_accuracy=MIN_VALIDATION_ACCURACY,
        min_target_tokens=MIN_TARGET_TOKENS,
    )
    status = "ELIGIBLE"
    checkpoint_bytes: bytes | None = None
    checkpoint_sha256: str | None = None
    try:
        require_release_quality(
            evaluation,
            criteria,
        )
    except VN97ReleaseQualityError:
        status = "REJECTED_QUALITY"
    else:
        checkpoint_bytes = build_deployment_checkpoint(
            model
        )
        loaded = load_deployment_checkpoint(
            checkpoint_bytes
        )
        checkpoint_sha256 = loaded.checkpoint_sha256

    result = VN97P3CandidateResult(
        candidate_index=args.candidate_index,
        candidate=candidate,
        corpus_manifest_id=corpus_manifest_id,
        corpus_manifest_sha256=corpus_manifest_sha256,
        training_dataset_sha256=str(
            train_spec["dataset_sha256"]
        ),
        validation_dataset_sha256=str(
            validation_spec["dataset_sha256"]
        ),
        release_split_sha256=release_split_sha256,
        tokenizer_sha256=tokenizer_sha256,
        profile_sha256=profile_sha256(),
        parameter_count=parameter_count,
        mobile_footprint=footprint.canonical_object(),
        status=status,
        training_steps=training.steps,
        training_target_tokens=training.target_tokens,
        training_mean_loss=training.mean_loss,
        training_final_loss=training.final_loss,
        evaluation=evaluation,
        checkpoint_sha256=checkpoint_sha256,
        device=args.device,
    )

    _atomic_write(
        output / "tokenizer.vn97tk1",
        tokenizer_bytes,
    )
    if checkpoint_bytes is not None:
        _atomic_write(
            output / "candidate.vn97ck1",
            checkpoint_bytes,
        )
    _atomic_write(
        output / "candidate-report.vn97p3cand1.json",
        result.to_bytes(),
    )

    print(
        "VN97P3CAND1 "
        f"index={args.candidate_index} "
        f"candidate={result.candidate_id} "
        f"status={result.status} "
        f"validation_loss={evaluation.mean_loss:.10f} "
        f"validation_top1={evaluation.top1_accuracy:.10f} "
        f"elapsed_s={time.time()-started:.1f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(
            f"vn97-p3-kaggle-cached-candidate: {exc}",
            file=sys.stderr,
        )
        raise
