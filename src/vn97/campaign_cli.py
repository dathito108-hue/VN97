from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import sys

import torch

from .campaign import (
    VN97CampaignCandidate,
    VN97CampaignObservation,
    VN97CampaignError,
    select_best_campaign_candidate,
)
from .config import VN97Config
from .deployment_checkpoint import (
    build_deployment_checkpoint,
    load_deployment_checkpoint,
)
from .dataset_split import require_disjoint_dataset_splits
from .evaluation import (
    VN97ReleaseCriteria,
    VN97ReleaseQualityError,
    evaluate_vn97_language,
    require_release_quality,
)
from .model import VN97LanguageCore
from .mobile_budget import (
    VN97MobileBudget,
    estimate_vn97_mobile_footprint,
)
from .tokenizer import (
    VN97Tokenizer,
    learn_byte_bpe,
    select_deterministic_tokenizer_corpus,
)
from .training import (
    VN97TrainingConfig,
    build_training_windows,
    encode_causal_text,
    encode_chat_messages,
    render_chat_text,
    train_vn97_language,
)
from .training_cli import (
    _atomic_write,
    _canonical_json,
    _load_records,
    _read_bounded_regular_file,
)


_MAX_CAMPAIGN_MANIFEST_BYTES = 256 * 1024
_MAX_CANDIDATES = 64


def _strict_json(data: bytes) -> object:
    duplicates: list[str] = []

    def hook(pairs):
        out = {}
        for key, value in pairs:
            if key in out:
                duplicates.append(key)
            out[key] = value
        return out

    try:
        value = json.loads(
            data.decode("utf-8", errors="strict"),
            object_pairs_hook=hook,
            parse_constant=lambda raw: (_ for _ in ()).throw(
                ValueError(raw)
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("campaign manifest must be strict UTF-8 JSON") from exc
    if duplicates:
        raise ValueError("campaign manifest contains duplicate object keys")
    return value


def _load_candidates(path: Path) -> tuple[VN97CampaignCandidate, ...]:
    raw = _read_bounded_regular_file(
        path,
        max_bytes=_MAX_CAMPAIGN_MANIFEST_BYTES,
    )
    root = _strict_json(raw)
    if (
        not isinstance(root, dict)
        or set(root) != {"candidates", "schema"}
        or root["schema"] != "VN97CAMPDEF1"
        or not isinstance(root["candidates"], list)
    ):
        raise ValueError("campaign manifest schema is invalid")
    items = root["candidates"]
    if not 1 <= len(items) <= _MAX_CANDIDATES:
        raise ValueError("campaign candidate count is outside bounds")

    result: list[VN97CampaignCandidate] = []
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, dict) or set(item) != {
            "d_model",
            "d_state",
            "embedding_rank",
            "learning_rate",
            "n_layers",
            "seed",
        }:
            raise ValueError("campaign candidate fields are invalid")
        if any(
            type(item[key]) is not int
            for key in ("d_model", "d_state", "n_layers", "seed")
        ):
            raise ValueError(
                "campaign integer candidate fields must be integers"
            )
        rank = item["embedding_rank"]
        if rank is not None and type(rank) is not int:
            raise ValueError(
                "campaign embedding_rank must be integer or null"
            )
        lr = item["learning_rate"]
        if type(lr) not in (int, float):
            raise ValueError(
                "campaign learning_rate must be numeric"
            )
        candidate = VN97CampaignCandidate(
            d_model=item["d_model"],
            n_layers=item["n_layers"],
            d_state=item["d_state"],
            embedding_rank=rank,
            seed=item["seed"],
            learning_rate=float(lr),
        )
        if candidate.candidate_id in seen:
            raise ValueError("campaign contains duplicate candidate")
        seen.add(candidate.candidate_id)
        result.append(candidate)
    return tuple(result)


def _file_identities(
    paths: list[Path],
    *,
    label: str,
) -> set[tuple[int, int]]:
    identities: set[tuple[int, int]] = set()
    for path in paths:
        try:
            info = os.stat(path, follow_symlinks=False)
        except OSError as exc:
            raise ValueError(
                f"{label} path is unavailable: {path}"
            ) from exc
        if not stat.S_ISREG(info.st_mode):
            raise ValueError(
                f"{label} path must be a regular non-symlink file: {path}"
            )
        identity = (int(info.st_dev), int(info.st_ino))
        if identity in identities:
            raise ValueError(
                f"{label} contains the same physical file more than once"
            )
        identities.add(identity)
    return identities


def _estimate_parameter_count(
    *,
    vocab_size: int,
    candidate: VN97CampaignCandidate,
) -> int:
    config = VN97Config(
        vocab_size=vocab_size,
        d_model=candidate.d_model,
        n_layers=candidate.n_layers,
        d_state=candidate.d_state,
        embedding_rank=candidate.embedding_rank,
    )
    try:
        with torch.device("meta"):
            probe = VN97LanguageCore(config)
    except Exception as exc:
        raise VN97CampaignError(
            "candidate parameter budget probe failed"
        ) from exc
    count = sum(
        int(parameter.numel())
        for parameter in probe.parameters()
    )
    del probe
    if count <= 0:
        raise VN97CampaignError(
            "candidate parameter count must be positive"
        )
    return count


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train/evaluate multiple canonical VN97 candidates and promote one best checkpoint."
        )
    )
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--validation-input", action="append", required=True)
    parser.add_argument("--release-input", action="append", required=True)
    parser.add_argument("--format", choices=("text", "chat"), default="chat")
    parser.add_argument(
        "--validation-format",
        choices=("text", "chat"),
        default=None,
    )
    parser.add_argument(
        "--release-format",
        choices=("text", "chat"),
        default=None,
    )
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--output-dir", required=True)

    parser.add_argument("--max-input-bytes", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--max-examples", type=int, default=100_000)
    parser.add_argument("--max-windows", type=int, default=10_000)
    parser.add_argument(
        "--validation-max-input-bytes",
        type=int,
        default=64 * 1024 * 1024,
    )
    parser.add_argument(
        "--validation-max-examples",
        type=int,
        default=100_000,
    )
    parser.add_argument(
        "--validation-max-windows",
        type=int,
        default=10_000,
    )
    parser.add_argument(
        "--release-max-input-bytes",
        type=int,
        default=64 * 1024 * 1024,
    )
    parser.add_argument(
        "--release-max-examples",
        type=int,
        default=100_000,
    )
    parser.add_argument(
        "--release-max-windows",
        type=int,
        default=10_000,
    )

    parser.add_argument("--learned-tokens", type=int, default=2048)
    parser.add_argument("--min-pair-count", type=int, default=2)
    parser.add_argument(
        "--tokenizer-max-records",
        type=int,
        default=0,
        help=(
            "When positive, learn the tokenizer from a deterministic "
            "training-only hash-ranked subset of at most this many records."
        ),
    )
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--validation-batch-size", type=int, default=4)
    parser.add_argument("--release-batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-parameters", type=int, required=True)
    parser.add_argument(
        "--max-model-image-bytes",
        type=int,
        default=512 * 1024 * 1024,
    )
    parser.add_argument(
        "--max-recurrent-state-bytes",
        type=int,
        default=512 * 1024 * 1024,
    )
    parser.add_argument("--deployment-tile-rows", type=int, default=16)
    parser.add_argument("--deployment-tile-cols", type=int, default=16)

    parser.add_argument("--max-validation-loss", type=float, required=True)
    parser.add_argument(
        "--min-validation-accuracy",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--min-validation-target-tokens",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--release-max-validation-loss",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--release-min-validation-accuracy",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--release-min-validation-target-tokens",
        type=int,
        default=None,
    )
    return parser


def _examples(records, *, mode: str, tokenizer: VN97Tokenizer):
    if mode == "text":
        return [
            encode_causal_text(tokenizer, value)
            for value in records
        ]
    return [
        encode_chat_messages(tokenizer, value)
        for value in records
    ]


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if (
        args.max_parameters <= 0
        or args.max_model_image_bytes <= 0
        or args.max_recurrent_state_bytes <= 0
        or args.max_input_bytes <= 0
        or args.max_examples <= 0
        or args.max_windows <= 0
        or args.validation_max_input_bytes <= 0
        or args.validation_max_examples <= 0
        or args.validation_max_windows <= 0
        or args.release_max_input_bytes <= 0
        or args.release_max_examples <= 0
        or args.release_max_windows <= 0
        or args.epochs <= 0
        or args.batch_size <= 0
        or args.validation_batch_size <= 0
        or args.release_batch_size <= 0
        or args.tokenizer_max_records < 0
        or not 0 < args.deployment_tile_rows <= 256
        or not 0 < args.deployment_tile_cols <= 256
    ):
        raise ValueError("campaign resource/training bounds must be positive")

    train_paths = [Path(value) for value in args.input]
    validation_paths = [Path(value) for value in args.validation_input]
    release_paths = [Path(value) for value in args.release_input]
    train_ids = _file_identities(
        train_paths,
        label="campaign training input",
    )
    validation_ids = _file_identities(
        validation_paths,
        label="campaign validation input",
    )
    release_ids = _file_identities(
        release_paths,
        label="campaign release input",
    )
    if (
        train_ids.intersection(validation_ids)
        or train_ids.intersection(release_ids)
        or validation_ids.intersection(release_ids)
    ):
        raise ValueError(
            "campaign training, validation and release inputs must be physically distinct files"
        )

    train_records, train_sha256 = _load_records(
        train_paths,
        mode=args.format,
        max_input_bytes=args.max_input_bytes,
        max_examples=args.max_examples,
    )
    validation_mode = args.validation_format or args.format
    validation_records, validation_sha256 = _load_records(
        validation_paths,
        mode=validation_mode,
        max_input_bytes=args.validation_max_input_bytes,
        max_examples=args.validation_max_examples,
    )
    release_mode = args.release_format or validation_mode
    release_records, release_sha256 = _load_records(
        release_paths,
        mode=release_mode,
        max_input_bytes=args.release_max_input_bytes,
        max_examples=args.release_max_examples,
    )
    require_disjoint_dataset_splits(
        train_records,
        training_mode=args.format,
        validation_records=validation_records,
        validation_mode=validation_mode,
        release_records=release_records,
        release_mode=release_mode,
    )
    corpus = (
        train_records
        if args.format == "text"
        else [render_chat_text(value) for value in train_records]
    )
    tokenizer_corpus = (
        corpus
        if args.tokenizer_max_records == 0
        else select_deterministic_tokenizer_corpus(
            corpus,
            max_samples=args.tokenizer_max_records,
        )
    )
    tokenizer_package = learn_byte_bpe(
        tokenizer_corpus,
        max_learned_tokens=args.learned_tokens,
        min_pair_count=args.min_pair_count,
    )
    tokenizer = VN97Tokenizer(tokenizer_package)
    tokenizer_bytes = tokenizer_package.to_bytes()
    tokenizer_sha256 = hashlib.sha256(tokenizer_bytes).hexdigest()

    train_examples = _examples(
        train_records,
        mode=args.format,
        tokenizer=tokenizer,
    )
    validation_examples = _examples(
        validation_records,
        mode=validation_mode,
        tokenizer=tokenizer,
    )

    candidates = _load_candidates(Path(args.campaign))
    criteria = VN97ReleaseCriteria(
        max_validation_loss=args.max_validation_loss,
        min_top1_accuracy=args.min_validation_accuracy,
        min_target_tokens=args.min_validation_target_tokens,
    )
    release_criteria = VN97ReleaseCriteria(
        max_validation_loss=(
            criteria.max_validation_loss
            if args.release_max_validation_loss is None
            else args.release_max_validation_loss
        ),
        min_top1_accuracy=(
            criteria.min_top1_accuracy
            if args.release_min_validation_accuracy is None
            else args.release_min_validation_accuracy
        ),
        min_target_tokens=(
            criteria.min_target_tokens
            if args.release_min_validation_target_tokens is None
            else args.release_min_validation_target_tokens
        ),
    )
    mobile_budget = VN97MobileBudget(
        max_model_image_bytes=args.max_model_image_bytes,
        max_recurrent_state_bytes=args.max_recurrent_state_bytes,
    )

    # Materialize the exact same supervised windows once for every candidate.
    window_config = VN97TrainingConfig(
        sequence_length=args.sequence_length,
        stride=args.stride,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=1e-4,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        seed=0,
        shuffle=False,
        max_windows=args.max_windows,
    )
    train_windows = build_training_windows(
        train_examples,
        window_config,
        pad_token_id=tokenizer.pad_id,
    )
    validation_window_config = VN97TrainingConfig(
        sequence_length=args.sequence_length,
        stride=args.stride,
        batch_size=args.validation_batch_size,
        epochs=1,
        seed=0,
        shuffle=False,
        max_windows=args.validation_max_windows,
    )
    validation_windows = build_training_windows(
        validation_examples,
        validation_window_config,
        pad_token_id=tokenizer.pad_id,
    )

    observations: list[VN97CampaignObservation] = []
    rows: list[dict[str, object]] = []
    best_checkpoint: bytes | None = None
    best_observation: VN97CampaignObservation | None = None

    for candidate in candidates:
        candidate_config = VN97Config(
            vocab_size=tokenizer.vocab_size,
            d_model=candidate.d_model,
            n_layers=candidate.n_layers,
            d_state=candidate.d_state,
            embedding_rank=candidate.embedding_rank,
        )
        parameter_count = _estimate_parameter_count(
            vocab_size=tokenizer.vocab_size,
            candidate=candidate,
        )
        footprint = estimate_vn97_mobile_footprint(
            candidate_config,
            tokenizer_nbytes=len(tokenizer_bytes),
            tile_rows=args.deployment_tile_rows,
            tile_cols=args.deployment_tile_cols,
        )
        admission = {
            "candidate": candidate.canonical_object(),
            "candidate_id": candidate.candidate_id,
            "mobile_footprint": footprint.canonical_object(),
            "parameter_count": parameter_count,
        }
        if parameter_count > args.max_parameters:
            rows.append(
                {
                    **admission,
                    "status": "REJECTED_PARAMETER_BUDGET",
                }
            )
            continue

        mobile_rejection = mobile_budget.rejection_status(footprint)
        if mobile_rejection is not None:
            rows.append(
                {
                    **admission,
                    "status": mobile_rejection,
                }
            )
            continue

        torch.manual_seed(candidate.seed)
        model = VN97LanguageCore(candidate_config)
        actual_parameter_count = sum(
            int(parameter.numel())
            for parameter in model.parameters()
        )
        if actual_parameter_count != parameter_count:
            raise VN97CampaignError(
                "candidate parameter count changed between meta probe and materialization"
            )

        training_config = VN97TrainingConfig(
            sequence_length=args.sequence_length,
            stride=args.stride,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=candidate.learning_rate,
            weight_decay=args.weight_decay,
            max_grad_norm=args.max_grad_norm,
            seed=candidate.seed,
            shuffle=True,
            max_windows=args.max_windows,
        )
        training = train_vn97_language(
            model,
            train_windows,
            training_config,
            device=args.device,
        )
        evaluation = evaluate_vn97_language(
            model,
            validation_windows,
            batch_size=args.validation_batch_size,
            device=args.device,
        )

        status = "ELIGIBLE"
        try:
            require_release_quality(evaluation, criteria)
        except VN97ReleaseQualityError:
            status = "REJECTED_QUALITY"

        row = {
            "candidate": candidate.canonical_object(),
            "candidate_id": candidate.candidate_id,
            "evaluation": {
                "mean_loss": evaluation.mean_loss,
                "target_tokens": evaluation.target_tokens,
                "top1_accuracy": evaluation.top1_accuracy,
                "windows": evaluation.windows,
            },
            "mobile_footprint": footprint.canonical_object(),
            "parameter_count": parameter_count,
            "status": status,
            "training": {
                "final_loss": training.final_loss,
                "mean_loss": training.mean_loss,
                "steps": training.steps,
                "target_tokens": training.target_tokens,
            },
        }

        if status == "ELIGIBLE":
            checkpoint = build_deployment_checkpoint(model)
            restored = load_deployment_checkpoint(checkpoint)
            checkpoint_sha256 = restored.checkpoint_sha256
            observation = VN97CampaignObservation(
                candidate=candidate,
                parameter_count=parameter_count,
                evaluation=evaluation,
                checkpoint_sha256=checkpoint_sha256,
            )
            observations.append(observation)
            row["checkpoint_sha256"] = checkpoint_sha256

            selected = select_best_campaign_candidate(observations)
            if (
                best_observation is None
                or selected.candidate.candidate_id
                != best_observation.candidate.candidate_id
            ):
                if (
                    selected.candidate.candidate_id
                    == observation.candidate.candidate_id
                ):
                    best_checkpoint = checkpoint
                best_observation = selected

        rows.append(row)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    best = select_best_campaign_candidate(observations)
    if (
        best_checkpoint is None
        or best_observation is None
        or best.candidate.candidate_id
        != best_observation.candidate.candidate_id
    ):
        raise VN97CampaignError(
            "internal campaign best-checkpoint tracking mismatch"
        )

    # The release split is deliberately first encoded/evaluated only after the
    # validation-selected winner is fixed. A release failure never falls back to
    # another candidate, because that would turn the sealed set into a selector.
    release_examples = _examples(
        release_records,
        mode=release_mode,
        tokenizer=tokenizer,
    )
    release_window_config = VN97TrainingConfig(
        sequence_length=args.sequence_length,
        stride=args.stride,
        batch_size=args.release_batch_size,
        epochs=1,
        seed=0,
        shuffle=False,
        max_windows=args.release_max_windows,
    )
    release_windows = build_training_windows(
        release_examples,
        release_window_config,
        pad_token_id=tokenizer.pad_id,
    )
    release_loaded = load_deployment_checkpoint(best_checkpoint)
    if release_loaded.checkpoint_sha256 != best.checkpoint_sha256:
        raise VN97CampaignError(
            "selected checkpoint identity changed before sealed release evaluation"
        )
    release_evaluation = evaluate_vn97_language(
        release_loaded.model,
        release_windows,
        batch_size=args.release_batch_size,
        device=args.device,
    )
    try:
        require_release_quality(
            release_evaluation,
            release_criteria,
        )
    except VN97ReleaseQualityError as exc:
        raise VN97CampaignError(
            "selected campaign winner failed sealed release gate"
        ) from exc

    output = Path(args.output_dir)
    if output.is_symlink():
        raise ValueError("campaign output-dir must not be a symlink")
    output.mkdir(parents=True, exist_ok=True)

    _atomic_write(output / "tokenizer.vn97tk1", tokenizer_bytes)
    _atomic_write(output / "model.vn97ck1", best_checkpoint)

    report = {
        "candidates": rows,
        "criteria": {
            "deployment_tile_cols": args.deployment_tile_cols,
            "deployment_tile_rows": args.deployment_tile_rows,
            "max_model_image_bytes": mobile_budget.max_model_image_bytes,
            "max_parameters": args.max_parameters,
            "max_recurrent_state_bytes": mobile_budget.max_recurrent_state_bytes,
            "max_validation_loss": criteria.max_validation_loss,
            "min_top1_accuracy": criteria.min_top1_accuracy,
            "min_validation_target_tokens": criteria.min_target_tokens,
            "release_max_validation_loss": release_criteria.max_validation_loss,
            "release_min_top1_accuracy": release_criteria.min_top1_accuracy,
            "release_min_validation_target_tokens": release_criteria.min_target_tokens,
        },
        "release_evaluation": {
            "mean_loss": release_evaluation.mean_loss,
            "target_tokens": release_evaluation.target_tokens,
            "top1_accuracy": release_evaluation.top1_accuracy,
            "windows": release_evaluation.windows,
        },
        "schema": "VN97CAMP2",
        "selected_candidate_id": best.candidate.candidate_id,
        "selected_checkpoint_sha256": best.checkpoint_sha256,
        "tokenizer_sha256": tokenizer_sha256,
        "training_dataset_sha256": train_sha256,
        "validation_dataset_sha256": validation_sha256,
        "release_dataset_sha256": release_sha256,
    }
    _atomic_write(
        output / "campaign-report.json",
        _canonical_json(report),
    )

    print(
        "VN97CAMP2 "
        f"selected={best.candidate.candidate_id} "
        f"checkpoint_sha256={best.checkpoint_sha256}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"vn97-campaign: {exc}", file=sys.stderr)
        raise
