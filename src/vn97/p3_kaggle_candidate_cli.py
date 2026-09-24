from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch

from .campaign import VN97CampaignCandidate
from .campaign_cli import (
    _estimate_parameter_count,
    _examples,
)
from .config import VN97Config
from .deployment_checkpoint import (
    build_deployment_checkpoint,
    load_deployment_checkpoint,
)
from .evaluation import (
    VN97ReleaseCriteria,
    VN97ReleaseQualityError,
    evaluate_vn97_language,
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
    CORPUS_PROFILE_ID,
    EPOCHS,
    LEARNED_TOKENS,
    MAX_MODEL_IMAGE_BYTES,
    MAX_PARAMETERS,
    MAX_RECURRENT_STATE_BYTES,
    MAX_TRAIN_WINDOWS,
    MAX_VALIDATION_LOSS,
    MAX_VALIDATION_WINDOWS,
    MIN_PAIR_COUNT,
    MIN_TARGET_TOKENS,
    MIN_VALIDATION_ACCURACY,
    SEQUENCE_LENGTH,
    TILE_COLS,
    TILE_ROWS,
    profile_sha256,
)
from .p3_language_campaign_cli import (
    _strict_canonical_json,
    _validate_corpus,
)
from .tokenizer import VN97Tokenizer, learn_byte_bpe
from .training import (
    VN97TrainingConfig,
    build_training_windows,
    render_chat_text,
    train_vn97_language,
)
from .training_cli import (
    _atomic_write,
    _load_records,
    _read_bounded_regular_file,
)


_MAX_CORPUS_MANIFEST_BYTES = 4 * 1024 * 1024


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
        learning_rate=float(
            raw["learning_rate"]
        ),
    )


def _manifest_release_sha256(
    corpus_dir: Path,
) -> str:
    data = _read_bounded_regular_file(
        corpus_dir
        / "corpus.vn97corpus1.json",
        max_bytes=_MAX_CORPUS_MANIFEST_BYTES,
    )
    manifest = _strict_canonical_json(
        data,
        label="VN97CORPUS1 manifest",
    )
    if (
        manifest.get("profile_id")
        != CORPUS_PROFILE_ID
    ):
        raise VN97P3KaggleError(
            "P3 corpus profile mismatch"
        )
    splits = manifest.get("splits")
    if not isinstance(splits, dict):
        raise VN97P3KaggleError(
            "P3 corpus split map is invalid"
        )
    release = splits.get("release")
    if (
        not isinstance(release, dict)
        or set(release)
        != {
            "bytes",
            "mode",
            "records",
            "sha256",
        }
        or release.get("mode") != "chat"
    ):
        raise VN97P3KaggleError(
            "P3 release split contract is invalid"
        )
    value = release.get("sha256")
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(
            ch not in "0123456789abcdef"
            for ch in value
        )
    ):
        raise VN97P3KaggleError(
            "P3 release split SHA-256 is invalid"
        )
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train exactly one frozen P3 VN97 candidate on CUDA and "
            "evaluate validation only. The sealed release split is "
            "not encoded or evaluated here."
        )
    )
    parser.add_argument(
        "--corpus-dir",
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
        default="cuda:0",
    )
    return parser


def main(
    argv: list[str] | None = None,
) -> int:
    args = _parser().parse_args(argv)
    if not args.device.startswith("cuda"):
        raise VN97P3KaggleError(
            "P3 sharded candidate training requires CUDA"
        )
    if not torch.cuda.is_available():
        raise VN97P3KaggleError(
            "CUDA is not available on this host"
        )

    candidate = _candidate(
        args.candidate_index
    )
    corpus_dir = Path(
        args.corpus_dir
    ).resolve(strict=True)
    (
        corpus_manifest_id,
        corpus_manifest_sha256,
    ) = _validate_corpus(corpus_dir)
    release_split_sha256 = (
        _manifest_release_sha256(
            corpus_dir
        )
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

    train_records, training_dataset_sha256 = (
        _load_records(
            [corpus_dir / "training.jsonl"],
            mode="chat",
            max_input_bytes=64
            * 1024
            * 1024,
            max_examples=100_000,
        )
    )
    (
        validation_records,
        validation_dataset_sha256,
    ) = _load_records(
        [corpus_dir / "validation.jsonl"],
        mode="chat",
        max_input_bytes=64
        * 1024
        * 1024,
        max_examples=100_000,
    )

    corpus = [
        render_chat_text(value)
        for value in train_records
    ]
    tokenizer_package = learn_byte_bpe(
        corpus,
        max_learned_tokens=LEARNED_TOKENS,
        min_pair_count=MIN_PAIR_COUNT,
    )
    tokenizer = VN97Tokenizer(
        tokenizer_package
    )
    tokenizer_bytes = (
        tokenizer_package.to_bytes()
    )
    tokenizer_sha256 = hashlib.sha256(
        tokenizer_bytes
    ).hexdigest()

    train_examples = _examples(
        train_records,
        mode="chat",
        tokenizer=tokenizer,
    )
    validation_examples = _examples(
        validation_records,
        mode="chat",
        tokenizer=tokenizer,
    )

    window_config = VN97TrainingConfig(
        sequence_length=SEQUENCE_LENGTH,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        learning_rate=1e-4,
        seed=0,
        shuffle=False,
        max_windows=MAX_TRAIN_WINDOWS,
    )
    train_windows = build_training_windows(
        train_examples,
        window_config,
        pad_token_id=tokenizer.pad_id,
    )
    validation_config = VN97TrainingConfig(
        sequence_length=SEQUENCE_LENGTH,
        batch_size=BATCH_SIZE,
        epochs=1,
        seed=0,
        shuffle=False,
        max_windows=MAX_VALIDATION_WINDOWS,
    )
    validation_windows = (
        build_training_windows(
            validation_examples,
            validation_config,
            pad_token_id=tokenizer.pad_id,
        )
    )

    model_config = VN97Config(
        vocab_size=tokenizer.vocab_size,
        d_model=candidate.d_model,
        n_layers=candidate.n_layers,
        d_state=candidate.d_state,
        embedding_rank=(
            candidate.embedding_rank
        ),
    )
    parameter_count = (
        _estimate_parameter_count(
            vocab_size=tokenizer.vocab_size,
            candidate=candidate,
        )
    )
    if parameter_count > MAX_PARAMETERS:
        raise VN97P3KaggleError(
            "frozen P3 candidate exceeds parameter budget"
        )

    footprint = (
        estimate_vn97_mobile_footprint(
            model_config,
            tokenizer_nbytes=len(
                tokenizer_bytes
            ),
            tile_rows=TILE_ROWS,
            tile_cols=TILE_COLS,
        )
    )
    budget = VN97MobileBudget(
        max_model_image_bytes=
            MAX_MODEL_IMAGE_BYTES,
        max_recurrent_state_bytes=
            MAX_RECURRENT_STATE_BYTES,
    )
    rejection = budget.rejection_status(
        footprint
    )
    if rejection is not None:
        raise VN97P3KaggleError(
            "frozen P3 candidate failed mobile admission: "
            f"{rejection}"
        )

    torch.manual_seed(candidate.seed)
    torch.cuda.manual_seed_all(
        candidate.seed
    )
    model = VN97LanguageCore(
        model_config
    )
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
        learning_rate=
            candidate.learning_rate,
        weight_decay=0.01,
        max_grad_norm=1.0,
        seed=candidate.seed,
        shuffle=True,
        max_windows=MAX_TRAIN_WINDOWS,
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
        batch_size=BATCH_SIZE,
        device=args.device,
    )

    criteria = VN97ReleaseCriteria(
        max_validation_loss=
            MAX_VALIDATION_LOSS,
        min_top1_accuracy=
            MIN_VALIDATION_ACCURACY,
        min_target_tokens=
            MIN_TARGET_TOKENS,
    )
    status = "ELIGIBLE"
    checkpoint_bytes: (
        bytes | None
    ) = None
    checkpoint_sha256: (
        str | None
    ) = None
    try:
        require_release_quality(
            evaluation,
            criteria,
        )
    except VN97ReleaseQualityError:
        status = "REJECTED_QUALITY"
    else:
        checkpoint_bytes = (
            build_deployment_checkpoint(
                model
            )
        )
        loaded = (
            load_deployment_checkpoint(
                checkpoint_bytes
            )
        )
        checkpoint_sha256 = (
            loaded.checkpoint_sha256
        )

    result = VN97P3CandidateResult(
        candidate_index=
            args.candidate_index,
        candidate=candidate,
        corpus_manifest_id=
            corpus_manifest_id,
        corpus_manifest_sha256=
            corpus_manifest_sha256,
        training_dataset_sha256=
            training_dataset_sha256,
        validation_dataset_sha256=
            validation_dataset_sha256,
        release_split_sha256=
            release_split_sha256,
        tokenizer_sha256=
            tokenizer_sha256,
        profile_sha256=
            profile_sha256(),
        parameter_count=
            parameter_count,
        mobile_footprint=
            footprint.canonical_object(),
        status=status,
        training_steps=training.steps,
        training_target_tokens=
            training.target_tokens,
        training_mean_loss=
            training.mean_loss,
        training_final_loss=
            training.final_loss,
        evaluation=evaluation,
        checkpoint_sha256=
            checkpoint_sha256,
        device=args.device,
    )

    _atomic_write(
        output / "tokenizer.vn97tk1",
        tokenizer_bytes,
    )
    if checkpoint_bytes is not None:
        _atomic_write(
            output
            / "candidate.vn97ck1",
            checkpoint_bytes,
        )
    _atomic_write(
        output
        / "candidate-report.vn97p3cand1.json",
        result.to_bytes(),
    )

    print(
        "VN97P3CAND1 "
        f"index={args.candidate_index} "
        f"candidate={result.candidate_id} "
        f"status={result.status} "
        f"validation_loss={evaluation.mean_loss:.10f} "
        f"validation_top1={evaluation.top1_accuracy:.10f}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(
            f"vn97-p3-kaggle-candidate: {exc}",
            file=sys.stderr,
        )
        raise
