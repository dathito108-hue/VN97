from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch

from .campaign import (
    VN97CampaignObservation,
    select_best_campaign_candidate,
)
from .campaign_cli import _examples
from .deployment_checkpoint import (
    load_deployment_checkpoint_file,
)
from .evaluation import (
    VN97ReleaseCriteria,
    VN97ReleaseQualityError,
    evaluate_vn97_language,
    require_release_quality,
)
from .model_image import build_model_image
from .p3_kaggle import (
    VN97P3KaggleError,
    parse_candidate_result,
)
from .p3_language_campaign import (
    BATCH_SIZE,
    CANDIDATES,
    MAX_MODEL_IMAGE_BYTES,
    MAX_PARAMETERS,
    MAX_RECURRENT_STATE_BYTES,
    MAX_RELEASE_WINDOWS,
    MAX_VALIDATION_LOSS,
    MIN_TARGET_TOKENS,
    MIN_VALIDATION_ACCURACY,
    RELEASE_MAX_VALIDATION_LOSS,
    RELEASE_MIN_VALIDATION_ACCURACY,
    SEQUENCE_LENGTH,
    TILE_COLS,
    TILE_ROWS,
    VN97P3CampaignReceipt,
    candidate_ids,
    profile_sha256,
)
from .p3_language_campaign_cli import (
    _validate_corpus,
)
from .tokenizer import (
    VN97Tokenizer,
    VN97TokenizerPackage,
)
from .training import (
    VN97TrainingConfig,
    build_training_windows,
)
from .training_cli import (
    _atomic_write,
    _canonical_json,
    _load_records,
    _read_bounded_regular_file,
)


_MAX_CANDIDATE_REPORT_BYTES = 4 * 1024 * 1024
_MAX_TOKENIZER_BYTES = 128 * 1024 * 1024


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Combine the four validation-only P3 candidate shards, "
            "select one deterministic winner, and only then evaluate "
            "the sealed release split."
        )
    )
    parser.add_argument(
        "--corpus-dir",
        required=True,
    )
    parser.add_argument(
        "--candidate-dir",
        action="append",
        required=True,
        help=(
            "Repeat exactly four times, one for each VN97P3CAND1 output."
        ),
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


def _release_manifest_sha256(
    corpus_dir: Path,
) -> str:
    data = _read_bounded_regular_file(
        corpus_dir
        / "corpus.vn97corpus1.json",
        max_bytes=4 * 1024 * 1024,
    )
    try:
        manifest = json.loads(
            data.decode("utf-8"),
        )
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
    ) as exc:
        raise VN97P3KaggleError(
            "could not parse VN97CORPUS1 manifest"
        ) from exc
    splits = manifest.get("splits")
    if not isinstance(splits, dict):
        raise VN97P3KaggleError(
            "VN97CORPUS1 split map is invalid"
        )
    release = splits.get("release")
    if not isinstance(release, dict):
        raise VN97P3KaggleError(
            "VN97CORPUS1 release split is invalid"
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
            "VN97CORPUS1 release SHA-256 is invalid"
        )
    return value


def _load_shards(
    paths: list[Path],
    *,
    corpus_manifest_id: str,
    corpus_manifest_sha256: str,
    release_split_sha256: str,
) -> tuple[
    list[dict[str, object]],
    list[VN97CampaignObservation],
    bytes,
    str,
    str,
    str,
]:
    if len(paths) != len(CANDIDATES):
        raise VN97P3KaggleError(
            "P3 finalization requires exactly four candidate directories"
        )

    expected_ids = set(candidate_ids())
    seen_ids: set[str] = set()
    seen_indexes: set[int] = set()
    rows: list[dict[str, object]] = []
    observations: list[
        VN97CampaignObservation
    ] = []
    tokenizer_bytes: bytes | None = None
    tokenizer_sha256: str | None = None
    training_dataset_sha256: (
        str | None
    ) = None
    validation_dataset_sha256: (
        str | None
    ) = None

    for raw_path in paths:
        path = raw_path.resolve(strict=True)
        if path.is_symlink() or not path.is_dir():
            raise VN97P3KaggleError(
                "candidate shard must be a real directory"
            )
        names = {
            item.name
            for item in path.iterdir()
        }
        if "candidate-report.vn97p3cand1.json" not in names:
            raise VN97P3KaggleError(
                "candidate shard is missing VN97P3CAND1"
            )
        if "tokenizer.vn97tk1" not in names:
            raise VN97P3KaggleError(
                "candidate shard is missing tokenizer"
            )
        if names - {
            "candidate-report.vn97p3cand1.json",
            "candidate.vn97ck1",
            "tokenizer.vn97tk1",
        }:
            raise VN97P3KaggleError(
                "candidate shard contains unexpected files"
            )

        report_bytes = (
            _read_bounded_regular_file(
                path
                / "candidate-report.vn97p3cand1.json",
                max_bytes=
                    _MAX_CANDIDATE_REPORT_BYTES,
            )
        )
        result = parse_candidate_result(
            report_bytes
        )
        if result.candidate_id in seen_ids:
            raise VN97P3KaggleError(
                "candidate shards contain duplicate candidate IDs"
            )
        if (
            result.candidate_index
            in seen_indexes
        ):
            raise VN97P3KaggleError(
                "candidate shards contain duplicate indexes"
            )
        seen_ids.add(result.candidate_id)
        seen_indexes.add(
            result.candidate_index
        )

        if result.candidate_id not in expected_ids:
            raise VN97P3KaggleError(
                "candidate shard is outside frozen P3 set"
            )
        expected_candidate = CANDIDATES[
            result.candidate_index
        ]
        if (
            result.candidate.canonical_object()
            != expected_candidate
        ):
            raise VN97P3KaggleError(
                "candidate shard index/object mismatch"
            )
        if (
            result.profile_sha256
            != profile_sha256()
        ):
            raise VN97P3KaggleError(
                "candidate shard profile mismatch"
            )
        if (
            result.corpus_manifest_id
            != corpus_manifest_id
            or result.corpus_manifest_sha256
            != corpus_manifest_sha256
        ):
            raise VN97P3KaggleError(
                "candidate shard corpus identity mismatch"
            )
        if (
            result.release_split_sha256
            != release_split_sha256
        ):
            raise VN97P3KaggleError(
                "candidate shard sealed-release identity mismatch"
            )

        current_tokenizer = (
            _read_bounded_regular_file(
                path / "tokenizer.vn97tk1",
                max_bytes=
                    _MAX_TOKENIZER_BYTES,
            )
        )
        current_tokenizer_sha = (
            hashlib.sha256(
                current_tokenizer
            ).hexdigest()
        )
        if (
            current_tokenizer_sha
            != result.tokenizer_sha256
        ):
            raise VN97P3KaggleError(
                "candidate tokenizer SHA-256 mismatch"
            )
        if tokenizer_sha256 is None:
            tokenizer_sha256 = (
                current_tokenizer_sha
            )
            tokenizer_bytes = (
                current_tokenizer
            )
            training_dataset_sha256 = (
                result.training_dataset_sha256
            )
            validation_dataset_sha256 = (
                result.validation_dataset_sha256
            )
        elif (
            current_tokenizer_sha
            != tokenizer_sha256
            or result.training_dataset_sha256
            != training_dataset_sha256
            or result.validation_dataset_sha256
            != validation_dataset_sha256
        ):
            raise VN97P3KaggleError(
                "candidate shards do not share one tokenizer/dataset identity"
            )

        row = {
            "candidate":
                result.candidate.canonical_object(),
            "candidate_id":
                result.candidate_id,
            "evaluation": {
                "mean_loss":
                    result.evaluation.mean_loss,
                "target_tokens":
                    result.evaluation.target_tokens,
                "top1_accuracy":
                    result.evaluation.top1_accuracy,
                "windows":
                    result.evaluation.windows,
            },
            "mobile_footprint":
                result.mobile_footprint,
            "parameter_count":
                result.parameter_count,
            "status": result.status,
            "training": {
                "final_loss":
                    result.training_final_loss,
                "mean_loss":
                    result.training_mean_loss,
                "steps":
                    result.training_steps,
                "target_tokens":
                    result.training_target_tokens,
            },
        }

        checkpoint_path = (
            path / "candidate.vn97ck1"
        )
        if result.status == "ELIGIBLE":
            if not checkpoint_path.is_file():
                raise VN97P3KaggleError(
                    "eligible candidate shard is missing checkpoint"
                )
            loaded = (
                load_deployment_checkpoint_file(
                    checkpoint_path
                )
            )
            if (
                loaded.checkpoint_sha256
                != result.checkpoint_sha256
            ):
                raise VN97P3KaggleError(
                    "candidate checkpoint SHA-256 mismatch"
                )
            row["checkpoint_sha256"] = (
                loaded.checkpoint_sha256
            )
            observations.append(
                VN97CampaignObservation(
                    candidate=result.candidate,
                    parameter_count=
                        result.parameter_count,
                    evaluation=result.evaluation,
                    checkpoint_sha256=
                        loaded.checkpoint_sha256,
                )
            )
        elif checkpoint_path.exists():
            raise VN97P3KaggleError(
                "rejected candidate shard must not contain checkpoint"
            )

        row["_candidate_dir"] = str(
            path
        )
        rows.append(row)

    if seen_ids != expected_ids:
        raise VN97P3KaggleError(
            "candidate shard set does not cover frozen P3 candidates"
        )
    if seen_indexes != set(
        range(len(CANDIDATES))
    ):
        raise VN97P3KaggleError(
            "candidate shard indexes are incomplete"
        )
    if (
        tokenizer_bytes is None
        or tokenizer_sha256 is None
        or training_dataset_sha256 is None
        or validation_dataset_sha256 is None
    ):
        raise VN97P3KaggleError(
            "candidate shard identities were not established"
        )

    return (
        rows,
        observations,
        tokenizer_bytes,
        tokenizer_sha256,
        training_dataset_sha256,
        validation_dataset_sha256,
    )


def main(
    argv: list[str] | None = None,
) -> int:
    args = _parser().parse_args(argv)
    if not args.device.startswith("cuda"):
        raise VN97P3KaggleError(
            "P3 sharded finalization requires CUDA"
        )
    if not torch.cuda.is_available():
        raise VN97P3KaggleError(
            "CUDA is not available on this host"
        )

    corpus_dir = Path(
        args.corpus_dir
    ).resolve(strict=True)
    (
        corpus_manifest_id,
        corpus_manifest_sha256,
    ) = _validate_corpus(corpus_dir)
    release_split_sha256 = (
        _release_manifest_sha256(
            corpus_dir
        )
    )

    (
        rows,
        observations,
        tokenizer_bytes,
        tokenizer_sha256,
        training_dataset_sha256,
        validation_dataset_sha256,
    ) = _load_shards(
        [
            Path(value)
            for value in args.candidate_dir
        ],
        corpus_manifest_id=
            corpus_manifest_id,
        corpus_manifest_sha256=
            corpus_manifest_sha256,
        release_split_sha256=
            release_split_sha256,
    )

    best = select_best_campaign_candidate(
        observations
    )
    selected_rows = [
        row
        for row in rows
        if row["candidate_id"]
        == best.candidate.candidate_id
    ]
    if len(selected_rows) != 1:
        raise VN97P3KaggleError(
            "selected candidate shard is ambiguous"
        )
    selected_dir = Path(
        str(
            selected_rows[0][
                "_candidate_dir"
            ]
        )
    )
    selected_checkpoint_path = (
        selected_dir
        / "candidate.vn97ck1"
    )
    selected_checkpoint = (
        load_deployment_checkpoint_file(
            selected_checkpoint_path
        )
    )
    if (
        selected_checkpoint.checkpoint_sha256
        != best.checkpoint_sha256
    ):
        raise VN97P3KaggleError(
            "selected checkpoint identity changed before release"
        )

    tokenizer_package = (
        VN97TokenizerPackage.from_bytes(
            tokenizer_bytes
        )
    )
    tokenizer = VN97Tokenizer(
        tokenizer_package
    )
    if (
        tokenizer.vocab_size
        != selected_checkpoint.config.vocab_size
    ):
        raise VN97P3KaggleError(
            "selected checkpoint/tokenizer vocabulary mismatch"
        )

    release_records, (
        release_dataset_sha256
    ) = _load_records(
        [corpus_dir / "release.jsonl"],
        mode="chat",
        max_input_bytes=64
        * 1024
        * 1024,
        max_examples=100_000,
    )
    release_examples = _examples(
        release_records,
        mode="chat",
        tokenizer=tokenizer,
    )
    release_config = VN97TrainingConfig(
        sequence_length=SEQUENCE_LENGTH,
        batch_size=BATCH_SIZE,
        epochs=1,
        seed=0,
        shuffle=False,
        max_windows=MAX_RELEASE_WINDOWS,
    )
    release_windows = (
        build_training_windows(
            release_examples,
            release_config,
            pad_token_id=
                tokenizer.pad_id,
        )
    )
    release_evaluation = (
        evaluate_vn97_language(
            selected_checkpoint.model,
            release_windows,
            batch_size=BATCH_SIZE,
            device=args.device,
        )
    )
    release_criteria = (
        VN97ReleaseCriteria(
            max_validation_loss=
                RELEASE_MAX_VALIDATION_LOSS,
            min_top1_accuracy=
                RELEASE_MIN_VALIDATION_ACCURACY,
            min_target_tokens=
                MIN_TARGET_TOKENS,
        )
    )
    try:
        require_release_quality(
            release_evaluation,
            release_criteria,
        )
    except VN97ReleaseQualityError as exc:
        raise VN97P3KaggleError(
            "validation-selected sharded winner failed sealed release gate"
        ) from exc

    output = Path(args.output_dir)
    if output.exists():
        if (
            output.is_symlink()
            or not output.is_dir()
            or any(output.iterdir())
        ):
            raise VN97P3KaggleError(
                "P3 final output-dir must be new or empty"
            )
    else:
        output.mkdir(
            parents=True,
            exist_ok=False,
        )
    output = output.resolve(strict=True)

    clean_rows: list[
        dict[str, object]
    ] = []
    for row in rows:
        clean = dict(row)
        clean.pop("_candidate_dir")
        clean_rows.append(clean)
    clean_rows.sort(
        key=lambda row: str(
            row["candidate_id"]
        )
    )

    report = {
        "candidates": clean_rows,
        "criteria": {
            "deployment_tile_cols":
                TILE_COLS,
            "deployment_tile_rows":
                TILE_ROWS,
            "max_model_image_bytes":
                MAX_MODEL_IMAGE_BYTES,
            "max_parameters":
                MAX_PARAMETERS,
            "max_recurrent_state_bytes":
                MAX_RECURRENT_STATE_BYTES,
            "max_validation_loss":
                MAX_VALIDATION_LOSS,
            "min_top1_accuracy":
                MIN_VALIDATION_ACCURACY,
            "min_validation_target_tokens":
                MIN_TARGET_TOKENS,
            "release_max_validation_loss":
                RELEASE_MAX_VALIDATION_LOSS,
            "release_min_top1_accuracy":
                RELEASE_MIN_VALIDATION_ACCURACY,
            "release_min_validation_target_tokens":
                MIN_TARGET_TOKENS,
        },
        "release_evaluation": {
            "mean_loss":
                release_evaluation.mean_loss,
            "target_tokens":
                release_evaluation.target_tokens,
            "top1_accuracy":
                release_evaluation.top1_accuracy,
            "windows":
                release_evaluation.windows,
        },
        "schema": "VN97CAMP2",
        "selected_candidate_id":
            best.candidate.candidate_id,
        "selected_checkpoint_sha256":
            best.checkpoint_sha256,
        "tokenizer_sha256":
            tokenizer_sha256,
        "training_dataset_sha256":
            training_dataset_sha256,
        "validation_dataset_sha256":
            validation_dataset_sha256,
        "release_dataset_sha256":
            release_dataset_sha256,
    }
    report_bytes = _canonical_json(
        report
    )

    checkpoint_bytes = (
        _read_bounded_regular_file(
            selected_checkpoint_path,
            max_bytes=512 * 1024 * 1024,
        )
    )
    _atomic_write(
        output / "tokenizer.vn97tk1",
        tokenizer_bytes,
    )
    _atomic_write(
        output / "model.vn97ck1",
        checkpoint_bytes,
    )
    _atomic_write(
        output / "campaign-report.json",
        report_bytes,
    )

    image = build_model_image(
        selected_checkpoint.model,
        tokenizer=tokenizer_package,
        tile_rows=TILE_ROWS,
        tile_cols=TILE_COLS,
    )
    image_bytes = image.data
    _atomic_write(
        output / "model.vn97mi1",
        image_bytes,
    )

    selected_validation = (
        best.evaluation
    )
    receipt = VN97P3CampaignReceipt(
        corpus_manifest_id=
            corpus_manifest_id,
        corpus_manifest_sha256=
            corpus_manifest_sha256,
        campaign_report_sha256=
            hashlib.sha256(
                report_bytes
            ).hexdigest(),
        selected_candidate_id=
            best.candidate.candidate_id,
        checkpoint_sha256=
            best.checkpoint_sha256,
        tokenizer_sha256=
            tokenizer_sha256,
        model_image_sha256=
            hashlib.sha256(
                image_bytes
            ).hexdigest(),
        model_image_bytes=
            len(image_bytes),
        validation_mean_loss=
            selected_validation.mean_loss,
        validation_top1_accuracy=
            selected_validation.top1_accuracy,
        release_mean_loss=
            release_evaluation.mean_loss,
        release_top1_accuracy=
            release_evaluation.top1_accuracy,
        device=args.device,
        profile_sha256=
            profile_sha256(),
    )
    _atomic_write(
        output
        / "p3-run.vn97p3run1.json",
        receipt.to_bytes(),
    )

    print(
        "VN97P3RUN1 "
        f"selected={best.candidate.candidate_id} "
        f"checkpoint_sha256={best.checkpoint_sha256} "
        f"release_loss={release_evaluation.mean_loss:.10f} "
        f"release_top1={release_evaluation.top1_accuracy:.10f}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(
            f"vn97-p3-kaggle-finalize: {exc}",
            file=sys.stderr,
        )
        raise
