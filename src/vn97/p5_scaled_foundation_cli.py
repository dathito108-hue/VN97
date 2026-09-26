from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time

import torch
import torch.nn.functional as F

from .campaign import VN97CampaignCandidate
from .campaign_cli import _estimate_parameter_count
from .config import VN97Config
from .deployment_checkpoint import save_deployment_checkpoint
from .mobile_budget import (
    VN97MobileBudget,
    estimate_vn97_mobile_footprint,
)
from .model import VN97LanguageCore
from .model_image import build_model_image
from .p3_language_campaign_cli import _validate_corpus
from .p4_arithmetic_coverage_curriculum import extract_expression
from .p4_arithmetic_mechanism_audit_cli import _verify_artifact
from .p4_compositional_repair_cli import (
    _canonical_measure_dev,
    _canonical_measure_records,
)
from .p4_generalization_cli import (
    _probe_records,
    _select_replay,
    _sha256,
)
from .p4_generalization_curriculum import (
    P4D_CATEGORIES,
    default_training as p4_training,
    default_validation as p4_validation,
)
from .p4_numeric_arithmetic_repair_cli import _measure_copy_suite
from .p4_task_evaluation import load_p4_task_suite
from .p5_scaled_foundation import (
    CANDIDATES,
    EPOCHS,
    LOGICAL_BATCH_SIZE,
    MAX_MODEL_IMAGE_BYTES,
    MAX_RECURRENT_STATE_BYTES,
    MAX_TRAIN_WINDOWS,
    MICRO_BATCH_SIZE,
    P3_TRAIN_RECORDS,
    P3_VALIDATION_MAX_WINDOWS,
    P4_TRAIN_RECORDS,
    P5A_PROFILE_ID,
    SEQUENCE_LENGTH,
    TILE_COLS,
    TILE_ROWS,
    profile_sha256,
)
from .tokenizer import VN97Tokenizer
from .training import (
    IGNORE_INDEX,
    VN97TrainingConfig,
    VN97TrainingWindow,
    build_training_windows,
    encode_chat_completion_messages,
    encode_chat_messages,
)
from .training_cli import (
    _atomic_write,
    _canonical_json,
    _load_records,
)


P5A_REPORT_SCHEMA = "VN97P5ACAND1"


class VN97P5AError(RuntimeError):
    pass


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train one scaled VN97 foundation candidate with low-memory "
            "sequential SSM execution and FP16 gradient accumulation."
        )
    )
    parser.add_argument("--parent-p4e-k-dir", required=True)
    parser.add_argument("--p3-corpus-dir", required=True)
    parser.add_argument("--dev-suite", default=None)
    parser.add_argument("--candidate-index", type=int, required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--progress-interval-steps",
        type=int,
        default=25,
    )
    parser.add_argument(
        "--checkpoint-interval-steps",
        type=int,
        default=50,
    )
    return parser


def _window_digest(
    window: VN97TrainingWindow,
) -> bytes:
    digest = hashlib.sha256()
    for value in window.input_ids:
        digest.update(
            int(value).to_bytes(
                4,
                "little",
                signed=False,
            )
        )
    digest.update(b"\0")
    for value in window.labels:
        digest.update(
            int(value).to_bytes(
                4,
                "little",
                signed=True,
            )
        )
    return digest.digest()


def _select_windows(
    windows: tuple[
        VN97TrainingWindow,
        ...
    ],
    count: int,
) -> tuple[
    VN97TrainingWindow,
    ...
]:
    if count <= 0:
        raise ValueError(
            "window selection count must be positive"
        )
    ordered = sorted(
        windows,
        key=_window_digest,
    )
    if len(ordered) < count:
        raise VN97P5AError(
            f"not enough windows: need {count}, have {len(ordered)}"
        )
    return tuple(
        ordered[:count]
    )


def _held_out_dev(
    suite_path: Path | None,
) -> tuple[
    set[str],
    set[str],
]:
    prompts: set[str] = set()
    expressions: set[str] = set()

    for record in p4_validation():
        prompts.add(record.prompt)
        if (
            record.category
            == "reasoning_planning"
        ):
            expression = extract_expression(
                record.prompt
            )
            if expression is not None:
                expressions.add(
                    expression
                )

    if suite_path is not None:
        suite = load_p4_task_suite(
            suite_path
        )
        for task in suite.tasks:
            prompts.add(task.prompt)
            if (
                task.category
                == "reasoning_planning"
            ):
                expression = extract_expression(
                    task.prompt
                )
                if expression is not None:
                    expressions.add(
                        expression
                    )

    return (
        prompts,
        expressions,
    )


def _select_p4_records(
    *,
    count: int,
    held_out_prompts: set[str],
    held_out_expressions: set[str],
) -> tuple[object, ...]:
    if count % len(P4D_CATEGORIES) != 0:
        raise VN97P5AError(
            "P5A P4 record count must be category-balanced"
        )
    per_category = (
        count
        // len(P4D_CATEGORIES)
    )

    output: list[object] = []
    training = p4_training()

    for category in P4D_CATEGORIES:
        rows = [
            record
            for record in training
            if record.category
            == category
            and record.prompt
            not in held_out_prompts
        ]
        if (
            category
            == "reasoning_planning"
        ):
            filtered = []
            for record in rows:
                expression = extract_expression(
                    record.prompt
                )
                if (
                    expression is not None
                    and expression
                    in held_out_expressions
                ):
                    continue
                filtered.append(record)
            rows = filtered

        rows.sort(
            key=lambda record:
                hashlib.sha256(
                    (
                        record.category
                        + "\0"
                        + record.prompt
                    ).encode("utf-8")
                ).digest()
        )
        if len(rows) < per_category:
            raise VN97P5AError(
                f"not enough P4 rows for {category}"
            )
        output.extend(
            rows[:per_category]
        )

    output.sort(
        key=lambda record:
            hashlib.sha256(
                (
                    record.category
                    + "\0"
                    + record.prompt
                ).encode("utf-8")
            ).digest()
    )
    return tuple(output)


def _tensor_batch(
    window: VN97TrainingWindow,
    *,
    device: str,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
]:
    return (
        torch.tensor(
            [window.input_ids],
            dtype=torch.long,
            device=device,
        ),
        torch.tensor(
            [window.labels],
            dtype=torch.long,
            device=device,
        ),
    )


@torch.inference_mode()
def _evaluate_windows(
    model: VN97LanguageCore,
    windows: tuple[
        VN97TrainingWindow,
        ...
    ],
    *,
    device: str,
) -> dict[str, object]:
    if not windows:
        raise VN97P5AError(
            "evaluation windows are empty"
        )

    model.eval()
    loss_sum = 0.0
    target_tokens = 0
    correct = 0

    for window in windows:
        inputs, labels = _tensor_batch(
            window,
            device=device,
        )
        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
        ):
            logits, _ = (
                model.forward_sequential_reference(
                    inputs
                )
            )

        mask = labels != IGNORE_INDEX
        if not bool(mask.any()):
            continue

        selected_logits = logits[
            mask
        ].float()
        selected_labels = labels[
            mask
        ]
        loss = F.cross_entropy(
            selected_logits,
            selected_labels,
            reduction="sum",
        )
        count = int(
            selected_labels.numel()
        )
        loss_sum += float(
            loss.item()
        )
        target_tokens += count
        correct += int(
            (
                selected_logits.argmax(
                    dim=-1
                )
                == selected_labels
            ).sum().item()
        )

        del (
            inputs,
            labels,
            logits,
            selected_logits,
            selected_labels,
            loss,
        )

    if target_tokens <= 0:
        raise VN97P5AError(
            "evaluation produced no target tokens"
        )

    return {
        "mean_loss":
            loss_sum
            / target_tokens,
        "target_tokens":
            target_tokens,
        "top1_accuracy":
            correct
            / target_tokens,
        "windows":
            len(windows),
    }


def _save_resume(
    path: Path,
    *,
    identity: str,
    model: VN97LanguageCore,
    optimizer,
    scaler,
    next_step: int,
    loss_sum: float,
    target_tokens: int,
) -> None:
    temp = path.with_suffix(
        path.suffix + ".tmp"
    )
    torch.save(
        {
            "identity":
                identity,
            "loss_sum":
                loss_sum,
            "model":
                model.state_dict(),
            "next_step":
                next_step,
            "optimizer":
                optimizer.state_dict(),
            "scaler":
                scaler.state_dict(),
            "target_tokens":
                target_tokens,
        },
        temp,
    )
    temp.replace(path)


def _load_resume(
    path: Path,
    *,
    identity: str,
    model: VN97LanguageCore,
    optimizer,
    scaler,
) -> tuple[
    int,
    float,
    int,
]:
    if not path.exists():
        return (
            0,
            0.0,
            0,
        )

    payload = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )
    if (
        not isinstance(payload, dict)
        or payload.get("identity")
        != identity
    ):
        raise VN97P5AError(
            "P5A resume identity mismatch"
        )

    model.load_state_dict(
        payload["model"]
    )
    optimizer.load_state_dict(
        payload["optimizer"]
    )
    scaler.load_state_dict(
        payload["scaler"]
    )
    return (
        int(
            payload["next_step"]
        ),
        float(
            payload["loss_sum"]
        ),
        int(
            payload["target_tokens"]
        ),
    )


def main(
    argv: list[str] | None = None,
) -> int:
    args = _parser().parse_args(
        argv
    )

    if (
        not 0
        <= args.candidate_index
        < len(CANDIDATES)
    ):
        raise VN97P5AError(
            "candidate index is outside the P5A ladder"
        )
    if (
        not torch.cuda.is_available()
        or not str(
            args.device
        ).startswith("cuda")
    ):
        raise VN97P5AError(
            "P5A requires CUDA"
        )
    if (
        args.progress_interval_steps
        <= 0
        or args.checkpoint_interval_steps
        <= 0
    ):
        raise VN97P5AError(
            "P5A progress/checkpoint intervals must be positive"
        )

    output = Path(
        args.output_dir
    )
    if (
        output.exists()
        and (
            output.is_symlink()
            or not output.is_dir()
            or any(output.iterdir())
        )
    ):
        raise VN97P5AError(
            "P5A output-dir must be new or empty"
        )

    (
        parent_root,
        parent_checkpoint,
        parent_package,
        parent_report,
    ) = _verify_artifact(
        Path(
            args.parent_p4e_k_dir
        )
    )
    tokenizer = VN97Tokenizer(
        parent_package
    )
    tokenizer_bytes = (
        parent_package.to_bytes()
    )
    tokenizer_sha256 = (
        hashlib.sha256(
            tokenizer_bytes
        ).hexdigest()
    )

    p3_root = Path(
        args.p3_corpus_dir
    ).resolve(
        strict=True
    )
    (
        corpus_manifest_id,
        corpus_manifest_sha256,
    ) = _validate_corpus(
        p3_root
    )

    (
        p3_training_records,
        p3_training_sha256,
    ) = _load_records(
        [
            p3_root
            / "training.jsonl"
        ],
        mode="chat",
        max_input_bytes=
            64 * 1024 * 1024,
        max_examples=100_000,
    )
    (
        p3_validation_records,
        p3_validation_sha256,
    ) = _load_records(
        [
            p3_root
            / "validation.jsonl"
        ],
        mode="chat",
        max_input_bytes=
            64 * 1024 * 1024,
        max_examples=100_000,
    )

    p3_selected = _select_replay(
        p3_training_records,
        count=
            P3_TRAIN_RECORDS,
    )

    suite_path = (
        Path(
            args.dev_suite
        )
        if args.dev_suite
        is not None
        else None
    )
    (
        held_out_prompts,
        held_out_expressions,
    ) = _held_out_dev(
        suite_path
    )

    p4_selected = _select_p4_records(
        count=P4_TRAIN_RECORDS,
        held_out_prompts=
            held_out_prompts,
        held_out_expressions=
            held_out_expressions,
    )

    window_config = VN97TrainingConfig(
        sequence_length=
            SEQUENCE_LENGTH,
        batch_size=
            LOGICAL_BATCH_SIZE,
        epochs=EPOCHS,
        learning_rate=1e-4,
        seed=0,
        shuffle=False,
        max_windows=30_000,
    )

    p3_examples = [
        encode_chat_messages(
            tokenizer,
            record,
        )
        for record in p3_selected
    ]
    p4_examples = [
        encode_chat_completion_messages(
            tokenizer,
            record.messages,
        )
        for record in p4_selected
    ]

    p3_windows_all = (
        build_training_windows(
            p3_examples,
            window_config,
            pad_token_id=
                tokenizer.pad_id,
        )
    )
    p4_windows_all = (
        build_training_windows(
            p4_examples,
            window_config,
            pad_token_id=
                tokenizer.pad_id,
        )
    )

    half = (
        MAX_TRAIN_WINDOWS
        // 2
    )
    p3_windows = _select_windows(
        p3_windows_all,
        half,
    )
    p4_windows = _select_windows(
        p4_windows_all,
        MAX_TRAIN_WINDOWS
        - half,
    )
    train_windows = [
        *p3_windows,
        *p4_windows,
    ]
    train_windows.sort(
        key=_window_digest
    )

    validation_examples = [
        encode_chat_messages(
            tokenizer,
            record,
        )
        for record
        in p3_validation_records
    ]
    validation_config = (
        VN97TrainingConfig(
            sequence_length=
                SEQUENCE_LENGTH,
            batch_size=1,
            epochs=1,
            seed=0,
            shuffle=False,
            max_windows=20_000,
        )
    )
    validation_windows_all = (
        build_training_windows(
            validation_examples,
            validation_config,
            pad_token_id=
                tokenizer.pad_id,
        )
    )
    validation_windows = _select_windows(
        validation_windows_all,
        min(
            P3_VALIDATION_MAX_WINDOWS,
            len(
                validation_windows_all
            ),
        ),
    )

    candidate = CANDIDATES[
        args.candidate_index
    ]
    campaign_candidate = (
        VN97CampaignCandidate(
            d_model=
                candidate.d_model,
            n_layers=
                candidate.n_layers,
            d_state=
                candidate.d_state,
            embedding_rank=
                candidate.embedding_rank,
            seed=
                candidate.seed,
            learning_rate=
                candidate.learning_rate,
        )
    )

    parameter_count = (
        _estimate_parameter_count(
            vocab_size=
                tokenizer.vocab_size,
            candidate=
                campaign_candidate,
        )
    )

    model_config = VN97Config(
        vocab_size=
            tokenizer.vocab_size,
        d_model=
            candidate.d_model,
        n_layers=
            candidate.n_layers,
        d_state=
            candidate.d_state,
        embedding_rank=
            candidate.embedding_rank,
    )

    footprint = (
        estimate_vn97_mobile_footprint(
            model_config,
            tokenizer_nbytes=
                len(
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
    rejection = (
        budget.rejection_status(
            footprint
        )
    )
    if rejection is not None:
        raise VN97P5AError(
            "P5A candidate exceeds mobile budget: "
            + rejection
        )

    torch.manual_seed(
        candidate.seed
    )
    torch.cuda.manual_seed_all(
        candidate.seed
    )
    model = VN97LanguageCore(
        model_config
    ).to(
        args.device
    )
    actual_parameter_count = sum(
        int(
            parameter.numel()
        )
        for parameter
        in model.parameters()
    )
    if (
        actual_parameter_count
        != parameter_count
    ):
        raise VN97P5AError(
            "P5A parameter-count probe mismatch"
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=
            candidate.learning_rate,
        weight_decay=0.01,
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=True,
    )

    training_identity = _sha256(
        _canonical_json(
            {
                "candidate":
                    candidate.canonical_object(),
                "corpus_manifest_sha256":
                    corpus_manifest_sha256,
                "p3_training_sha256":
                    p3_training_sha256,
                "p3_validation_sha256":
                    p3_validation_sha256,
                "profile_sha256":
                    profile_sha256(),
                "tokenizer_sha256":
                    tokenizer_sha256,
                "window_digests": [
                    _window_digest(
                        window
                    ).hex()
                    for window
                    in train_windows
                ],
            }
        )
    )

    work = Path(
        args.work_dir
    )
    if work.is_symlink():
        raise VN97P5AError(
            "P5A work-dir must not be a symlink"
        )
    work.mkdir(
        parents=True,
        exist_ok=True,
    )
    resume_path = (
        work
        / "state.p5a.pt"
    )

    (
        start_step,
        cumulative_loss_sum,
        cumulative_target_tokens,
    ) = _load_resume(
        resume_path,
        identity=
            training_identity,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
    )

    rng = random.Random(
        candidate.seed
    )
    order = list(
        range(
            len(
                train_windows
            )
        )
    )
    rng.shuffle(order)

    total_steps = math.ceil(
        len(order)
        / LOGICAL_BATCH_SIZE
    )
    if start_step > total_steps:
        raise VN97P5AError(
            "P5A resume step exceeds training length"
        )

    print(
        "VN97 P5A START "
        f"candidate_index={args.candidate_index} "
        f"candidate={candidate.candidate_id} "
        f"parameters={parameter_count} "
        f"d_model={candidate.d_model} "
        f"layers={candidate.n_layers} "
        f"d_state={candidate.d_state} "
        f"rank={candidate.embedding_rank} "
        f"windows={len(train_windows)} "
        f"steps={total_steps} "
        f"resume_step={start_step}",
        flush=True,
    )

    model.train()
    started = time.monotonic()

    for step in range(
        start_step,
        total_steps,
    ):
        batch_indices = order[
            step
            * LOGICAL_BATCH_SIZE:
            min(
                (
                    step + 1
                )
                * LOGICAL_BATCH_SIZE,
                len(order),
            )
        ]
        batch = [
            train_windows[index]
            for index
            in batch_indices
        ]
        logical_targets = sum(
            window.target_tokens
            for window in batch
        )
        if logical_targets <= 0:
            raise VN97P5AError(
                "P5A logical batch has no targets"
            )

        optimizer.zero_grad(
            set_to_none=True
        )
        logical_loss_sum = 0.0

        for window in batch:
            inputs, labels = (
                _tensor_batch(
                    window,
                    device=args.device,
                )
            )
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
            ):
                logits, _ = (
                    model.forward_sequential_reference(
                        inputs
                    )
                )
                token_loss_sum = (
                    F.cross_entropy(
                        logits.reshape(
                            -1,
                            logits.shape[-1],
                        ),
                        labels.reshape(
                            -1
                        ),
                        ignore_index=
                            IGNORE_INDEX,
                        reduction="sum",
                    )
                )
                loss = (
                    token_loss_sum
                    / logical_targets
                )

            if not bool(
                torch.isfinite(
                    loss
                )
            ):
                raise VN97P5AError(
                    "P5A loss became non-finite"
                )

            scaler.scale(
                loss
            ).backward()

            logical_loss_sum += (
                float(
                    token_loss_sum
                    .detach()
                    .float()
                    .item()
                )
            )

            del (
                inputs,
                labels,
                logits,
                token_loss_sum,
                loss,
            )

        scaler.unscale_(
            optimizer
        )
        grad_norm = (
            torch.nn.utils
            .clip_grad_norm_(
                model.parameters(),
                1.0,
            )
        )
        if not bool(
            torch.isfinite(
                torch.as_tensor(
                    grad_norm
                )
            )
        ):
            raise VN97P5AError(
                "P5A gradient norm became non-finite"
            )

        scaler.step(
            optimizer
        )
        scaler.update()

        cumulative_loss_sum += (
            logical_loss_sum
        )
        cumulative_target_tokens += (
            logical_targets
        )
        completed = step + 1

        if (
            completed
            % args.progress_interval_steps
            == 0
            or completed
            == total_steps
        ):
            elapsed = (
                time.monotonic()
                - started
            )
            local_steps = max(
                1,
                completed
                - start_step,
            )
            eta = (
                total_steps
                - completed
            ) * (
                elapsed
                / local_steps
            )
            print(
                "VN97 P5A PROGRESS "
                f"step={completed}/{total_steps} "
                f"percent={100.0 * completed / total_steps:.2f} "
                f"elapsed_s={elapsed:.1f} "
                f"eta_s={eta:.1f} "
                f"batch_loss={logical_loss_sum / logical_targets:.6f} "
                f"mean_loss={cumulative_loss_sum / cumulative_target_tokens:.6f} "
                f"scale={float(scaler.get_scale()):.1f}",
                flush=True,
            )

        if (
            completed
            % args.checkpoint_interval_steps
            == 0
            and completed
            < total_steps
        ):
            _save_resume(
                resume_path,
                identity=
                    training_identity,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                next_step=
                    completed,
                loss_sum=
                    cumulative_loss_sum,
                target_tokens=
                    cumulative_target_tokens,
            )
            print(
                "VN97 P5A CHECKPOINT "
                f"step={completed}/{total_steps} "
                f"path={resume_path}",
                flush=True,
            )

    model.eval()
    p3_validation = (
        _evaluate_windows(
            model,
            validation_windows,
            device=args.device,
        )
    )

    probe_records = _probe_records(
        p4_validation(),
        per_category=30,
    )
    p4_result = (
        _canonical_measure_records(
            model=model,
            tokenizer=tokenizer,
            records=probe_records,
            device=args.device,
        )
    )
    copy_result = (
        _measure_copy_suite(
            model=model,
            tokenizer=tokenizer,
            device=args.device,
        )
    )
    dev_result = None
    if suite_path is not None:
        dev_result = (
            _canonical_measure_dev(
                model=model,
                tokenizer=tokenizer,
                suite_path=suite_path,
                device=args.device,
            )
        )

    reasoning = int(
        p4_result[
            "categories"
        ][
            "reasoning_planning"
        ]["passed"]
    )
    overall = int(
        p4_result[
            "passed"
        ]
    )
    copy_passed = int(
        copy_result[
            "passed"
        ]
    )

    status = (
        "PROMISING_SCALE_SIGNAL"
        if (
            reasoning >= 5
            and overall >= 60
            and copy_passed >= 40
            and float(
                p3_validation[
                    "top1_accuracy"
                ]
            ) >= 0.08
        )
        else "NO_SCALE_SIGNAL"
    )

    print(
        "VN97 P5A RESULT "
        f"status={status} "
        f"candidate={candidate.candidate_id} "
        f"parameters={parameter_count} "
        f"canonical={overall}/180 "
        f"reasoning={reasoning}/30 "
        f"copy={copy_passed}/60 "
        f"p3_loss={p3_validation['mean_loss']:.6f} "
        f"p3_top1={p3_validation['top1_accuracy']:.6f}",
        flush=True,
    )
    for category in P4D_CATEGORIES:
        row = (
            p4_result[
                "categories"
            ][category]
        )
        print(
            "VN97 P5A CATEGORY "
            f"{category} "
            f"{row['passed']}/{row['tasks']}",
            flush=True,
        )

    model.cpu()
    torch.cuda.empty_cache()

    output.mkdir(
        parents=True,
        exist_ok=True,
    )
    checkpoint_path = (
        output
        / "model.vn97ck1"
    )
    checkpoint_sha256 = (
        save_deployment_checkpoint(
            model,
            checkpoint_path,
        )
    )
    _atomic_write(
        output
        / "tokenizer.vn97tk1",
        tokenizer_bytes,
    )
    image_bytes = (
        build_model_image(
            model,
            tokenizer=
                parent_package,
            tile_rows=TILE_ROWS,
            tile_cols=TILE_COLS,
        ).data
    )
    if len(image_bytes) > (
        MAX_MODEL_IMAGE_BYTES
    ):
        raise VN97P5AError(
            "P5A model image exceeded frozen budget"
        )
    _atomic_write(
        output
        / "model.vn97mi1",
        image_bytes,
    )

    parent_metrics = (
        parent_report.get(
            "selected_metrics"
        )
    )
    report = {
        "candidate":
            candidate.canonical_object(),
        "candidate_id":
            candidate.candidate_id,
        "checkpoint_sha256":
            checkpoint_sha256,
        "corpus_manifest_id":
            corpus_manifest_id,
        "corpus_manifest_sha256":
            corpus_manifest_sha256,
        "dev":
            dev_result,
        "model_image_bytes":
            len(image_bytes),
        "model_image_sha256":
            _sha256(
                image_bytes
            ),
        "mobile_footprint":
            footprint.canonical_object(),
        "p3_training_sha256":
            p3_training_sha256,
        "p3_validation":
            p3_validation,
        "p3_validation_sha256":
            p3_validation_sha256,
        "p4":
            p4_result,
        "numeric_copy":
            copy_result,
        "parameter_count":
            parameter_count,
        "parent_p4e_k": {
            "checkpoint_sha256":
                parent_checkpoint.checkpoint_sha256,
            "metrics":
                parent_metrics,
            "selected_candidate":
                parent_report.get(
                    "selected_candidate"
                ),
            "status":
                parent_report.get(
                    "status"
                ),
        },
        "profile_id":
            P5A_PROFILE_ID,
        "profile_sha256":
            profile_sha256(),
        "schema":
            P5A_REPORT_SCHEMA,
        "status":
            status,
        "tokenizer_sha256":
            tokenizer_sha256,
        "training": {
            "epochs":
                EPOCHS,
            "final_mean_loss":
                cumulative_loss_sum
                / cumulative_target_tokens,
            "logical_batch_size":
                LOGICAL_BATCH_SIZE,
            "micro_batch_size":
                MICRO_BATCH_SIZE,
            "sequence_length":
                SEQUENCE_LENGTH,
            "steps":
                total_steps,
            "target_tokens":
                cumulative_target_tokens,
            "windows":
                len(
                    train_windows
                ),
        },
    }
    report_bytes = (
        _canonical_json(
            report
        )
        + b"\n"
    )
    _atomic_write(
        output
        / "p5a-report.json",
        report_bytes,
    )

    files = {
        "model.vn97ck1":
            checkpoint_path.read_bytes(),
        "model.vn97mi1":
            image_bytes,
        "p5a-report.json":
            report_bytes,
        "tokenizer.vn97tk1":
            tokenizer_bytes,
    }
    sums = b"".join(
        (
            _sha256(
                files[name]
            )
            + "  "
            + name
            + "\n"
        ).encode("ascii")
        for name in sorted(files)
    )
    _atomic_write(
        output
        / "SHA256SUMS",
        sums,
    )

    resume_path.unlink(
        missing_ok=True
    )

    print(
        "VN97P5ACAND1 "
        f"status={status} "
        f"index={args.candidate_index} "
        f"candidate={candidate.candidate_id} "
        f"parameters={parameter_count} "
        f"checkpoint={checkpoint_sha256}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(
            main()
        )
    except Exception as exc:
        print(
            "vn97-p5a-scaled-foundation: "
            f"{exc}",
            file=sys.stderr,
        )
        raise
