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

from .deployment_checkpoint import (
    load_deployment_checkpoint_file,
    save_deployment_checkpoint,
)
from .model_image import build_model_image
from .p3_language_campaign import (
    BATCH_SIZE as P3_BATCH_SIZE,
    MAX_VALIDATION_WINDOWS as P3_MAX_VALIDATION_WINDOWS,
    SEQUENCE_LENGTH as P3_SEQUENCE_LENGTH,
)
from .p3_language_campaign_cli import _validate_corpus
from .p3_tensor_cache import evaluate_vn97_from_tensors
from .p4_arithmetic_mechanism_audit_cli import _verify_artifact
from .p4_arithmetic_coverage_curriculum import extract_expression
from .p4_compositional_repair_cli import (
    _canonical_measure_dev,
    _canonical_measure_records,
)
from .p4_generalization_cli import (
    _canonical_json,
    _probe_records,
    _select_replay,
    _sha256,
    _windows_to_tensors,
)
from .p4_generalization_curriculum import (
    P4D_CATEGORIES,
    default_validation,
)
from .p4_numeric_arithmetic_repair_cli import _measure_copy_suite
from .p4_sequence_ranking_curriculum import (
    P4E_N_PROFILE_ID,
    P4E_N_SEED,
    ranking_training,
    preservation_records,
)
from .p4_task_evaluation import load_p4_task_suite
from .tokenizer import VN97Tokenizer
from .training import (
    IGNORE_INDEX,
    VN97ChatMessage,
    VN97TrainingConfig,
    VN97TrainingExample,
    build_training_windows,
    encode_chat_completion_messages,
    encode_chat_messages,
)
from .training_cli import (
    _atomic_write,
    _load_records,
)


P4E_N_REPORT_SCHEMA = "VN97P4EN1"


class VN97P4ENError(RuntimeError):
    pass


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Sequence-ranking arithmetic repair from the safe P4E-K checkpoint."
        )
    )
    parser.add_argument("--p4e-k-dir", required=True)
    parser.add_argument("--p3-corpus-dir", required=True)
    parser.add_argument("--dev-suite", default=None)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--ranking-records", type=int, default=1800)
    parser.add_argument("--numeric-copy-records", type=int, default=1400)
    parser.add_argument("--anchor-records", type=int, default=1600)
    parser.add_argument("--p3-replay-records", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--preserve-batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-6)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--ranking-margin", type=float, default=0.25)
    parser.add_argument("--ranking-weight", type=float, default=0.75)
    parser.add_argument("--preserve-weight", type=float, default=0.70)
    parser.add_argument("--checkpoint-interval-steps", type=int, default=50)
    parser.add_argument("--progress-interval-steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=P4E_N_SEED)
    parser.add_argument("--min-reasoning-passes", type=int, default=5)
    parser.add_argument("--min-numeric-copy-passes", type=int, default=52)
    parser.add_argument("--max-overall-drop", type=int, default=2)
    parser.add_argument("--max-copy-drop", type=int, default=3)
    parser.add_argument("--max-tool-drop", type=int, default=1)
    parser.add_argument("--max-authority-drop", type=int, default=1)
    parser.add_argument("--max-other-category-drop", type=int, default=2)
    parser.add_argument("--max-dev-drop", type=int, default=1)
    parser.add_argument(
        "--max-token-validation-loss-increase",
        type=float,
        default=0.03,
    )
    parser.add_argument(
        "--max-token-top1-drop",
        type=float,
        default=0.003,
    )
    parser.add_argument(
        "--max-p3-validation-loss-increase",
        type=float,
        default=0.07,
    )
    parser.add_argument(
        "--max-p3-top1-drop",
        type=float,
        default=0.007,
    )
    return parser


def _example(
    tokenizer: VN97Tokenizer,
    prompt: str,
    answer: str,
) -> VN97TrainingExample:
    return encode_chat_completion_messages(
        tokenizer,
        (
            VN97ChatMessage(
                role="user",
                content=prompt,
            ),
            VN97ChatMessage(
                role="assistant",
                content=answer,
            ),
        ),
    )


def _padded_batch(
    examples: list[VN97TrainingExample],
    *,
    pad_token_id: int,
    device: str,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    if not examples:
        raise VN97P4ENError(
            "cannot construct an empty training batch"
        )

    length = max(
        len(item.token_ids) - 1
        for item in examples
    )
    inputs = torch.full(
        (len(examples), length),
        pad_token_id,
        dtype=torch.long,
        device=device,
    )
    targets = torch.full(
        (len(examples), length),
        pad_token_id,
        dtype=torch.long,
        device=device,
    )
    mask = torch.zeros(
        (len(examples), length),
        dtype=torch.bool,
        device=device,
    )

    for row, item in enumerate(examples):
        n = len(item.token_ids) - 1
        inputs[
            row,
            :n,
        ] = torch.tensor(
            item.token_ids[:-1],
            dtype=torch.long,
            device=device,
        )
        targets[
            row,
            :n,
        ] = torch.tensor(
            item.token_ids[1:],
            dtype=torch.long,
            device=device,
        )
        mask[
            row,
            :n,
        ] = torch.tensor(
            item.target_mask[1:],
            dtype=torch.bool,
            device=device,
        )

    if not bool(mask.any()):
        raise VN97P4ENError(
            "training batch contains no supervised targets"
        )

    return (
        inputs,
        targets,
        mask,
    )


def _scores_and_ce(
    model,
    examples: list[VN97TrainingExample],
    *,
    pad_token_id: int,
    device: str,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
]:
    (
        inputs,
        targets,
        mask,
    ) = _padded_batch(
        examples,
        pad_token_id=pad_token_id,
        device=device,
    )
    logits, _ = model(inputs)
    log_probs = F.log_softmax(
        logits,
        dim=-1,
    )
    token_logprob = log_probs.gather(
        -1,
        targets.unsqueeze(-1),
    ).squeeze(-1)
    weights = mask.to(
        dtype=token_logprob.dtype
    )
    counts = weights.sum(
        dim=1
    ).clamp_min(1.0)
    sequence_scores = (
        (
            token_logprob
            * weights
        ).sum(dim=1)
        / counts
    )
    ce = -(
        (
            token_logprob
            * weights
        ).sum()
        / weights.sum().clamp_min(1.0)
    )
    return (
        sequence_scores,
        ce,
    )


def _save_resume(
    path: Path,
    *,
    identity: str,
    model,
    optimizer,
    next_step: int,
    cumulative_loss: float,
    cumulative_rank_loss: float,
    cumulative_preserve_loss: float,
) -> None:
    tmp = path.with_suffix(
        path.suffix + ".tmp"
    )
    torch.save(
        {
            "identity": identity,
            "model":
                model.state_dict(),
            "optimizer":
                optimizer.state_dict(),
            "next_step":
                next_step,
            "cumulative_loss":
                cumulative_loss,
            "cumulative_rank_loss":
                cumulative_rank_loss,
            "cumulative_preserve_loss":
                cumulative_preserve_loss,
        },
        tmp,
    )
    tmp.replace(path)


def _load_resume(
    path: Path,
    *,
    identity: str,
    model,
    optimizer,
) -> tuple[
    int,
    float,
    float,
    float,
]:
    if not path.exists():
        return (
            0,
            0.0,
            0.0,
            0.0,
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
        raise VN97P4ENError(
            "P4E-N resume identity mismatch"
        )

    model.load_state_dict(
        payload["model"]
    )
    optimizer.load_state_dict(
        payload["optimizer"]
    )
    return (
        int(
            payload["next_step"]
        ),
        float(
            payload[
                "cumulative_loss"
            ]
        ),
        float(
            payload[
                "cumulative_rank_loss"
            ]
        ),
        float(
            payload[
                "cumulative_preserve_loss"
            ]
        ),
    )


def _dev_reasoning_expressions(
    suite_path: Path | None,
) -> tuple[str, ...]:
    if suite_path is None:
        return ()

    suite = load_p4_task_suite(
        suite_path
    )
    output: list[str] = []
    for task in suite.tasks:
        if (
            task.category
            == "reasoning_planning"
        ):
            expression = extract_expression(
                task.prompt
            )
            if expression is not None:
                output.append(
                    expression
                )
    return tuple(output)


def _safe_status(
    *,
    before: dict[str, object],
    after: dict[str, object],
    copy_before: dict[str, object],
    copy_after: dict[str, object],
    dev_before: dict[str, object] | None,
    dev_after: dict[str, object] | None,
    token_before,
    token_after,
    p3_before,
    p3_after,
    args: argparse.Namespace,
) -> tuple[str, list[str]]:
    reasons: list[str] = []

    if int(after["passed"]) < (
        int(before["passed"])
        - args.max_overall_drop
    ):
        reasons.append("overall")

    copy_floor = max(
        args.min_numeric_copy_passes,
        int(copy_before["passed"])
        - args.max_copy_drop,
    )
    if int(copy_after["passed"]) < copy_floor:
        reasons.append("numeric_copy")

    before_cats = before[
        "categories"
    ]
    after_cats = after[
        "categories"
    ]
    allowed = {
        "tool_intent":
            args.max_tool_drop,
        "authority_behavior":
            args.max_authority_drop,
        "instruction_following":
            args.max_other_category_drop,
        "memory_use":
            args.max_other_category_drop,
        "structured_cognition":
            args.max_other_category_drop,
    }
    for category, drop in (
        allowed.items()
    ):
        if int(
            after_cats[
                category
            ]["passed"]
        ) < (
            int(
                before_cats[
                    category
                ]["passed"]
            )
            - drop
        ):
            reasons.append(category)

    if (
        token_after.mean_loss
        > token_before.mean_loss
        + args.max_token_validation_loss_increase
        or token_after.top1_accuracy
        < token_before.top1_accuracy
        - args.max_token_top1_drop
    ):
        reasons.append(
            "token_validation"
        )

    if (
        p3_after.mean_loss
        > p3_before.mean_loss
        + args.max_p3_validation_loss_increase
        or p3_after.top1_accuracy
        < p3_before.top1_accuracy
        - args.max_p3_top1_drop
    ):
        reasons.append(
            "p3_retention"
        )

    if (
        dev_before is not None
        and dev_after is not None
        and int(
            dev_after["passed"]
        ) < (
            int(
                dev_before["passed"]
            )
            - args.max_dev_drop
        )
    ):
        reasons.append("dev")

    reasoning = int(
        after_cats[
            "reasoning_planning"
        ]["passed"]
    )

    if reasons:
        return (
            "REJECTED_REGRESSION",
            reasons,
        )
    if reasoning < (
        args.min_reasoning_passes
    ):
        return (
            "REJECTED_REASONING",
            [],
        )
    return (
        "ELIGIBLE",
        [],
    )


def main(
    argv: list[str] | None = None,
) -> int:
    args = _parser().parse_args(
        argv
    )

    numeric_values = (
        args.ranking_records,
        args.numeric_copy_records,
        args.anchor_records,
        args.p3_replay_records,
        args.batch_size,
        args.preserve_batch_size,
        args.checkpoint_interval_steps,
        args.progress_interval_steps,
    )
    if (
        any(value <= 0 for value in numeric_values)
        or not torch.cuda.is_available()
        or not str(
            args.device
        ).startswith("cuda")
        or not math.isfinite(
            args.learning_rate
        )
        or args.learning_rate <= 0.0
        or not math.isfinite(
            args.ranking_margin
        )
        or args.ranking_margin <= 0.0
        or not math.isfinite(
            args.ranking_weight
        )
        or args.ranking_weight <= 0.0
        or not math.isfinite(
            args.preserve_weight
        )
        or args.preserve_weight <= 0.0
    ):
        raise VN97P4ENError(
            "P4E-N arguments are invalid"
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
        raise VN97P4ENError(
            "P4E-N output-dir must be new or empty"
        )

    (
        parent_root,
        parent_checkpoint,
        parent_package,
        parent_report,
    ) = _verify_artifact(
        Path(
            args.p4e_k_dir
        )
    )

    tokenizer = VN97Tokenizer(
        parent_package
    )
    model = (
        parent_checkpoint.model
        .to(args.device)
    )

    p3_root = Path(
        args.p3_corpus_dir
    ).resolve(
        strict=True
    )
    _validate_corpus(
        p3_root
    )

    (
        p3_training_records,
        p3_training_sha,
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
        p3_validation_sha,
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

    validation_records = (
        default_validation()
    )
    probe_records = _probe_records(
        validation_records,
        per_category=30,
    )

    suite_path = (
        Path(args.dev_suite)
        if args.dev_suite
        is not None
        else None
    )

    before = (
        _canonical_measure_records(
            model=model,
            tokenizer=tokenizer,
            records=probe_records,
            device=args.device,
        )
    )
    copy_before = (
        _measure_copy_suite(
            model=model,
            tokenizer=tokenizer,
            device=args.device,
        )
    )
    dev_before = None
    if suite_path is not None:
        dev_before = (
            _canonical_measure_dev(
                model=model,
                tokenizer=tokenizer,
                suite_path=suite_path,
                device=args.device,
            )
        )

    validation_config = VN97TrainingConfig(
        sequence_length=256,
        batch_size=4,
        epochs=1,
        seed=0,
        shuffle=False,
        max_windows=100_000,
    )
    validation_examples = [
        encode_chat_completion_messages(
            tokenizer,
            item.messages,
        )
        for item in validation_records
    ]
    validation_windows = (
        build_training_windows(
            validation_examples,
            validation_config,
            pad_token_id=
                tokenizer.pad_id,
        )
    )
    (
        validation_inputs,
        validation_labels,
    ) = _windows_to_tensors(
        validation_windows
    )

    p3_validation_config = (
        VN97TrainingConfig(
            sequence_length=
                P3_SEQUENCE_LENGTH,
            batch_size=
                P3_BATCH_SIZE,
            epochs=1,
            seed=0,
            shuffle=False,
            max_windows=
                P3_MAX_VALIDATION_WINDOWS,
        )
    )
    p3_validation_examples = [
        encode_chat_messages(
            tokenizer,
            record,
        )
        for record
        in p3_validation_records
    ]
    p3_validation_windows = (
        build_training_windows(
            p3_validation_examples,
            p3_validation_config,
            pad_token_id=
                tokenizer.pad_id,
        )
    )
    (
        p3_inputs,
        p3_labels,
    ) = _windows_to_tensors(
        p3_validation_windows
    )

    token_before = (
        evaluate_vn97_from_tensors(
            model,
            validation_inputs,
            validation_labels,
            batch_size=4,
            device=args.device,
            cpu_prefetch_workers=1,
            micro_batch_size=2,
        )
    )
    p3_before = (
        evaluate_vn97_from_tensors(
            model,
            p3_inputs,
            p3_labels,
            batch_size=
                P3_BATCH_SIZE,
            device=args.device,
            cpu_prefetch_workers=1,
            micro_batch_size=2,
        )
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(
        "VN97 P4E-N BASELINE "
        f"canonical={before['passed']}/{before['task_count']} "
        f"copy={copy_before['passed']}/{copy_before['task_count']} "
        "reasoning="
        f"{before['categories']['reasoning_planning']['passed']}/30",
        flush=True,
    )

    dev_prompts = (
        _dev_reasoning_expressions(
            suite_path
        )
    )
    ranking = ranking_training(
        count=args.ranking_records,
        forbidden_expressions=set(
            dev_prompts
        ),
        seed=args.seed,
    )
    preservation = (
        preservation_records(
            numeric_copy_count=
                args.numeric_copy_records,
            anchor_count=
                args.anchor_records,
            seed=args.seed + 1000,
        )
    )
    p3_replay = _select_replay(
        p3_training_records,
        count=
            args.p3_replay_records,
    )

    validation_prompts = {
        item.prompt
        for item in validation_records
    }
    dev_all_prompts: set[str] = set()
    if suite_path is not None:
        dev_suite = load_p4_task_suite(
            suite_path
        )
        dev_all_prompts = {
            task.prompt
            for task in dev_suite.tasks
        }

    if {
        item.prompt
        for item in ranking
    }.intersection(
        validation_prompts
    ):
        raise VN97P4ENError(
            "P4E-N ranking data overlaps validation prompts"
        )
    if (
        dev_all_prompts
        and {
            item.prompt
            for item in ranking
        }.intersection(
            dev_all_prompts
        )
    ):
        raise VN97P4ENError(
            "P4E-N ranking data overlaps held-out dev prompts"
        )

    correct_examples = [
        _example(
            tokenizer,
            item.prompt,
            item.correct,
        )
        for item in ranking
    ]
    negative_examples = [
        _example(
            tokenizer,
            item.prompt,
            item.negative,
        )
        for item in ranking
    ]
    preserve_examples = [
        encode_chat_completion_messages(
            tokenizer,
            item.messages,
        )
        for item in preservation
    ]
    preserve_examples.extend(
        encode_chat_messages(
            tokenizer,
            record,
        )
        for record in p3_replay
    )

    ranking_identity = _sha256(
        b"VN97P4ENRANK1\0"
        + _canonical_json(
            [
                {
                    "correct":
                        item.correct,
                    "negative":
                        item.negative,
                    "operation":
                        item.operation,
                    "prompt":
                        item.prompt,
                }
                for item in ranking
            ]
        )
    )
    preserve_identity = _sha256(
        b"VN97P4ENPRESERVE1\0"
        + _canonical_json(
            {
                "records":
                    len(
                        preserve_examples
                    ),
                "p3_training_sha256":
                    p3_training_sha,
                "p3_validation_sha256":
                    p3_validation_sha,
            }
        )
    )
    resume_identity = _sha256(
        (
            P4E_N_PROFILE_ID
            + "\0"
            + parent_checkpoint.checkpoint_sha256
            + "\0"
            + ranking_identity
            + "\0"
            + preserve_identity
            + "\0"
            + str(
                args.learning_rate
            )
            + "\0"
            + str(
                args.ranking_margin
            )
            + "\0"
            + str(
                args.ranking_weight
            )
            + "\0"
            + str(
                args.preserve_weight
            )
            + "\0"
            + str(
                args.batch_size
            )
            + "\0"
            + str(
                args.preserve_batch_size
            )
            + "\0memory_safe_detached_negative_v1"
        ).encode("utf-8")
    )

    work = Path(
        args.work_dir
    )
    if work.is_symlink():
        raise VN97P4ENError(
            "P4E-N work-dir must not be a symlink"
        )
    work.mkdir(
        parents=True,
        exist_ok=True,
    )
    resume_path = (
        work
        / "state.p4en.pt"
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=
            args.weight_decay,
    )

    total_steps = math.ceil(
        len(ranking)
        / args.batch_size
    )

    (
        start_step,
        cumulative_loss,
        cumulative_rank_loss,
        cumulative_preserve_loss,
    ) = _load_resume(
        resume_path,
        identity=
            resume_identity,
        model=model,
        optimizer=optimizer,
    )

    rank_order = list(
        range(
            len(ranking)
        )
    )
    preserve_order = list(
        range(
            len(
                preserve_examples
            )
        )
    )
    random.Random(
        args.seed + 2000
    ).shuffle(
        rank_order
    )
    random.Random(
        args.seed + 3000
    ).shuffle(
        preserve_order
    )

    if start_step > total_steps:
        raise VN97P4ENError(
            "P4E-N resume step exceeds training length"
        )

    started = time.monotonic()
    model.train()

    for step in range(
        start_step,
        total_steps,
    ):
        begin = (
            step
            * args.batch_size
        )
        end = min(
            begin
            + args.batch_size,
            len(rank_order),
        )
        indices = rank_order[
            begin:end
        ]

        correct_batch = [
            correct_examples[index]
            for index in indices
        ]
        negative_batch = [
            negative_examples[index]
            for index in indices
        ]

        preserve_begin = (
            step
            * args.preserve_batch_size
        )
        preserve_indices = [
            preserve_order[
                (
                    preserve_begin
                    + offset
                )
                % len(
                    preserve_order
                )
            ]
            for offset in range(
                args.preserve_batch_size
            )
        ]
        preserve_batch = [
            preserve_examples[index]
            for index
            in preserve_indices
        ]

        optimizer.zero_grad(
            set_to_none=True
        )

        # Memory-safe ordering for 16 GB-class GPUs:
        # 1) preserve loss backward immediately so its recurrent graph can die;
        # 2) score hard negatives without gradients;
        # 3) keep gradients only for the correct-answer path and ranking term.
        (
            _preserve_scores,
            preserve_ce,
        ) = _scores_and_ce(
            model,
            preserve_batch,
            pad_token_id=
                tokenizer.pad_id,
            device=args.device,
        )
        preserve_loss = (
            args.preserve_weight
            * preserve_ce
        )
        if not bool(
            torch.isfinite(
                preserve_loss
            )
        ):
            raise VN97P4ENError(
                "P4E-N preservation loss is non-finite"
            )
        preserve_loss.backward()
        preserve_value = float(
            preserve_ce.detach().item()
        )
        del (
            _preserve_scores,
            preserve_ce,
            preserve_loss,
        )

        with torch.no_grad():
            (
                negative_scores,
                _negative_ce,
            ) = _scores_and_ce(
                model,
                negative_batch,
                pad_token_id=
                    tokenizer.pad_id,
                device=args.device,
            )
            negative_scores = (
                negative_scores.detach()
            )
            del _negative_ce

        (
            correct_scores,
            correct_ce,
        ) = _scores_and_ce(
            model,
            correct_batch,
            pad_token_id=
                tokenizer.pad_id,
            device=args.device,
        )

        rank_loss = F.softplus(
            args.ranking_margin
            - (
                correct_scores
                - negative_scores
            )
        ).mean()

        reason_loss = (
            correct_ce
            + args.ranking_weight
            * rank_loss
        )

        if not bool(
            torch.isfinite(
                reason_loss
            )
        ):
            raise VN97P4ENError(
                "P4E-N reasoning loss is non-finite"
            )

        reason_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            args.max_grad_norm,
        )
        optimizer.step()

        loss_value = (
            float(
                reason_loss.detach().item()
            )
            + args.preserve_weight
            * preserve_value
        )
        rank_value = float(
            rank_loss.detach().item()
        )

        cumulative_loss += (
            loss_value
        )
        cumulative_rank_loss += (
            rank_value
        )
        cumulative_preserve_loss += (
            preserve_value
        )

        del (
            negative_scores,
            correct_scores,
            correct_ce,
            rank_loss,
            reason_loss,
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
            seconds_per = (
                elapsed
                / local_steps
            )
            eta = (
                total_steps
                - completed
            ) * seconds_per
            print(
                "VN97 P4E-N PROGRESS "
                f"step={completed}/{total_steps} "
                f"percent={100.0 * completed / total_steps:.2f} "
                f"elapsed_s={elapsed:.1f} "
                f"eta_s={eta:.1f} "
                f"loss={loss_value:.6f} "
                f"rank_loss={rank_value:.6f} "
                f"preserve_loss={preserve_value:.6f}",
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
                    resume_identity,
                model=model,
                optimizer=optimizer,
                next_step=
                    completed,
                cumulative_loss=
                    cumulative_loss,
                cumulative_rank_loss=
                    cumulative_rank_loss,
                cumulative_preserve_loss=
                    cumulative_preserve_loss,
            )
            print(
                "VN97 P4E-N CHECKPOINT "
                f"step={completed}/{total_steps} "
                f"path={resume_path}",
                flush=True,
            )

    model.eval()

    token_after = (
        evaluate_vn97_from_tensors(
            model,
            validation_inputs,
            validation_labels,
            batch_size=4,
            device=args.device,
            cpu_prefetch_workers=1,
            micro_batch_size=2,
        )
    )
    p3_after = (
        evaluate_vn97_from_tensors(
            model,
            p3_inputs,
            p3_labels,
            batch_size=
                P3_BATCH_SIZE,
            device=args.device,
            cpu_prefetch_workers=1,
            micro_batch_size=2,
        )
    )
    after = (
        _canonical_measure_records(
            model=model,
            tokenizer=tokenizer,
            records=probe_records,
            device=args.device,
        )
    )
    copy_after = (
        _measure_copy_suite(
            model=model,
            tokenizer=tokenizer,
            device=args.device,
        )
    )
    dev_after = None
    if suite_path is not None:
        dev_after = (
            _canonical_measure_dev(
                model=model,
                tokenizer=tokenizer,
                suite_path=suite_path,
                device=args.device,
            )
        )

    status, unsafe = _safe_status(
        before=before,
        after=after,
        copy_before=
            copy_before,
        copy_after=
            copy_after,
        dev_before=
            dev_before,
        dev_after=
            dev_after,
        token_before=
            token_before,
        token_after=
            token_after,
        p3_before=
            p3_before,
        p3_after=
            p3_after,
        args=args,
    )

    reasoning_before = int(
        before[
            "categories"
        ][
            "reasoning_planning"
        ]["passed"]
    )
    reasoning_after = int(
        after[
            "categories"
        ][
            "reasoning_planning"
        ]["passed"]
    )

    print(
        "VN97 P4E-N RESULT "
        f"status={status} "
        f"canonical={before['passed']}->{after['passed']}/180 "
        f"copy={copy_before['passed']}->{copy_after['passed']}/60 "
        f"reasoning={reasoning_before}->{reasoning_after}/30 "
        f"unsafe={','.join(unsafe) if unsafe else 'none'}",
        flush=True,
    )
    for category in P4D_CATEGORIES:
        row = after[
            "categories"
        ][category]
        print(
            "VN97 P4E-N CATEGORY AFTER "
            f"{category} "
            f"{row['passed']}/{row['tasks']}",
            flush=True,
        )

    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_path = (
        output
        / "model.vn97ck1"
    )
    checkpoint_sha = (
        save_deployment_checkpoint(
            model,
            checkpoint_path,
        )
    )
    tokenizer_bytes = (
        parent_package.to_bytes()
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
            tile_rows=16,
            tile_cols=16,
        ).data
    )
    _atomic_write(
        output
        / "model.vn97mi1",
        image_bytes,
    )

    report = {
        "after": {
            "canonical":
                after,
            "dev":
                dev_after,
            "numeric_copy":
                copy_after,
            "p3_retention": {
                "mean_loss":
                    p3_after.mean_loss,
                "top1_accuracy":
                    p3_after.top1_accuracy,
            },
            "token_validation": {
                "mean_loss":
                    token_after.mean_loss,
                "top1_accuracy":
                    token_after.top1_accuracy,
            },
        },
        "before": {
            "canonical":
                before,
            "dev":
                dev_before,
            "numeric_copy":
                copy_before,
            "p3_retention": {
                "mean_loss":
                    p3_before.mean_loss,
                "top1_accuracy":
                    p3_before.top1_accuracy,
            },
            "token_validation": {
                "mean_loss":
                    token_before.mean_loss,
                "top1_accuracy":
                    token_before.top1_accuracy,
            },
        },
        "model_image_sha256":
            _sha256(
                image_bytes
            ),
        "objective": {
            "anchor_records":
                args.anchor_records,
            "batch_size":
                args.batch_size,
            "learning_rate":
                args.learning_rate,
            "numeric_copy_records":
                args.numeric_copy_records,
            "p3_replay_records":
                args.p3_replay_records,
            "preserve_batch_size":
                args.preserve_batch_size,
            "preserve_weight":
                args.preserve_weight,
            "ranking_margin":
                args.ranking_margin,
            "ranking_records":
                args.ranking_records,
            "ranking_weight":
                args.ranking_weight,
            "steps":
                total_steps,
        },
        "output_checkpoint_sha256":
            checkpoint_sha,
        "parent_p4e_k": {
            "checkpoint_sha256":
                parent_checkpoint.checkpoint_sha256,
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
            P4E_N_PROFILE_ID,
        "schema":
            P4E_N_REPORT_SCHEMA,
        "status":
            status,
        "unsafe_reasons":
            unsafe,
    }

    report_bytes = (
        _canonical_json(
            report
        )
        + b"\n"
    )
    _atomic_write(
        output
        / "p4n-report.json",
        report_bytes,
    )

    files = {
        "model.vn97ck1":
            checkpoint_path.read_bytes(),
        "model.vn97mi1":
            image_bytes,
        "p4n-report.json":
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
        "VN97P4EN1 "
        f"status={status} "
        f"checkpoint={checkpoint_sha} "
        f"canonical_before={before['passed']}/180 "
        f"canonical_after={after['passed']}/180 "
        f"copy_before={copy_before['passed']}/60 "
        f"copy_after={copy_after['passed']}/60 "
        f"reasoning_before={reasoning_before}/30 "
        f"reasoning_after={reasoning_after}/30",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(
            "vn97-p4e-sequence-ranking-repair: "
            f"{exc}",
            file=sys.stderr,
        )
        raise
