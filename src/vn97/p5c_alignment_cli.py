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
from .p3_language_campaign_cli import _validate_corpus
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
    default_validation as p4_validation,
)
from .p4_numeric_arithmetic_repair_cli import _measure_copy_suite
from .p5_scaled_foundation_cli import (
    _evaluate_windows,
    _held_out_dev,
    _select_p4_records,
    _select_windows,
    _tensor_batch,
    _window_digest,
)
from .p5c_alignment import (
    CHECKPOINT_INTERVAL_STEPS,
    CORE_LR,
    LEXICAL_LR,
    LEXICAL_WARMUP_STEPS,
    LOGICAL_BATCH_SIZE,
    MAX_GRAD_NORM,
    MAX_TRAIN_WINDOWS,
    MIN_CANONICAL_PASS,
    MIN_COPY_PASS,
    MIN_P3_TOP1,
    MIN_REASONING_PASS,
    P3_TRAIN_RECORDS,
    P3_VALIDATION_WINDOWS,
    P4_TRAIN_RECORDS,
    P5C_PROFILE_ID,
    P5C_SCHEMA,
    PROGRESS_INTERVAL_STEPS,
    SEED,
    SEQUENCE_LENGTH,
    WEIGHT_DECAY,
)
from .tokenizer import (
    VN97Tokenizer,
    VN97TokenizerPackage,
)
from .training import (
    IGNORE_INDEX,
    VN97TrainingConfig,
    build_training_windows,
    encode_chat_completion_messages,
    encode_chat_messages,
)
from .training_cli import (
    _atomic_write,
    _canonical_json,
    _load_records,
)


class VN97P5CError(RuntimeError):
    pass


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Short low-rate alignment/calibration of a P5B transplanted "
            "native VN97 checkpoint."
        )
    )
    parser.add_argument("--p5b-dir", required=True)
    parser.add_argument("--p3-corpus-dir", required=True)
    parser.add_argument("--dev-suite", default=None)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser


def _verify_p5b(
    root: Path,
):
    resolved = root.resolve(strict=True)
    if (
        not resolved.is_dir()
        or resolved.is_symlink()
    ):
        raise VN97P5CError(
            "P5B parent must be a real directory"
        )

    required = {
        "SHA256SUMS",
        "model.vn97ck1",
        "model.vn97mi1",
        "p5b-report.json",
        "tokenizer.vn97tk1",
    }
    names = {
        item.name
        for item in resolved.iterdir()
    }
    if names != required:
        raise VN97P5CError(
            "P5B parent file set mismatch"
        )

    sums: dict[str, str] = {}
    text = (
        resolved
        / "SHA256SUMS"
    ).read_text(
        encoding="ascii"
    )
    if not text.endswith("\n"):
        raise VN97P5CError(
            "P5B SHA256SUMS must end with newline"
        )

    for line in text.splitlines():
        if (
            len(line) < 67
            or line[64:66] != "  "
        ):
            raise VN97P5CError(
                "malformed P5B SHA256SUMS"
            )
        digest = line[:64]
        name = line[66:]
        if (
            len(digest) != 64
            or any(
                char not in
                "0123456789abcdef"
                for char in digest
            )
        ):
            raise VN97P5CError(
                "invalid P5B SHA256 digest"
            )
        sums[name] = digest

    if set(sums) != (
        required
        - {"SHA256SUMS"}
    ):
        raise VN97P5CError(
            "P5B SHA256SUMS file set mismatch"
        )

    for name, expected in sums.items():
        actual = hashlib.sha256(
            (
                resolved
                / name
            ).read_bytes()
        ).hexdigest()
        if actual != expected:
            raise VN97P5CError(
                f"P5B SHA256 mismatch: {name}"
            )

    report = json.loads(
        (
            resolved
            / "p5b-report.json"
        ).read_text(
            encoding="utf-8"
        )
    )
    if (
        not isinstance(report, dict)
        or report.get("schema")
        != "VN97P5B1"
        or report.get("status")
        != "TRANSPLANT_READY_FOR_ALIGNMENT"
        or report.get("alignment_required")
        is not True
    ):
        raise VN97P5CError(
            "parent is not an alignment-ready P5B artifact"
        )

    checkpoint = (
        load_deployment_checkpoint_file(
            resolved
            / "model.vn97ck1"
        )
    )
    package = (
        VN97TokenizerPackage
        .from_bytes(
            (
                resolved
                / "tokenizer.vn97tk1"
            ).read_bytes()
        )
    )
    if (
        checkpoint.config.vocab_size
        != package.vocab_size
    ):
        raise VN97P5CError(
            "P5B checkpoint/tokenizer vocabulary mismatch"
        )

    return (
        resolved,
        checkpoint,
        package,
        report,
    )


def _save_resume(
    path: Path,
    *,
    identity: str,
    model,
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
    model,
    optimizer,
    scaler,
) -> tuple[int, float, int]:
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
        raise VN97P5CError(
            "P5C resume identity mismatch"
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


def _status(
    *,
    canonical: int,
    reasoning: int,
    copy_passed: int,
    p3_top1: float,
    p3_before_top1: float,
) -> str:
    if (
        canonical
        >= MIN_CANONICAL_PASS
        and reasoning
        >= MIN_REASONING_PASS
        and copy_passed
        >= MIN_COPY_PASS
        and p3_top1
        >= MIN_P3_TOP1
    ):
        return "ELIGIBLE_FOR_P5D"

    if (
        math.isfinite(
            p3_top1
        )
        and p3_top1
        >= p3_before_top1
        - 0.01
    ):
        return (
            "ALIGNMENT_COMPLETED_BELOW_GATE"
        )

    return "REJECTED_ALIGNMENT"


def main(
    argv: list[str] | None = None,
) -> int:
    args = _parser().parse_args(
        argv
    )

    if (
        not torch.cuda.is_available()
        or not str(
            args.device
        ).startswith("cuda")
    ):
        raise VN97P5CError(
            "P5C requires CUDA"
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
        raise VN97P5CError(
            "P5C output-dir must be new or empty"
        )

    (
        parent_root,
        parent_checkpoint,
        package,
        parent_report,
    ) = _verify_p5b(
        Path(
            args.p5b_dir
        )
    )
    tokenizer = VN97Tokenizer(
        package
    )
    tokenizer_bytes = (
        package.to_bytes()
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

    p3_selected = _select_replay(
        p3_training_records,
        count=
            P3_TRAIN_RECORDS,
    )
    p4_selected = _select_p4_records(
        count=P4_TRAIN_RECORDS,
        held_out_prompts=
            held_out_prompts,
        held_out_expressions=
            held_out_expressions,
    )

    build_config = VN97TrainingConfig(
        sequence_length=
            SEQUENCE_LENGTH,
        batch_size=
            LOGICAL_BATCH_SIZE,
        epochs=1,
        learning_rate=
            CORE_LR,
        seed=SEED,
        shuffle=False,
        max_windows=20_000,
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
            build_config,
            pad_token_id=
                tokenizer.pad_id,
        )
    )
    p4_windows_all = (
        build_training_windows(
            p4_examples,
            build_config,
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
            P3_VALIDATION_WINDOWS,
            len(
                validation_windows_all
            ),
        ),
    )

    model = (
        parent_checkpoint.model
        .to(args.device)
    )

    p3_before = _evaluate_windows(
        model,
        validation_windows,
        device=args.device,
    )
    baseline_probe = (
        _canonical_measure_records(
            model=model,
            tokenizer=tokenizer,
            records=_probe_records(
                p4_validation(),
                per_category=5,
            ),
            device=args.device,
        )
    )
    baseline_copy = (
        _measure_copy_suite(
            model=model,
            tokenizer=tokenizer,
            device=args.device,
        )
    )

    print(
        "VN97 P5C BASELINE "
        f"probe={baseline_probe['passed']}/{baseline_probe['task_count']} "
        f"copy={baseline_copy['passed']}/{baseline_copy['task_count']} "
        f"p3_loss={p3_before['mean_loss']:.6f} "
        f"p3_top1={p3_before['top1_accuracy']:.6f}",
        flush=True,
    )

    lexical_params = []
    core_params = []
    for name, parameter in (
        model.named_parameters()
    ):
        if name.startswith(
            "embedding."
        ):
            lexical_params.append(
                parameter
            )
        else:
            core_params.append(
                parameter
            )

    optimizer = torch.optim.AdamW(
        [
            {
                "params":
                    lexical_params,
                "lr":
                    LEXICAL_LR,
            },
            {
                "params":
                    core_params,
                "lr":
                    CORE_LR,
            },
        ],
        weight_decay=
            WEIGHT_DECAY,
    )
    # FP16 backward overflowed on the transplanted recurrent core before the
    # first optimizer step. P5C v2 deliberately trains in FP32. A disabled
    # GradScaler is retained only so resume serialization stays simple.
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=False,
    )

    identity = _sha256(
        _canonical_json(
            {
                "core_lr":
                    CORE_LR,
                "lexical_lr":
                    LEXICAL_LR,
                "lexical_warmup_steps":
                    LEXICAL_WARMUP_STEPS,
                "logical_batch_size":
                    LOGICAL_BATCH_SIZE,
                "max_train_windows":
                    MAX_TRAIN_WINDOWS,
                "p3_training_sha256":
                    p3_training_sha256,
                "p3_validation_sha256":
                    p3_validation_sha256,
                "parent_checkpoint_sha256":
                    parent_checkpoint
                    .checkpoint_sha256,
                "parent_report_sha256":
                    parent_report.get(
                        "report_sha256"
                    ),
                "profile_id":
                    P5C_PROFILE_ID,
                "seed":
                    SEED,
                "sequence_length":
                    SEQUENCE_LENGTH,
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
        raise VN97P5CError(
            "P5C work-dir must not be a symlink"
        )
    work.mkdir(
        parents=True,
        exist_ok=True,
    )
    resume_path = (
        work
        / "state.p5c.pt"
    )

    (
        start_step,
        cumulative_loss_sum,
        cumulative_target_tokens,
    ) = _load_resume(
        resume_path,
        identity=identity,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
    )

    order = list(
        range(
            len(train_windows)
        )
    )
    random.Random(
        SEED
    ).shuffle(
        order
    )

    total_steps = math.ceil(
        len(order)
        / LOGICAL_BATCH_SIZE
    )
    if start_step > total_steps:
        raise VN97P5CError(
            "P5C resume step exceeds training length"
        )

    print(
        "VN97 P5C START "
        f"parameters={sum(int(p.numel()) for p in model.parameters())} "
        f"windows={len(train_windows)} "
        f"steps={total_steps} "
        f"resume_step={start_step} "
        f"sequence_length={SEQUENCE_LENGTH} "
        f"precision=fp32 "
        f"lexical_warmup_steps={LEXICAL_WARMUP_STEPS} "
        f"lexical_lr={LEXICAL_LR} "
        f"core_lr={CORE_LR}",
        flush=True,
    )

    for parameter in core_params:
        parameter.requires_grad_(
            start_step >= LEXICAL_WARMUP_STEPS
        )
    optimizer.param_groups[1]["lr"] = (
        CORE_LR
        if start_step >= LEXICAL_WARMUP_STEPS
        else 0.0
    )

    model.train()
    started = time.monotonic()

    for step in range(
        start_step,
        total_steps,
    ):
        if step == LEXICAL_WARMUP_STEPS:
            for parameter in core_params:
                parameter.requires_grad_(
                    True
                )
            optimizer.param_groups[1]["lr"] = (
                CORE_LR
            )
            print(
                "VN97 P5C CORE UNFREEZE "
                f"step={step}/{total_steps} "
                f"core_lr={CORE_LR}",
                flush=True,
            )

        begin = (
            step
            * LOGICAL_BATCH_SIZE
        )
        indices = order[
            begin:
            min(
                begin
                + LOGICAL_BATCH_SIZE,
                len(order),
            )
        ]
        batch = [
            train_windows[index]
            for index in indices
        ]
        logical_targets = sum(
            window.target_tokens
            for window in batch
        )
        if logical_targets <= 0:
            raise VN97P5CError(
                "P5C logical batch has no targets"
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
                raise VN97P5CError(
                    "P5C loss became non-finite"
                )

            loss.backward()

            logical_loss_sum += float(
                token_loss_sum
                .detach()
                .float()
                .item()
            )
            del (
                inputs,
                labels,
                logits,
                token_loss_sum,
                loss,
            )

        grad_norm = (
            torch.nn.utils
            .clip_grad_norm_(
                model.parameters(),
                MAX_GRAD_NORM,
            )
        )
        if not bool(
            torch.isfinite(
                torch.as_tensor(
                    grad_norm
                )
            )
        ):
            raise VN97P5CError(
                "P5C gradient norm became non-finite"
            )

        optimizer.step()

        cumulative_loss_sum += (
            logical_loss_sum
        )
        cumulative_target_tokens += (
            logical_targets
        )
        completed = step + 1

        if (
            completed
            % PROGRESS_INTERVAL_STEPS
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
                "VN97 P5C PROGRESS "
                f"step={completed}/{total_steps} "
                f"percent={100.0 * completed / total_steps:.2f} "
                f"elapsed_s={elapsed:.1f} "
                f"eta_s={eta:.1f} "
                f"batch_loss={logical_loss_sum / logical_targets:.6f} "
                f"mean_loss={cumulative_loss_sum / cumulative_target_tokens:.6f} "
                f"phase={'lexical_warmup' if completed <= LEXICAL_WARMUP_STEPS else 'full_alignment'}",
                flush=True,
            )

        if (
            completed
            % CHECKPOINT_INTERVAL_STEPS
            == 0
            and completed
            < total_steps
        ):
            _save_resume(
                resume_path,
                identity=identity,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                next_step=completed,
                loss_sum=
                    cumulative_loss_sum,
                target_tokens=
                    cumulative_target_tokens,
            )
            print(
                "VN97 P5C CHECKPOINT "
                f"step={completed}/{total_steps} "
                f"path={resume_path}",
                flush=True,
            )

    del optimizer
    del scaler
    torch.cuda.empty_cache()
    model.eval()

    p3_after = _evaluate_windows(
        model,
        validation_windows,
        device=args.device,
    )
    canonical_after = (
        _canonical_measure_records(
            model=model,
            tokenizer=tokenizer,
            records=_probe_records(
                p4_validation(),
                per_category=30,
            ),
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

    reasoning = int(
        canonical_after[
            "categories"
        ][
            "reasoning_planning"
        ]["passed"]
    )
    canonical_passed = int(
        canonical_after[
            "passed"
        ]
    )
    copy_passed = int(
        copy_after[
            "passed"
        ]
    )
    status = _status(
        canonical=
            canonical_passed,
        reasoning=
            reasoning,
        copy_passed=
            copy_passed,
        p3_top1=float(
            p3_after[
                "top1_accuracy"
            ]
        ),
        p3_before_top1=float(
            p3_before[
                "top1_accuracy"
            ]
        ),
    )

    print(
        "VN97 P5C RESULT "
        f"status={status} "
        f"canonical={canonical_passed}/180 "
        f"reasoning={reasoning}/30 "
        f"copy={copy_passed}/60 "
        f"p3_loss={p3_after['mean_loss']:.6f} "
        f"p3_top1={p3_after['top1_accuracy']:.6f}",
        flush=True,
    )
    for category in P4D_CATEGORIES:
        row = (
            canonical_after[
                "categories"
            ][category]
        )
        print(
            "VN97 P5C CATEGORY "
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
                package,
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
                canonical_after,
            "dev":
                dev_after,
            "numeric_copy":
                copy_after,
            "p3_validation":
                p3_after,
        },
        "baseline": {
            "numeric_copy":
                baseline_copy,
            "p3_validation":
                p3_before,
            "probe":
                baseline_probe,
        },
        "checkpoint_sha256":
            checkpoint_sha256,
        "corpus_manifest_id":
            corpus_manifest_id,
        "corpus_manifest_sha256":
            corpus_manifest_sha256,
        "model_image_bytes":
            len(image_bytes),
        "model_image_sha256":
            hashlib.sha256(
                image_bytes
            ).hexdigest(),
        "parent_p5b": {
            "checkpoint_sha256":
                parent_checkpoint
                .checkpoint_sha256,
            "report_sha256":
                parent_report.get(
                    "report_sha256"
                ),
            "source":
                parent_report.get(
                    "source"
                ),
        },
        "profile_id":
            P5C_PROFILE_ID,
        "schema":
            P5C_SCHEMA,
        "status":
            status,
        "training": {
            "core_lr":
                CORE_LR,
            "final_mean_loss":
                cumulative_loss_sum
                / cumulative_target_tokens,
            "lexical_lr":
                LEXICAL_LR,
            "lexical_warmup_steps":
                LEXICAL_WARMUP_STEPS,
            "logical_batch_size":
                LOGICAL_BATCH_SIZE,
            "precision":
                "fp32",
            "steps":
                total_steps,
            "target_tokens":
                cumulative_target_tokens,
            "windows":
                len(train_windows),
        },
    }
    report_bytes = (
        json.dumps(
            report,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    _atomic_write(
        output
        / "p5c-report.json",
        report_bytes,
    )

    files = {
        "model.vn97ck1":
            checkpoint_path.read_bytes(),
        "model.vn97mi1":
            image_bytes,
        "p5c-report.json":
            report_bytes,
        "tokenizer.vn97tk1":
            tokenizer_bytes,
    }
    sums = b"".join(
        (
            hashlib.sha256(
                files[name]
            ).hexdigest()
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
        "VN97P5C1 "
        f"status={status} "
        f"checkpoint={checkpoint_sha256} "
        f"canonical={canonical_passed}/180 "
        f"reasoning={reasoning}/30 "
        f"copy={copy_passed}/60",
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
            "vn97-p5c-align: "
            f"{exc}",
            file=sys.stderr,
        )
        raise
