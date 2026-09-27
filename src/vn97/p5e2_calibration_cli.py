from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time

import torch
import torch.nn.functional as F

from .config import VN97Config
from .model import VN97LanguageCore
from .p3_language_campaign_cli import _validate_corpus
from .p5_scaled_foundation_cli import _select_windows, _tensor_batch
from .p5d_behavioral_distillation_cli import (
    _evaluate_fp32,
    _safe_torch_save,
    _windows_from_messages,
)
from .quantization import set_float_shadow_mode
from .tokenizer import VN97Tokenizer, VN97TokenizerPackage
from .training import IGNORE_INDEX
from .training_cli import _atomic_write, _load_records


P5E2_SCHEMA = "VN97P5E2"
P5E2_FLOAT_SCHEMA = "VN97P5E2FLOAT1"
P5E2_PROFILE_ID = "vn97-p5e2-short-post-transplant-calibration-v1"

TRAIN_WINDOWS = 192
VALIDATION_WINDOWS = 32
TRAIN_STEPS = 96
LEARNING_RATE = 2e-6
WEIGHT_DECAY = 1e-4
MAX_GRAD_NORM = 0.5
TRAINABLE_TAIL_LAYERS = 8
PROGRESS_INTERVAL = 12


class VN97P5E2Error(RuntimeError):
    pass


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Short post-transplant calibration for the P5E1 309M VN97 "
            "float-shadow model. The calibration uses only native VN97 "
            "training data; Falcon3-Mamba is not loaded or executed."
        )
    )
    parser.add_argument("--p5e1-dir", required=True)
    parser.add_argument("--p3-corpus-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _load_p5e1(
    root: Path,
) -> tuple[
    VN97LanguageCore,
    VN97Tokenizer,
    bytes,
    dict[str, object],
    str,
]:
    resolved = root.resolve(strict=True)
    report_path = resolved / "p5e1-report.json"
    student_path = resolved / "student-float.pt"
    tokenizer_path = resolved / "tokenizer.vn97tk1"
    if not report_path.is_file() or not student_path.is_file() or not tokenizer_path.is_file():
        raise VN97P5E2Error("P5E1 artifact is incomplete")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (
        not isinstance(report, dict)
        or report.get("schema") != "VN97P5E1"
        or report.get("status") != "DIRECT_SSM_TRANSPLANT_READY_FOR_CALIBRATION"
    ):
        raise VN97P5E2Error("P5E1 report is not calibratable")

    payload = torch.load(
        student_path,
        map_location="cpu",
        weights_only=False,
    )
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != "VN97P5E1FLOAT1"
        or payload.get("float_shadow_required") is not True
    ):
        raise VN97P5E2Error("invalid P5E1 float-shadow artifact")

    tokenizer_bytes = tokenizer_path.read_bytes()
    package = VN97TokenizerPackage.from_bytes(tokenizer_bytes)
    tokenizer_sha = hashlib.sha256(tokenizer_bytes).hexdigest()
    if payload.get("tokenizer_sha256") != tokenizer_sha:
        raise VN97P5E2Error("P5E1 tokenizer hash mismatch")

    config_raw = payload.get("config")
    state_dict = payload.get("state_dict")
    if not isinstance(config_raw, dict) or not isinstance(state_dict, dict):
        raise VN97P5E2Error("P5E1 student payload is incomplete")
    config = VN97Config(**config_raw)
    if (
        config.d_model != 1536
        or config.n_layers != 32
        or config.d_state != 16
        or package.vocab_size != config.vocab_size
    ):
        raise VN97P5E2Error("P5E1 student is not the canonical 309M VN97 target")

    model = VN97LanguageCore(config)
    set_float_shadow_mode(model, True)
    model.load_state_dict(state_dict)
    return (
        model,
        VN97Tokenizer(package),
        tokenizer_bytes,
        report,
        _sha256(student_path),
    )


def _load_windows(
    *,
    corpus_root: Path,
    tokenizer: VN97Tokenizer,
) -> tuple[tuple[object, ...], tuple[object, ...], str]:
    _validate_corpus(corpus_root)
    train_raw, train_sha = _load_records(
        [corpus_root / "training.jsonl"],
        mode="chat",
        max_input_bytes=256 * 1024 * 1024,
        max_examples=500_000,
    )
    validation_raw, validation_sha = _load_records(
        [corpus_root / "validation.jsonl"],
        mode="chat",
        max_input_bytes=64 * 1024 * 1024,
        max_examples=100_000,
    )

    train_all = _windows_from_messages(
        tokenizer=tokenizer,
        messages_list=[tuple(messages) for messages in train_raw],
    )
    validation_all = _windows_from_messages(
        tokenizer=tokenizer,
        messages_list=[tuple(messages) for messages in validation_raw],
    )
    if not train_all or not validation_all:
        raise VN97P5E2Error("P3 calibration windows are empty")

    train = _select_windows(
        train_all,
        min(TRAIN_WINDOWS, len(train_all)),
    )
    validation = _select_windows(
        validation_all,
        min(VALIDATION_WINDOWS, len(validation_all)),
    )
    corpus_identity = hashlib.sha256(
        (
            "VN97P5E2CORPUS\0"
            + train_sha
            + "\0"
            + validation_sha
        ).encode("ascii")
    ).hexdigest()
    return train, validation, corpus_identity


def _configure_trainable_tail(model: VN97LanguageCore) -> tuple[list[torch.nn.Parameter], list[str]]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    names: list[str] = []
    parameters: list[torch.nn.Parameter] = []

    start = max(0, len(model.layers) - TRAINABLE_TAIL_LAYERS)
    for layer_index in range(start, len(model.layers)):
        for name, parameter in model.layers[layer_index].named_parameters():
            parameter.requires_grad_(True)
            parameters.append(parameter)
            names.append(f"layers.{layer_index}.{name}")

    for name, parameter in model.final_norm.named_parameters():
        parameter.requires_grad_(True)
        parameters.append(parameter)
        names.append(f"final_norm.{name}")

    if not parameters:
        raise VN97P5E2Error("no trainable calibration parameters")
    return parameters, names


def _train(
    *,
    model: VN97LanguageCore,
    windows: tuple[object, ...],
    device: str,
) -> dict[str, float | int]:
    trainable, names = _configure_trainable_tail(model)
    optimizer = torch.optim.AdamW(
        trainable,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    model.to(device)
    set_float_shadow_mode(model, True)
    model.train()

    loss_sum = 0.0
    target_tokens = 0
    started = time.monotonic()

    for step in range(TRAIN_STEPS):
        window = windows[step % len(windows)]
        inputs, labels = _tensor_batch(window, device=device)
        optimizer.zero_grad(set_to_none=True)

        logits, _ = model.forward_sequential_reference(inputs)
        mask = labels != IGNORE_INDEX
        if not bool(mask.any()):
            raise VN97P5E2Error("calibration window has no supervised target tokens")

        selected_logits = logits[mask].float()
        selected_labels = labels[mask]
        loss = F.cross_entropy(
            selected_logits,
            selected_labels,
            reduction="mean",
        )
        if not bool(torch.isfinite(loss)):
            raise VN97P5E2Error("P5E2 calibration loss became non-finite")

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            trainable,
            MAX_GRAD_NORM,
        )
        if not bool(torch.isfinite(torch.as_tensor(grad_norm))):
            raise VN97P5E2Error("P5E2 gradient norm became non-finite")
        optimizer.step()

        count = int(selected_labels.numel())
        loss_sum += float(loss.item()) * count
        target_tokens += count

        if (
            (step + 1) % PROGRESS_INTERVAL == 0
            or step + 1 == TRAIN_STEPS
        ):
            elapsed = time.monotonic() - started
            mean_loss = loss_sum / max(target_tokens, 1)
            print(
                "VN97 P5E2 PROGRESS "
                f"step={step + 1}/{TRAIN_STEPS} "
                f"percent={(step + 1) * 100.0 / TRAIN_STEPS:.2f} "
                f"elapsed_s={elapsed:.1f} "
                f"loss={float(loss.item()):.6f} "
                f"mean_loss={mean_loss:.6f} "
                f"grad_norm={float(grad_norm):.6f}",
                flush=True,
            )

        del inputs, labels, logits, selected_logits, selected_labels, loss

    model.cpu()
    torch.cuda.empty_cache()
    return {
        "mean_loss": loss_sum / max(target_tokens, 1),
        "steps": TRAIN_STEPS,
        "target_tokens": target_tokens,
        "trainable_parameters": sum(int(p.numel()) for p in trainable),
        "trainable_tensor_count": len(trainable),
        "trainable_name_count": len(names),
    }


def _status(
    before: dict[str, float | int],
    after: dict[str, float | int],
) -> str:
    before_loss = float(before["mean_loss"])
    after_loss = float(after["mean_loss"])
    before_top1 = float(before["top1_accuracy"])
    after_top1 = float(after["top1_accuracy"])
    if not all(
        torch.isfinite(torch.tensor(v)).item()
        for v in (before_loss, after_loss, before_top1, after_top1)
    ):
        return "REJECTED_NONFINITE"
    if after_loss <= before_loss * 1.02 and after_top1 >= before_top1 - 0.01:
        if after_loss < before_loss or after_top1 > before_top1:
            return "CALIBRATION_SIGNAL"
        return "CALIBRATION_STABLE"
    return "REJECTED_REGRESSION"


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not torch.cuda.is_available() and str(args.device).startswith("cuda"):
        raise VN97P5E2Error("P5E2 CUDA device requested but CUDA is unavailable")

    model, tokenizer, tokenizer_bytes, p5e1_report, parent_sha = _load_p5e1(
        Path(args.p5e1_dir)
    )
    train_windows, validation_windows, corpus_identity = _load_windows(
        corpus_root=Path(args.p3_corpus_dir).resolve(strict=True),
        tokenizer=tokenizer,
    )

    model.to(args.device)
    before = _evaluate_fp32(
        model,
        validation_windows,
        device=args.device,
    )
    model.cpu()
    torch.cuda.empty_cache()

    print(
        "VN97 P5E2 BASELINE "
        f"loss={float(before['mean_loss']):.6f} "
        f"top1={float(before['top1_accuracy']):.6f} "
        f"train_windows={len(train_windows)} "
        f"validation_windows={len(validation_windows)} "
        f"teacher_loaded=false",
        flush=True,
    )

    training = _train(
        model=model,
        windows=train_windows,
        device=args.device,
    )

    model.to(args.device)
    after = _evaluate_fp32(
        model,
        validation_windows,
        device=args.device,
    )
    model.cpu()
    torch.cuda.empty_cache()

    status = _status(before, after)
    output = Path(args.output_dir)
    if output.exists() and (
        output.is_symlink()
        or not output.is_dir()
        or any(output.iterdir())
    ):
        raise VN97P5E2Error("output-dir must be new or empty")
    output.mkdir(parents=True, exist_ok=True)

    report = {
        "calibration": {
            "learning_rate": LEARNING_RATE,
            "max_grad_norm": MAX_GRAD_NORM,
            "steps": TRAIN_STEPS,
            "trainable_tail_layers": TRAINABLE_TAIL_LAYERS,
            "weight_decay": WEIGHT_DECAY,
        },
        "corpus_identity": corpus_identity,
        "evaluation": {
            "after": after,
            "before": before,
        },
        "p5e1": {
            "blend": p5e1_report.get("blend"),
            "student_sha256": parent_sha,
        },
        "profile_id": P5E2_PROFILE_ID,
        "schema": P5E2_SCHEMA,
        "status": status,
        "teacher_loaded": False,
        "training": training,
    }
    report_bytes = (
        json.dumps(
            report,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    report_sha = hashlib.sha256(report_bytes).hexdigest()
    _atomic_write(output / "p5e2-report.json", report_bytes)

    payload = {
        "config": asdict(model.config),
        "float_shadow_required": True,
        "p5e1_student_sha256": parent_sha,
        "profile_id": P5E2_PROFILE_ID,
        "report_sha256": report_sha,
        "schema": P5E2_FLOAT_SCHEMA,
        "state_dict": model.state_dict(),
        "status": status,
        "tokenizer_sha256": hashlib.sha256(tokenizer_bytes).hexdigest(),
    }
    _safe_torch_save(
        payload,
        output / "student-float.pt",
    )
    _atomic_write(
        output / "tokenizer.vn97tk1",
        tokenizer_bytes,
    )

    files = {
        "p5e2-report.json": output / "p5e2-report.json",
        "student-float.pt": output / "student-float.pt",
        "tokenizer.vn97tk1": output / "tokenizer.vn97tk1",
    }
    sums = "".join(
        f"{_sha256(path)}  {name}\n"
        for name, path in sorted(files.items())
    ).encode("ascii")
    _atomic_write(output / "SHA256SUMS", sums)

    print(
        "VN97P5E2 "
        f"status={status} "
        f"loss_before={float(before['mean_loss']):.6f} "
        f"loss_after={float(after['mean_loss']):.6f} "
        f"top1_before={float(before['top1_accuracy']):.6f} "
        f"top1_after={float(after['top1_accuracy']):.6f} "
        f"teacher_loaded=false "
        f"student_sha256={_sha256(output / 'student-float.pt')}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"vn97-p5e2-calibrate: {exc}", file=sys.stderr)
        raise
