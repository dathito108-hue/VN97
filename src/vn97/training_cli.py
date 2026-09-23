from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch

from .config import VN97Config
from .deployment_checkpoint import save_deployment_checkpoint
from .model import VN97LanguageCore
from .tokenizer import VN97Tokenizer, learn_byte_bpe
from .training import (
    VN97ChatMessage,
    VN97TrainingConfig,
    build_training_windows,
    encode_causal_text,
    encode_chat_messages,
    render_chat_text,
    train_vn97_language,
)


_MAX_DEFAULT_INPUT_BYTES = 64 * 1024 * 1024


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _load_records(
    paths: list[Path],
    *,
    mode: str,
    max_input_bytes: int,
    max_examples: int,
) -> tuple[list[object], str]:
    if not paths:
        raise ValueError("at least one --input path is required")
    total = 0
    records: list[object] = []
    digest = hashlib.sha256()

    for path in paths:
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"training input must be a regular non-symlink file: {path}")
        data = path.read_bytes()
        total += len(data)
        if total > max_input_bytes:
            raise ValueError("training inputs exceed --max-input-bytes")
        digest.update(len(data).to_bytes(8, "little"))
        digest.update(data)

        try:
            text = data.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise ValueError(f"training input is not UTF-8: {path}") from exc

        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid JSONL at {path}:{line_number}"
                ) from exc
            if not isinstance(value, dict):
                raise ValueError(f"JSONL record must be object at {path}:{line_number}")

            if mode == "text":
                if set(value) != {"text"} or not isinstance(value["text"], str) or not value["text"]:
                    raise ValueError(
                        f"text record must be exactly {{\"text\": non-empty string}} at {path}:{line_number}"
                    )
                records.append(value["text"])
            else:
                if set(value) != {"messages"} or not isinstance(value["messages"], list):
                    raise ValueError(
                        f"chat record must be exactly {{\"messages\": [...]}} at {path}:{line_number}"
                    )
                messages: list[VN97ChatMessage] = []
                for raw in value["messages"]:
                    if not isinstance(raw, dict) or set(raw) != {"role", "content"}:
                        raise ValueError(
                            f"chat message keys are invalid at {path}:{line_number}"
                        )
                    messages.append(
                        VN97ChatMessage(
                            role=raw["role"],
                            content=raw["content"],
                        )
                    )
                if not messages:
                    raise ValueError(f"chat record has no messages at {path}:{line_number}")
                records.append(tuple(messages))

            if len(records) > max_examples:
                raise ValueError("training record count exceeds --max-examples")

    if not records:
        raise ValueError("training inputs contain no usable records")
    return records, digest.hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train canonical VN97LanguageCore and export VN97CK1/VN97TK1."
    )
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--format", choices=("text", "chat"), default="chat")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-input-bytes", type=int, default=_MAX_DEFAULT_INPUT_BYTES)
    parser.add_argument("--max-examples", type=int, default=100_000)
    parser.add_argument("--learned-tokens", type=int, default=2048)
    parser.add_argument("--min-pair-count", type=int, default=2)

    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--d-state", type=int, default=8)
    parser.add_argument("--embedding-rank", type=int, default=None)

    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=97)
    parser.add_argument("--device", default="auto")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.max_input_bytes <= 0 or args.max_examples <= 0:
        raise ValueError("input bounds must be positive")

    records, dataset_sha256 = _load_records(
        [Path(value) for value in args.input],
        mode=args.format,
        max_input_bytes=args.max_input_bytes,
        max_examples=args.max_examples,
    )

    corpus = (
        records
        if args.format == "text"
        else [render_chat_text(value) for value in records]
    )
    package = learn_byte_bpe(
        corpus,
        max_learned_tokens=args.learned_tokens,
        min_pair_count=args.min_pair_count,
    )
    tokenizer = VN97Tokenizer(package)

    torch.manual_seed(args.seed)
    model = VN97LanguageCore(
        VN97Config(
            vocab_size=tokenizer.vocab_size,
            d_model=args.d_model,
            n_layers=args.layers,
            d_state=args.d_state,
            embedding_rank=args.embedding_rank,
        )
    )

    examples = (
        [encode_causal_text(tokenizer, value) for value in records]
        if args.format == "text"
        else [encode_chat_messages(tokenizer, value) for value in records]
    )
    training_config = VN97TrainingConfig(
        sequence_length=args.sequence_length,
        stride=args.stride,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        seed=args.seed,
        shuffle=True,
    )
    windows = build_training_windows(
        examples,
        training_config,
        pad_token_id=tokenizer.pad_id,
    )
    result = train_vn97_language(
        model,
        windows,
        training_config,
        device=args.device,
    )

    output = Path(args.output_dir)
    if output.is_symlink():
        raise ValueError("output-dir must not be a symlink")
    output.mkdir(parents=True, exist_ok=True)

    tokenizer_bytes = package.to_bytes()
    tokenizer_path = output / "tokenizer.vn97tk1"
    tokenizer_path.write_bytes(tokenizer_bytes)

    checkpoint_path = output / "model.vn97ck1"
    checkpoint_sha256 = save_deployment_checkpoint(model, checkpoint_path)

    report = {
        "checkpoint_sha256": checkpoint_sha256,
        "dataset_sha256": dataset_sha256,
        "examples": len(examples),
        "model": {
            "d_model": model.config.d_model,
            "d_state": model.config.d_state,
            "embedding_rank": model.config.embedding_rank,
            "n_layers": model.config.n_layers,
            "vocab_size": model.config.vocab_size,
        },
        "schema": "VN97TRAIN1",
        "tokenizer_sha256": hashlib.sha256(tokenizer_bytes).hexdigest(),
        "training": {
            "batch_size": training_config.batch_size,
            "epochs": training_config.epochs,
            "final_loss": result.final_loss,
            "learning_rate": training_config.learning_rate,
            "max_grad_norm": training_config.max_grad_norm,
            "mean_loss": result.mean_loss,
            "seed": training_config.seed,
            "sequence_length": training_config.sequence_length,
            "steps": result.steps,
            "stride": training_config.stride,
            "target_tokens": result.target_tokens,
            "weight_decay": training_config.weight_decay,
            "windows": len(windows),
        },
    }
    report_path = output / "training-report.json"
    report_path.write_bytes(_canonical_json(report))

    print(f"VN97TRAIN1 checkpoint={checkpoint_path}")
    print(f"checkpoint_sha256={checkpoint_sha256}")
    print(f"tokenizer={tokenizer_path}")
    print(f"final_loss={result.final_loss:.6f}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"vn97-train: {exc}", file=sys.stderr)
        raise
