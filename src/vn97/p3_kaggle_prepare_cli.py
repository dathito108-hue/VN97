from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import torch

from .campaign_cli import _examples
from .p3_kaggle import VN97P3KaggleError
from .p3_language_campaign import (
    BATCH_SIZE,
    CORPUS_PROFILE_ID,
    EPOCHS,
    LEARNED_TOKENS,
    MAX_TRAIN_WINDOWS,
    MAX_VALIDATION_WINDOWS,
    MIN_PAIR_COUNT,
    SEQUENCE_LENGTH,
    TOKENIZER_TRAIN_RECORDS,
    profile_sha256,
)
from .p3_language_campaign_cli import (
    _strict_canonical_json,
    _validate_corpus,
)
from .p3_tensor_cache import (
    _canonical_json,
    tensor_bundle_bytes,
)
from .tokenizer import (
    VN97Tokenizer,
    learn_byte_bpe_fast,
    select_deterministic_tokenizer_corpus,
)
from .training import (
    VN97TrainingConfig,
    build_training_windows,
    render_chat_text,
)
from .training_cli import (
    _atomic_write,
    _load_records,
    _read_bounded_regular_file,
)


_MAX_MANIFEST_BYTES = 4 * 1024 * 1024


def _release_sha256(
    corpus_dir: Path,
) -> str:
    data = _read_bounded_regular_file(
        corpus_dir / "corpus.vn97corpus1.json",
        max_bytes=_MAX_MANIFEST_BYTES,
    )
    manifest = _strict_canonical_json(
        data,
        label="VN97CORPUS1 manifest",
    )
    if manifest.get("profile_id") != CORPUS_PROFILE_ID:
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


def _windows_to_tensors(
    windows,
) -> tuple[torch.Tensor, torch.Tensor]:
    inputs = torch.tensor(
        [window.input_ids for window in windows],
        dtype=torch.int32,
    )
    labels = torch.tensor(
        [window.labels for window in windows],
        dtype=torch.int32,
    )
    return inputs, labels


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare one shared P3 tokenizer + tensor-window cache so "
            "all four CUDA candidates avoid repeating CPU preprocessing."
        )
    )
    parser.add_argument(
        "--corpus-dir",
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        required=True,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    corpus_dir = Path(
        args.corpus_dir
    ).resolve(strict=True)
    (
        corpus_manifest_id,
        corpus_manifest_sha256,
    ) = _validate_corpus(corpus_dir)
    release_split_sha256 = _release_sha256(
        corpus_dir
    )

    output = Path(args.output_dir)
    if output.exists():
        if (
            output.is_symlink()
            or not output.is_dir()
            or any(output.iterdir())
        ):
            raise VN97P3KaggleError(
                "P3 cache output-dir must be new or empty"
            )
    else:
        output.mkdir(
            parents=True,
            exist_ok=False,
        )
    output = output.resolve(strict=True)

    started = time.time()
    print(
        "P3 CACHE stage=records "
        "loading training/validation JSONL",
        flush=True,
    )
    train_records, training_dataset_sha256 = _load_records(
        [corpus_dir / "training.jsonl"],
        mode="chat",
        max_input_bytes=64 * 1024 * 1024,
        max_examples=100_000,
    )
    validation_records, validation_dataset_sha256 = _load_records(
        [corpus_dir / "validation.jsonl"],
        mode="chat",
        max_input_bytes=64 * 1024 * 1024,
        max_examples=100_000,
    )

    rendered_training = [
        render_chat_text(value)
        for value in train_records
    ]
    tokenizer_corpus = (
        select_deterministic_tokenizer_corpus(
            rendered_training,
            max_samples=TOKENIZER_TRAIN_RECORDS,
        )
    )
    print(
        "P3 CACHE stage=tokenizer "
        f"records={len(tokenizer_corpus)}/{len(train_records)} "
        f"merges<={LEARNED_TOKENS}",
        flush=True,
    )
    tokenizer_package = learn_byte_bpe_fast(
        tokenizer_corpus,
        max_learned_tokens=LEARNED_TOKENS,
        min_pair_count=MIN_PAIR_COUNT,
    )
    tokenizer = VN97Tokenizer(
        tokenizer_package
    )
    tokenizer_bytes = tokenizer_package.to_bytes()
    tokenizer_sha256 = hashlib.sha256(
        tokenizer_bytes
    ).hexdigest()
    print(
        "P3 CACHE stage=tokenizer-done "
        f"vocab={tokenizer.vocab_size} "
        f"elapsed_s={time.time()-started:.1f}",
        flush=True,
    )

    print(
        "P3 CACHE stage=encode-windows",
        flush=True,
    )
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
    train_windows = build_training_windows(
        train_examples,
        VN97TrainingConfig(
            sequence_length=SEQUENCE_LENGTH,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            learning_rate=1e-4,
            seed=0,
            shuffle=False,
            max_windows=MAX_TRAIN_WINDOWS,
        ),
        pad_token_id=tokenizer.pad_id,
    )
    validation_windows = build_training_windows(
        validation_examples,
        VN97TrainingConfig(
            sequence_length=SEQUENCE_LENGTH,
            batch_size=BATCH_SIZE,
            epochs=1,
            seed=0,
            shuffle=False,
            max_windows=MAX_VALIDATION_WINDOWS,
        ),
        pad_token_id=tokenizer.pad_id,
    )

    train_inputs, train_labels = _windows_to_tensors(
        train_windows
    )
    validation_inputs, validation_labels = _windows_to_tensors(
        validation_windows
    )
    training_target_tokens = int(
        (train_labels != -100).sum().item()
    )
    validation_target_tokens = int(
        (validation_labels != -100).sum().item()
    )

    train_blob = tensor_bundle_bytes(
        train_inputs,
        train_labels,
    )
    validation_blob = tensor_bundle_bytes(
        validation_inputs,
        validation_labels,
    )
    train_cache_sha256 = hashlib.sha256(
        train_blob
    ).hexdigest()
    validation_cache_sha256 = hashlib.sha256(
        validation_blob
    ).hexdigest()

    _atomic_write(
        output / "tokenizer.vn97tk1",
        tokenizer_bytes,
    )
    _atomic_write(
        output / "training-windows.pt",
        train_blob,
    )
    _atomic_write(
        output / "validation-windows.pt",
        validation_blob,
    )

    body = {
        "corpus_manifest_id": corpus_manifest_id,
        "corpus_manifest_sha256": corpus_manifest_sha256,
        "profile_sha256": profile_sha256(),
        "release_split_sha256": release_split_sha256,
        "schema": "VN97P3CACHE1",
        "sequence_length": SEQUENCE_LENGTH,
        "tokenizer_sha256": tokenizer_sha256,
        "training": {
            "cache_sha256": train_cache_sha256,
            "dataset_sha256": training_dataset_sha256,
            "target_tokens": training_target_tokens,
            "windows": int(train_inputs.shape[0]),
        },
        "validation": {
            "cache_sha256": validation_cache_sha256,
            "dataset_sha256": validation_dataset_sha256,
            "target_tokens": validation_target_tokens,
            "windows": int(validation_inputs.shape[0]),
        },
    }
    manifest = dict(body)
    manifest["cache_id"] = hashlib.sha256(
        b"VN97P3CACHE1\0"
        + _canonical_json(body)
    ).hexdigest()
    _atomic_write(
        output / "cache.vn97p3cache1.json",
        _canonical_json(manifest) + b"\n",
    )

    print(
        "VN97P3CACHE1 "
        f"id={manifest['cache_id']} "
        f"vocab={tokenizer.vocab_size} "
        f"train_windows={train_inputs.shape[0]} "
        f"validation_windows={validation_inputs.shape[0]} "
        f"elapsed_s={time.time()-started:.1f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(
            f"vn97-p3-kaggle-prepare: {exc}",
            file=sys.stderr,
        )
        raise
