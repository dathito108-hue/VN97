from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import tempfile

import torch

from .deployment_checkpoint import (
    load_deployment_checkpoint_file,
    save_deployment_checkpoint,
)
from .modality import VisionAdapterConfig, VisionPatchAdapter
from .tokenizer import VN97Tokenizer, VN97TokenizerPackage
from .vision_training import (
    VN97VisionExample,
    VN97VisionTrainingConfig,
    train_vn97_vision_adapter,
)

_MAX_MANIFEST_BYTES = 64 * 1024 * 1024
_MAX_TOKENIZER_BYTES = 64 * 1024 * 1024
_MAX_RGB_BYTES = 224 * 224 * 3


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _read_regular_file(path: Path, *, max_bytes: int, label: str) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise ValueError(f"{label} could not be opened safely: {path}") from exc
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise ValueError(f"{label} must be a regular file")
        if not 0 < info.st_size <= max_bytes:
            raise ValueError(f"{label} size is outside bounds")
        out = bytearray()
        while len(out) < info.st_size:
            chunk = os.read(fd, min(1024 * 1024, info.st_size - len(out)))
            if not chunk:
                break
            out.extend(chunk)
        after = os.fstat(fd)
        if (
            len(out) != info.st_size
            or after.st_size != info.st_size
            or after.st_ino != info.st_ino
            or after.st_dev != info.st_dev
        ):
            raise ValueError(f"{label} changed while being read")
        return bytes(out)
    finally:
        os.close(fd)


def load_vision_manifest(
    path: Path,
    *,
    max_examples: int,
) -> tuple[list[VN97VisionExample], str]:
    data = _read_regular_file(
        path,
        max_bytes=_MAX_MANIFEST_BYTES,
        label="vision manifest",
    )
    text = data.decode("utf-8", errors="strict")
    root = path.parent.resolve(strict=True)
    examples: list[VN97VisionExample] = []
    digest = hashlib.sha256(data)

    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
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
                line,
                object_pairs_hook=hook,
                parse_constant=lambda raw: (_ for _ in ()).throw(ValueError(raw)),
            )
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(
                f"invalid vision JSONL at line {line_number}"
            ) from exc
        if duplicates or not isinstance(value, dict) or set(value) != {
            "height",
            "image",
            "text",
            "width",
        }:
            raise ValueError(
                "vision records must contain image,width,height,text exactly"
            )
        if (
            not isinstance(value["image"], str)
            or not value["image"]
            or type(value["width"]) is not int
            or type(value["height"]) is not int
            or not isinstance(value["text"], str)
            or not value["text"].strip()
        ):
            raise ValueError("vision record field types are invalid")
        width = value["width"]
        height = value["height"]
        if (
            width <= 0 or height <= 0 or
            width > 224 or height > 224 or
            width % 16 != 0 or height % 16 != 0
        ):
            raise ValueError(
                "vision dimensions must be positive <=224 and divisible by 16"
            )

        relative = Path(value["image"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("vision image path must stay inside manifest directory")
        requested = root / relative
        if requested.is_symlink():
            raise ValueError("vision image path must not be a symlink")
        resolved = requested.resolve(strict=True)
        resolved.relative_to(root)
        raw = _read_regular_file(
            resolved,
            max_bytes=_MAX_RGB_BYTES,
            label="vision RGB888",
        )
        expected = width * height * 3
        if len(raw) != expected:
            raise ValueError(
                "vision RGB888 byte count does not match width/height"
            )
        tensor = torch.tensor(
            list(raw),
            dtype=torch.float32,
        ).reshape(height, width, 3).permute(2, 0, 1) / 255.0
        examples.append(
            VN97VisionExample(
                image=tensor.contiguous(),
                description=value["text"],
            )
        )
        digest.update(hashlib.sha256(raw).digest())
        digest.update(value["text"].encode("utf-8"))
        if len(examples) > max_examples:
            raise ValueError("vision example count exceeds max_examples")
    if not examples:
        raise ValueError("vision manifest contains no examples")
    return examples, digest.hexdigest()


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temp = Path(temp_name)
    try:
        with os.fdopen(fd, "wb", closefd=True) as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)
    if path.read_bytes() != data:
        raise IOError("vision output post-write verification failed")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train canonical VN97 visual perception weights against a frozen "
            "VN97 language core using local RGB888 supervision."
        )
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-examples", type=int, default=100_000)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-patches", type=int, default=196)
    parser.add_argument("--max-target-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=97)
    parser.add_argument("--device", default="auto")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    loaded = load_deployment_checkpoint_file(args.checkpoint)
    tokenizer_bytes = _read_regular_file(
        Path(args.tokenizer),
        max_bytes=_MAX_TOKENIZER_BYTES,
        label="VN97TK1 tokenizer",
    )
    package = VN97TokenizerPackage.from_bytes(tokenizer_bytes)
    if package.vocab_size != loaded.config.vocab_size:
        raise ValueError("VN97TK1 vocabulary does not match checkpoint")
    tokenizer = VN97Tokenizer(package)
    examples, dataset_sha256 = load_vision_manifest(
        Path(args.input),
        max_examples=args.max_examples,
    )

    adapter = loaded.vision_adapter
    if adapter is None:
        adapter = VisionPatchAdapter(
            loaded.config.d_model,
            ternary_threshold=loaded.config.ternary_threshold,
            config=VisionAdapterConfig(
                channels=3,
                patch_size=16,
                eps=1e-5,
            ),
            rms_eps=loaded.config.rms_eps,
        )
    config = VN97VisionTrainingConfig(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        seed=args.seed,
        max_patches=args.max_patches,
        max_target_tokens=args.max_target_tokens,
    )
    result = train_vn97_vision_adapter(
        loaded.model,
        adapter,
        tokenizer,
        examples,
        config,
        device=args.device,
    )

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "model.vn97ck1"
    checkpoint_sha256 = save_deployment_checkpoint(
        loaded.model,
        checkpoint_path,
        audio_adapter=loaded.audio_adapter,
        vision_adapter=adapter,
    )
    report = {
        "base_checkpoint_sha256": loaded.checkpoint_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "dataset_sha256": dataset_sha256,
        "examples": result.examples,
        "final_loss": result.final_loss,
        "mean_loss": result.mean_loss,
        "schema": "VN97VISIONTRAIN1",
        "steps": result.steps,
        "target_tokens": result.target_tokens,
        "tokenizer_sha256": hashlib.sha256(tokenizer_bytes).hexdigest(),
        "training": {
            "epochs": config.epochs,
            "learning_rate": config.learning_rate,
            "max_grad_norm": config.max_grad_norm,
            "max_patches": config.max_patches,
            "max_target_tokens": config.max_target_tokens,
            "seed": config.seed,
            "weight_decay": config.weight_decay,
        },
    }
    _atomic_write(
        output / "vision-training-report.json",
        _canonical_json(report),
    )
    print(_canonical_json(report).decode("utf-8"))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"vn97-vision-train: {exc}", file=sys.stderr)
        raise
