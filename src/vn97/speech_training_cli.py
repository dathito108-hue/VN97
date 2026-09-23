from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import struct
import sys
import tempfile
import wave

import torch

from .deployment_checkpoint import (
    load_deployment_checkpoint_file,
    save_deployment_checkpoint,
)
from .modality import AudioAdapterConfig, AudioFrameAdapter
from .speech_training import (
    VN97SpeechExample,
    VN97SpeechTrainingConfig,
    train_vn97_speech_adapter,
)
from .tokenizer import VN97Tokenizer, VN97TokenizerPackage


_MAX_MANIFEST_BYTES = 64 * 1024 * 1024
_MAX_TOKENIZER_BYTES = 64 * 1024 * 1024
_MAX_WAV_BYTES = 64 * 1024 * 1024


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _atomic_write(path: Path, data: bytes) -> None:
    if path.is_symlink():
        raise ValueError(f"output target must not be a symlink: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.is_symlink():
        raise ValueError(f"output parent must not be a symlink: {path.parent}")
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
        if path.is_symlink():
            raise ValueError(f"output target became a symlink: {path}")
        os.replace(temp, path)
        dir_fd = os.open(
            path.parent.resolve(strict=True),
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    finally:
        temp.unlink(missing_ok=True)
    if path.read_bytes() != data:
        raise IOError(f"output post-write verification failed: {path}")


def _read_regular_file(path: Path, *, max_bytes: int, label: str) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise ValueError(f"{label} could not be opened safely: {path}") from exc
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise ValueError(f"{label} must be a regular file: {path}")
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
            raise ValueError(f"{label} changed while being read: {path}")
        return bytes(out)
    finally:
        os.close(fd)


def _load_pcm16_mono_16k(path: Path) -> torch.Tensor:
    data = _read_regular_file(
        path,
        max_bytes=_MAX_WAV_BYTES,
        label="speech WAV",
    )
    import io

    try:
        with wave.open(io.BytesIO(data), "rb") as wav:
            if wav.getnchannels() != 1:
                raise ValueError("speech WAV must be mono")
            if wav.getsampwidth() != 2:
                raise ValueError("speech WAV must be PCM16")
            if wav.getframerate() != 16_000:
                raise ValueError("speech WAV must use 16 kHz sample rate")
            if wav.getcomptype() != "NONE":
                raise ValueError("speech WAV must be uncompressed PCM")
            frames = wav.getnframes()
            if frames < 320 or frames > 16_000 * 30:
                raise ValueError(
                    "speech WAV duration must be between 20 ms and 30 seconds"
                )
            raw = wav.readframes(frames)
            if len(raw) != frames * 2:
                raise ValueError("speech WAV PCM payload is truncated")
    except wave.Error as exc:
        raise ValueError(f"speech WAV is invalid: {path}") from exc

    samples = torch.frombuffer(
        bytearray(raw),
        dtype=torch.int16,
    ).clone().to(dtype=torch.float32)
    return samples / 32768.0


def load_speech_manifest(
    path: Path,
    *,
    max_examples: int,
) -> tuple[list[VN97SpeechExample], str]:
    if max_examples <= 0:
        raise ValueError("max_examples must be positive")
    data = _read_regular_file(
        path,
        max_bytes=_MAX_MANIFEST_BYTES,
        label="speech manifest",
    )
    try:
        text = data.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ValueError("speech manifest must be strict UTF-8") from exc

    root = path.parent.resolve(strict=True)
    examples: list[VN97SpeechExample] = []
    digest = hashlib.sha256()
    digest.update(data)

    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        duplicates: list[str] = []

        def object_hook(
            pairs: list[tuple[str, object]],
        ) -> dict[str, object]:
            output: dict[str, object] = {}
            for key, item in pairs:
                if key in output:
                    duplicates.append(key)
                output[key] = item
            return output

        try:
            value = json.loads(
                line,
                object_pairs_hook=object_hook,
                parse_constant=lambda value: (_ for _ in ()).throw(
                    ValueError(value)
                ),
            )
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(
                f"invalid speech JSONL at line {line_number}"
            ) from exc
        if duplicates:
            raise ValueError(
                f"duplicate speech JSON key at line {line_number}"
            )
        if (
            not isinstance(value, dict)
            or set(value) != {"audio", "text"}
            or not isinstance(value["audio"], str)
            or not value["audio"]
            or not isinstance(value["text"], str)
            or not value["text"].strip()
        ):
            raise ValueError(
                "speech records must be exactly "
                '{"audio":"relative.wav","text":"transcript"}'
            )
        audio_rel = Path(value["audio"])
        if audio_rel.is_absolute() or ".." in audio_rel.parts:
            raise ValueError("speech audio path must stay inside manifest directory")
        audio_path = root.joinpath(audio_rel)
        if audio_path.is_symlink():
            raise ValueError("speech audio path must not be a symlink")
        resolved = audio_path.resolve(strict=True)
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise ValueError("speech audio path escapes manifest directory") from exc
        waveform = _load_pcm16_mono_16k(resolved)
        audio_digest = hashlib.sha256()
        audio_digest.update(b"VN97PCMF32LE1\0")
        for sample in waveform.tolist():
            audio_digest.update(struct.pack("<f", float(sample)))
        wav_hash = audio_digest.digest()
        digest.update(len(value["audio"].encode("utf-8")).to_bytes(8, "little"))
        digest.update(value["audio"].encode("utf-8"))
        digest.update(wav_hash)
        digest.update(value["text"].encode("utf-8"))

        examples.append(
            VN97SpeechExample(
                waveform=waveform,
                transcript=value["text"],
                audio_sha256=wav_hash.hex(),
            )
        )
        if len(examples) > max_examples:
            raise ValueError("speech example count exceeds max_examples")

    if not examples:
        raise ValueError("speech manifest contains no usable examples")
    return examples, digest.hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train the canonical VN97 AudioFrameAdapter against a frozen "
            "VN97LanguageCore and persist both in one VN97CK1 checkpoint."
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
    parser.add_argument("--max-frames", type=int, default=1500)
    parser.add_argument("--max-target-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=97)
    parser.add_argument("--device", default="auto")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    checkpoint = load_deployment_checkpoint_file(args.checkpoint)
    tokenizer_bytes = _read_regular_file(
        Path(args.tokenizer),
        max_bytes=_MAX_TOKENIZER_BYTES,
        label="VN97TK1 tokenizer",
    )
    tokenizer_package = VN97TokenizerPackage.from_bytes(tokenizer_bytes)
    if tokenizer_package.vocab_size != checkpoint.config.vocab_size:
        raise ValueError(
            "VN97TK1 tokenizer vocabulary does not match VN97CK1 checkpoint"
        )
    tokenizer = VN97Tokenizer(tokenizer_package)
    examples, dataset_sha256 = load_speech_manifest(
        Path(args.input),
        max_examples=args.max_examples,
    )

    adapter = checkpoint.audio_adapter
    if adapter is None:
        adapter = AudioFrameAdapter(
            checkpoint.config.d_model,
            ternary_threshold=checkpoint.config.ternary_threshold,
            config=AudioAdapterConfig(),
            rms_eps=checkpoint.config.rms_eps,
        )

    config = VN97SpeechTrainingConfig(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        seed=args.seed,
        max_frames=args.max_frames,
        max_target_tokens=args.max_target_tokens,
    )
    result = train_vn97_speech_adapter(
        checkpoint.model,
        adapter,
        tokenizer,
        examples,
        config,
        device=args.device,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_checkpoint = output_dir / "model.vn97ck1"
    checkpoint_sha256 = save_deployment_checkpoint(
        checkpoint.model,
        output_checkpoint,
        audio_adapter=adapter,
    )

    report = {
        "base_checkpoint_sha256": checkpoint.checkpoint_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "dataset_sha256": dataset_sha256,
        "examples": result.examples,
        "final_loss": result.final_loss,
        "mean_loss": result.mean_loss,
        "schema": "VN97SPEECHTRAIN1",
        "steps": result.steps,
        "target_tokens": result.target_tokens,
        "tokenizer_sha256": hashlib.sha256(tokenizer_bytes).hexdigest(),
        "training": {
            "epochs": config.epochs,
            "learning_rate": config.learning_rate,
            "max_frames": config.max_frames,
            "max_grad_norm": config.max_grad_norm,
            "max_target_tokens": config.max_target_tokens,
            "seed": config.seed,
            "weight_decay": config.weight_decay,
        },
    }
    _atomic_write(
        output_dir / "speech-training-report.json",
        _canonical_json(report),
    )
    print(_canonical_json(report).decode("utf-8"))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"vn97-speech-train: {exc}", file=sys.stderr)
        raise
