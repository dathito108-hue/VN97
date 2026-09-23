from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import sys

from .bootstrap_bundle import (
    Ed25519PrivateKeySigner,
    build_bootstrap_bundle,
    write_bootstrap_assets,
)
from .capability_package import CapabilitySource
from .deployment_checkpoint import load_deployment_checkpoint_file
from .tokenizer import VN97TokenizerPackage


_MAX_TOKENIZER_BYTES = 64 * 1024 * 1024
_MAX_PRIVATE_KEY_FILE_BYTES = 256


def _read_regular_file(
    path: Path,
    *,
    max_bytes: int,
    label: str,
) -> bytes:
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
            raise ValueError(f"{label} changed while being read")
        return bytes(out)
    finally:
        os.close(fd)


def _parse_private_key(data: bytes) -> bytes:
    if len(data) == 32:
        return bytes(data)
    try:
        text = data.decode("ascii", errors="strict")
    except UnicodeDecodeError as exc:
        raise ValueError(
            "Ed25519 private key must be 32 raw bytes or 64 lowercase hex characters"
        ) from exc
    if (
        len(text) == 64
        and all(ch in "0123456789abcdef" for ch in text)
    ):
        return bytes.fromhex(text)
    raise ValueError(
        "Ed25519 private key must be 32 raw bytes or 64 lowercase hex characters"
    )


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create the exact signed VN97 bootstrap assets consumed by the M10J Android app."
        )
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--private-key", required=True)
    parser.add_argument("--key-id", required=True)
    parser.add_argument("--capability-version", type=int, required=True)
    parser.add_argument("--source-origin", required=True)
    parser.add_argument("--source-license", required=True)
    parser.add_argument("--assets-dir", required=True)
    parser.add_argument("--tile-rows", type=int, default=16)
    parser.add_argument("--tile-cols", type=int, default=16)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)

    checkpoint_path = Path(args.checkpoint)
    tokenizer_path = Path(args.tokenizer)
    private_key_path = Path(args.private_key)
    assets_dir = Path(args.assets_dir)

    loaded = load_deployment_checkpoint_file(checkpoint_path)

    tokenizer_bytes = _read_regular_file(
        tokenizer_path,
        max_bytes=_MAX_TOKENIZER_BYTES,
        label="VN97TK1 tokenizer",
    )
    try:
        tokenizer = VN97TokenizerPackage.from_bytes(tokenizer_bytes)
    except ValueError as exc:
        raise ValueError("VN97TK1 tokenizer is invalid") from exc
    if tokenizer.vocab_size != loaded.config.vocab_size:
        raise ValueError(
            "VN97TK1 tokenizer vocabulary does not match VN97CK1 checkpoint"
        )

    private_key = _parse_private_key(
        _read_regular_file(
            private_key_path,
            max_bytes=_MAX_PRIVATE_KEY_FILE_BYTES,
            label="Ed25519 private key",
        )
    )
    signer = Ed25519PrivateKeySigner(
        args.key_id,
        private_key,
    )
    source = CapabilitySource(
        args.source_origin,
        loaded.checkpoint_sha256,
        args.source_license,
    )

    bundle = build_bootstrap_bundle(
        loaded.model,
        tokenizer=tokenizer,
        source=source,
        capability_version=args.capability_version,
        signer=signer,
        tile_rows=args.tile_rows,
        tile_cols=args.tile_cols,
    )
    output = write_bootstrap_assets(bundle, assets_dir)

    report = {
        "assets_dir": str(output),
        "capability_version": bundle.capability_version,
        "checkpoint_sha256": loaded.checkpoint_sha256,
        "model_image_sha256": bundle.model_image_sha256,
        "package_sha256": bundle.package_sha256,
        "publisher_key_id": bundle.publisher_key_id,
        "publisher_public_key_sha256": hashlib.sha256(
            bundle.publisher_public_key
        ).hexdigest(),
        "schema": "VN97BOOTREL1",
        "tokenizer_sha256": hashlib.sha256(tokenizer_bytes).hexdigest(),
    }
    print(_canonical_json(report))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"vn97-bootstrap-release: {exc}", file=sys.stderr)
        raise
