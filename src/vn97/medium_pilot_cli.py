from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import sys

from .campaign_cli import main as campaign_main
from .deployment_checkpoint import load_deployment_checkpoint_file
from .medium_pilot import (
    CORPUS_PROFILE_ID,
    VN97PilotReceipt,
    candidate_id,
    candidate_manifest_bytes,
    campaign_argv,
    profile_sha256,
    TILE_COLS,
    TILE_ROWS,
)
from .model_image import build_model_image
from .tokenizer import VN97TokenizerPackage
from .training_cli import (
    _atomic_write,
    _read_bounded_regular_file,
)


_MAX_CORPUS_MANIFEST_BYTES = 4 * 1024 * 1024
_MAX_CAMPAIGN_REPORT_BYTES = 4 * 1024 * 1024
_MAX_TOKENIZER_BYTES = 64 * 1024 * 1024


class VN97MediumPilotError(RuntimeError):
    pass


def _strict_canonical_json(data: bytes, *, label: str) -> dict[str, object]:
    duplicates: list[str] = []

    def hook(pairs):
        out = {}
        for key, value in pairs:
            if key in out:
                duplicates.append(key)
            out[key] = value
        return out

    try:
        text = data.decode("utf-8", errors="strict")
        stripped = text[:-1] if text.endswith("\n") else text
        value = json.loads(
            stripped,
            object_pairs_hook=hook,
            parse_constant=lambda raw: (_ for _ in ()).throw(
                ValueError(raw)
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise VN97MediumPilotError(
            f"{label} must be strict UTF-8 JSON"
        ) from exc
    if duplicates or not isinstance(value, dict):
        raise VN97MediumPilotError(
            f"{label} must be one object without duplicate keys"
        )
    canonical = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    expected_text = canonical + ("\n" if text.endswith("\n") else "")
    if text != expected_text:
        raise VN97MediumPilotError(
            f"{label} must use canonical JSON"
        )
    return value


def _require_sha256(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(ch not in "0123456789abcdef" for ch in value)
    ):
        raise VN97MediumPilotError(
            f"{label} must be lowercase SHA-256"
        )
    return value


def _validate_split_file(
    corpus_dir: Path,
    *,
    split: str,
    spec: object,
) -> None:
    if (
        not isinstance(spec, dict)
        or set(spec) != {"bytes", "mode", "records", "sha256"}
        or spec["mode"] != "chat"
        or type(spec["bytes"]) is not int
        or type(spec["records"]) is not int
        or spec["bytes"] <= 0
        or spec["records"] <= 0
    ):
        raise VN97MediumPilotError(
            f"{split} corpus split contract is invalid for P2"
        )
    expected_sha = _require_sha256(
        spec["sha256"],
        label=f"{split} split SHA-256",
    )
    path = corpus_dir / f"{split}.jsonl"
    data = _read_bounded_regular_file(
        path,
        max_bytes=max(int(spec["bytes"]), 1),
    )
    if len(data) != int(spec["bytes"]):
        raise VN97MediumPilotError(
            f"{split} split byte count changed after sealing"
        )
    if hashlib.sha256(data).hexdigest() != expected_sha:
        raise VN97MediumPilotError(
            f"{split} split SHA-256 changed after sealing"
        )
    records = sum(
        1
        for line in data.splitlines()
        if line.strip()
    )
    if records != int(spec["records"]):
        raise VN97MediumPilotError(
            f"{split} split record count changed after sealing"
        )


def _validate_corpus(corpus_dir: Path) -> tuple[str, str]:
    if corpus_dir.is_symlink():
        raise VN97MediumPilotError(
            "P2 corpus directory must not be a symlink"
        )
    root = corpus_dir.resolve(strict=True)
    if not root.is_dir():
        raise VN97MediumPilotError(
            "P2 corpus path must be a directory"
        )

    expected = {
        "corpus.vn97corpus1.json",
        "training.jsonl",
        "validation.jsonl",
        "release.jsonl",
    }
    entries = {item.name for item in root.iterdir()}
    if entries != expected:
        raise VN97MediumPilotError(
            "P2 corpus directory must contain exactly the VN97CORPUS1 output set"
        )

    manifest_path = root / "corpus.vn97corpus1.json"
    data = _read_bounded_regular_file(
        manifest_path,
        max_bytes=_MAX_CORPUS_MANIFEST_BYTES,
    )
    manifest = _strict_canonical_json(
        data,
        label="VN97CORPUS1 manifest",
    )
    if set(manifest) != {
        "manifest_id",
        "profile_id",
        "schema",
        "sources",
        "splits",
    }:
        raise VN97MediumPilotError(
            "VN97CORPUS1 manifest fields are invalid"
        )
    if manifest.get("schema") != "VN97CORPUS1":
        raise VN97MediumPilotError(
            "P2 requires VN97CORPUS1"
        )
    if manifest.get("profile_id") != CORPUS_PROFILE_ID:
        raise VN97MediumPilotError(
            "P2 corpus profile does not match production intelligence v1"
        )
    manifest_id = _require_sha256(
        manifest.get("manifest_id"),
        label="corpus manifest ID",
    )
    identity = {
        key: value
        for key, value in manifest.items()
        if key != "manifest_id"
    }
    expected_manifest_id = hashlib.sha256(
        b"VN97CORPUS1\0"
        + json.dumps(
            identity,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    if manifest_id != expected_manifest_id:
        raise VN97MediumPilotError(
            "VN97CORPUS1 manifest identity mismatch"
        )
    splits = manifest.get("splits")
    if (
        not isinstance(splits, dict)
        or set(splits) != {"training", "validation", "release"}
    ):
        raise VN97MediumPilotError(
            "VN97CORPUS1 split map is invalid"
        )
    for split in ("training", "validation", "release"):
        _validate_split_file(
            root,
            split=split,
            spec=splits[split],
        )
    return manifest_id, hashlib.sha256(data).hexdigest()


def _load_campaign_report(
    path: Path,
) -> tuple[dict[str, object], bytes]:
    data = _read_bounded_regular_file(
        path,
        max_bytes=_MAX_CAMPAIGN_REPORT_BYTES,
    )
    report = _strict_canonical_json(
        data,
        label="VN97CAMP2 report",
    )
    if report.get("schema") != "VN97CAMP2":
        raise VN97MediumPilotError(
            "P2 campaign did not produce VN97CAMP2"
        )
    if report.get("selected_candidate_id") != candidate_id():
        raise VN97MediumPilotError(
            "P2 campaign selected an unexpected candidate"
        )
    _require_sha256(
        report.get("selected_checkpoint_sha256"),
        label="selected checkpoint SHA-256",
    )
    _require_sha256(
        report.get("tokenizer_sha256"),
        label="tokenizer SHA-256",
    )
    return report, data


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the fixed VN97 P2 medium CPU-first pilot against a sealed "
            "VN97CORPUS1 directory."
        )
    )
    parser.add_argument("--corpus-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--device",
        default="cpu",
        help="Explicit torch device; use cpu by default or cuda on rented GPU.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not args.device or args.device == "auto":
        raise VN97MediumPilotError(
            "P2 device must be explicit; auto is not accepted"
        )

    corpus_dir = Path(args.corpus_dir)
    corpus_manifest_id, corpus_manifest_sha256 = _validate_corpus(
        corpus_dir
    )
    corpus_dir = corpus_dir.resolve(strict=True)

    output = Path(args.output_dir)
    if output.exists():
        if output.is_symlink() or not output.is_dir():
            raise VN97MediumPilotError(
                "P2 output-dir must be a real directory"
            )
        if any(output.iterdir()):
            raise VN97MediumPilotError(
                "P2 output-dir must be new or empty"
            )
    else:
        output.mkdir(parents=True, exist_ok=False)
    output = output.resolve(strict=True)

    with tempfile.TemporaryDirectory(
        prefix="vn97-p2-campaign-"
    ) as temp:
        campaign_path = (
            Path(temp)
            / "p2-medium.vn97campdef1.json"
        )
        campaign_path.write_bytes(
            candidate_manifest_bytes()
        )
        result = campaign_main(
            campaign_argv(
                corpus_dir=corpus_dir,
                output_dir=output,
                campaign_path=campaign_path,
                device=args.device,
            )
        )
        if result != 0:
            raise VN97MediumPilotError(
                f"vn97-campaign returned nonzero status {result}"
            )

    report, report_bytes = _load_campaign_report(
        output / "campaign-report.json"
    )

    checkpoint = load_deployment_checkpoint_file(
        output / "model.vn97ck1"
    )
    expected_checkpoint = str(
        report["selected_checkpoint_sha256"]
    )
    if checkpoint.checkpoint_sha256 != expected_checkpoint:
        raise VN97MediumPilotError(
            "P2 checkpoint does not match VN97CAMP2"
        )

    tokenizer_bytes = _read_bounded_regular_file(
        output / "tokenizer.vn97tk1",
        max_bytes=_MAX_TOKENIZER_BYTES,
    )
    tokenizer_sha256 = hashlib.sha256(
        tokenizer_bytes
    ).hexdigest()
    if tokenizer_sha256 != report["tokenizer_sha256"]:
        raise VN97MediumPilotError(
            "P2 tokenizer does not match VN97CAMP2"
        )
    tokenizer = VN97TokenizerPackage.from_bytes(
        tokenizer_bytes
    )
    if tokenizer.vocab_size != checkpoint.config.vocab_size:
        raise VN97MediumPilotError(
            "P2 tokenizer/checkpoint vocabulary mismatch"
        )

    image = build_model_image(
        checkpoint.model,
        tokenizer=tokenizer,
        tile_rows=TILE_ROWS,
        tile_cols=TILE_COLS,
    )
    image_bytes = image.data
    _atomic_write(
        output / "model.vn97mi1",
        image_bytes,
    )

    receipt = VN97PilotReceipt(
        corpus_manifest_id=corpus_manifest_id,
        corpus_manifest_sha256=corpus_manifest_sha256,
        campaign_report_sha256=hashlib.sha256(
            report_bytes
        ).hexdigest(),
        candidate_id=candidate_id(),
        checkpoint_sha256=checkpoint.checkpoint_sha256,
        tokenizer_sha256=tokenizer_sha256,
        model_image_sha256=hashlib.sha256(
            image_bytes
        ).hexdigest(),
        model_image_bytes=len(image_bytes),
        device=args.device,
        profile_sha256=profile_sha256(),
    )
    _atomic_write(
        output / "pilot.vn97pilot1.json",
        receipt.to_bytes(),
    )

    print(
        "VN97PILOT1 "
        f"candidate={receipt.candidate_id} "
        f"checkpoint_sha256={receipt.checkpoint_sha256} "
        f"model_image_sha256={receipt.model_image_sha256}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"vn97-medium-pilot: {exc}", file=sys.stderr)
        raise
