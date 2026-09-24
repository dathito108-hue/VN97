from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import sys

import torch

from .campaign_cli import main as campaign_main
from .deployment_checkpoint import load_deployment_checkpoint_file
from .model_image import build_model_image
from .p3_language_campaign import (
    CORPUS_PROFILE_ID,
    MIN_RELEASE_RECORDS,
    MIN_TRAIN_RECORDS,
    MIN_VALIDATION_RECORDS,
    TILE_COLS,
    TILE_ROWS,
    VN97P3CampaignReceipt,
    campaign_argv,
    candidate_ids,
    candidate_manifest_bytes,
    profile_sha256,
)
from .tokenizer import VN97TokenizerPackage
from .training_cli import (
    _atomic_write,
    _read_bounded_regular_file,
)


_MAX_CORPUS_MANIFEST_BYTES = 4 * 1024 * 1024
_MAX_CAMPAIGN_REPORT_BYTES = 8 * 1024 * 1024
_MAX_TOKENIZER_BYTES = 128 * 1024 * 1024


class VN97P3LanguageCampaignError(RuntimeError):
    pass


def _strict_canonical_json(
    data: bytes,
    *,
    label: str,
) -> dict[str, object]:
    duplicates: list[str] = []

    def hook(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                duplicates.append(key)
            result[key] = value
        return result

    try:
        text = data.decode("utf-8", errors="strict")
        body = text[:-1] if text.endswith("\n") else text
        value = json.loads(
            body,
            object_pairs_hook=hook,
            parse_constant=lambda raw: (
                _ for _ in ()
            ).throw(ValueError(raw)),
        )
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        ValueError,
    ) as exc:
        raise VN97P3LanguageCampaignError(
            f"{label} must be strict UTF-8 JSON"
        ) from exc

    if duplicates or not isinstance(value, dict):
        raise VN97P3LanguageCampaignError(
            f"{label} must be one object without duplicate keys"
        )

    canonical = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    expected = canonical + ("\n" if text.endswith("\n") else "")
    if text != expected:
        raise VN97P3LanguageCampaignError(
            f"{label} must use canonical JSON"
        )
    return value


def _require_sha256(
    value: object,
    *,
    label: str,
) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(ch not in "0123456789abcdef" for ch in value)
    ):
        raise VN97P3LanguageCampaignError(
            f"{label} must be lowercase SHA-256"
        )
    return value


def _validate_split_file(
    corpus_dir: Path,
    *,
    split: str,
    spec: object,
    min_records: int,
) -> None:
    if (
        not isinstance(spec, dict)
        or set(spec) != {"bytes", "mode", "records", "sha256"}
        or spec.get("mode") != "chat"
        or type(spec.get("bytes")) is not int
        or type(spec.get("records")) is not int
        or spec["bytes"] <= 0
        or spec["records"] < min_records
    ):
        raise VN97P3LanguageCampaignError(
            f"{split} corpus split contract is invalid for P3"
        )

    path = corpus_dir / f"{split}.jsonl"
    data = _read_bounded_regular_file(
        path,
        max_bytes=max(int(spec["bytes"]), 1),
    )
    if len(data) != int(spec["bytes"]):
        raise VN97P3LanguageCampaignError(
            f"{split} split byte count changed after sealing"
        )
    expected_sha = _require_sha256(
        spec.get("sha256"),
        label=f"{split} split SHA-256",
    )
    if hashlib.sha256(data).hexdigest() != expected_sha:
        raise VN97P3LanguageCampaignError(
            f"{split} split SHA-256 changed after sealing"
        )
    records = sum(
        1
        for line in data.split(b"\n")
        if line.strip()
    )
    if records != int(spec["records"]):
        raise VN97P3LanguageCampaignError(
            f"{split} split record count changed after sealing"
        )


def _validate_corpus(
    corpus_dir: Path,
) -> tuple[str, str]:
    if corpus_dir.is_symlink():
        raise VN97P3LanguageCampaignError(
            "P3 corpus directory must not be a symlink"
        )
    root = corpus_dir.resolve(strict=True)
    if not root.is_dir():
        raise VN97P3LanguageCampaignError(
            "P3 corpus path must be a directory"
        )

    expected = {
        "corpus.vn97corpus1.json",
        "training.jsonl",
        "validation.jsonl",
        "release.jsonl",
    }
    if {item.name for item in root.iterdir()} != expected:
        raise VN97P3LanguageCampaignError(
            "P3 corpus directory must contain exactly VN97CORPUS1 outputs"
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
        raise VN97P3LanguageCampaignError(
            "VN97CORPUS1 manifest fields are invalid"
        )
    if manifest.get("schema") != "VN97CORPUS1":
        raise VN97P3LanguageCampaignError(
            "P3 requires VN97CORPUS1"
        )
    if manifest.get("profile_id") != CORPUS_PROFILE_ID:
        raise VN97P3LanguageCampaignError(
            "P3 corpus profile does not match production intelligence v1"
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
        raise VN97P3LanguageCampaignError(
            "VN97CORPUS1 manifest identity mismatch"
        )

    splits = manifest.get("splits")
    if (
        not isinstance(splits, dict)
        or set(splits) != {
            "training",
            "validation",
            "release",
        }
    ):
        raise VN97P3LanguageCampaignError(
            "VN97CORPUS1 split map is invalid"
        )

    _validate_split_file(
        root,
        split="training",
        spec=splits["training"],
        min_records=MIN_TRAIN_RECORDS,
    )
    _validate_split_file(
        root,
        split="validation",
        spec=splits["validation"],
        min_records=MIN_VALIDATION_RECORDS,
    )
    _validate_split_file(
        root,
        split="release",
        spec=splits["release"],
        min_records=MIN_RELEASE_RECORDS,
    )

    return manifest_id, hashlib.sha256(data).hexdigest()


def _load_report(
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
        raise VN97P3LanguageCampaignError(
            "P3 campaign did not produce VN97CAMP2"
        )
    selected = report.get("selected_candidate_id")
    if selected not in candidate_ids():
        raise VN97P3LanguageCampaignError(
            "P3 campaign selected a candidate outside the frozen set"
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


def _selected_validation(
    report: dict[str, object],
) -> tuple[float, float]:
    rows = report.get("candidates")
    selected = report.get("selected_candidate_id")
    if not isinstance(rows, list):
        raise VN97P3LanguageCampaignError(
            "VN97CAMP2 candidate rows are invalid"
        )
    matches = [
        row
        for row in rows
        if (
            isinstance(row, dict)
            and row.get("candidate_id") == selected
        )
    ]
    if len(matches) != 1:
        raise VN97P3LanguageCampaignError(
            "VN97CAMP2 selected candidate row is ambiguous"
        )
    row = matches[0]
    if row.get("status") != "ELIGIBLE":
        raise VN97P3LanguageCampaignError(
            "VN97CAMP2 selected candidate is not eligible"
        )
    evaluation = row.get("evaluation")
    if not isinstance(evaluation, dict):
        raise VN97P3LanguageCampaignError(
            "selected validation evaluation is invalid"
        )
    return (
        float(evaluation["mean_loss"]),
        float(evaluation["top1_accuracy"]),
    )


def _release_metrics(
    report: dict[str, object],
) -> tuple[float, float]:
    value = report.get("release_evaluation")
    if not isinstance(value, dict):
        raise VN97P3LanguageCampaignError(
            "VN97CAMP2 release evaluation is invalid"
        )
    return (
        float(value["mean_loss"]),
        float(value["top1_accuracy"]),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen VN97 P3 production-language campaign "
            "against a sealed expanded VN97CORPUS1."
        )
    )
    parser.add_argument("--corpus-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--device",
        default="cuda",
        help="Explicit CUDA device, for example cuda or cuda:0.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not args.device.startswith("cuda"):
        raise VN97P3LanguageCampaignError(
            "P3 production campaign requires CUDA"
        )
    if not torch.cuda.is_available():
        raise VN97P3LanguageCampaignError(
            "CUDA is not available on this host"
        )

    corpus_dir = Path(args.corpus_dir)
    (
        corpus_manifest_id,
        corpus_manifest_sha256,
    ) = _validate_corpus(corpus_dir)
    corpus_dir = corpus_dir.resolve(strict=True)

    output = Path(args.output_dir)
    if output.exists():
        if (
            output.is_symlink()
            or not output.is_dir()
            or any(output.iterdir())
        ):
            raise VN97P3LanguageCampaignError(
                "P3 output-dir must be new or empty"
            )
    else:
        output.mkdir(parents=True, exist_ok=False)
    output = output.resolve(strict=True)

    with tempfile.TemporaryDirectory(
        prefix="vn97-p3-campaign-"
    ) as temporary:
        campaign_path = (
            Path(temporary)
            / "p3-language.vn97campdef1.json"
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
            raise VN97P3LanguageCampaignError(
                f"vn97-campaign returned nonzero status {result}"
            )

    report, report_bytes = _load_report(
        output / "campaign-report.json"
    )
    selected_checkpoint_sha256 = str(
        report["selected_checkpoint_sha256"]
    )

    checkpoint = load_deployment_checkpoint_file(
        output / "model.vn97ck1"
    )
    if (
        checkpoint.checkpoint_sha256
        != selected_checkpoint_sha256
    ):
        raise VN97P3LanguageCampaignError(
            "P3 checkpoint does not match VN97CAMP2"
        )

    tokenizer_bytes = _read_bounded_regular_file(
        output / "tokenizer.vn97tk1",
        max_bytes=_MAX_TOKENIZER_BYTES,
    )
    tokenizer_sha256 = hashlib.sha256(
        tokenizer_bytes
    ).hexdigest()
    if tokenizer_sha256 != report["tokenizer_sha256"]:
        raise VN97P3LanguageCampaignError(
            "P3 tokenizer does not match VN97CAMP2"
        )

    tokenizer = VN97TokenizerPackage.from_bytes(
        tokenizer_bytes
    )
    if tokenizer.vocab_size != checkpoint.config.vocab_size:
        raise VN97P3LanguageCampaignError(
            "P3 tokenizer/checkpoint vocabulary mismatch"
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

    validation_loss, validation_accuracy = (
        _selected_validation(report)
    )
    release_loss, release_accuracy = (
        _release_metrics(report)
    )

    receipt = VN97P3CampaignReceipt(
        corpus_manifest_id=corpus_manifest_id,
        corpus_manifest_sha256=corpus_manifest_sha256,
        campaign_report_sha256=hashlib.sha256(
            report_bytes
        ).hexdigest(),
        selected_candidate_id=str(
            report["selected_candidate_id"]
        ),
        checkpoint_sha256=checkpoint.checkpoint_sha256,
        tokenizer_sha256=tokenizer_sha256,
        model_image_sha256=hashlib.sha256(
            image_bytes
        ).hexdigest(),
        model_image_bytes=len(image_bytes),
        validation_mean_loss=validation_loss,
        validation_top1_accuracy=validation_accuracy,
        release_mean_loss=release_loss,
        release_top1_accuracy=release_accuracy,
        device=args.device,
        profile_sha256=profile_sha256(),
    )
    _atomic_write(
        output / "p3-run.vn97p3run1.json",
        receipt.to_bytes(),
    )

    print(
        "VN97P3RUN1 "
        f"candidate={receipt.selected_candidate_id} "
        f"checkpoint_sha256={receipt.checkpoint_sha256} "
        f"model_image_sha256={receipt.model_image_sha256}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(
            f"vn97-p3-language-campaign: {exc}",
            file=sys.stderr,
        )
        raise
