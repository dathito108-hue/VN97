from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "vn97"

pkg = types.ModuleType("vn97")
pkg.__path__ = [str(SRC)]
sys.modules["vn97"] = pkg

spec = importlib.util.spec_from_file_location(
    "vn97.p3_language_campaign",
    SRC / "p3_language_campaign.py",
)
if spec is None or spec.loader is None:
    raise RuntimeError("could not load p3_language_campaign.py")
module = importlib.util.module_from_spec(spec)
sys.modules["vn97.p3_language_campaign"] = module
spec.loader.exec_module(module)


def expect_failure(label: str, fn) -> None:
    try:
        fn()
    except Exception:
        return
    raise AssertionError(f"expected failure: {label}")


def canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def main() -> None:
    config = json.loads(
        (
            ROOT
            / "configs"
            / "p3-language-production.vn97campdef1.json"
        ).read_text(encoding="utf-8")
    )
    generated = json.loads(
        module.candidate_manifest_bytes().decode("utf-8")
    )
    assert generated == config
    assert len(module.CANDIDATES) == 4
    assert len(set(module.candidate_ids())) == 4
    assert all(len(value) == 16 for value in module.candidate_ids())

    expected_ids = tuple(
        hashlib.sha256(
            canonical(candidate)
        ).hexdigest()[:16]
        for candidate in module.CANDIDATES
    )
    assert module.candidate_ids() == expected_ids
    assert module.LEARNED_TOKENS == 4096
    assert module.TOKENIZER_TRAIN_RECORDS == 2048
    assert module.profile_object()["tokenizer_sampling"] == "sha256-vn97toksample1"

    argv = module.campaign_argv(
        corpus_dir=Path("/corpus"),
        output_dir=Path("/output"),
        campaign_path=Path("/tmp/p3.json"),
        device="cuda:0",
    )
    joined = " ".join(argv)
    for expected in (
        "--input /corpus/training.jsonl",
        "--validation-input /corpus/validation.jsonl",
        "--release-input /corpus/release.jsonl",
        "--learned-tokens 4096",
        "--tokenizer-max-records 2048",
        "--sequence-length 512",
        "--batch-size 8",
        "--epochs 2",
        "--max-windows 100000",
        "--max-parameters 50000000",
        "--max-model-image-bytes 134217728",
        "--max-recurrent-state-bytes 8388608",
        "--max-validation-loss 6.76",
        "--release-max-validation-loss 6.79",
        "--device cuda:0",
    ):
        assert expected in joined, expected

    expect_failure(
        "CPU campaign",
        lambda: module.campaign_argv(
            corpus_dir=Path("/corpus"),
            output_dir=Path("/output"),
            campaign_path=Path("/tmp/p3.json"),
            device="cpu",
        ),
    )

    receipt = module.VN97P3CampaignReceipt(
        corpus_manifest_id="1" * 64,
        corpus_manifest_sha256="2" * 64,
        campaign_report_sha256="3" * 64,
        selected_candidate_id=module.candidate_ids()[1],
        checkpoint_sha256="4" * 64,
        tokenizer_sha256="5" * 64,
        model_image_sha256="6" * 64,
        model_image_bytes=10_000_000,
        validation_mean_loss=6.1,
        validation_top1_accuracy=0.1,
        release_mean_loss=6.2,
        release_top1_accuracy=0.09,
        device="cuda:0",
        profile_sha256=module.profile_sha256(),
    )
    payload = json.loads(receipt.to_bytes().decode("utf-8"))
    assert payload["schema"] == "VN97P3RUN1"
    assert payload["selected_candidate_id"] == module.candidate_ids()[1]
    assert payload["p2_baseline_run_identity"] == (
        "e0509769f3f819516c7e41c287690536"
        "73f634e41051d8e469b95bd8421db5a2"
    )

    expect_failure(
        "unknown selected candidate",
        lambda: module.VN97P3CampaignReceipt(
            corpus_manifest_id="1" * 64,
            corpus_manifest_sha256="2" * 64,
            campaign_report_sha256="3" * 64,
            selected_candidate_id="deadbeefdeadbeef",
            checkpoint_sha256="4" * 64,
            tokenizer_sha256="5" * 64,
            model_image_sha256="6" * 64,
            model_image_bytes=10_000_000,
            validation_mean_loss=6.1,
            validation_top1_accuracy=0.1,
            release_mean_loss=6.2,
            release_top1_accuracy=0.09,
            device="cuda:0",
            profile_sha256=module.profile_sha256(),
        ),
    )

    expect_failure(
        "non-CUDA receipt",
        lambda: module.VN97P3CampaignReceipt(
            corpus_manifest_id="1" * 64,
            corpus_manifest_sha256="2" * 64,
            campaign_report_sha256="3" * 64,
            selected_candidate_id=module.candidate_ids()[0],
            checkpoint_sha256="4" * 64,
            tokenizer_sha256="5" * 64,
            model_image_sha256="6" * 64,
            model_image_bytes=10_000_000,
            validation_mean_loss=6.1,
            validation_top1_accuracy=0.1,
            release_mean_loss=6.2,
            release_top1_accuracy=0.09,
            device="cpu",
            profile_sha256=module.profile_sha256(),
        ),
    )

    print(
        "VN97 P3 language campaign host contract PASS "
        f"profile={module.profile_sha256()}"
    )


if __name__ == "__main__":
    main()
