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
    "vn97.medium_pilot",
    SRC / "medium_pilot.py",
)
if spec is None or spec.loader is None:
    raise RuntimeError("could not load medium_pilot.py")
module = importlib.util.module_from_spec(spec)
sys.modules["vn97.medium_pilot"] = module
spec.loader.exec_module(module)


def expect_failure(label: str, fn) -> None:
    try:
        fn()
    except Exception:
        return
    raise AssertionError(f"expected failure: {label}")


def main() -> None:
    assert module.PROFILE_ID == "vn97-p2-medium-cpu-first-v1"
    assert module.CORPUS_PROFILE_ID == "vn97-production-intelligence-v1"
    assert module.CANDIDATE == {
        "d_model": 192,
        "d_state": 16,
        "embedding_rank": 96,
        "learning_rate": 0.0003,
        "n_layers": 6,
        "seed": 97,
    }
    assert module.SEQUENCE_LENGTH == 256
    assert module.BATCH_SIZE == 2
    assert module.EPOCHS == 2
    assert module.MAX_TRAIN_WINDOWS == 2048
    assert module.MAX_VALIDATION_WINDOWS == 256
    assert module.MAX_RELEASE_WINDOWS == 256
    assert module.LEARNED_TOKENS == 4096

    candidate_bytes = module.candidate_manifest_bytes()
    manifest = json.loads(candidate_bytes.decode("utf-8"))
    assert manifest == {
        "candidates": [module.CANDIDATE],
        "schema": "VN97CAMPDEF1",
    }
    assert candidate_bytes.endswith(b"\n")
    expected_candidate_id = hashlib.sha256(
        json.dumps(
            module.CANDIDATE,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()[:16]
    assert module.candidate_id() == expected_candidate_id

    argv = module.campaign_argv(
        corpus_dir=Path("/corpus"),
        output_dir=Path("/output"),
        campaign_path=Path("/tmp/campaign.json"),
        device="cpu",
    )
    joined = " ".join(argv)
    for expected in (
        "--input /corpus/training.jsonl",
        "--validation-input /corpus/validation.jsonl",
        "--release-input /corpus/release.jsonl",
        "--sequence-length 256",
        "--batch-size 2",
        "--epochs 2",
        "--max-windows 2048",
        "--validation-max-windows 256",
        "--release-max-windows 256",
        "--learned-tokens 4096",
        "--max-parameters 5000000",
        "--max-model-image-bytes 67108864",
        "--max-recurrent-state-bytes 8388608",
        "--device cpu",
    ):
        assert expected in joined, expected

    sha = "0" * 64
    receipt = module.VN97PilotReceipt(
        corpus_manifest_id=sha,
        corpus_manifest_sha256="1" * 64,
        campaign_report_sha256="2" * 64,
        candidate_id=module.candidate_id(),
        checkpoint_sha256="3" * 64,
        tokenizer_sha256="4" * 64,
        model_image_sha256="5" * 64,
        model_image_bytes=123456,
        device="cpu",
        profile_sha256=module.profile_sha256(),
    )
    payload = json.loads(receipt.to_bytes().decode("utf-8"))
    assert payload["schema"] == "VN97PILOT1"
    assert payload["candidate_id"] == module.candidate_id()
    assert payload["profile_sha256"] == module.profile_sha256()

    expect_failure(
        "wrong candidate",
        lambda: module.VN97PilotReceipt(
            corpus_manifest_id=sha,
            corpus_manifest_sha256="1" * 64,
            campaign_report_sha256="2" * 64,
            candidate_id="deadbeefdeadbeef",
            checkpoint_sha256="3" * 64,
            tokenizer_sha256="4" * 64,
            model_image_sha256="5" * 64,
            model_image_bytes=123456,
            device="cpu",
            profile_sha256=module.profile_sha256(),
        ),
    )
    expect_failure(
        "wrong profile",
        lambda: module.VN97PilotReceipt(
            corpus_manifest_id=sha,
            corpus_manifest_sha256="1" * 64,
            campaign_report_sha256="2" * 64,
            candidate_id=module.candidate_id(),
            checkpoint_sha256="3" * 64,
            tokenizer_sha256="4" * 64,
            model_image_sha256="5" * 64,
            model_image_bytes=123456,
            device="cpu",
            profile_sha256="f" * 64,
        ),
    )
    expect_failure(
        "oversize image",
        lambda: module.VN97PilotReceipt(
            corpus_manifest_id=sha,
            corpus_manifest_sha256="1" * 64,
            campaign_report_sha256="2" * 64,
            candidate_id=module.candidate_id(),
            checkpoint_sha256="3" * 64,
            tokenizer_sha256="4" * 64,
            model_image_sha256="5" * 64,
            model_image_bytes=module.MAX_MODEL_IMAGE_BYTES + 1,
            device="cpu",
            profile_sha256=module.profile_sha256(),
        ),
    )

    print(
        "VN97 P2 medium pilot host contract PASS "
        f"candidate={module.candidate_id()} "
        f"profile={module.profile_sha256()}"
    )


if __name__ == "__main__":
    main()
