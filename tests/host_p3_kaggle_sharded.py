from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import types


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "vn97"

pkg = types.ModuleType("vn97")
pkg.__path__ = [str(SRC)]
sys.modules["vn97"] = pkg

campaign = types.ModuleType("vn97.campaign")


@dataclass(frozen=True)
class Candidate:
    d_model: int
    n_layers: int
    d_state: int
    embedding_rank: int | None
    seed: int
    learning_rate: float

    def canonical_object(self):
        return {
            "d_model": self.d_model,
            "d_state": self.d_state,
            "embedding_rank": self.embedding_rank,
            "learning_rate": self.learning_rate,
            "n_layers": self.n_layers,
            "seed": self.seed,
        }

    @property
    def candidate_id(self):
        blob = json.dumps(
            self.canonical_object(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:16]


campaign.VN97CampaignCandidate = Candidate
sys.modules["vn97.campaign"] = campaign

evaluation = types.ModuleType("vn97.evaluation")


@dataclass(frozen=True)
class Eval:
    windows: int
    target_tokens: int
    mean_loss: float
    top1_accuracy: float

    def __post_init__(self):
        assert self.windows > 0
        assert self.target_tokens > 0
        assert math.isfinite(self.mean_loss)
        assert 0 <= self.top1_accuracy <= 1


evaluation.VN97EvaluationResult = Eval
sys.modules["vn97.evaluation"] = evaluation

spec = importlib.util.spec_from_file_location(
    "vn97.p3_kaggle",
    SRC / "p3_kaggle.py",
)
if spec is None or spec.loader is None:
    raise RuntimeError("could not load p3_kaggle.py")
module = importlib.util.module_from_spec(spec)
sys.modules["vn97.p3_kaggle"] = module
spec.loader.exec_module(module)


def expect_failure(label: str, fn) -> None:
    try:
        fn()
    except Exception:
        return
    raise AssertionError(f"expected failure: {label}")


def result(status: str, checkpoint: str | None):
    return module.VN97P3CandidateResult(
        candidate_index=1,
        candidate=Candidate(
            d_model=320,
            n_layers=10,
            d_state=24,
            embedding_rank=160,
            seed=98,
            learning_rate=0.00025,
        ),
        corpus_manifest_id="1" * 64,
        corpus_manifest_sha256="2" * 64,
        training_dataset_sha256="3" * 64,
        validation_dataset_sha256="4" * 64,
        release_split_sha256="5" * 64,
        tokenizer_sha256="6" * 64,
        profile_sha256="7" * 64,
        parameter_count=12_345_678,
        mobile_footprint={
            "model_image_bytes": 10_000_000,
            "recurrent_state_bytes": 1_000_000,
        },
        status=status,
        training_steps=100,
        training_target_tokens=50_000,
        training_mean_loss=6.2,
        training_final_loss=6.0,
        evaluation=Eval(
            windows=50,
            target_tokens=20_000,
            mean_loss=6.1,
            top1_accuracy=0.08,
        ),
        checkpoint_sha256=checkpoint,
        device="cuda:0",
    )


def main() -> None:
    eligible = result(
        "ELIGIBLE",
        "8" * 64,
    )
    data = eligible.to_bytes()
    parsed = module.parse_candidate_result(data)
    assert parsed.candidate_id == eligible.candidate_id
    assert parsed.candidate_index == 1
    assert parsed.status == "ELIGIBLE"
    assert parsed.checkpoint_sha256 == "8" * 64
    assert data.endswith(b"\n")

    rejected = result(
        "REJECTED_QUALITY",
        None,
    )
    parsed_rejected = module.parse_candidate_result(
        rejected.to_bytes()
    )
    assert parsed_rejected.status == "REJECTED_QUALITY"
    assert parsed_rejected.checkpoint_sha256 is None

    expect_failure(
        "rejected result with checkpoint",
        lambda: result(
            "REJECTED_QUALITY",
            "9" * 64,
        ),
    )

    tampered = json.loads(data.decode("utf-8"))
    tampered["evaluation"]["mean_loss"] = 1.0
    expect_failure(
        "tampered result identity",
        lambda: module.parse_candidate_result(
            (
                json.dumps(
                    tampered,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            ).encode("utf-8")
        ),
    )

    candidate_cli = (
        SRC / "p3_kaggle_candidate_cli.py"
    ).read_text(encoding="utf-8")
    finalizer_cli = (
        SRC / "p3_kaggle_finalize_cli.py"
    ).read_text(encoding="utf-8")
    launcher = (
        ROOT / "tools" / "kaggle_p3.sh"
    ).read_text(encoding="utf-8")

    # Candidate shards may bind the release hash from the manifest, but must
    # never open/encode the release JSONL itself.
    assert '"release.jsonl"' not in candidate_cli
    assert '"release.jsonl"' in finalizer_cli
    assert "select_best_campaign_candidate" in finalizer_cli
    assert "require_release_quality" in finalizer_cli
    assert "VN97P3CAND1" in candidate_cli
    assert "VN97P3RUN1" in finalizer_cli

    for value in (
        "candidate <0|1|2|3>",
        "vn97-p3-kaggle-candidate",
        "vn97-p3-kaggle-finalize",
        "VN97-P3-candidate-$INDEX.zip",
        "VN97-P3-final.zip",
    ):
        assert value in launcher, value

    print(
        "VN97 P3 Kaggle sharded campaign host contract PASS "
        f"candidate={eligible.candidate_id}"
    )


if __name__ == "__main__":
    main()
