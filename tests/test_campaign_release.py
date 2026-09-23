import json

import pytest

from vn97 import VN97EvaluationResult, VN97TrainingResult
from vn97 import campaign_cli
from vn97.campaign import VN97CampaignError


def _write_campaign_inputs(tmp_path):
    train = tmp_path / "train.jsonl"
    validation = tmp_path / "validation.jsonl"
    release = tmp_path / "release.jsonl"
    manifest = tmp_path / "campaign.json"

    train.write_text(
        json.dumps({"text": "training alpha beta gamma delta"}) + "\n",
        encoding="utf-8",
    )
    validation.write_text(
        json.dumps({"text": "validation epsilon zeta eta theta"}) + "\n",
        encoding="utf-8",
    )
    release.write_text(
        json.dumps({"text": "release iota kappa lambda mu"}) + "\n",
        encoding="utf-8",
    )
    manifest.write_text(
        json.dumps(
            {
                "schema": "VN97CAMPDEF1",
                "candidates": [
                    {
                        "d_model": 8,
                        "n_layers": 1,
                        "d_state": 2,
                        "embedding_rank": None,
                        "seed": 97,
                        "learning_rate": 0.001,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return train, validation, release, manifest


def _argv(train, validation, release, manifest, output):
    return [
        "--input",
        str(train),
        "--validation-input",
        str(validation),
        "--release-input",
        str(release),
        "--format",
        "text",
        "--validation-format",
        "text",
        "--release-format",
        "text",
        "--campaign",
        str(manifest),
        "--output-dir",
        str(output),
        "--sequence-length",
        "16",
        "--batch-size",
        "1",
        "--validation-batch-size",
        "1",
        "--release-batch-size",
        "1",
        "--epochs",
        "1",
        "--max-parameters",
        "1000000",
        "--max-validation-loss",
        "2.0",
        "--min-validation-target-tokens",
        "1",
        "--release-max-validation-loss",
        "1.5",
        "--release-min-validation-target-tokens",
        "1",
    ]


def _fake_training(*args, **kwargs):
    return VN97TrainingResult(
        steps=1,
        target_tokens=10,
        mean_loss=1.0,
        final_loss=1.0,
    )


def test_campaign_evaluates_sealed_release_only_after_selection(
    tmp_path,
    monkeypatch,
):
    train, validation, release, manifest = _write_campaign_inputs(tmp_path)
    output = tmp_path / "out"
    calls = []

    def fake_evaluation(model, windows, *, batch_size, device):
        calls.append(len(windows))
        if len(calls) == 1:
            return VN97EvaluationResult(
                windows=len(windows),
                target_tokens=100,
                mean_loss=1.0,
                top1_accuracy=0.25,
            )
        return VN97EvaluationResult(
            windows=len(windows),
            target_tokens=100,
            mean_loss=0.9,
            top1_accuracy=0.30,
        )

    monkeypatch.setattr(
        campaign_cli,
        "train_vn97_language",
        _fake_training,
    )
    monkeypatch.setattr(
        campaign_cli,
        "evaluate_vn97_language",
        fake_evaluation,
    )

    assert campaign_cli.main(
        _argv(train, validation, release, manifest, output)
    ) == 0
    assert len(calls) == 2

    report = json.loads(
        (output / "campaign-report.json").read_text(encoding="utf-8")
    )
    assert report["schema"] == "VN97CAMP2"
    assert report["release_evaluation"]["mean_loss"] == 0.9
    assert report["release_dataset_sha256"]
    assert (output / "model.vn97ck1").is_file()
    assert (output / "tokenizer.vn97tk1").is_file()


def test_campaign_release_failure_writes_nothing_and_does_not_fallback(
    tmp_path,
    monkeypatch,
):
    train, validation, release, manifest = _write_campaign_inputs(tmp_path)
    output = tmp_path / "out"
    calls = []

    def fake_evaluation(model, windows, *, batch_size, device):
        calls.append(len(windows))
        if len(calls) == 1:
            return VN97EvaluationResult(
                windows=len(windows),
                target_tokens=100,
                mean_loss=1.0,
                top1_accuracy=0.25,
            )
        return VN97EvaluationResult(
            windows=len(windows),
            target_tokens=100,
            mean_loss=5.0,
            top1_accuracy=0.0,
        )

    monkeypatch.setattr(
        campaign_cli,
        "train_vn97_language",
        _fake_training,
    )
    monkeypatch.setattr(
        campaign_cli,
        "evaluate_vn97_language",
        fake_evaluation,
    )

    with pytest.raises(
        VN97CampaignError,
        match="failed sealed release gate",
    ):
        campaign_cli.main(
            _argv(train, validation, release, manifest, output)
        )

    assert len(calls) == 2
    assert not output.exists()
