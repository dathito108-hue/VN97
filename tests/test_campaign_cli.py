import json
import os

import pytest

from vn97 import (
    VN97CampaignCandidate,
    VN97Config,
    VN97LanguageCore,
)
from vn97.campaign_cli import (
    _estimate_parameter_count,
    _file_identities,
    _load_candidates,
)


def test_campaign_manifest_is_strict_and_candidate_ids_stable(tmp_path):
    manifest = tmp_path / "campaign.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "VN97CAMPDEF1",
                "candidates": [
                    {
                        "d_model": 16,
                        "n_layers": 2,
                        "d_state": 4,
                        "embedding_rank": None,
                        "seed": 97,
                        "learning_rate": 0.001,
                    },
                    {
                        "d_model": 24,
                        "n_layers": 3,
                        "d_state": 4,
                        "embedding_rank": 8,
                        "seed": 98,
                        "learning_rate": 0.0005,
                    },
                ],
            },
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    candidates = _load_candidates(manifest)
    assert len(candidates) == 2
    assert candidates[0].candidate_id != candidates[1].candidate_id
    assert candidates == _load_candidates(manifest)


def test_campaign_manifest_rejects_duplicate_candidate(tmp_path):
    candidate = {
        "d_model": 16,
        "n_layers": 2,
        "d_state": 4,
        "embedding_rank": None,
        "seed": 97,
        "learning_rate": 0.001,
    }
    manifest = tmp_path / "campaign.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "VN97CAMPDEF1",
                "candidates": [candidate, candidate],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate candidate"):
        _load_candidates(manifest)


def test_campaign_meta_parameter_probe_matches_materialized_model():
    candidate = VN97CampaignCandidate(
        d_model=16,
        n_layers=2,
        d_state=4,
        embedding_rank=8,
        seed=97,
        learning_rate=0.001,
    )
    estimated = _estimate_parameter_count(
        vocab_size=300,
        candidate=candidate,
    )
    model = VN97LanguageCore(
        VN97Config(
            vocab_size=300,
            d_model=16,
            n_layers=2,
            d_state=4,
            embedding_rank=8,
        )
    )
    actual = sum(
        int(parameter.numel())
        for parameter in model.parameters()
    )
    assert estimated == actual


def test_campaign_file_identity_detects_hardlink_alias(tmp_path):
    training = tmp_path / "train.jsonl"
    training.write_text('{"text":"train"}\n', encoding="utf-8")
    alias = tmp_path / "validation-alias.jsonl"
    os.link(training, alias)

    train_ids = _file_identities(
        [training],
        label="training",
    )
    validation_ids = _file_identities(
        [alias],
        label="validation",
    )
    assert train_ids.intersection(validation_ids)
