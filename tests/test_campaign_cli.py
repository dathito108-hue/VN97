import json

import pytest

from vn97.campaign_cli import _load_candidates


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
