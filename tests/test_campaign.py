import pytest

from vn97 import (
    VN97CampaignCandidate,
    VN97CampaignError,
    VN97CampaignObservation,
    VN97EvaluationResult,
    select_best_campaign_candidate,
)


def _obs(loss, acc, params, seed, digest):
    return VN97CampaignObservation(
        candidate=VN97CampaignCandidate(
            d_model=16,
            n_layers=2,
            d_state=4,
            embedding_rank=None,
            seed=seed,
            learning_rate=1e-3,
        ),
        parameter_count=params,
        evaluation=VN97EvaluationResult(
            windows=2,
            target_tokens=100,
            mean_loss=loss,
            top1_accuracy=acc,
        ),
        checkpoint_sha256=digest * 64,
    )


def test_campaign_ranking_is_deterministic():
    lower_loss = _obs(1.0, 0.10, 1000, 1, "a")
    higher_loss = _obs(1.1, 0.90, 100, 2, "b")
    assert select_best_campaign_candidate(
        [higher_loss, lower_loss]
    ) is lower_loss

    higher_acc = _obs(1.0, 0.20, 2000, 3, "c")
    assert select_best_campaign_candidate(
        [lower_loss, higher_acc]
    ) is higher_acc

    smaller = _obs(1.0, 0.20, 1500, 4, "d")
    assert select_best_campaign_candidate(
        [higher_acc, smaller]
    ) is smaller


def test_campaign_rejects_empty_and_duplicate_candidates():
    with pytest.raises(VN97CampaignError, match="no campaign"):
        select_best_campaign_candidate([])

    candidate = VN97CampaignCandidate(
        d_model=8,
        n_layers=1,
        d_state=2,
        embedding_rank=None,
        seed=97,
        learning_rate=1e-3,
    )
    first = VN97CampaignObservation(
        candidate,
        10,
        VN97EvaluationResult(1, 10, 1.0, 0.1),
        "a" * 64,
    )
    second = VN97CampaignObservation(
        candidate,
        10,
        VN97EvaluationResult(1, 10, 0.9, 0.2),
        "b" * 64,
    )
    with pytest.raises(VN97CampaignError, match="duplicate"):
        select_best_campaign_candidate([first, second])
