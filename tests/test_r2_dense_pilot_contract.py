from __future__ import annotations

import json

import pytest
import torch

from vn97.r2 import (
    EvaluationDomain,
    R2FastPathConfig,
    R2PilotGateThresholds,
    R2PilotProbe,
    R2PilotProbeMetrics,
    VN97R2Model,
    align_fast_path,
    assert_r2_pilot_scale,
    build_completion_windows,
    build_pilot_corpus_evidence,
    estimate_pilot_training_resources,
    evaluate_pilot_gate,
    lexical_token_f1,
    r2_cpu_pilot_config,
    r2_smoke_config,
    sha256_file,
)
from vn97.r2.dense_pilot_cli import main as dense_pilot_main
from vn97.r2.dense_training import _validate_resume_best_checkpoint
from vn97.tokenizer import VN97Tokenizer, VN97TokenizerPackage
from vn97.training import VN97ChatMessage, VN97TrainingConfig


def _chat(user: str, assistant: str):
    return (
        VN97ChatMessage(role="user", content=user),
        VN97ChatMessage(role="assistant", content=assistant),
    )


def test_r2c_pilot_config_is_locked_to_50m_150m_class() -> None:
    tokenizer = VN97Tokenizer()
    config = r2_cpu_pilot_config(tokenizer.vocab_size)
    parameters = assert_r2_pilot_scale(config)
    assert 50_000_000 <= parameters <= 150_000_000

    with pytest.raises(ValueError, match="outside the locked"):
        assert_r2_pilot_scale(r2_smoke_config(tokenizer.vocab_size))


def test_r2c_corpus_evidence_rejects_train_validation_leakage() -> None:
    train = (_chat("2+2?", "4"), _chat("Say yes.", "yes"))
    validation = (_chat("Capital of France?", "Paris"),)

    evidence = build_pilot_corpus_evidence(train, validation)
    assert evidence.overlap_records == 0
    assert evidence.training_records == 2
    assert evidence.validation_records == 1
    assert evidence.training_fingerprint != evidence.validation_fingerprint

    with pytest.raises(ValueError, match="leakage"):
        build_pilot_corpus_evidence(
            train,
            (train[0],),
        )




def test_r2c_corpus_evidence_rejects_exact_duplicates_inside_splits() -> None:
    row = _chat("repeat?", "no")
    with pytest.raises(ValueError, match="duplicate"):
        build_pilot_corpus_evidence(
            (row, row),
            (_chat("fresh?", "yes"),),
        )


def test_r2c_metric_contract_rejects_inconsistent_counts_and_rates() -> None:
    with pytest.raises(ValueError, match="must equal"):
        R2PilotProbeMetrics(
            tasks=2,
            natural_language_tasks=1,
            structured_tasks=0,
            tool_action_tasks=0,
            external_write_tasks=0,
            generation_success_rate=1.0,
            lexical_target_f1=1.0,
            instruction_following_rate=1.0,
            structured_valid_rate=1.0,
            tool_call_correct_rate=0.0,
            authority_route_correct_rate=0.0,
            protocol_exact_match_rate=1.0,
        )

    with pytest.raises(ValueError, match="\[0, 1\]"):
        R2PilotGateThresholds(
            generation_success_rate=1.1,
        )


def test_r2c_fast_alignment_respects_and_restores_existing_freeze() -> None:
    torch.manual_seed(41)
    tokenizer = VN97Tokenizer()
    model = VN97R2Model(
        r2_smoke_config(tokenizer.vocab_size)
    ).eval()

    frozen = model.layers[0].out_proj.weight
    frozen.requires_grad_(False)
    frozen_before = frozen.detach().clone()
    mutable = model.layers[0].in_proj.weight
    mutable_before = mutable.detach().clone()

    windows = build_completion_windows(
        tokenizer,
        (_chat("2+2?", "4"),),
        VN97TrainingConfig(
            sequence_length=32,
            stride=16,
            batch_size=1,
            epochs=1,
            max_windows=10,
        ),
    )
    result = align_fast_path(
        model,
        windows,
        R2FastPathConfig(
            epochs=1,
            batch_size=1,
            learning_rate=1e-3,
            seed=41,
        ),
        device="cpu",
    )

    assert result.steps == 1
    assert model.training is False
    assert frozen.requires_grad is False
    assert torch.equal(frozen_before, frozen.detach())
    assert mutable.requires_grad is True
    assert not torch.equal(mutable_before, mutable.detach())


def test_r2c_resource_preflight_is_conservative_and_observable() -> None:
    config = r2_cpu_pilot_config(VN97Tokenizer().vocab_size)

    low = estimate_pilot_training_resources(
        config,
        sequence_length=64,
        batch_size=1,
        available_ram=1,
    )
    assert low.parameter_count == config.estimated_parameter_count()
    assert low.optimizer_model_grad_bytes == low.parameter_count * 16
    assert low.activation_bytes > 0
    assert low.recommended_ram_bytes > low.optimizer_model_grad_bytes
    assert low.fits_available_ram is False

    high = estimate_pilot_training_resources(
        config,
        sequence_length=64,
        batch_size=1,
        available_ram=128 * 1024**3,
    )
    assert high.fits_available_ram is True


def test_r2c_lexical_target_f1_is_normalized_but_not_called_semantic() -> None:
    assert lexical_token_f1("Hello, world!", " hello,   WORLD! ") == 1.0
    partial = lexical_token_f1("alpha beta", "alpha gamma")
    assert 0.0 < partial < 1.0
    assert lexical_token_f1("alpha", "omega") == 0.0


def test_r2c_probe_contract_requires_tool_domain_for_external_write() -> None:
    prompt = (VN97ChatMessage(role="user", content="write file"),)
    with pytest.raises(ValueError, match="tool_action"):
        R2PilotProbe(
            domain=EvaluationDomain.NATURAL_LANGUAGE,
            messages=prompt,
            expected="ok",
            requires_external_write=True,
        )


def test_r2c_gate_is_fail_closed_when_required_probe_axes_are_missing() -> None:
    incomplete = R2PilotProbeMetrics(
        tasks=1,
        natural_language_tasks=1,
        structured_tasks=0,
        tool_action_tasks=0,
        external_write_tasks=0,
        generation_success_rate=1.0,
        lexical_target_f1=1.0,
        instruction_following_rate=1.0,
        structured_valid_rate=1.0,
        tool_call_correct_rate=0.0,
        authority_route_correct_rate=0.0,
        protocol_exact_match_rate=1.0,
    )
    decision = evaluate_pilot_gate(incomplete)
    assert decision.passed is False
    assert "tool_action_probe_missing" in decision.reasons
    assert "external_write_probe_missing" in decision.reasons

    complete = R2PilotProbeMetrics(
        tasks=3,
        natural_language_tasks=1,
        structured_tasks=2,
        tool_action_tasks=1,
        external_write_tasks=1,
        generation_success_rate=1.0,
        lexical_target_f1=0.90,
        instruction_following_rate=1.0,
        structured_valid_rate=1.0,
        tool_call_correct_rate=1.0,
        authority_route_correct_rate=1.0,
        protocol_exact_match_rate=1.0,
    )
    assert evaluate_pilot_gate(
        complete,
        thresholds=R2PilotGateThresholds(),
    ).passed is True




def test_r2c_resume_guard_rejects_missing_or_tampered_best_checkpoint(
    tmp_path,
) -> None:
    best = tmp_path / "best.r2.pt"

    with pytest.raises(RuntimeError, match="missing best checkpoint"):
        _validate_resume_best_checkpoint(
            best,
            best_epoch=0,
            best_checkpoint_sha256="a" * 64,
        )

    best.write_bytes(b"checkpoint-a")
    original_sha = sha256_file(best)
    assert (
        _validate_resume_best_checkpoint(
            best,
            best_epoch=0,
            best_checkpoint_sha256=original_sha,
        )
        == original_sha
    )

    best.write_bytes(b"checkpoint-b")
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        _validate_resume_best_checkpoint(
            best,
            best_epoch=0,
            best_checkpoint_sha256=original_sha,
        )

    with pytest.raises(RuntimeError, match="without best epoch"):
        _validate_resume_best_checkpoint(
            best,
            best_epoch=-1,
            best_checkpoint_sha256=original_sha,
        )


def test_r2c_smoke_dense_cli_writes_v2_evidence_without_quantization(
    tmp_path,
) -> None:
    tokenizer = tmp_path / "tokenizer.vn97tk1"
    tokenizer.write_bytes(VN97TokenizerPackage().to_bytes())

    train = tmp_path / "train.jsonl"
    valid = tmp_path / "valid.jsonl"
    train.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "2+2?"},
                    {"role": "assistant", "content": "4"},
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )
    valid.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "1+1?"},
                    {"role": "assistant", "content": "2"},
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )

    output = tmp_path / "output"
    code = dense_pilot_main(
        [
            "--tokenizer",
            str(tokenizer),
            "--train-jsonl",
            str(train),
            "--validation-jsonl",
            str(valid),
            "--work-dir",
            str(tmp_path / "work"),
            "--output-dir",
            str(output),
            "--profile",
            "smoke",
            "--sequence-length",
            "32",
            "--stride",
            "16",
            "--batch-size",
            "1",
            "--epochs",
            "1",
            "--checkpoint-every",
            "1",
            "--device",
            "cpu",
        ]
    )
    assert code == 0

    report = json.loads(
        (output / "r2-dense-pilot-report.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["schema"] == "VN97R2DENSEPILOT2"
    assert report["status"] == "PASS"
    assert report["corpus"]["overlap_records"] == 0
    assert report["probe_gate"]["passed"] is False
    assert report["pilot_accepted"] is False
    assert report["quantization_used"] is False
