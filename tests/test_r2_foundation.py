from __future__ import annotations

import hashlib

import torch

from vn97.r2 import (
    CapabilityManifest,
    CognitionMode,
    EvaluationDomain,
    CognitionRequest,
    R2EvaluationMetrics,
    R2PromotionPolicy,
    R2TrainingStage,
    R2ValidationGate,
    StageEvidence,
    VN97R2ExecutionPolicy,
    VN97R2Model,
    load_r2_checkpoint,
    merge_capability_delta,
    r2_smoke_config,
    save_r2_checkpoint,
)


def test_r2_full_and_recurrent_step_are_equivalent() -> None:
    torch.manual_seed(7)
    config = r2_smoke_config(vocab_size=96)
    model = VN97R2Model(config).eval()
    tokens = torch.randint(0, config.vocab_size, (2, 7))

    with torch.inference_mode():
        full, _ = model(tokens, profile="deep")
        state = None
        pieces = []
        for index in range(tokens.shape[1]):
            logits, state = model.step(
                tokens[:, index],
                state,
                profile="deep",
            )
            pieces.append(logits.unsqueeze(1))
        recurrent = torch.cat(pieces, dim=1)

    torch.testing.assert_close(
        full,
        recurrent,
        rtol=1e-5,
        atol=1e-6,
    )


def test_r2_fast_path_is_same_model_with_fewer_layers() -> None:
    torch.manual_seed(8)
    config = r2_smoke_config(vocab_size=80)
    model = VN97R2Model(config).eval()
    tokens = torch.randint(0, config.vocab_size, (1, 5))

    with torch.inference_mode():
        fast, fast_state = model(tokens, profile="fast")
        deep, deep_state = model(tokens, profile="deep")

    assert fast.shape == deep.shape
    assert fast_state.active_layers == config.fast_layers
    assert deep_state.active_layers == config.n_layers
    assert len(fast_state.layers) == config.fast_layers
    assert len(deep_state.layers) == config.n_layers


def test_r2_runtime_policy_forces_deep_on_external_write() -> None:
    config = r2_smoke_config(vocab_size=64)
    policy = VN97R2ExecutionPolicy(config)

    fast = policy.decide(CognitionRequest(deadline_ms=20.0))
    write = policy.decide(
        CognitionRequest(
            deadline_ms=20.0,
            requires_external_write=True,
        )
    )

    assert fast.mode is CognitionMode.FAST
    assert fast.active_layers == config.fast_layers
    assert write.mode is CognitionMode.DEEP
    assert write.active_layers == config.n_layers


def test_r2_capability_delta_bakes_into_single_state() -> None:
    config = r2_smoke_config(vocab_size=64)
    model = VN97R2Model(config)
    base = model.state_dict()
    name = "final_norm.weight"
    delta = {name: torch.full_like(base[name], 0.125)}

    merged = merge_capability_delta(base, delta, alpha=0.5)

    torch.testing.assert_close(
        merged[name],
        base[name] + 0.0625,
    )
    assert set(merged) == set(base)

    digest = hashlib.sha256(b"x").hexdigest()
    manifest = CapabilityManifest(
        name="TEST",
        version=1,
        parent_checkpoint_sha256=digest,
        training_recipe_sha256=digest,
        dataset_sha256=(digest,),
        task_families=("test",),
    )
    assert len(manifest.fingerprint()) == 64


def test_r2_checkpoint_roundtrip(tmp_path) -> None:
    torch.manual_seed(9)
    config = r2_smoke_config(vocab_size=72)
    model = VN97R2Model(config).eval()
    tokens = torch.randint(0, config.vocab_size, (1, 4))

    with torch.inference_mode():
        before, _ = model(tokens)

    checkpoint = tmp_path / "r2.pt"
    digest = save_r2_checkpoint(
        checkpoint,
        model,
        stage="unit-test",
    )
    loaded, evidence = load_r2_checkpoint(checkpoint)
    loaded.eval()

    with torch.inference_mode():
        after, _ = loaded(tokens)

    assert evidence["sha256"] == digest
    assert evidence["config_fingerprint"] == config.fingerprint()
    torch.testing.assert_close(before, after, rtol=0.0, atol=0.0)


def test_r2_qat_is_blocked_until_fresh_validation_passes() -> None:
    policy = R2PromotionPolicy()
    failed = StageEvidence(
        stage=R2TrainingStage.FRESH_VALIDATION,
        generation_valid=False,
        semantic_valid=True,
        tool_protocol_valid=True,
        authority_valid=True,
        regression_valid=True,
    )
    passed = StageEvidence(
        stage=R2TrainingStage.FRESH_VALIDATION,
        generation_valid=True,
        semantic_valid=True,
        tool_protocol_valid=True,
        authority_valid=True,
        regression_valid=True,
    )

    assert policy.qat_allowed(failed) is False
    assert policy.next_stage(
        R2TrainingStage.FRESH_VALIDATION,
        failed,
    ) is R2TrainingStage.FRESH_VALIDATION

    assert policy.qat_allowed(passed) is True
    assert policy.next_stage(
        R2TrainingStage.FRESH_VALIDATION,
        passed,
    ) is R2TrainingStage.QAT


def test_r2_natural_language_gate_does_not_require_exact_match() -> None:
    metrics = R2EvaluationMetrics(
        tasks=100,
        generation_success_rate=0.99,
        semantic_score=0.90,
        instruction_following_rate=0.95,
        structured_valid_rate=0.0,
        tool_call_correct_rate=0.0,
        authority_correct_rate=1.0,
        exact_match_rate=0.0,
        regression_score=0.99,
    )
    decision = R2ValidationGate().evaluate(
        metrics,
        domain=EvaluationDomain.NATURAL_LANGUAGE,
    )
    assert decision.passed is True


def test_r2_structured_protocol_keeps_exact_gate() -> None:
    metrics = R2EvaluationMetrics(
        tasks=100,
        generation_success_rate=0.99,
        semantic_score=0.90,
        instruction_following_rate=0.95,
        structured_valid_rate=0.99,
        tool_call_correct_rate=0.99,
        authority_correct_rate=1.0,
        exact_match_rate=0.50,
        regression_score=0.99,
    )
    decision = R2ValidationGate().evaluate(
        metrics,
        domain=EvaluationDomain.STRUCTURED_PROTOCOL,
    )
    assert decision.passed is False
    assert "protocol_exact_match_below_threshold" in decision.reasons



def test_r2_parallel_scan_matches_reference_and_chunked_continuation() -> None:
    torch.manual_seed(17)
    config = r2_smoke_config(vocab_size=112)
    model = VN97R2Model(config).eval()
    tokens = torch.randint(0, config.vocab_size, (2, 11))

    with torch.inference_mode():
        scan_hidden, scan_state = model.forward_hidden(
            tokens,
            profile="deep",
        )
        ref_hidden, ref_state = model.forward_hidden_reference(
            tokens,
            profile="deep",
        )

        first_hidden, first_state = model.forward_hidden(
            tokens[:, :6],
            profile="deep",
        )
        second_hidden, second_state = model.forward_hidden(
            tokens[:, 6:],
            first_state,
            profile="deep",
        )
        chunked_hidden = torch.cat(
            (first_hidden, second_hidden),
            dim=1,
        )

    torch.testing.assert_close(
        scan_hidden,
        ref_hidden,
        rtol=2e-5,
        atol=2e-6,
    )
    torch.testing.assert_close(
        scan_hidden,
        chunked_hidden,
        rtol=2e-5,
        atol=2e-6,
    )

    assert scan_state.active_layers == ref_state.active_layers
    assert scan_state.active_layers == second_state.active_layers
    for scan_layer, ref_layer, chunk_layer in zip(
        scan_state.layers,
        ref_state.layers,
        second_state.layers,
    ):
        torch.testing.assert_close(
            scan_layer.conv,
            ref_layer.conv,
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            scan_layer.ssm,
            ref_layer.ssm,
            rtol=2e-5,
            atol=2e-6,
        )
        torch.testing.assert_close(
            scan_layer.conv,
            chunk_layer.conv,
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            scan_layer.ssm,
            chunk_layer.ssm,
            rtol=2e-5,
            atol=2e-6,
        )


def test_r2_fast_parallel_scan_matches_recurrent_reference() -> None:
    torch.manual_seed(18)
    config = r2_smoke_config(vocab_size=104)
    model = VN97R2Model(config).eval()
    tokens = torch.randint(0, config.vocab_size, (1, 9))

    with torch.inference_mode():
        scan_logits, _ = model(tokens, profile="fast")
        state = None
        pieces = []
        for position in range(tokens.shape[1]):
            logits, state = model.step(
                tokens[:, position],
                state,
                profile="fast",
            )
            pieces.append(logits.unsqueeze(1))
        recurrent_logits = torch.cat(pieces, dim=1)

    torch.testing.assert_close(
        scan_logits,
        recurrent_logits,
        rtol=2e-5,
        atol=2e-6,
    )
