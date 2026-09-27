from __future__ import annotations

from pathlib import Path

import pytest
import torch

from vn97.r2 import VN97R2Model, r2_smoke_config
from vn97.r2.production_contract import (
    R2ProductionCorpusManifest,
    R2ProductionShard,
    R2ProductionTrainingRecipe,
)
from vn97.r2.production_training import (
    CPUOffloadedAdamW,
    R2MeasuredMemoryEvidence,
    R2ProductionTrainerConfig,
    train_production_stage,
)
from vn97.training import VN97TrainingWindow


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


def _window(seed: int, *, vocab_size: int = 96) -> VN97TrainingWindow:
    generator = torch.Generator().manual_seed(seed)
    tokens = torch.randint(
        0,
        vocab_size,
        (64,),
        generator=generator,
    ).tolist()
    labels = torch.randint(
        0,
        vocab_size,
        (64,),
        generator=generator,
    ).tolist()
    return VN97TrainingWindow(
        input_ids=tuple(int(value) for value in tokens),
        labels=tuple(int(value) for value in labels),
        target_tokens=64,
    )


def _manifest() -> R2ProductionCorpusManifest:
    return R2ProductionCorpusManifest(
        stage="dense_pretrain",
        tokenizer_sha256=SHA_A,
        shards=(
            R2ProductionShard(
                split="train",
                sha256=SHA_B,
                bytes=1024,
                records=4,
                task_families=("language",),
            ),
            R2ProductionShard(
                split="validation",
                sha256=SHA_C,
                bytes=512,
                records=1,
                task_families=("language",),
            ),
        ),
    )


def _recipe() -> R2ProductionTrainingRecipe:
    return R2ProductionTrainingRecipe(
        sequence_length=64,
        micro_batch_size=1,
        gradient_accumulation_steps=2,
        precision="fp32",
        activation_checkpointing=True,
        memory_efficient_scan=True,
        optimizer_state_offload=True,
    )


def test_cpu_offloaded_adamw_matches_torch_adamw_one_step() -> None:
    torch.manual_seed(9711)
    initial = torch.randn(3, 5)
    gradient = torch.randn(3, 5)

    reference = torch.nn.Parameter(initial.clone())
    candidate = torch.nn.Parameter(initial.clone())

    reference.grad = gradient.clone()
    candidate.grad = gradient.clone()

    torch_optimizer = torch.optim.AdamW(
        [reference],
        lr=3e-4,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.01,
    )
    offloaded = CPUOffloadedAdamW(
        [("weight", candidate)],
        learning_rate=3e-4,
        weight_decay=0.01,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
    )

    torch_optimizer.step()
    offloaded.step()

    torch.testing.assert_close(
        candidate,
        reference,
        rtol=2e-6,
        atol=2e-7,
    )


def test_cpu_offloaded_adamw_state_roundtrip() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    parameter.grad = torch.tensor([0.25, -0.5])
    optimizer = CPUOffloadedAdamW(
        [("p", parameter)],
        learning_rate=1e-3,
        weight_decay=0.02,
        beta1=0.9,
        beta2=0.99,
        eps=1e-8,
    )
    optimizer.step()
    state = optimizer.state_dict()

    restored_parameter = torch.nn.Parameter(parameter.detach().clone())
    restored = CPUOffloadedAdamW(
        [("p", restored_parameter)],
        learning_rate=1e-3,
        weight_decay=0.02,
        beta1=0.9,
        beta2=0.99,
        eps=1e-8,
    )
    restored.load_state_dict(state)

    assert restored.state["p"]["step"] == 1
    torch.testing.assert_close(
        restored.state["p"]["exp_avg"],
        optimizer.state["p"]["exp_avg"],
    )
    torch.testing.assert_close(
        restored.state["p"]["exp_avg_sq"],
        optimizer.state["p"]["exp_avg_sq"],
    )


def test_measured_memory_evidence_rejects_bad_fraction() -> None:
    with pytest.raises(ValueError, match="safety_fraction"):
        R2MeasuredMemoryEvidence(
            architecture_fingerprint=SHA_A,
            recipe_fingerprint=SHA_B,
            sequence_length=64,
            micro_batch_size=1,
            device_name="test",
            free_device_bytes_before=1,
            total_device_bytes=1,
            peak_allocated_bytes=1,
            peak_reserved_bytes=1,
            safety_fraction=0.0,
            passed=False,
            reason="test",
        )


def test_production_trainer_pauses_and_resumes_at_accumulation_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vn97.r2.production_training as production_training

    # The production-scale lock is covered independently. This orchestration
    # test intentionally uses the tiny smoke model so CI never allocates 1B.
    monkeypatch.setattr(
        production_training,
        "assert_r2_production_scale",
        lambda _config: 1,
    )
    monkeypatch.setattr(
        production_training,
        "assert_production_training_contract",
        lambda *_args, **_kwargs: None,
    )

    torch.manual_seed(9712)
    model = VN97R2Model(r2_smoke_config(vocab_size=96))
    training = tuple(_window(100 + index) for index in range(4))
    validation = (_window(200),)
    manifest = _manifest()
    recipe = _recipe()
    trainer = R2ProductionTrainerConfig(
        epochs=1,
        learning_rate=3e-4,
        checkpoint_every_optimizer_steps=1,
        require_measured_cuda_preflight=False,
    )
    work = tmp_path / "work"
    best = tmp_path / "best.r2.pt"

    paused = train_production_stage(
        model,
        training,
        validation,
        manifest,
        recipe,
        trainer,
        work_dir=work,
        best_checkpoint_path=best,
        device="cpu",
        max_run_seconds=1e-9,
    )
    assert paused.completed is False
    assert paused.optimizer_steps == 1
    assert paused.micro_steps == 2
    assert paused.resume_checkpoint_sha256
    assert (work / "production-resume.pt").is_file()
    assert best.is_file()

    resumed_model = VN97R2Model(r2_smoke_config(vocab_size=96))
    completed = train_production_stage(
        resumed_model,
        training,
        validation,
        manifest,
        recipe,
        trainer,
        work_dir=work,
        best_checkpoint_path=best,
        device="cpu",
    )
    assert completed.completed is True
    assert completed.optimizer_steps == 2
    assert completed.micro_steps == 4
    assert completed.resume_checkpoint_sha256 == ""
    assert not (work / "production-resume.pt").exists()
