from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile

import torch
import torch.nn.functional as F

from .capability import CapabilityManifest, merge_capability_delta
from .checkpoint import load_r2_checkpoint, save_r2_checkpoint
from .config import (
    r2_cpu_pilot_config,
    r2_mobile_1b_config,
    r2_smoke_config,
)
from .model import VN97R2Model


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "CPU-only VN97-R2 foundation pilot. Verifies recurrent/full "
            "parity, fast/deep paths, one dense learning step, checkpoint "
            "roundtrip, and mergeable capability-delta semantics."
        )
    )
    parser.add_argument(
        "--profile",
        choices=("smoke", "pilot"),
        default="smoke",
    )
    parser.add_argument("--vocab-size", type=int, default=None)
    parser.add_argument("--sequence-length", type=int, default=12)
    parser.add_argument("--output", default=None)
    parser.add_argument("--seed", type=int, default=9702)
    return parser


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _step_logits(
    model: VN97R2Model,
    input_ids: torch.Tensor,
    *,
    profile: str,
) -> torch.Tensor:
    state = None
    outputs: list[torch.Tensor] = []
    for position in range(input_ids.shape[1]):
        logits, state = model.step(
            input_ids[:, position],
            state,
            profile=profile,
        )
        outputs.append(logits.unsqueeze(1))
    return torch.cat(outputs, dim=1)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.sequence_length < 3:
        raise ValueError("sequence-length must be >= 3")
    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, min(torch.get_num_threads(), 4)))

    if args.profile == "pilot":
        config = r2_cpu_pilot_config(
            vocab_size=args.vocab_size or 4096
        )
    else:
        config = r2_smoke_config(
            vocab_size=args.vocab_size or 320
        )

    model = VN97R2Model(config)
    model.eval()
    input_ids = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(2, args.sequence_length),
        dtype=torch.long,
    )

    with torch.inference_mode():
        deep_logits, _ = model(input_ids, profile="deep")
        deep_step = _step_logits(
            model,
            input_ids,
            profile="deep",
        )
        fast_logits, fast_state = model(
            input_ids,
            profile="fast",
        )
        fast_step = _step_logits(
            model,
            input_ids,
            profile="fast",
        )

    deep_error = float(
        (deep_logits - deep_step).abs().max().item()
    )
    fast_error = float(
        (fast_logits - fast_step).abs().max().item()
    )
    torch.testing.assert_close(
        deep_logits,
        deep_step,
        rtol=1e-5,
        atol=1e-6,
    )
    torch.testing.assert_close(
        fast_logits,
        fast_step,
        rtol=1e-5,
        atol=1e-6,
    )
    if fast_state.active_layers != config.fast_layers:
        raise AssertionError("fast state did not honor fast_layers")

    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
    )
    before = {
        name: tensor.detach().clone()
        for name, tensor in model.state_dict().items()
        if name in {
            "embedding.weight",
            "layers.0.out_proj.weight",
        }
    }
    logits, _ = model(input_ids, profile="deep")
    loss = F.cross_entropy(
        logits[:, :-1].reshape(-1, config.vocab_size),
        input_ids[:, 1:].reshape(-1),
    )
    if not bool(torch.isfinite(loss)):
        raise AssertionError("R2 smoke loss is non-finite")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        1.0,
    )
    if not bool(torch.isfinite(torch.as_tensor(grad_norm))):
        raise AssertionError("R2 smoke gradient is non-finite")
    optimizer.step()

    changed = 0
    state_after = model.state_dict()
    for name, tensor in before.items():
        changed += int(
            not torch.equal(
                tensor,
                state_after[name].detach(),
            )
        )
    if changed == 0:
        raise AssertionError("dense training step changed no tracked weights")

    model.eval()
    with torch.inference_mode():
        post_train_logits, _ = model(
            input_ids,
            profile="deep",
        )

    temporary = None
    if args.output is None:
        temporary = tempfile.TemporaryDirectory(
            prefix="vn97-r2-pilot-"
        )
        checkpoint_path = Path(temporary.name) / "r2-smoke.pt"
    else:
        checkpoint_path = Path(args.output)

    checkpoint_sha = save_r2_checkpoint(
        checkpoint_path,
        model,
        stage="r2-foundation-smoke",
        metadata={
            "profile": args.profile,
            "seed": args.seed,
        },
    )
    loaded, evidence = load_r2_checkpoint(checkpoint_path)
    loaded.eval()
    with torch.inference_mode():
        loaded_logits, _ = loaded(
            input_ids,
            profile="deep",
        )
    torch.testing.assert_close(
        post_train_logits,
        loaded_logits,
        rtol=0.0,
        atol=0.0,
    )

    base_state = model.state_dict()
    delta_name = "final_norm.weight"
    capability_delta = {
        delta_name: torch.zeros_like(base_state[delta_name])
    }
    manifest = CapabilityManifest(
        name="FOUNDATION_SMOKE",
        version=1,
        parent_checkpoint_sha256=checkpoint_sha,
        training_recipe_sha256=_digest("r2-smoke-recipe"),
        dataset_sha256=(_digest("r2-smoke-dataset"),),
        task_families=("foundation",),
    )
    merged = merge_capability_delta(
        base_state,
        capability_delta,
    )
    if not torch.equal(
        merged[delta_name],
        base_state[delta_name],
    ):
        raise AssertionError("zero capability delta changed model weights")

    mobile = r2_mobile_1b_config(config.vocab_size)
    result = {
        "schema": "VN97R2PILOT1",
        "status": "PASS",
        "profile": args.profile,
        "architecture_id": config.architecture_id,
        "config_fingerprint": config.fingerprint(),
        "parameters": model.parameter_count(),
        "recurrent_state_bytes_fp16_batch1": (
            config.recurrent_state_bytes(
                1,
                bytes_per_value=2,
            )
        ),
        "deep_step_parity_max_abs": deep_error,
        "fast_step_parity_max_abs": fast_error,
        "fast_layers": config.fast_layers,
        "deep_layers": config.n_layers,
        "training_loss": float(loss.detach().cpu()),
        "grad_norm": float(grad_norm),
        "tracked_weights_changed": changed,
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_schema": evidence["schema"],
        "capability_manifest_fingerprint": manifest.fingerprint(),
        "mobile_target_parameter_estimate": (
            mobile.estimated_parameter_count()
        ),
        "mobile_target_2bit_weight_bytes": (
            mobile.estimated_weight_bytes(2.0)
        ),
        "mobile_target_int4_weight_bytes": (
            mobile.estimated_weight_bytes(4.0)
        ),
    }
    print(
        json.dumps(
            result,
            sort_keys=True,
            indent=2,
        ),
        flush=True,
    )

    if temporary is not None:
        temporary.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
