from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from .mamba2_kaggle_real_parity import _mixer_tensors, _max_abs
from .mamba2_source_integrity import (
    PINNED_ORACLE_MAMBA_COMMIT,
    PINNED_SOURCE_REVISION,
    PINNED_WEIGHT_SHA256,
    inspect_pinned_source,
)
from .mamba2_ssd_reference import (
    Mamba2LayerReferenceState,
    Mamba2ReferenceConfig,
    mamba2_mixer_step_ref,
    rms_norm_gated_ref,
    rms_norm_ref,
)


SCHEMA = "VN97M2K1GSTAGE1"


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _load_transfer_evidence(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "VN97R2G03REALEVIDENCE1":
        raise ValueError("K1G requires real G0.3 evidence")
    if payload.get("status") != "REAL_G0_3_TRANSFER_PASS":
        raise ValueError("K1G G0.3 evidence is not passing")
    if payload.get("source_revision") != PINNED_SOURCE_REVISION:
        raise ValueError("K1G source revision mismatch")
    if payload.get("source_weight_sha256") != PINNED_WEIGHT_SHA256:
        raise ValueError("K1G source weight mismatch")
    return payload


@torch.inference_mode()
def run_stage_trace(
    *,
    source_root: Path,
    evidence_path: Path,
    output_path: Path,
    trace_layer: int,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("K1G requires CUDA")
    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(device)
    if "T4" not in gpu_name.upper():
        raise RuntimeError(f"K1G expects Kaggle T4, got {gpu_name!r}")
    if trace_layer < 0 or trace_layer >= 64:
        raise ValueError("trace_layer must be in [0,63]")

    transfer = _load_transfer_evidence(evidence_path)
    spec, source_state, receipt = inspect_pinned_source(
        source_root,
        source_revision=PINNED_SOURCE_REVISION,
        verify_weight_sha256=True,
    )
    del source_state
    if receipt.weight_sha256 != transfer["source_weight_sha256"]:
        raise ValueError("K1G source differs from real transfer")
    if spec.d_model != 2560 or spec.n_layers != 64:
        raise ValueError("K1G source geometry mismatch")

    from transformers import AutoTokenizer
    from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
    import mamba_ssm.modules.mamba2 as official_mamba2_module

    # Force the official recurrent path to use its unfused Python fallback.
    official_mamba2_module.causal_conv1d_update = None
    official_mamba2_module.selective_state_update = None

    tokenizer = AutoTokenizer.from_pretrained(
        str(source_root / "tokenizer"),
        local_files_only=True,
        use_fast=True,
    )
    torch.cuda.empty_cache()
    model = MambaLMHeadModel.from_pretrained(
        str(source_root),
        device=str(device),
        dtype=torch.float16,
    ).eval()

    config = Mamba2ReferenceConfig(
        d_model=int(model.config.d_model),
        d_state=128,
        d_conv=4,
        expand=2,
        head_dim=64,
        n_groups=1,
        rms_eps=1e-5,
        norm_before_gate=False,
    )

    prompt = "The quick brown fox"
    token_ids = tokenizer(
        prompt,
        add_special_tokens=False,
        return_attention_mask=False,
    )["input_ids"]
    if not token_ids:
        raise RuntimeError("K1G probe tokenization is empty")
    token_id = int(token_ids[0])
    token = torch.tensor([token_id], device=device, dtype=torch.long)

    states: list[Mamba2LayerReferenceState] = []
    for layer in model.backbone.layers:
        conv, ssm = layer.mixer.allocate_inference_cache(
            1,
            1,
            dtype=torch.float16,
        )
        states.append(
            Mamba2LayerReferenceState(
                conv=conv.zero_(),
                ssm=ssm.zero_(),
            )
        )

    hidden = model.backbone.embedding(token)
    residual: torch.Tensor | None = None

    for index in range(trace_layer):
        layer = model.backbone.layers[index]
        residual_next = hidden if residual is None else hidden + residual
        normalized = layer.norm(
            residual_next.to(dtype=layer.norm.weight.dtype)
        )
        state = states[index]
        out, conv, ssm = layer.mixer.step(
            normalized.unsqueeze(1),
            state.conv,
            state.ssm,
        )
        states[index] = Mamba2LayerReferenceState(
            conv=conv,
            ssm=ssm,
        )
        hidden = out.squeeze(1)
        residual = residual_next.float()

    layer = model.backbone.layers[trace_layer]
    residual_next = hidden if residual is None else hidden + residual
    residual_input = residual_next.to(dtype=layer.norm.weight.dtype)

    official_block_norm = layer.norm(residual_input)
    reference_block_norm = rms_norm_ref(
        residual_input,
        layer.norm.weight,
        eps=float(layer.norm.eps),
    )

    mixer = layer.mixer
    same_input_in_proj_official = mixer.in_proj(official_block_norm)
    same_input_in_proj_reference = F.linear(
        official_block_norm,
        mixer.in_proj.weight,
    )

    source_state = states[trace_layer]
    source_conv_initial = source_state.conv.clone()
    source_ssm_initial = source_state.ssm.clone()

    captured: dict[str, torch.Tensor] = {}

    def norm_pre_hook(_module: torch.nn.Module, args: tuple[Any, ...]) -> None:
        captured["gated_norm_x"] = args[0].detach().clone()
        captured["gated_norm_z"] = args[1].detach().clone()

    def norm_hook(
        _module: torch.nn.Module,
        _args: tuple[Any, ...],
        output: torch.Tensor,
    ) -> None:
        captured["gated_norm_out"] = output.detach().clone()

    def out_proj_pre_hook(
        _module: torch.nn.Module,
        args: tuple[Any, ...],
    ) -> None:
        captured["out_proj_input"] = args[0].detach().clone()

    def out_proj_hook(
        _module: torch.nn.Module,
        _args: tuple[Any, ...],
        output: torch.Tensor,
    ) -> None:
        captured["out_proj_out"] = output.detach().clone()

    h1 = mixer.norm.register_forward_pre_hook(norm_pre_hook)
    h2 = mixer.norm.register_forward_hook(norm_hook)
    h3 = mixer.out_proj.register_forward_pre_hook(out_proj_pre_hook)
    h4 = mixer.out_proj.register_forward_hook(out_proj_hook)
    try:
        official_out, official_conv, official_ssm = mixer.step(
            official_block_norm.unsqueeze(1),
            source_conv_initial.clone(),
            source_ssm_initial.clone(),
        )
    finally:
        h1.remove()
        h2.remove()
        h3.remove()
        h4.remove()
    official_out = official_out.squeeze(1)

    # Run the VN97 reference on the *same* official normalized input and exact
    # same initial states. Any delta here is an operator-semantic delta, not
    # trajectory drift from earlier layers.
    vn97_out_same_input, vn97_state_same_input = mamba2_mixer_step_ref(
        official_block_norm,
        Mamba2LayerReferenceState(
            conv=source_conv_initial.clone(),
            ssm=source_ssm_initial.clone(),
        ),
        _mixer_tensors(layer),
        config,
    )

    gated_norm_reference_same_input = rms_norm_gated_ref(
        captured["gated_norm_x"],
        captured["gated_norm_z"],
        mixer.norm.weight,
        eps=float(mixer.norm.eps),
        n_groups=config.n_groups,
        norm_before_gate=bool(mixer.norm.norm_before_gate),
    )
    out_proj_reference_same_input = F.linear(
        captured["out_proj_input"],
        mixer.out_proj.weight,
    )

    stage_errors = {
        "block_norm_same_input": _max_abs(
            official_block_norm,
            reference_block_norm,
        ),
        "in_proj_same_input": _max_abs(
            same_input_in_proj_official,
            same_input_in_proj_reference,
        ),
        "mixer_out_same_input": _max_abs(
            official_out,
            vn97_out_same_input,
        ),
        "conv_state_same_input": _max_abs(
            official_conv,
            vn97_state_same_input.conv,
        ),
        "ssm_state_same_input": _max_abs(
            official_ssm,
            vn97_state_same_input.ssm,
        ),
        "gated_norm_same_input": _max_abs(
            captured["gated_norm_out"],
            gated_norm_reference_same_input,
        ),
        "out_proj_same_input": _max_abs(
            captured["out_proj_out"],
            out_proj_reference_same_input,
        ),
    }

    ranked = sorted(
        stage_errors.items(),
        key=lambda item: item[1],
        reverse=True,
    )
    first_nonzero = next(
        (
            {"stage": name, "max_abs_error": value}
            for name, value in stage_errors.items()
            if value > 0.0
        ),
        None,
    )

    body: dict[str, Any] = {
        "schema": SCHEMA,
        "source_model_id": "state-spaces/mamba2-2.7b",
        "source_revision": PINNED_SOURCE_REVISION,
        "source_weight_sha256": PINNED_WEIGHT_SHA256,
        "capsule_id": transfer["capsule_id"],
        "oracle_commit": PINNED_ORACLE_MAMBA_COMMIT,
        "gpu_name": gpu_name,
        "dtype": "float16",
        "official_kernel_mode": "fallback",
        "trace_layer": trace_layer,
        "probe_prompt_sha256": hashlib.sha256(
            prompt.encode("utf-8")
        ).hexdigest(),
        "probe_token_id": token_id,
        "stage_errors": stage_errors,
        "ranked_stage_errors": [
            {"stage": name, "max_abs_error": value}
            for name, value in ranked
        ],
        "first_nonzero_stage": first_nonzero,
        "production_activation_authorized": False,
    }
    body["receipt_id"] = hashlib.sha256(
        SCHEMA.encode("ascii") + b"\0" + _canonical_json(body)
    ).hexdigest()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(_canonical_json(body) + b"\n")
    return body


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument(
        "--evidence",
        type=Path,
        default=Path("evidence/r2-g03-real-transfer.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/kaggle/working/vn97-k1g-stage-trace.json"),
    )
    parser.add_argument("--trace-layer", type=int, default=2)
    args = parser.parse_args()

    receipt = run_stage_trace(
        source_root=args.source_root.resolve(strict=True),
        evidence_path=args.evidence.resolve(strict=True),
        output_path=args.output,
        trace_layer=args.trace_layer,
    )
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
