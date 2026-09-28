from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from .mamba2_kaggle_real_parity import (
    _default_prompts,
    _mixer_tensors,
)
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
    rms_norm_ref,
)


SCHEMA = "VN97M2K1ISEMANTIC1"


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
        raise ValueError("K1I requires real G0.3 transfer evidence")
    if payload.get("status") != "REAL_G0_3_TRANSFER_PASS":
        raise ValueError("K1I G0.3 transfer evidence is not passing")
    if payload.get("source_revision") != PINNED_SOURCE_REVISION:
        raise ValueError("K1I source revision mismatch")
    if payload.get("source_weight_sha256") != PINNED_WEIGHT_SHA256:
        raise ValueError("K1I source weight mismatch")
    if payload.get("transfer_semantics") != "tensor_value_identity_1to1":
        raise ValueError("K1I requires exact 1:1 transfer semantics")
    if payload.get("core_reinitialized") is not False:
        raise ValueError("K1I transferred core must not be reinitialized")
    if payload.get("lossy_mapping_used") is not False:
        raise ValueError("K1I transfer must be lossless")
    if payload.get("quantization_used") is not False:
        raise ValueError("K1I transfer must be dense/unquantized")
    return payload


def _epsilon_units(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    *,
    epsilon: float,
) -> tuple[float, float]:
    ref = reference.detach().float()
    cand = candidate.detach().float()
    diff = (ref - cand).abs()
    scale = torch.maximum(torch.ones_like(ref), ref.abs())
    units = diff / (epsilon * scale)
    return float(diff.max().item()), float(units.max().item())


def _new_layer_record(index: int) -> dict[str, Any]:
    return {
        "layer": index,
        "block_norm_max_abs_error": 0.0,
        "block_norm_max_epsilon_units": 0.0,
        "mixer_out_max_abs_error": 0.0,
        "mixer_out_max_epsilon_units": 0.0,
        "conv_state_max_abs_error": 0.0,
        "conv_state_max_epsilon_units": 0.0,
        "ssm_state_max_abs_error": 0.0,
        "ssm_state_max_epsilon_units": 0.0,
        "samples": 0,
    }


def _update_record(
    record: dict[str, Any],
    *,
    block_norm: tuple[float, float],
    mixer_out: tuple[float, float],
    conv_state: tuple[float, float],
    ssm_state: tuple[float, float],
) -> None:
    for prefix, values in (
        ("block_norm", block_norm),
        ("mixer_out", mixer_out),
        ("conv_state", conv_state),
        ("ssm_state", ssm_state),
    ):
        record[f"{prefix}_max_abs_error"] = max(
            float(record[f"{prefix}_max_abs_error"]),
            values[0],
        )
        record[f"{prefix}_max_epsilon_units"] = max(
            float(record[f"{prefix}_max_epsilon_units"]),
            values[1],
        )
    record["samples"] = int(record["samples"]) + 1


@torch.inference_mode()
def run_all_layer_semantic_parity(
    *,
    source_root: Path,
    evidence_path: Path,
    output_path: Path,
    max_prompt_tokens: int,
    max_epsilon_units: float,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("K1I requires CUDA")
    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(device)
    if "T4" not in gpu_name.upper():
        raise RuntimeError(f"K1I expects Kaggle T4, got {gpu_name!r}")
    if max_prompt_tokens <= 0:
        raise ValueError("max_prompt_tokens must be positive")
    if not (0.0 < max_epsilon_units <= 8.0):
        raise ValueError("max_epsilon_units must be in (0,8]")

    transfer = _load_transfer_evidence(evidence_path)
    spec, source_state, source_receipt = inspect_pinned_source(
        source_root,
        source_revision=PINNED_SOURCE_REVISION,
        verify_weight_sha256=True,
    )
    del source_state
    if source_receipt.weight_sha256 != transfer["source_weight_sha256"]:
        raise ValueError("K1I source differs from real G0.3 transfer")
    if spec.d_model != 2560 or spec.n_layers != 64:
        raise ValueError("K1I source geometry mismatch")

    from transformers import AutoTokenizer
    from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
    import mamba_ssm.modules.mamba2 as official_mamba2_module

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
    epsilon = float(torch.finfo(torch.float16).eps)
    layer_records = [_new_layer_record(index) for index in range(64)]
    probe_cases = 0
    probe_tokens = 0

    for prompt in _default_prompts():
        token_ids = tokenizer(
            prompt,
            add_special_tokens=False,
            return_attention_mask=False,
        )["input_ids"]
        tokens = [int(value) for value in token_ids[:max_prompt_tokens]]
        if not tokens:
            continue
        probe_cases += 1

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

        for token_id in tokens:
            token = torch.tensor(
                [token_id],
                dtype=torch.long,
                device=device,
            )
            hidden = model.backbone.embedding(token)
            residual: torch.Tensor | None = None

            for index, layer in enumerate(model.backbone.layers):
                residual_next = hidden if residual is None else hidden + residual
                residual_input = residual_next.to(
                    dtype=layer.norm.weight.dtype
                )

                official_norm = layer.norm(residual_input)
                vn97_norm = rms_norm_ref(
                    residual_input,
                    layer.norm.weight,
                    eps=float(layer.norm.eps),
                )

                state = states[index]
                conv_initial = state.conv.clone()
                ssm_initial = state.ssm.clone()

                official_out, official_conv, official_ssm = layer.mixer.step(
                    official_norm.unsqueeze(1),
                    conv_initial.clone(),
                    ssm_initial.clone(),
                )
                official_out = official_out.squeeze(1)

                vn97_out, vn97_state = mamba2_mixer_step_ref(
                    official_norm,
                    Mamba2LayerReferenceState(
                        conv=conv_initial.clone(),
                        ssm=ssm_initial.clone(),
                    ),
                    _mixer_tensors(layer),
                    config,
                )

                _update_record(
                    layer_records[index],
                    block_norm=_epsilon_units(
                        official_norm,
                        vn97_norm,
                        epsilon=epsilon,
                    ),
                    mixer_out=_epsilon_units(
                        official_out,
                        vn97_out,
                        epsilon=epsilon,
                    ),
                    conv_state=_epsilon_units(
                        official_conv,
                        vn97_state.conv,
                        epsilon=epsilon,
                    ),
                    ssm_state=_epsilon_units(
                        official_ssm,
                        vn97_state.ssm,
                        epsilon=epsilon,
                    ),
                )

                states[index] = Mamba2LayerReferenceState(
                    conv=official_conv,
                    ssm=official_ssm,
                )
                hidden = official_out
                residual = residual_next.float()

            probe_tokens += 1

    if probe_cases <= 0 or probe_tokens <= 0:
        raise RuntimeError("K1I produced no probe data")

    stage_names = (
        "block_norm",
        "mixer_out",
        "conv_state",
        "ssm_state",
    )
    violations: list[dict[str, object]] = []
    for record in layer_records:
        for stage in stage_names:
            units = float(record[f"{stage}_max_epsilon_units"])
            if units > max_epsilon_units:
                violations.append(
                    {
                        "layer": int(record["layer"]),
                        "stage": stage,
                        "max_epsilon_units": units,
                        "max_abs_error": float(
                            record[f"{stage}_max_abs_error"]
                        ),
                    }
                )

    worst = max(
        (
            {
                "layer": int(record["layer"]),
                "stage": stage,
                "max_epsilon_units":
                    float(record[f"{stage}_max_epsilon_units"]),
                "max_abs_error": float(record[f"{stage}_max_abs_error"]),
            }
            for record in layer_records
            for stage in stage_names
        ),
        key=lambda item: float(item["max_epsilon_units"]),
    )
    passed = not violations

    body: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "PASS" if passed else "FAIL",
        "source_model_id": "state-spaces/mamba2-2.7b",
        "source_revision": PINNED_SOURCE_REVISION,
        "source_weight_sha256": PINNED_WEIGHT_SHA256,
        "capsule_id": transfer["capsule_id"],
        "oracle_commit": PINNED_ORACLE_MAMBA_COMMIT,
        "gpu_name": gpu_name,
        "dtype": "float16",
        "official_kernel_mode": "fallback",
        "fp16_epsilon": epsilon,
        "max_epsilon_units_allowed": max_epsilon_units,
        "probe_cases": probe_cases,
        "probe_tokens": probe_tokens,
        "layer_samples": probe_tokens * 64,
        "layer_records": layer_records,
        "violations": violations,
        "worst_same_input_result": worst,
        "same_input_all_layers_passed": passed,
        "source_to_vn97_semantic_parity_passed": passed,
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
        default=Path("/kaggle/working/vn97-k1i-semantic-parity.json"),
    )
    parser.add_argument("--max-prompt-tokens", type=int, default=4)
    parser.add_argument("--max-epsilon-units", type=float, default=2.0)
    args = parser.parse_args()

    receipt = run_all_layer_semantic_parity(
        source_root=args.source_root.resolve(strict=True),
        evidence_path=args.evidence.resolve(strict=True),
        output_path=args.output,
        max_prompt_tokens=args.max_prompt_tokens,
        max_epsilon_units=args.max_epsilon_units,
    )
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "probe_cases": receipt["probe_cases"],
                "probe_tokens": receipt["probe_tokens"],
                "layer_samples": receipt["layer_samples"],
                "max_epsilon_units_allowed":
                    receipt["max_epsilon_units_allowed"],
                "worst_same_input_result":
                    receipt["worst_same_input_result"],
                "violations": receipt["violations"][:20],
                "receipt_id": receipt["receipt_id"],
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    if receipt["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
