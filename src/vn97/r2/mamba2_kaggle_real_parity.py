from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

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


SCHEMA = "VN97M2K1PARITY1"


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite(value: float, label: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{label} must be finite and non-negative")
    return value


def _load_transfer_evidence(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "VN97R2G03REALEVIDENCE1":
        raise ValueError("real G0.3 transfer evidence schema mismatch")
    if payload.get("status") != "REAL_G0_3_TRANSFER_PASS":
        raise ValueError("real G0.3 transfer evidence is not passing")
    if payload.get("source_revision") != PINNED_SOURCE_REVISION:
        raise ValueError("real G0.3 source revision mismatch")
    if payload.get("source_weight_sha256") != PINNED_WEIGHT_SHA256:
        raise ValueError("real G0.3 source weight mismatch")
    if payload.get("transfer_semantics") != "tensor_value_identity_1to1":
        raise ValueError("real G0.3 transfer is not 1:1")
    if payload.get("core_reinitialized") is not False:
        raise ValueError("real G0.3 core was reinitialized")
    if payload.get("lossy_mapping_used") is not False:
        raise ValueError("real G0.3 used lossy mapping")
    if payload.get("quantization_used") is not False:
        raise ValueError("real G0.3 used quantization")
    return payload


def _mixer_tensors(layer: Any) -> dict[str, torch.Tensor]:
    mixer = layer.mixer
    return {
        "in_proj.weight": mixer.in_proj.weight,
        "conv1d.weight": mixer.conv1d.weight,
        "conv1d.bias": mixer.conv1d.bias,
        "dt_bias": mixer.dt_bias,
        "A_log": mixer.A_log,
        "D": mixer.D,
        "norm.weight": mixer.norm.weight,
        "out_proj.weight": mixer.out_proj.weight,
    }


def _max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(
        (left.detach().float() - right.detach().float())
        .abs()
        .max()
        .item()
    )


def _mean_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(
        (left.detach().float() - right.detach().float())
        .abs()
        .mean()
        .item()
    )


class RealParityRunner:
    def __init__(
        self,
        *,
        model: Any,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        self.model = model
        self.dtype = dtype
        self.device = device
        self.config = Mamba2ReferenceConfig(
            d_model=int(model.config.d_model),
            d_state=128,
            d_conv=4,
            expand=2,
            head_dim=64,
            n_groups=1,
            rms_eps=1e-5,
            norm_before_gate=False,
        )
        if len(model.backbone.layers) != 64:
            raise ValueError("official Mamba-2 model must have exactly 64 layers")
        self.source_states: list[Mamba2LayerReferenceState] = []
        self.vn97_states: list[Mamba2LayerReferenceState] = []
        self.reset()

    def reset(self) -> None:
        self.source_states.clear()
        self.vn97_states.clear()
        for layer in self.model.backbone.layers:
            conv, ssm = layer.mixer.allocate_inference_cache(
                1,
                1,
                dtype=self.dtype,
            )
            source = Mamba2LayerReferenceState(
                conv=conv.zero_(),
                ssm=ssm.zero_(),
            )
            target = Mamba2LayerReferenceState(
                conv=source.conv.clone(),
                ssm=source.ssm.clone(),
            )
            self.source_states.append(source)
            self.vn97_states.append(target)

    @torch.inference_mode()
    def source_step(
        self,
        token_id: int,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        tuple[torch.Tensor, ...],
    ]:
        token = torch.tensor(
            [token_id],
            dtype=torch.long,
            device=self.device,
        )
        hidden = self.model.backbone.embedding(token)
        residual: torch.Tensor | None = None
        layer0_hidden: torch.Tensor | None = None
        layer_hiddens: list[torch.Tensor] = []

        for index, layer in enumerate(self.model.backbone.layers):
            residual_next = hidden if residual is None else hidden + residual
            normalized = layer.norm(
                residual_next.to(dtype=layer.norm.weight.dtype)
            )
            state = self.source_states[index]
            out, conv, ssm = layer.mixer.step(
                normalized.unsqueeze(1),
                state.conv,
                state.ssm,
            )
            self.source_states[index] = Mamba2LayerReferenceState(
                conv=conv,
                ssm=ssm,
            )
            hidden = out.squeeze(1)
            residual = residual_next.float()
            layer_hiddens.append(hidden)
            if index == 0:
                layer0_hidden = hidden

        assert layer0_hidden is not None
        final_residual = hidden + residual
        final_hidden = self.model.backbone.norm_f(
            final_residual.to(
                dtype=self.model.backbone.norm_f.weight.dtype
            )
        )
        logits = F.linear(final_hidden, self.model.lm_head.weight)
        first = self.source_states[0]
        return logits, layer0_hidden, first.conv, first.ssm, tuple(layer_hiddens)

    @torch.inference_mode()
    def vn97_step(
        self,
        token_id: int,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        tuple[torch.Tensor, ...],
    ]:
        token = torch.tensor(
            [token_id],
            dtype=torch.long,
            device=self.device,
        )
        hidden = F.embedding(
            token,
            self.model.backbone.embedding.weight,
        )
        residual: torch.Tensor | None = None
        layer0_hidden: torch.Tensor | None = None
        layer_hiddens: list[torch.Tensor] = []

        for index, layer in enumerate(self.model.backbone.layers):
            residual_next = hidden if residual is None else hidden + residual
            normalized = rms_norm_ref(
                residual_next.to(dtype=layer.norm.weight.dtype),
                layer.norm.weight,
                eps=float(layer.norm.eps),
            )
            out, state = mamba2_mixer_step_ref(
                normalized,
                self.vn97_states[index],
                _mixer_tensors(layer),
                self.config,
            )
            self.vn97_states[index] = state
            hidden = out
            residual = residual_next.float()
            layer_hiddens.append(hidden)
            if index == 0:
                layer0_hidden = hidden

        assert layer0_hidden is not None
        final_residual = hidden + residual
        final_hidden = rms_norm_ref(
            final_residual.to(
                dtype=self.model.backbone.norm_f.weight.dtype
            ),
            self.model.backbone.norm_f.weight,
            eps=float(self.model.backbone.norm_f.eps),
        )
        logits = F.linear(final_hidden, self.model.lm_head.weight)
        first = self.vn97_states[0]
        return logits, layer0_hidden, first.conv, first.ssm, tuple(layer_hiddens)


def _default_prompts() -> list[str]:
    return [
        "The quick brown fox",
        "State space models",
        "2 + 2 =",
        "Mobile inference",
        "Vietnam is",
        "Explain memory",
        "Reason step by step:",
        "A small program",
    ]


def run_real_parity(
    *,
    source_root: Path,
    evidence_path: Path,
    output_path: Path,
    max_prompt_tokens: int,
    generation_tokens: int,
    max_logit_error: float,
    max_hidden_error: float,
    max_state_error: float,
    official_kernel_mode: str,
    compute_dtype: str,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("K1 real parity requires CUDA")
    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(device)
    if "T4" not in gpu_name.upper():
        raise RuntimeError(
            f"K1 is locked to Kaggle T4 for this campaign, got {gpu_name!r}"
        )

    transfer = _load_transfer_evidence(evidence_path)
    spec, source_state, source_receipt = inspect_pinned_source(
        source_root,
        source_revision=PINNED_SOURCE_REVISION,
        verify_weight_sha256=True,
    )
    del source_state
    if source_receipt.weight_sha256 != transfer["source_weight_sha256"]:
        raise ValueError("K1 source weight differs from real G0.3 transfer")
    if int(transfer["unique_core_parameters"]) != 2_702_599_680:
        raise ValueError("K1 real G0.3 parameter count mismatch")
    if spec.d_model != 2560 or spec.n_layers != 64:
        raise ValueError("K1 source geometry mismatch")

    from transformers import AutoTokenizer
    from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
    import mamba_ssm.modules.mamba2 as official_mamba2_module

    if official_kernel_mode not in {"optimized", "fallback"}:
        raise ValueError("official kernel mode must be optimized or fallback")
    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if compute_dtype not in dtype_map:
        raise ValueError("compute dtype must be float16 or float32")
    dtype = dtype_map[compute_dtype]
    if official_kernel_mode == "fallback":
        official_mamba2_module.causal_conv1d_update = None
        official_mamba2_module.selective_state_update = None

    tokenizer = AutoTokenizer.from_pretrained(
        str(source_root / "tokenizer"),
        local_files_only=True,
        use_fast=True,
    )
    torch.cuda.empty_cache()
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    if compute_dtype == "float32":
        if total_bytes < 20 * 1024**3:
            raise RuntimeError(
                "K1F full-model float32 parity is intentionally disabled on "
                "GPUs below 20 GiB total VRAM; use float16 fallback with the "
                "64-layer first-divergence trace on Kaggle T4"
            )
        if free_bytes < 16 * 1024**3:
            raise RuntimeError(
                "K1F float32 structural parity requires at least 16 GiB free "
                "VRAM before model load"
            )

    model = MambaLMHeadModel.from_pretrained(
        str(source_root),
        device=str(device),
        dtype=dtype,
    ).eval()

    runner = RealParityRunner(
        model=model,
        dtype=dtype,
        device=device,
    )

    maxima = {
        "logits": 0.0,
        "layer0_hidden": 0.0,
        "layer0_conv_state": 0.0,
        "layer0_ssm_state": 0.0,
    }
    means = {
        "logits": [],
        "layer0_hidden": [],
        "layer0_conv_state": [],
        "layer0_ssm_state": [],
    }
    total_probe_tokens = 0
    generation_equal = True
    generated_pairs: list[dict[str, object]] = []
    first_token_layerwise: list[dict[str, object]] | None = None

    for prompt in _default_prompts():
        encoded = tokenizer(
            prompt,
            add_special_tokens=False,
            return_attention_mask=False,
        )["input_ids"]
        tokens = [int(value) for value in encoded[:max_prompt_tokens]]
        if not tokens:
            continue
        runner.reset()
        source_logits: torch.Tensor | None = None
        vn97_logits: torch.Tensor | None = None

        for token_id in tokens:
            source = runner.source_step(token_id)
            target = runner.vn97_step(token_id)
            total_probe_tokens += 1

            if first_token_layerwise is None:
                first_token_layerwise = []
                for layer_index, (source_hidden, vn97_hidden) in enumerate(
                    zip(source[4], target[4], strict=True)
                ):
                    first_token_layerwise.append(
                        {
                            "layer": layer_index,
                            "hidden_max_abs_error": _max_abs(
                                source_hidden,
                                vn97_hidden,
                            ),
                            "conv_state_max_abs_error": _max_abs(
                                runner.source_states[layer_index].conv,
                                runner.vn97_states[layer_index].conv,
                            ),
                            "ssm_state_max_abs_error": _max_abs(
                                runner.source_states[layer_index].ssm,
                                runner.vn97_states[layer_index].ssm,
                            ),
                        }
                    )

            for name, left, right in (
                ("logits", source[0], target[0]),
                ("layer0_hidden", source[1], target[1]),
                ("layer0_conv_state", source[2], target[2]),
                ("layer0_ssm_state", source[3], target[3]),
            ):
                maxima[name] = max(maxima[name], _max_abs(left, right))
                means[name].append(_mean_abs(left, right))
            source_logits = source[0]
            vn97_logits = target[0]

        source_generated: list[int] = []
        vn97_generated: list[int] = []
        for _ in range(generation_tokens):
            assert source_logits is not None and vn97_logits is not None
            source_token = int(source_logits.argmax(dim=-1).item())
            vn97_token = int(vn97_logits.argmax(dim=-1).item())
            source_generated.append(source_token)
            vn97_generated.append(vn97_token)
            if source_token != vn97_token:
                generation_equal = False
                break
            source = runner.source_step(source_token)
            target = runner.vn97_step(vn97_token)
            for name, left, right in (
                ("logits", source[0], target[0]),
                ("layer0_hidden", source[1], target[1]),
                ("layer0_conv_state", source[2], target[2]),
                ("layer0_ssm_state", source[3], target[3]),
            ):
                maxima[name] = max(maxima[name], _max_abs(left, right))
                means[name].append(_mean_abs(left, right))
            source_logits = source[0]
            vn97_logits = target[0]

        generated_pairs.append(
            {
                "prompt_sha256": hashlib.sha256(
                    prompt.encode("utf-8")
                ).hexdigest(),
                "prompt_tokens": tokens,
                "source_generated": source_generated,
                "vn97_generated": vn97_generated,
                "exact": source_generated == vn97_generated,
            }
        )

    if total_probe_tokens <= 0:
        raise RuntimeError("K1 produced no probe tokens")

    assert first_token_layerwise is not None
    first_layer_over_threshold = next(
        (
            int(item["layer"])
            for item in first_token_layerwise
            if max(
                float(item["hidden_max_abs_error"]),
                float(item["conv_state_max_abs_error"]),
                float(item["ssm_state_max_abs_error"]),
            )
            > max(max_hidden_error, max_state_error)
        ),
        None,
    )
    worst_first_token_layer = max(
        first_token_layerwise,
        key=lambda item: max(
            float(item["hidden_max_abs_error"]),
            float(item["conv_state_max_abs_error"]),
            float(item["ssm_state_max_abs_error"]),
        ),
    )

    metrics = {
        "max_logits_abs_error": _finite(maxima["logits"], "logits"),
        "mean_logits_abs_error": _finite(
            sum(means["logits"]) / len(means["logits"]),
            "mean logits",
        ),
        "max_layer0_hidden_abs_error": _finite(
            maxima["layer0_hidden"],
            "layer0 hidden",
        ),
        "max_layer0_conv_state_abs_error": _finite(
            maxima["layer0_conv_state"],
            "layer0 conv state",
        ),
        "max_layer0_ssm_state_abs_error": _finite(
            maxima["layer0_ssm_state"],
            "layer0 ssm state",
        ),
    }
    passed = (
        generation_equal
        and metrics["max_logits_abs_error"] <= max_logit_error
        and metrics["max_layer0_hidden_abs_error"] <= max_hidden_error
        and max(
            metrics["max_layer0_conv_state_abs_error"],
            metrics["max_layer0_ssm_state_abs_error"],
        )
        <= max_state_error
    )

    body: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "PASS" if passed else "FAIL",
        "source_model_id": "state-spaces/mamba2-2.7b",
        "source_revision": PINNED_SOURCE_REVISION,
        "source_weight_sha256": PINNED_WEIGHT_SHA256,
        "capsule_id": transfer["capsule_id"],
        "transfer_manifest_id": transfer["transfer_manifest_id"],
        "oracle_commit": PINNED_ORACLE_MAMBA_COMMIT,
        "gpu_name": gpu_name,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_total_bytes": total_bytes,
        "gpu_free_bytes_before_model_load": free_bytes,
        "dtype": compute_dtype,
        "official_kernel_mode": official_kernel_mode,
        "probe_cases": len(generated_pairs),
        "probe_tokens": total_probe_tokens,
        "generation_tokens_per_case": generation_tokens,
        "generation_exact": generation_equal,
        "metrics": metrics,
        "thresholds": {
            "max_logit_error": max_logit_error,
            "max_hidden_error": max_hidden_error,
            "max_state_error": max_state_error,
        },
        "generated_pairs": generated_pairs,
        "first_token_layerwise": first_token_layerwise,
        "first_layer_over_threshold": first_layer_over_threshold,
        "worst_first_token_layer": worst_first_token_layer,
        "source_to_vn97_passed": passed,
        "vn97_to_ort_passed": False,
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
        default=Path("/kaggle/working/vn97-k1-real-parity.json"),
    )
    parser.add_argument("--max-prompt-tokens", type=int, default=12)
    parser.add_argument("--generation-tokens", type=int, default=4)
    parser.add_argument("--max-logit-error", type=float, default=5.0e-3)
    parser.add_argument("--max-hidden-error", type=float, default=5.0e-3)
    parser.add_argument("--max-state-error", type=float, default=5.0e-3)
    parser.add_argument(
        "--official-kernel-mode",
        choices=("optimized", "fallback"),
        default="optimized",
    )
    parser.add_argument(
        "--compute-dtype",
        choices=("float16", "float32"),
        default="float16",
    )
    args = parser.parse_args()

    receipt = run_real_parity(
        source_root=args.source_root.resolve(strict=True),
        evidence_path=args.evidence.resolve(strict=True),
        output_path=args.output,
        max_prompt_tokens=args.max_prompt_tokens,
        generation_tokens=args.generation_tokens,
        max_logit_error=args.max_logit_error,
        max_hidden_error=args.max_hidden_error,
        max_state_error=args.max_state_error,
        official_kernel_mode=args.official_kernel_mode,
        compute_dtype=args.compute_dtype,
    )
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "receipt_id": receipt["receipt_id"],
                "gpu_name": receipt["gpu_name"],
                "probe_cases": receipt["probe_cases"],
                "probe_tokens": receipt["probe_tokens"],
                "generation_exact": receipt["generation_exact"],
                "metrics": receipt["metrics"],
                "official_kernel_mode": receipt["official_kernel_mode"],
                "dtype": receipt["dtype"],
                "first_token_layerwise": receipt["first_token_layerwise"],
                "first_layer_over_threshold":
                    receipt["first_layer_over_threshold"],
                "worst_first_token_layer":
                    receipt["worst_first_token_layer"],
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    if receipt["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
