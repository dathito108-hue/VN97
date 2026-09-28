from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from .mamba2_kaggle_real_parity import (
    RealParityRunner,
    _default_prompts,
    _max_abs,
)
from .mamba2_source_integrity import (
    PINNED_ORACLE_MAMBA_COMMIT,
    PINNED_SOURCE_REVISION,
    PINNED_WEIGHT_SHA256,
    inspect_pinned_source,
)


SCHEMA = "VN97M2K1HENVELOPE1"


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
        raise ValueError("K1H requires real G0.3 transfer evidence")
    if payload.get("status") != "REAL_G0_3_TRANSFER_PASS":
        raise ValueError("K1H G0.3 transfer evidence is not passing")
    if payload.get("source_revision") != PINNED_SOURCE_REVISION:
        raise ValueError("K1H source revision mismatch")
    if payload.get("source_weight_sha256") != PINNED_WEIGHT_SHA256:
        raise ValueError("K1H source weight mismatch")
    if payload.get("transfer_semantics") != "tensor_value_identity_1to1":
        raise ValueError("K1H requires exact 1:1 transfer semantics")
    return payload


def _metric_template() -> dict[str, float]:
    return {
        "logits": 0.0,
        "layer0_hidden": 0.0,
        "layer0_conv_state": 0.0,
        "layer0_ssm_state": 0.0,
    }


def _update_metric(
    maxima: dict[str, float],
    *,
    left: tuple[torch.Tensor, ...],
    right: tuple[torch.Tensor, ...],
) -> None:
    for name, lhs, rhs in (
        ("logits", left[0], right[0]),
        ("layer0_hidden", left[1], right[1]),
        ("layer0_conv_state", left[2], right[2]),
        ("layer0_ssm_state", left[3], right[3]),
    ):
        maxima[name] = max(maxima[name], _max_abs(lhs, rhs))


def _max_abs_value(tensor: torch.Tensor) -> float:
    return float(tensor.detach().float().abs().max().item())


def run_backend_envelope(
    *,
    source_root: Path,
    evidence_path: Path,
    output_path: Path,
    max_prompt_tokens: int,
    continuation_tokens: int,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("K1H requires CUDA")
    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(device)
    if "T4" not in gpu_name.upper():
        raise RuntimeError(f"K1H expects Kaggle T4, got {gpu_name!r}")

    transfer = _load_transfer_evidence(evidence_path)
    spec, source_state, receipt = inspect_pinned_source(
        source_root,
        source_revision=PINNED_SOURCE_REVISION,
        verify_weight_sha256=True,
    )
    del source_state
    if receipt.weight_sha256 != transfer["source_weight_sha256"]:
        raise ValueError("K1H source differs from real G0.3 transfer")
    if spec.d_model != 2560 or spec.n_layers != 64:
        raise ValueError("K1H source geometry mismatch")

    from transformers import AutoTokenizer
    from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
    import mamba_ssm.modules.mamba2 as official_mamba2_module

    optimized_conv = official_mamba2_module.causal_conv1d_update
    optimized_ssm = official_mamba2_module.selective_state_update
    if optimized_conv is None or optimized_ssm is None:
        raise RuntimeError(
            "K1H requires official optimized CUDA kernels to establish "
            "the backend numerical envelope"
        )

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

    optimized = RealParityRunner(
        model=model,
        dtype=torch.float16,
        device=device,
    )
    fallback = RealParityRunner(
        model=model,
        dtype=torch.float16,
        device=device,
    )
    vn97 = RealParityRunner(
        model=model,
        dtype=torch.float16,
        device=device,
    )

    official_backend_drift = _metric_template()
    vn97_fallback_drift = _metric_template()
    fallback_scales = _metric_template()
    all_argmax_equal = True
    probe_count = 0
    token_count = 0

    try:
        for prompt in _default_prompts():
            ids = tokenizer(
                prompt,
                add_special_tokens=False,
                return_attention_mask=False,
            )["input_ids"]
            tokens = [int(value) for value in ids[:max_prompt_tokens]]
            if not tokens:
                continue

            optimized.reset()
            fallback.reset()
            vn97.reset()

            optimized_out: tuple[torch.Tensor, ...] | None = None
            fallback_out: tuple[torch.Tensor, ...] | None = None
            vn97_out: tuple[torch.Tensor, ...] | None = None

            for token_id in tokens:
                official_mamba2_module.causal_conv1d_update = optimized_conv
                official_mamba2_module.selective_state_update = optimized_ssm
                optimized_out = optimized.source_step(token_id)

                official_mamba2_module.causal_conv1d_update = None
                official_mamba2_module.selective_state_update = None
                fallback_out = fallback.source_step(token_id)

                vn97_out = vn97.vn97_step(token_id)

                _update_metric(
                    official_backend_drift,
                    left=optimized_out,
                    right=fallback_out,
                )
                _update_metric(
                    vn97_fallback_drift,
                    left=fallback_out,
                    right=vn97_out,
                )
                for name, value in (
                    ("logits", fallback_out[0]),
                    ("layer0_hidden", fallback_out[1]),
                    ("layer0_conv_state", fallback_out[2]),
                    ("layer0_ssm_state", fallback_out[3]),
                ):
                    fallback_scales[name] = max(
                        fallback_scales[name],
                        _max_abs_value(value),
                    )
                token_count += 1

            probe_count += 1

            for _ in range(continuation_tokens):
                assert optimized_out is not None
                assert fallback_out is not None
                assert vn97_out is not None

                optimized_token = int(optimized_out[0].argmax(dim=-1).item())
                fallback_token = int(fallback_out[0].argmax(dim=-1).item())
                vn97_token = int(vn97_out[0].argmax(dim=-1).item())
                if not (
                    optimized_token == fallback_token == vn97_token
                ):
                    all_argmax_equal = False

                # Advance all three trajectories with one common token so
                # numerical comparisons remain input-identical.
                common_token = fallback_token

                official_mamba2_module.causal_conv1d_update = optimized_conv
                official_mamba2_module.selective_state_update = optimized_ssm
                optimized_out = optimized.source_step(common_token)

                official_mamba2_module.causal_conv1d_update = None
                official_mamba2_module.selective_state_update = None
                fallback_out = fallback.source_step(common_token)

                vn97_out = vn97.vn97_step(common_token)

                _update_metric(
                    official_backend_drift,
                    left=optimized_out,
                    right=fallback_out,
                )
                _update_metric(
                    vn97_fallback_drift,
                    left=fallback_out,
                    right=vn97_out,
                )
                for name, value in (
                    ("logits", fallback_out[0]),
                    ("layer0_hidden", fallback_out[1]),
                    ("layer0_conv_state", fallback_out[2]),
                    ("layer0_ssm_state", fallback_out[3]),
                ):
                    fallback_scales[name] = max(
                        fallback_scales[name],
                        _max_abs_value(value),
                    )
                token_count += 1
    finally:
        official_mamba2_module.causal_conv1d_update = optimized_conv
        official_mamba2_module.selective_state_update = optimized_ssm

    if probe_count <= 0 or token_count <= 0:
        raise RuntimeError("K1H produced no probes")

    fp16_eps = float(torch.finfo(torch.float16).eps)
    envelope: dict[str, dict[str, object]] = {}
    envelope_pass = True
    for name in official_backend_drift:
        baseline = float(official_backend_drift[name])
        target = float(vn97_fallback_drift[name])
        scale = max(1.0, float(fallback_scales[name]))

        # The official optimized-vs-fallback delta is the empirical backend
        # envelope. One additional FP16 epsilon at the observed tensor scale
        # is allowed only as ordering/rounding slack.
        rounding_slack = fp16_eps * scale
        limit = baseline + rounding_slack
        passed = target <= limit
        envelope_pass = envelope_pass and passed
        envelope[name] = {
            "official_optimized_vs_fallback_max_abs_error": baseline,
            "vn97_vs_official_fallback_max_abs_error": target,
            "fallback_max_abs_scale": scale,
            "fp16_rounding_slack": rounding_slack,
            "allowed_limit": limit,
            "passed": passed,
        }

    passed = envelope_pass and all_argmax_equal

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
        "probe_cases": probe_count,
        "tokens_compared": token_count,
        "continuation_tokens_per_case": continuation_tokens,
        "all_argmax_equal": all_argmax_equal,
        "fp16_epsilon": fp16_eps,
        "envelope": envelope,
        "envelope_pass": envelope_pass,
        "source_to_vn97_backend_envelope_passed": passed,
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
        default=Path(
            "/kaggle/working/vn97-k1h-backend-envelope.json"
        ),
    )
    parser.add_argument("--max-prompt-tokens", type=int, default=12)
    parser.add_argument("--continuation-tokens", type=int, default=4)
    args = parser.parse_args()

    receipt = run_backend_envelope(
        source_root=args.source_root.resolve(strict=True),
        evidence_path=args.evidence.resolve(strict=True),
        output_path=args.output,
        max_prompt_tokens=args.max_prompt_tokens,
        continuation_tokens=args.continuation_tokens,
    )
    print(json.dumps(receipt, sort_keys=True))
    if receipt["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
