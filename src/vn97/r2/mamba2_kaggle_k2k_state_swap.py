from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .mamba2_g03_capsule import load_g03_capsule
from .mamba2_kaggle_k2a_ort import (
    _canonical_json,
    _epsilon_metrics,
)
from .mamba2_kaggle_k2f_dbx_operands import _verify_receipt
from .mamba2_kaggle_k2j_closed_loop import (
    SCHEMA as K2J_SCHEMA,
    TOP_K,
    _topk_ids,
)
from .mamba2_onnx import (
    Mamba2OnnxConfig,
    VN97Mamba2StepOnnx,
    verify_mamba2_g04_bundle,
)


SCHEMA = "VN97M2K2KSWAP1"


def _earliest_divergence(k2j: dict[str, Any]) -> dict[str, Any]:
    candidates = [
        prompt
        for prompt in k2j.get("prompts", [])
        if prompt.get("first_divergence_step") is not None
    ]
    if not candidates:
        raise ValueError("K2K requires at least one K2J closed-loop divergence")
    candidates.sort(
        key=lambda item: (
            int(item["first_divergence_step"]),
            int(item["prompt_index"]),
        )
    )
    return candidates[0]


def _classify_top1(
    token_id: int,
    *,
    reference_token: int,
    ort_token: int,
) -> str:
    if token_id == reference_token:
        return "reference"
    if token_id == ort_token:
        return "ort"
    return "other"


def _diagnose(matrix: dict[str, dict[str, Any]]) -> str:
    def label(runtime: str, conv: str, ssm: str) -> str:
        return str(matrix[f"{runtime}_{conv}{ssm}"]["top1_class"])

    py_all_ref = all(
        label("py", conv, ssm) == "reference"
        for conv in ("p", "o")
        for ssm in ("p", "o")
    )
    ort_all_ort = all(
        label("ort", conv, ssm) == "ort"
        for conv in ("p", "o")
        for ssm in ("p", "o")
    )
    if py_all_ref and ort_all_ort:
        return "runtime_step_dominant"

    ssm_py_ref = all(
        label(runtime, conv, "p") == "reference"
        for runtime in ("py", "ort")
        for conv in ("p", "o")
    )
    ssm_ort_ort = all(
        label(runtime, conv, "o") == "ort"
        for runtime in ("py", "ort")
        for conv in ("p", "o")
    )
    if ssm_py_ref and ssm_ort_ort:
        return "ssm_state_dominant"

    conv_py_ref = all(
        label(runtime, "p", ssm) == "reference"
        for runtime in ("py", "ort")
        for ssm in ("p", "o")
    )
    conv_ort_ort = all(
        label(runtime, "o", ssm) == "ort"
        for runtime in ("py", "ort")
        for ssm in ("p", "o")
    )
    if conv_py_ref and conv_ort_ort:
        return "conv_state_dominant"

    pp_ref = all(
        label(runtime, "p", "p") == "reference"
        for runtime in ("py", "ort")
    )
    oo_ort = all(
        label(runtime, "o", "o") == "ort"
        for runtime in ("py", "ort")
    )
    if pp_ref and oo_ort:
        return "accumulated_state_dominant_mixed"

    return "mixed_runtime_and_state"


def _record_logits(
    logits: np.ndarray,
    *,
    reference_logits: np.ndarray,
    reference_token: int,
    ort_token: int,
    epsilon: float,
) -> dict[str, Any]:
    values = logits.astype(np.float32, copy=False).reshape(-1)
    top1 = int(values.argmax())
    return {
        "top1": top1,
        "top1_class": _classify_top1(
            top1,
            reference_token=reference_token,
            ort_token=ort_token,
        ),
        "reference_token_logit": float(values[reference_token]),
        "ort_token_logit": float(values[ort_token]),
        "reference_minus_ort_logit": float(
            values[reference_token] - values[ort_token]
        ),
        "top5_ids": _topk_ids(logits, TOP_K),
        "vs_pytorch_pp": _epsilon_metrics(
            reference_logits,
            logits,
            epsilon=epsilon,
        ),
    }


@torch.inference_mode()
def _capture_pytorch_prestate(
    *,
    capsule_root: Path,
    prompt_token_ids: list[int],
    common_generated_before_current: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    device = torch.device("cuda:0")
    capsule = load_g03_capsule(
        capsule_root,
        verify_large_weight_sha256=False,
    )
    model = VN97Mamba2StepOnnx(
        Mamba2OnnxConfig.from_source_spec(capsule.spec),
        capsule.tensors,
    ).eval().to(device)

    conv, ssm = model.initial_state(1)
    for token_id in prompt_token_ids + common_generated_before_current:
        token = torch.tensor(
            [token_id],
            dtype=torch.long,
            device=device,
        )
        _, conv, ssm = model(token, conv, ssm)

    conv_np = conv.detach().cpu().numpy().copy()
    ssm_np = ssm.detach().cpu().numpy().copy()
    del model
    del capsule
    gc.collect()
    torch.cuda.empty_cache()
    return conv_np, ssm_np


def _ort_session(graph_path: Path):
    import onnxruntime as ort

    if hasattr(ort, "preload_dlls"):
        ort.preload_dlls()
    session = ort.InferenceSession(
        str(graph_path),
        providers=[
            (
                "CUDAExecutionProvider",
                {
                    "device_id": 0,
                    "arena_extend_strategy": "kSameAsRequested",
                },
            )
        ],
    )
    active = session.get_providers()
    if not active or active[0] != "CUDAExecutionProvider":
        raise RuntimeError(
            "K2K ORT session did not activate CUDAExecutionProvider"
        )
    return session, active[0]


def _capture_ort_prestate(
    *,
    graph_path: Path,
    prompt_token_ids: list[int],
    common_generated_before_current: list[int],
    zero_conv: np.ndarray,
    zero_ssm: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, str]:
    session, provider = _ort_session(graph_path)
    conv = np.zeros_like(zero_conv)
    ssm = np.zeros_like(zero_ssm)
    for token_id in prompt_token_ids + common_generated_before_current:
        _, conv, ssm = session.run(
            ["logits", "next_conv_state", "next_ssm_state"],
            {
                "input_ids": np.asarray([token_id], dtype=np.int64),
                "conv_state": conv,
                "ssm_state": ssm,
            },
        )
        conv = np.asarray(conv).copy()
        ssm = np.asarray(ssm).copy()
    del session
    gc.collect()
    return conv, ssm, provider


@torch.inference_mode()
def _run_pytorch_matrix(
    *,
    capsule_root: Path,
    current_token: int,
    py_conv: np.ndarray,
    py_ssm: np.ndarray,
    ort_conv: np.ndarray,
    ort_ssm: np.ndarray,
) -> dict[str, np.ndarray]:
    device = torch.device("cuda:0")
    capsule = load_g03_capsule(
        capsule_root,
        verify_large_weight_sha256=False,
    )
    model = VN97Mamba2StepOnnx(
        Mamba2OnnxConfig.from_source_spec(capsule.spec),
        capsule.tensors,
    ).eval().to(device)

    token = torch.tensor(
        [current_token],
        dtype=torch.long,
        device=device,
    )
    states = {
        "pp": (py_conv, py_ssm),
        "po": (py_conv, ort_ssm),
        "op": (ort_conv, py_ssm),
        "oo": (ort_conv, ort_ssm),
    }
    outputs: dict[str, np.ndarray] = {}
    for key, (conv_np, ssm_np) in states.items():
        conv = torch.from_numpy(conv_np).to(device)
        ssm = torch.from_numpy(ssm_np).to(device)
        logits, _, _ = model(token, conv, ssm)
        outputs[f"py_{key}"] = logits.detach().cpu().numpy().copy()

    del model
    del capsule
    gc.collect()
    torch.cuda.empty_cache()
    return outputs


def _run_ort_matrix(
    *,
    graph_path: Path,
    current_token: int,
    py_conv: np.ndarray,
    py_ssm: np.ndarray,
    ort_conv: np.ndarray,
    ort_ssm: np.ndarray,
) -> tuple[dict[str, np.ndarray], str]:
    session, provider = _ort_session(graph_path)
    token = np.asarray([current_token], dtype=np.int64)
    states = {
        "pp": (py_conv, py_ssm),
        "po": (py_conv, ort_ssm),
        "op": (ort_conv, py_ssm),
        "oo": (ort_conv, ort_ssm),
    }
    outputs: dict[str, np.ndarray] = {}
    for key, (conv, ssm) in states.items():
        logits, _, _ = session.run(
            ["logits", "next_conv_state", "next_ssm_state"],
            {
                "input_ids": token,
                "conv_state": conv,
                "ssm_state": ssm,
            },
        )
        outputs[f"ort_{key}"] = np.asarray(logits).copy()
    del session
    gc.collect()
    return outputs, provider


def run_k2k(
    *,
    capsule_root: Path,
    bundle_dir: Path,
    k2j_receipt_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("K2K requires CUDA")
    gpu_name = torch.cuda.get_device_name(0)
    if "T4" not in gpu_name.upper():
        raise RuntimeError(f"K2K expects Kaggle T4, got {gpu_name!r}")

    k2j = _verify_receipt(
        k2j_receipt_path,
        schema=K2J_SCHEMA,
        label="K2J receipt",
    )
    if k2j.get("closed_loop_exact_all_prompts") is not False:
        raise ValueError("K2K requires a real K2J closed-loop divergence")
    if int(k2j.get("closed_loop_exact_prompts", -1)) < 1:
        raise ValueError("K2K requires at least one exact K2J control prompt")
    if k2j.get("production_graph_changed") is not False:
        raise ValueError("K2K requires unchanged G0.4 production graph")

    manifest = verify_mamba2_g04_bundle(bundle_dir)
    for field in ("g04_manifest_id", "capsule_id", "source_weight_sha256"):
        manifest_field = "manifest_id" if field == "g04_manifest_id" else field
        if manifest.get(manifest_field) != k2j.get(field):
            raise ValueError(f"K2K {field} lineage mismatch")

    target = _earliest_divergence(k2j)
    divergence_step = int(target["first_divergence_step"])
    if divergence_step < 2:
        raise ValueError("K2K requires divergence after at least one common token")

    prompt_token_ids = [
        int(value) for value in target["prompt_token_ids"]
    ]
    reference_generated = [
        int(value) for value in target["reference_generated_token_ids"]
    ]
    ort_generated = [
        int(value) for value in target["ort_generated_token_ids"]
    ]
    if reference_generated[: divergence_step - 1] != ort_generated[
        : divergence_step - 1
    ]:
        raise ValueError("K2K divergence prefix is not actually common")

    before_current = reference_generated[: divergence_step - 2]
    current_token = reference_generated[divergence_step - 2]
    reference_token = reference_generated[divergence_step - 1]
    ort_token = ort_generated[divergence_step - 1]
    if reference_token == ort_token:
        raise ValueError("K2K target divergence token is not divergent")

    py_conv, py_ssm = _capture_pytorch_prestate(
        capsule_root=capsule_root,
        prompt_token_ids=prompt_token_ids,
        common_generated_before_current=before_current,
    )

    ort_conv, ort_ssm, provider = _capture_ort_prestate(
        graph_path=bundle_dir / "step.onnx",
        prompt_token_ids=prompt_token_ids,
        common_generated_before_current=before_current,
        zero_conv=py_conv,
        zero_ssm=py_ssm,
    )

    epsilon = float(np.finfo(np.float16).eps)
    prestate_metrics = {
        "conv_state": _epsilon_metrics(
            py_conv,
            ort_conv,
            epsilon=epsilon,
        ),
        "ssm_state": _epsilon_metrics(
            py_ssm,
            ort_ssm,
            epsilon=epsilon,
        ),
    }

    py_outputs = _run_pytorch_matrix(
        capsule_root=capsule_root,
        current_token=current_token,
        py_conv=py_conv,
        py_ssm=py_ssm,
        ort_conv=ort_conv,
        ort_ssm=ort_ssm,
    )
    reference_logits = py_outputs["py_pp"]
    if int(reference_logits.argmax()) != reference_token:
        raise RuntimeError(
            "K2K PyTorch PP replay does not reproduce K2J reference token"
        )

    ort_outputs, provider2 = _run_ort_matrix(
        graph_path=bundle_dir / "step.onnx",
        current_token=current_token,
        py_conv=py_conv,
        py_ssm=py_ssm,
        ort_conv=ort_conv,
        ort_ssm=ort_ssm,
    )
    if provider2 != provider:
        raise RuntimeError("K2K ORT provider changed between phases")
    if int(ort_outputs["ort_oo"].argmax()) != ort_token:
        raise RuntimeError(
            "K2K ORT OO replay does not reproduce K2J divergent token"
        )

    raw_outputs = {**py_outputs, **ort_outputs}
    matrix = {
        key: _record_logits(
            logits,
            reference_logits=reference_logits,
            reference_token=reference_token,
            ort_token=ort_token,
            epsilon=epsilon,
        )
        for key, logits in raw_outputs.items()
    }
    diagnosis = _diagnose(matrix)

    body: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "MEASURED",
        "gpu_name": gpu_name,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "ort_provider": provider,
        "k2j_receipt_id": k2j["receipt_id"],
        "g04_manifest_id": manifest["manifest_id"],
        "capsule_id": manifest["capsule_id"],
        "source_weight_sha256": manifest["source_weight_sha256"],
        "dtype": str(manifest["state_contract"]["dtype"]),
        "fp16_epsilon": epsilon,
        "prompt_index": int(target["prompt_index"]),
        "divergence_step": divergence_step,
        "common_prefix_length": int(target["common_prefix_length"]),
        "current_input_token": current_token,
        "reference_next_token": reference_token,
        "ort_next_token": ort_token,
        "prestate_metrics": prestate_metrics,
        "matrix": matrix,
        "diagnosis": diagnosis,
        "weights_changed": False,
        "architecture_changed": False,
        "production_graph_changed": False,
        "acceptance_threshold_defined": False,
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
    parser.add_argument("--capsule-root", required=True, type=Path)
    parser.add_argument("--bundle-dir", required=True, type=Path)
    parser.add_argument("--k2j-receipt", required=True, type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/kaggle/working/vn97-k2k-state-swap.json"),
    )
    args = parser.parse_args()

    receipt = run_k2k(
        capsule_root=args.capsule_root.resolve(strict=True),
        bundle_dir=args.bundle_dir.resolve(strict=True),
        k2j_receipt_path=args.k2j_receipt.resolve(strict=True),
        output_path=args.output,
    )
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
