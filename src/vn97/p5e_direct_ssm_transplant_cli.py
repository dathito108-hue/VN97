from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

import torch

from .config import VN97Config
from .model import VN97LanguageCore
from .p5d_distillation import (
    TEACHER_D_MODEL,
    TEACHER_D_STATE,
    TEACHER_LAYERS,
    TEACHER_REPO,
)
from .p5d3_relational_distillation import P5D3B_FLOAT_ARTIFACT_SCHEMA
from .quantization import set_float_shadow_mode
from .tokenizer import CONTROL_TOKENS, VN97Tokenizer, VN97TokenizerPackage
from .training_cli import _atomic_write


P5E1_SCHEMA = "VN97P5E1"
P5E1_FLOAT_SCHEMA = "VN97P5E1FLOAT1"
P5E1_PROFILE_ID = "vn97-p5e1-direct-falcon3-mamba-operator-transplant-v1"


class VN97P5E1Error(RuntimeError):
    pass


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Directly graft Falcon3-Mamba selective-SSM operators into an "
            "existing 309M VN97 float-shadow student without autoregressive "
            "teacher generation. The result remains a native VN97 model and "
            "requires a short calibration stage before promotion."
        )
    )
    parser.add_argument("--teacher-snapshot", required=True)
    parser.add_argument("--base-student-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--blend", type=float, default=0.35)
    parser.add_argument("--svd-device", default="cuda:0")
    parser.add_argument("--smoke-device", default="cuda:0")
    parser.add_argument("--assess-only", action="store_true")
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise VN97P5E1Error(f"{path.name} must contain a JSON object")
    return value


def _source_dependencies():
    try:
        from safetensors import safe_open
        from tokenizers import Tokenizer
    except Exception as exc:
        raise VN97P5E1Error(
            "P5E1 requires safetensors and tokenizers"
        ) from exc
    return safe_open, Tokenizer


class _TensorStore:
    def __init__(self, root: Path) -> None:
        safe_open, _ = _source_dependencies()
        self._safe_open = safe_open
        self.root = root
        index_path = root / "model.safetensors.index.json"
        single_path = root / "model.safetensors"
        self.weight_map: dict[str, str] = {}
        if index_path.is_file():
            index = _json(index_path)
            raw = index.get("weight_map")
            if not isinstance(raw, dict):
                raise VN97P5E1Error("teacher safetensors index has no weight_map")
            self.weight_map = {
                str(name): str(filename)
                for name, filename in raw.items()
            }
        elif single_path.is_file():
            with self._safe_open(
                str(single_path),
                framework="pt",
                device="cpu",
            ) as handle:
                self.weight_map = {
                    str(name): single_path.name
                    for name in handle.keys()
                }
        else:
            raise VN97P5E1Error(
                "teacher snapshot must contain model.safetensors or "
                "model.safetensors.index.json"
            )

    def has(self, name: str) -> bool:
        return name in self.weight_map

    def tensor(self, name: str) -> torch.Tensor:
        filename = self.weight_map.get(name)
        if filename is None:
            raise VN97P5E1Error(f"teacher tensor missing: {name}")
        path = self.root / filename
        if not path.is_file():
            raise VN97P5E1Error(f"teacher shard missing: {filename}")
        with self._safe_open(
            str(path),
            framework="pt",
            device="cpu",
        ) as handle:
            return handle.get_tensor(name).detach().cpu()


def _source_spec(config: dict[str, Any]) -> dict[str, int]:
    hidden = int(config.get("hidden_size", config.get("d_model", 0)))
    inner = int(config.get("intermediate_size", config.get("d_inner", 0)))
    layers = int(config.get("num_hidden_layers", config.get("n_layer", 0)))
    state = int(config.get("state_size", 16))
    raw_dt_rank = config.get("time_step_rank", math.ceil(hidden / 16))
    if isinstance(raw_dt_rank, str):
        if raw_dt_rank != "auto":
            raise VN97P5E1Error("unsupported teacher time_step_rank string")
        dt_rank = math.ceil(hidden / 16)
    else:
        dt_rank = int(raw_dt_rank)
    vocab = int(config.get("vocab_size", 0))
    return {
        "hidden": hidden,
        "inner": inner,
        "layers": layers,
        "state": state,
        "dt_rank": dt_rank,
        "vocab": vocab,
    }


def _load_base_student(
    root: Path,
) -> tuple[
    VN97LanguageCore,
    VN97Tokenizer,
    bytes,
    dict[str, Any],
    str,
]:
    resolved = root.resolve(strict=True)
    model_path = resolved / "student-float.pt"
    tokenizer_path = resolved / "tokenizer.vn97tk1"
    report_path = resolved / "p5d3b-report.json"
    if not model_path.is_file() or not tokenizer_path.is_file():
        raise VN97P5E1Error(
            "base student must contain student-float.pt and tokenizer.vn97tk1"
        )
    report = _json(report_path) if report_path.is_file() else {}
    payload = torch.load(
        model_path,
        map_location="cpu",
        weights_only=False,
    )
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != P5D3B_FLOAT_ARTIFACT_SCHEMA
        or payload.get("float_shadow_required") is not True
    ):
        raise VN97P5E1Error(
            "base student must be a P5D3B float-shadow artifact"
        )
    config_raw = payload.get("config")
    state_dict = payload.get("state_dict")
    if not isinstance(config_raw, dict) or not isinstance(state_dict, dict):
        raise VN97P5E1Error("base student payload is incomplete")
    config = VN97Config(**config_raw)
    if (
        config.d_model != 1536
        or config.n_layers != 32
        or config.d_state != 16
    ):
        raise VN97P5E1Error(
            "P5E1 requires the canonical 1536x32x16 VN97 student"
        )
    tokenizer_bytes = tokenizer_path.read_bytes()
    package = VN97TokenizerPackage.from_bytes(tokenizer_bytes)
    if package.vocab_size != config.vocab_size:
        raise VN97P5E1Error("base tokenizer/model vocabulary mismatch")
    model = VN97LanguageCore(config)
    set_float_shadow_mode(model, True)
    model.load_state_dict(state_dict)
    return (
        model,
        VN97Tokenizer(package),
        tokenizer_bytes,
        report,
        _sha256(model_path),
    )


def _decode_target_token(
    tokenizer: VN97Tokenizer,
    token_id: int,
) -> str:
    if 0 <= token_id < len(CONTROL_TOKENS):
        return CONTROL_TOKENS[token_id]
    payload = tokenizer.decode_bytes([token_id], skip_control=True)
    try:
        return payload.decode("utf-8")
    except UnicodeDecodeError:
        return payload.decode("latin-1")


def _rms_match_rows(
    source: torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    source_f = source.float()
    reference_f = reference.float()
    if source_f.ndim == 1:
        source_rms = source_f.square().mean().sqrt().clamp_min(1e-8)
        reference_rms = reference_f.square().mean().sqrt().clamp_min(1e-8)
        return source_f * (reference_rms / source_rms)
    if source_f.ndim != 2:
        raise VN97P5E1Error("RMS matching only supports rank-1/2 tensors")
    source_rms = source_f.square().mean(dim=1).sqrt().clamp_min(1e-8)
    reference_rms = reference_f.square().mean(dim=1).sqrt().clamp_min(1e-8)
    return source_f * (reference_rms / source_rms).unsqueeze(1)


def _blend(
    reference: torch.Tensor,
    transplanted: torch.Tensor,
    alpha: float,
    *,
    rms_match: bool = True,
) -> torch.Tensor:
    src = transplanted.float()
    ref = reference.detach().float()
    if tuple(src.shape) != tuple(ref.shape):
        raise VN97P5E1Error(
            f"transplant shape mismatch: {tuple(src.shape)} != {tuple(ref.shape)}"
        )
    if rms_match:
        src = _rms_match_rows(src, ref)
    return (
        ref * (1.0 - alpha)
        + src * alpha
    ).to(dtype=reference.dtype)


def _hidden_indices(
    embedding: torch.Tensor,
    *,
    target_width: int,
) -> torch.Tensor:
    if embedding.ndim != 2 or target_width > embedding.shape[1]:
        raise VN97P5E1Error("invalid teacher embedding geometry")
    energy = embedding.float().square().mean(dim=0)
    chosen = torch.topk(
        energy,
        k=target_width,
        largest=True,
        sorted=False,
    ).indices
    return torch.sort(chosen).values


def _inner_indices(
    store: _TensorStore,
    *,
    layer: int,
    inner: int,
    target_width: int,
) -> torch.Tensor:
    mixer = f"backbone.layers.{layer}.mixer"
    in_proj = store.tensor(f"{mixer}.in_proj.weight").float()
    if tuple(in_proj.shape)[0] != inner * 2:
        raise VN97P5E1Error("teacher in_proj geometry mismatch")
    signal = in_proj[:inner]
    gate = in_proj[inner:]
    energy = (
        signal.square().mean(dim=1)
        + gate.square().mean(dim=1)
    )
    del in_proj, signal, gate

    out_proj = store.tensor(f"{mixer}.out_proj.weight").float()
    energy = energy + out_proj.square().mean(dim=0)
    del out_proj

    x_proj = store.tensor(f"{mixer}.x_proj.weight").float()
    energy = energy + x_proj.square().mean(dim=0)
    del x_proj

    dt_proj = store.tensor(f"{mixer}.dt_proj.weight").float()
    energy = energy + dt_proj.square().mean(dim=1)
    del dt_proj

    if store.has(f"{mixer}.D"):
        d_skip = store.tensor(f"{mixer}.D").float()
        energy = energy + d_skip.square()
        del d_skip
    if store.has(f"{mixer}.conv1d.weight"):
        conv = store.tensor(f"{mixer}.conv1d.weight").float()
        energy = energy + conv.flatten(1).square().mean(dim=1)
        del conv

    chosen = torch.topk(
        energy,
        k=target_width,
        largest=True,
        sorted=False,
    ).indices
    return torch.sort(chosen).values


def _project_norm(
    source: torch.Tensor,
    hidden_idx: torch.Tensor,
) -> torch.Tensor:
    return source.float().index_select(0, hidden_idx)


def _map_layer(
    store: _TensorStore,
    *,
    source_layer: int,
    hidden_idx: torch.Tensor,
    inner_idx: torch.Tensor,
    state: int,
    dt_rank: int,
) -> dict[str, torch.Tensor]:
    prefix = f"backbone.layers.{source_layer}"
    mixer = f"{prefix}.mixer"

    in_proj = store.tensor(f"{mixer}.in_proj.weight").float()
    inner = int(in_proj.shape[0] // 2)
    signal = (
        in_proj[:inner]
        .index_select(0, inner_idx)
        .index_select(1, hidden_idx)
    )
    gate = (
        in_proj[inner:]
        .index_select(0, inner_idx)
        .index_select(1, hidden_idx)
    )
    target_in = torch.cat([signal, gate], dim=0)
    del in_proj, signal, gate

    x_proj = store.tensor(f"{mixer}.x_proj.weight").float()
    if x_proj.shape[0] < dt_rank + state * 2:
        raise VN97P5E1Error("teacher x_proj geometry mismatch")
    dt_low = x_proj[:dt_rank].index_select(1, inner_idx)
    b = x_proj[dt_rank:dt_rank + state].index_select(1, inner_idx)
    c = x_proj[dt_rank + state:dt_rank + state * 2].index_select(
        1,
        inner_idx,
    )

    dt_high = (
        store.tensor(f"{mixer}.dt_proj.weight")
        .float()
        .index_select(0, inner_idx)
    )
    dt = dt_high @ dt_low
    del x_proj, dt_low, dt_high

    out = (
        store.tensor(f"{mixer}.out_proj.weight")
        .float()
        .index_select(0, hidden_idx)
        .index_select(1, inner_idx)
    )

    mapped = {
        "norm.weight":
            _project_norm(
                store.tensor(f"{prefix}.norm.weight"),
                hidden_idx,
            ),
        "core.a_log":
            store.tensor(f"{mixer}.A_log")
            .float()
            .index_select(0, inner_idx),
        "core.in_proj.weight":
            target_in,
        "core.dt_proj.weight":
            dt,
        "core.dt_proj.bias":
            store.tensor(f"{mixer}.dt_proj.bias")
            .float()
            .index_select(0, inner_idx),
        "core.b_proj.weight":
            b,
        "core.c_proj.weight":
            c,
        "core.out_proj.weight":
            out,
    }
    return mapped


def _lexical_embedding(
    *,
    source_embedding: torch.Tensor,
    source_tokenizer: Any,
    target_tokenizer: VN97Tokenizer,
    hidden_idx: torch.Tensor,
) -> torch.Tensor:
    source = source_embedding.detach().float()
    rows: list[torch.Tensor] = []
    for token_id in range(target_tokenizer.vocab_size):
        text = _decode_target_token(target_tokenizer, token_id)
        encoded = source_tokenizer.encode(
            text,
            add_special_tokens=False,
        )
        ids = [
            int(value)
            for value in encoded.ids
            if 0 <= int(value) < source.shape[0]
        ]
        if not ids:
            ids = [0]
        index = torch.tensor(ids, dtype=torch.long)
        row = source.index_select(0, index).mean(dim=0)
        rows.append(row.index_select(0, hidden_idx))
    return torch.stack(rows, dim=0)


def _refactor_embedding(
    *,
    model: VN97LanguageCore,
    teacher_matrix: torch.Tensor,
    alpha: float,
    device: str,
) -> float:
    if not hasattr(model.embedding, "token_factors"):
        raise VN97P5E1Error("P5E1 requires factorized VN97 embedding")
    base_matrix = (
        model.embedding.token_factors.detach().float()
        @ model.embedding.projection.detach().float()
    )
    mapped = _rms_match_rows(teacher_matrix, base_matrix)
    blended = base_matrix * (1.0 - alpha) + mapped * alpha
    rank = int(model.embedding.token_factors.shape[1])
    svd_input = blended.to(device)
    u, s, vh = torch.linalg.svd(
        svd_input,
        full_matrices=False,
    )
    u = u[:, :rank]
    s = s[:rank]
    vh = vh[:rank]
    factors = (u * s.unsqueeze(0)).cpu()
    projection = vh.cpu()
    reconstructed = factors @ projection
    relative_mse = float(
        (reconstructed - blended).square().mean()
        / blended.square().mean().clamp_min(1e-12)
    )
    with torch.no_grad():
        model.embedding.token_factors.copy_(
            factors.to(model.embedding.token_factors.dtype)
        )
        model.embedding.projection.copy_(
            projection.to(model.embedding.projection.dtype)
        )
    return relative_mse


def _apply_layer(
    target,
    mapped: dict[str, torch.Tensor],
    *,
    alpha: float,
) -> None:
    with torch.no_grad():
        target.norm.weight.copy_(
            _blend(
                target.norm.weight,
                mapped["norm.weight"],
                alpha,
                rms_match=False,
            )
        )
        target.core.a_log.copy_(
            _blend(
                target.core.a_log,
                mapped["core.a_log"],
                alpha,
                rms_match=False,
            )
        )
        target.core.in_proj.weight.copy_(
            _blend(
                target.core.in_proj.weight,
                mapped["core.in_proj.weight"],
                alpha,
            )
        )
        target.core.dt_proj.weight.copy_(
            _blend(
                target.core.dt_proj.weight,
                mapped["core.dt_proj.weight"],
                alpha,
            )
        )
        if target.core.dt_proj.bias is None:
            raise VN97P5E1Error("VN97 dt_proj unexpectedly has no bias")
        target.core.dt_proj.bias.copy_(
            _blend(
                target.core.dt_proj.bias,
                mapped["core.dt_proj.bias"],
                alpha,
                rms_match=False,
            )
        )
        target.core.b_proj.weight.copy_(
            _blend(
                target.core.b_proj.weight,
                mapped["core.b_proj.weight"],
                alpha,
            )
        )
        target.core.c_proj.weight.copy_(
            _blend(
                target.core.c_proj.weight,
                mapped["core.c_proj.weight"],
                alpha,
            )
        )
        target.core.out_proj.weight.copy_(
            _blend(
                target.core.out_proj.weight,
                mapped["core.out_proj.weight"],
                alpha,
            )
        )


def _finite_smoke(
    model: VN97LanguageCore,
    tokenizer: VN97Tokenizer,
    *,
    device: str,
) -> dict[str, Any]:
    model.to(device)
    set_float_shadow_mode(model, True)
    model.eval()
    ids = tokenizer.encode(
        "VN97",
        add_bos=True,
        add_text_tag=True,
    )[:8]
    input_ids = torch.tensor(
        [ids],
        dtype=torch.long,
        device=device,
    )
    with torch.inference_mode():
        logits, _ = model(input_ids)
    finite = bool(torch.isfinite(logits).all())
    max_abs = float(logits.abs().max().item())
    model.cpu()
    torch.cuda.empty_cache()
    if not finite:
        raise VN97P5E1Error("direct transplant produced non-finite logits")
    return {
        "finite": True,
        "max_abs_logit": max_abs,
        "tokens": len(ids),
    }


def _write_torch_atomic(path: Path, payload: dict[str, Any]) -> None:
    temp = path.with_name(path.name + ".tmp")
    temp.unlink(missing_ok=True)
    torch.save(payload, temp)
    os.replace(temp, path)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not 0.0 < args.blend <= 1.0:
        raise VN97P5E1Error("blend must satisfy 0 < blend <= 1")

    teacher_root = Path(args.teacher_snapshot).resolve(strict=True)
    config = _json(teacher_root / "config.json")
    spec = _source_spec(config)
    if (
        spec["hidden"] != TEACHER_D_MODEL
        or spec["layers"] != TEACHER_LAYERS
        or spec["state"] != TEACHER_D_STATE
        or spec["inner"] < 1536
    ):
        raise VN97P5E1Error(
            "teacher snapshot does not match the locked Falcon3-Mamba contract"
        )

    model, tokenizer, tokenizer_bytes, base_report, base_sha = (
        _load_base_student(Path(args.base_student_dir))
    )
    store = _TensorStore(teacher_root)

    required_probe = (
        "backbone.embeddings.weight",
        "backbone.norm_f.weight",
        "backbone.layers.0.norm.weight",
        "backbone.layers.0.mixer.A_log",
        "backbone.layers.0.mixer.in_proj.weight",
        "backbone.layers.0.mixer.x_proj.weight",
        "backbone.layers.0.mixer.dt_proj.weight",
        "backbone.layers.0.mixer.dt_proj.bias",
        "backbone.layers.0.mixer.out_proj.weight",
    )
    missing = [name for name in required_probe if not store.has(name)]
    if missing:
        raise VN97P5E1Error(
            "teacher tensor namespace is incompatible: " + ", ".join(missing)
        )

    print(
        "VN97 P5E1 COMPATIBILITY "
        f"teacher_hidden={spec['hidden']} "
        f"teacher_inner={spec['inner']} "
        f"teacher_layers={spec['layers']} "
        f"teacher_state={spec['state']} "
        f"target_hidden={model.config.d_model} "
        f"target_layers={model.config.n_layers} "
        f"target_state={model.config.d_state} "
        f"blend={args.blend:.3f}",
        flush=True,
    )
    print(
        "VN97 P5E1 STRATEGY "
        "hidden_projection=embedding_energy_coordinate_selection "
        "layer_compression=64_to_32_pair_endpoints "
        "inner_projection=per_layer_energy_selection "
        "runtime=single_native_vn97 "
        "teacher_generation=false",
        flush=True,
    )
    if args.assess_only:
        print(
            "VN97P5E1 status=STRUCTURALLY_FEASIBLE calibration_required=true",
            flush=True,
        )
        return 0

    source_embedding = store.tensor("backbone.embeddings.weight")
    hidden_idx = _hidden_indices(
        source_embedding,
        target_width=model.config.d_model,
    )
    hidden_idx_sha = hashlib.sha256(
        hidden_idx.numpy().tobytes()
    ).hexdigest()

    _, SourceTokenizer = _source_dependencies()
    tokenizer_json = teacher_root / "tokenizer.json"
    if not tokenizer_json.is_file():
        raise VN97P5E1Error(
            "teacher snapshot tokenizer.json is required for lexical graft"
        )
    source_tokenizer = SourceTokenizer.from_file(str(tokenizer_json))
    teacher_lexical = _lexical_embedding(
        source_embedding=source_embedding,
        source_tokenizer=source_tokenizer,
        target_tokenizer=tokenizer,
        hidden_idx=hidden_idx,
    )
    embedding_relative_mse = _refactor_embedding(
        model=model,
        teacher_matrix=teacher_lexical,
        alpha=args.blend,
        device=args.svd_device,
    )
    del teacher_lexical

    with torch.no_grad():
        teacher_final_norm = _project_norm(
            store.tensor("backbone.norm_f.weight"),
            hidden_idx,
        )
        model.final_norm.weight.copy_(
            _blend(
                model.final_norm.weight,
                teacher_final_norm,
                args.blend,
                rms_match=False,
            )
        )

    layer_map: list[dict[str, Any]] = []
    for target_layer, layer in enumerate(model.layers):
        source_layer = min(
            spec["layers"] - 1,
            target_layer * 2 + 1,
        )
        inner_idx = _inner_indices(
            store,
            layer=source_layer,
            inner=spec["inner"],
            target_width=model.config.d_model,
        )
        mapped = _map_layer(
            store,
            source_layer=source_layer,
            hidden_idx=hidden_idx,
            inner_idx=inner_idx,
            state=spec["state"],
            dt_rank=spec["dt_rank"],
        )
        _apply_layer(
            layer,
            mapped,
            alpha=args.blend,
        )
        layer_map.append(
            {
                "target_layer": target_layer,
                "source_layer": source_layer,
                "inner_index_sha256": hashlib.sha256(
                    inner_idx.numpy().tobytes()
                ).hexdigest(),
            }
        )
        if (
            (target_layer + 1) % 4 == 0
            or target_layer + 1 == model.config.n_layers
        ):
            print(
                "VN97 P5E1 LAYER "
                f"{target_layer + 1}/{model.config.n_layers} "
                f"source_layer={source_layer}",
                flush=True,
            )
        del mapped, inner_idx

    smoke = _finite_smoke(
        model,
        tokenizer,
        device=args.smoke_device,
    )

    output = Path(args.output_dir)
    if output.exists() and (
        output.is_symlink()
        or not output.is_dir()
        or any(output.iterdir())
    ):
        raise VN97P5E1Error("output-dir must be new or empty")
    output.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "base": {
            "student_sha256": base_sha,
            "status": base_report.get("status"),
        },
        "blend": float(args.blend),
        "calibration_required": True,
        "embedding_factorization_relative_mse": embedding_relative_mse,
        "hidden_index_sha256": hidden_idx_sha,
        "layer_map": layer_map,
        "profile_id": P5E1_PROFILE_ID,
        "schema": P5E1_SCHEMA,
        "smoke": smoke,
        "source": {
            "repo": TEACHER_REPO,
            "revision": teacher_root.name,
            "spec": spec,
        },
        "status": "DIRECT_SSM_TRANSPLANT_READY_FOR_CALIBRATION",
        "target": {
            "d_model": model.config.d_model,
            "d_state": model.config.d_state,
            "layers": model.config.n_layers,
            "parameters": sum(int(p.numel()) for p in model.parameters()),
        },
        "unsupported_teacher_operators": [
            "depthwise_causal_conv1d",
            "D_skip",
        ],
    }
    report_bytes = (
        json.dumps(
            report,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    report_sha = hashlib.sha256(report_bytes).hexdigest()
    _atomic_write(output / "p5e1-report.json", report_bytes)

    model.cpu()
    payload = {
        "base_student_sha256": base_sha,
        "blend": float(args.blend),
        "config": asdict(model.config),
        "float_shadow_required": True,
        "hidden_index_sha256": hidden_idx_sha,
        "profile_id": P5E1_PROFILE_ID,
        "report_sha256": report_sha,
        "schema": P5E1_FLOAT_SCHEMA,
        "source_revision": teacher_root.name,
        "state_dict": model.state_dict(),
        "tokenizer_sha256": hashlib.sha256(tokenizer_bytes).hexdigest(),
    }
    _write_torch_atomic(
        output / "student-float.pt",
        payload,
    )
    _atomic_write(
        output / "tokenizer.vn97tk1",
        tokenizer_bytes,
    )

    files = {
        "p5e1-report.json": output / "p5e1-report.json",
        "student-float.pt": output / "student-float.pt",
        "tokenizer.vn97tk1": output / "tokenizer.vn97tk1",
    }
    sums = "".join(
        f"{_sha256(path)}  {name}\n"
        for name, path in sorted(files.items())
    ).encode("ascii")
    _atomic_write(output / "SHA256SUMS", sums)

    print(
        "VN97P5E1 "
        "status=DIRECT_SSM_TRANSPLANT_READY_FOR_CALIBRATION "
        f"blend={args.blend:.3f} "
        f"layers={model.config.n_layers} "
        f"parameters={sum(int(p.numel()) for p in model.parameters())} "
        f"embedding_mse={embedding_relative_mse:.8f} "
        f"smoke_finite={str(bool(smoke['finite'])).lower()}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"vn97-p5e-direct-transplant: {exc}", file=sys.stderr)
        raise
