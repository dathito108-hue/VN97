from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import torch
import torch.nn as nn

from .checkpoint import load_r2_checkpoint, sha256_file
from .model import VN97R2Model, VN97R2State
from .ssm import R2LayerState


R2_ONNX_BUNDLE_SCHEMA = "VN97R2ONNX1"
R2_ONNX_OPSET = 18
R2_ONNX_SUPPORTED_CHUNKS = (8, 16, 32, 64, 128)
R2_ONNX_INPUTS = ("input_ids", "conv_state", "ssm_state")
R2_ONNX_OUTPUTS = ("logits", "next_conv_state", "next_ssm_state")


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _bundle_identity(body: Mapping[str, object]) -> str:
    return _sha256_bytes(
        b"VN97R2ONNX1\0" + _canonical_json(dict(body))
    )


def _require_sha256(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(ch not in "0123456789abcdef" for ch in value)
    ):
        raise ValueError(f"{label} must be lowercase SHA-256")
    return value


def _strict_json(path: Path, *, label: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular non-symlink file")
    duplicates: list[str] = []

    def hook(pairs):
        out: dict[str, object] = {}
        for key, value in pairs:
            if key in out:
                duplicates.append(key)
            out[key] = value
        return out

    try:
        value = json.loads(
            path.read_text(encoding="utf-8", errors="strict"),
            object_pairs_hook=hook,
            parse_constant=lambda raw: (
                _ for _ in ()
            ).throw(ValueError(raw)),
        )
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        ValueError,
    ) as exc:
        raise ValueError(f"{label} must be strict UTF-8 JSON") from exc
    if duplicates or not isinstance(value, dict):
        raise ValueError(
            f"{label} must be one object without duplicate keys"
        )
    return value


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.unlink(missing_ok=True)
    temp.write_bytes(_canonical_json(payload) + b"\n")
    temp.replace(path)


def _profile_name(profile: str | int | None) -> str:
    if profile is None or profile == "deep":
        return "deep"
    if profile == "fast":
        return "fast"
    if isinstance(profile, int):
        return f"layers-{profile}"
    raise ValueError("invalid ONNX export profile")


def stack_r2_state(
    state: VN97R2State,
) -> tuple[torch.Tensor, torch.Tensor]:
    if state.active_layers <= 0:
        raise ValueError("state.active_layers must be positive")
    if len(state.layers) != state.active_layers:
        raise ValueError("state layer count mismatch")
    return (
        torch.stack(
            [layer.conv for layer in state.layers],
            dim=0,
        ),
        torch.stack(
            [layer.ssm for layer in state.layers],
            dim=0,
        ),
    )


def unstack_r2_state(
    conv_state: torch.Tensor,
    ssm_state: torch.Tensor,
    *,
    active_layers: int,
) -> VN97R2State:
    if conv_state.ndim != 4 or ssm_state.ndim != 4:
        raise ValueError(
            "stacked recurrent state must be rank-4 "
            "[layers,batch,inner,state]"
        )
    if conv_state.shape[0] != active_layers:
        raise ValueError("conv state active-layer count mismatch")
    if ssm_state.shape[0] != active_layers:
        raise ValueError("SSM state active-layer count mismatch")
    if conv_state.shape[:2] != ssm_state.shape[:2]:
        raise ValueError("conv/SSM layer and batch dimensions differ")
    return VN97R2State(
        layers=tuple(
            R2LayerState(
                conv=conv_state[index],
                ssm=ssm_state[index],
            )
            for index in range(active_layers)
        ),
        active_layers=active_layers,
    )


class _OnnxStateAdapter(nn.Module):
    def __init__(
        self,
        model: VN97R2Model,
        *,
        active_layers: int,
    ) -> None:
        super().__init__()
        self.model = model
        self.active_layers = active_layers

    def _state(
        self,
        conv_state: torch.Tensor,
        ssm_state: torch.Tensor,
    ) -> VN97R2State:
        return unstack_r2_state(
            conv_state,
            ssm_state,
            active_layers=self.active_layers,
        )

    @staticmethod
    def _stack(
        state: VN97R2State,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return stack_r2_state(state)


class VN97OnnxStepAdapter(_OnnxStateAdapter):
    def forward(
        self,
        input_ids: torch.Tensor,
        conv_state: torch.Tensor,
        ssm_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        state = self._state(conv_state, ssm_state)
        logits, next_state = self.model.step(
            input_ids,
            state,
            profile=self.active_layers,
        )
        next_conv, next_ssm = self._stack(next_state)
        return logits, next_conv, next_ssm


class VN97OnnxChunkAdapter(_OnnxStateAdapter):
    def __init__(
        self,
        model: VN97R2Model,
        *,
        active_layers: int,
        chunk_size: int,
    ) -> None:
        super().__init__(
            model,
            active_layers=active_layers,
        )
        if chunk_size not in R2_ONNX_SUPPORTED_CHUNKS:
            raise ValueError(
                "chunk_size must be one of "
                f"{R2_ONNX_SUPPORTED_CHUNKS}"
            )
        self.chunk_size = chunk_size

    def forward(
        self,
        input_ids: torch.Tensor,
        conv_state: torch.Tensor,
        ssm_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if input_ids.ndim != 2:
            raise ValueError("chunk input_ids must be [batch, chunk]")
        if input_ids.shape[1] != self.chunk_size:
            raise ValueError(
                f"expected chunk size {self.chunk_size}"
            )
        state = self._state(conv_state, ssm_state)
        logits, next_state = self.model(
            input_ids,
            state,
            profile=self.active_layers,
        )
        next_conv, next_ssm = self._stack(next_state)
        return logits, next_conv, next_ssm


def _state_contract(
    model: VN97R2Model,
    *,
    active_layers: int,
) -> dict[str, object]:
    return {
        "active_layers": active_layers,
        "conv_state": {
            "dtype": "float32",
            "shape": [
                active_layers,
                "batch",
                model.config.d_inner,
                max(model.config.d_conv - 1, 0),
            ],
        },
        "ssm_state": {
            "dtype": "float32",
            "shape": [
                active_layers,
                "batch",
                model.config.d_inner,
                model.config.d_state,
            ],
        },
        "state_is_explicit": True,
        "state_carry_semantics": "exact_between_graph_invocations",
    }


def _graph_record(
    *,
    path: Path,
    kind: str,
    sequence_length: int,
) -> dict[str, object]:
    return {
        "kind": kind,
        "filename": path.name,
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "sequence_length": sequence_length,
        "inputs": list(R2_ONNX_INPUTS),
        "outputs": list(R2_ONNX_OUTPUTS),
        "batch_axis_dynamic": True,
        "sequence_axis_dynamic": False,
    }


def _export_graph(
    adapter: nn.Module,
    *,
    input_ids: torch.Tensor,
    conv_state: torch.Tensor,
    ssm_state: torch.Tensor,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp = output_path.with_name(output_path.name + ".tmp")
    temp.unlink(missing_ok=True)
    torch.onnx.export(
        adapter,
        (
            input_ids,
            conv_state,
            ssm_state,
        ),
        temp,
        export_params=True,
        opset_version=R2_ONNX_OPSET,
        do_constant_folding=True,
        input_names=list(R2_ONNX_INPUTS),
        output_names=list(R2_ONNX_OUTPUTS),
        dynamic_axes={
            "input_ids": {0: "batch"},
            "conv_state": {1: "batch"},
            "ssm_state": {1: "batch"},
            "logits": {0: "batch"},
            "next_conv_state": {1: "batch"},
            "next_ssm_state": {1: "batch"},
        },
    )
    temp.replace(output_path)


@torch.inference_mode()
def export_r2_onnx_bundle(
    model: VN97R2Model,
    output_dir: Path,
    *,
    profile: str | int | None = "deep",
    chunk_sizes: Sequence[int] = (32,),
    checkpoint_sha256: str | None = None,
    checkpoint_stage: str | None = None,
    example_batch_size: int = 1,
) -> dict[str, object]:
    if example_batch_size <= 0:
        raise ValueError("example_batch_size must be positive")
    resolved_chunks = tuple(int(value) for value in chunk_sizes)
    if not resolved_chunks:
        raise ValueError("at least one chunk graph is required")
    if len(set(resolved_chunks)) != len(resolved_chunks):
        raise ValueError("chunk_sizes must be unique")
    if any(
        value not in R2_ONNX_SUPPORTED_CHUNKS
        for value in resolved_chunks
    ):
        raise ValueError(
            "chunk_sizes must be selected from "
            f"{R2_ONNX_SUPPORTED_CHUNKS}"
        )
    resolved_chunks = tuple(sorted(resolved_chunks))

    if output_dir.exists():
        if (
            output_dir.is_symlink()
            or not output_dir.is_dir()
            or any(output_dir.iterdir())
        ):
            raise ValueError("ONNX output directory must be new or empty")
    else:
        output_dir.mkdir(parents=True, exist_ok=False)
    output = output_dir.resolve(strict=True)

    model = model.cpu().float().eval()
    active_layers = model.resolve_active_layers(profile)
    example_state = model.initial_state(
        example_batch_size,
        device="cpu",
        dtype=torch.float32,
        profile=active_layers,
    )
    conv_state, ssm_state = stack_r2_state(example_state)

    step_path = output / "step.onnx"
    _export_graph(
        VN97OnnxStepAdapter(
            model,
            active_layers=active_layers,
        ).eval(),
        input_ids=torch.zeros(
            example_batch_size,
            dtype=torch.long,
        ),
        conv_state=conv_state,
        ssm_state=ssm_state,
        output_path=step_path,
    )

    graphs: list[dict[str, object]] = [
        _graph_record(
            path=step_path,
            kind="step",
            sequence_length=1,
        )
    ]
    for chunk_size in resolved_chunks:
        path = output / f"chunk-{chunk_size}.onnx"
        _export_graph(
            VN97OnnxChunkAdapter(
                model,
                active_layers=active_layers,
                chunk_size=chunk_size,
            ).eval(),
            input_ids=torch.zeros(
                example_batch_size,
                chunk_size,
                dtype=torch.long,
            ),
            conv_state=conv_state,
            ssm_state=ssm_state,
            output_path=path,
        )
        graphs.append(
            _graph_record(
                path=path,
                kind="chunk",
                sequence_length=chunk_size,
            )
        )

    if checkpoint_sha256 is not None:
        _require_sha256(
            checkpoint_sha256,
            label="checkpoint_sha256",
        )
    body: dict[str, object] = {
        "schema": R2_ONNX_BUNDLE_SCHEMA,
        "architecture_id": model.config.architecture_id,
        "architecture_fingerprint": model.config.fingerprint(),
        "config": asdict(model.config),
        "profile": _profile_name(profile),
        "active_layers": active_layers,
        "opset": R2_ONNX_OPSET,
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_stage": checkpoint_stage,
        "state_contract": _state_contract(
            model,
            active_layers=active_layers,
        ),
        "supported_chunk_sizes": list(resolved_chunks),
        "graphs": graphs,
        "same_weights_semantics": True,
        "quantization_used": False,
        "production_lowering_used": False,
    }
    manifest = dict(body)
    manifest["bundle_id"] = _bundle_identity(body)
    _atomic_json(output / "manifest.vn97onnx1.json", manifest)
    verify_r2_onnx_bundle(output)
    return manifest


def export_r2_checkpoint_onnx_bundle(
    checkpoint_path: Path,
    output_dir: Path,
    *,
    profile: str | int | None = "deep",
    chunk_sizes: Sequence[int] = (32,),
    example_batch_size: int = 1,
) -> dict[str, object]:
    if checkpoint_path.is_symlink():
        raise ValueError("checkpoint must not be a symlink")
    checkpoint = checkpoint_path.resolve(strict=True)
    model, evidence = load_r2_checkpoint(
        checkpoint,
        map_location="cpu",
    )
    return export_r2_onnx_bundle(
        model,
        output_dir,
        profile=profile,
        chunk_sizes=chunk_sizes,
        checkpoint_sha256=str(evidence["sha256"]),
        checkpoint_stage=str(evidence["stage"]),
        example_batch_size=example_batch_size,
    )


def load_r2_onnx_manifest(
    bundle_dir: Path,
) -> dict[str, object]:
    root = bundle_dir.resolve(strict=True)
    manifest = _strict_json(
        root / "manifest.vn97onnx1.json",
        label="R2 ONNX manifest",
    )
    if manifest.get("schema") != R2_ONNX_BUNDLE_SCHEMA:
        raise ValueError("R2 ONNX manifest schema mismatch")
    bundle_id = _require_sha256(
        manifest.get("bundle_id"),
        label="R2 ONNX bundle ID",
    )
    body = dict(manifest)
    body.pop("bundle_id", None)
    if bundle_id != _bundle_identity(body):
        raise ValueError("R2 ONNX bundle identity mismatch")
    return manifest


def verify_r2_onnx_bundle(
    bundle_dir: Path,
) -> dict[str, object]:
    if bundle_dir.is_symlink():
        raise ValueError("ONNX bundle directory must not be a symlink")
    root = bundle_dir.resolve(strict=True)
    manifest = load_r2_onnx_manifest(root)

    config = manifest.get("config")
    if not isinstance(config, dict):
        raise ValueError("R2 ONNX config is missing")
    from .config import VN97R2Config

    parsed = VN97R2Config(**config)
    if parsed.fingerprint() != manifest.get(
        "architecture_fingerprint"
    ):
        raise ValueError("R2 ONNX architecture fingerprint mismatch")
    active_layers = int(manifest.get("active_layers", -1))
    if not 1 <= active_layers <= parsed.n_layers:
        raise ValueError("R2 ONNX active layer count is invalid")
    if int(manifest.get("opset", -1)) != R2_ONNX_OPSET:
        raise ValueError("R2 ONNX opset mismatch")
    if manifest.get("same_weights_semantics") is not True:
        raise ValueError("R2 ONNX same-weight semantic lock missing")
    if manifest.get("quantization_used") is not False:
        raise ValueError("R2 ONNX E2 must remain unquantized")
    if manifest.get("production_lowering_used") is not False:
        raise ValueError("R2 ONNX E2 must precede production lowering")

    chunks = manifest.get("supported_chunk_sizes")
    if (
        not isinstance(chunks, list)
        or chunks != sorted(set(chunks))
        or any(
            type(value) is not int
            or value not in R2_ONNX_SUPPORTED_CHUNKS
            for value in chunks
        )
    ):
        raise ValueError("R2 ONNX supported chunk sizes are invalid")

    records = manifest.get("graphs")
    if not isinstance(records, list) or len(records) != len(chunks) + 1:
        raise ValueError("R2 ONNX graph manifest is incomplete")
    seen: set[str] = set()
    step_count = 0
    observed_chunks: list[int] = []
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("R2 ONNX graph record is invalid")
        filename = record.get("filename")
        if (
            not isinstance(filename, str)
            or not filename
            or Path(filename).name != filename
            or filename in seen
        ):
            raise ValueError("R2 ONNX graph filename is invalid")
        seen.add(filename)
        graph_path = root / filename
        if graph_path.is_symlink() or not graph_path.is_file():
            raise ValueError("R2 ONNX graph file is missing")
        if _sha256_file(graph_path) != record.get("sha256"):
            raise ValueError("R2 ONNX graph SHA-256 mismatch")
        if graph_path.stat().st_size != int(record.get("bytes", -1)):
            raise ValueError("R2 ONNX graph byte count mismatch")
        if record.get("inputs") != list(R2_ONNX_INPUTS):
            raise ValueError("R2 ONNX graph input contract mismatch")
        if record.get("outputs") != list(R2_ONNX_OUTPUTS):
            raise ValueError("R2 ONNX graph output contract mismatch")
        kind = record.get("kind")
        length = int(record.get("sequence_length", -1))
        if kind == "step":
            step_count += 1
            if length != 1 or filename != "step.onnx":
                raise ValueError("R2 ONNX step graph contract mismatch")
        elif kind == "chunk":
            if length not in chunks:
                raise ValueError("R2 ONNX chunk graph is undeclared")
            if filename != f"chunk-{length}.onnx":
                raise ValueError("R2 ONNX chunk filename mismatch")
            observed_chunks.append(length)
        else:
            raise ValueError("R2 ONNX graph kind is invalid")
    if step_count != 1 or sorted(observed_chunks) != chunks:
        raise ValueError("R2 ONNX graph coverage mismatch")
    return manifest


def ort_run_graph(
    graph_path: Path,
    *,
    input_ids: torch.Tensor,
    conv_state: torch.Tensor,
    ssm_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError(
            "onnxruntime is required for ORT parity validation"
        ) from exc

    if graph_path.is_symlink() or not graph_path.is_file():
        raise ValueError("ONNX graph path must be a regular file")
    session = ort.InferenceSession(
        str(graph_path),
        providers=["CPUExecutionProvider"],
    )
    feeds = {
        "input_ids": input_ids.detach().cpu().numpy(),
        "conv_state": conv_state.detach().cpu().numpy(),
        "ssm_state": ssm_state.detach().cpu().numpy(),
    }
    logits, next_conv, next_ssm = session.run(
        list(R2_ONNX_OUTPUTS),
        feeds,
    )
    return (
        torch.from_numpy(logits),
        torch.from_numpy(next_conv),
        torch.from_numpy(next_ssm),
    )


@torch.inference_mode()
def validate_ort_parity(
    model: VN97R2Model,
    bundle_dir: Path,
    *,
    profile: str | int | None = "deep",
    chunk_size: int | None = None,
    batch_size: int = 1,
    seed: int = 9702,
    rtol: float = 5e-4,
    atol: float = 5e-5,
) -> dict[str, float]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    manifest = verify_r2_onnx_bundle(bundle_dir)
    active_layers = model.resolve_active_layers(profile)
    if active_layers != int(manifest["active_layers"]):
        raise ValueError("parity profile does not match ONNX bundle")

    graph = (
        Path(bundle_dir) / "step.onnx"
        if chunk_size is None
        else Path(bundle_dir) / f"chunk-{chunk_size}.onnx"
    )
    sequence_length = 1 if chunk_size is None else chunk_size
    if chunk_size is not None and chunk_size not in manifest[
        "supported_chunk_sizes"
    ]:
        raise ValueError("requested chunk is not present in bundle")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    input_ids = torch.randint(
        0,
        model.config.vocab_size,
        (batch_size, sequence_length),
        generator=generator,
        dtype=torch.long,
    )
    state = model.initial_state(
        batch_size,
        device="cpu",
        dtype=torch.float32,
        profile=active_layers,
    )
    conv, ssm = stack_r2_state(state)

    model = model.cpu().float().eval()
    if chunk_size is None:
        reference_logits, reference_state = model.step(
            input_ids[:, 0],
            state,
            profile=active_layers,
        )
    else:
        reference_logits, reference_state = model(
            input_ids,
            state,
            profile=active_layers,
        )
    reference_conv, reference_ssm = stack_r2_state(
        reference_state
    )
    ort_logits, ort_conv, ort_ssm = ort_run_graph(
        graph,
        input_ids=(
            input_ids[:, 0]
            if chunk_size is None
            else input_ids
        ),
        conv_state=conv,
        ssm_state=ssm,
    )
    torch.testing.assert_close(
        ort_logits,
        reference_logits,
        rtol=rtol,
        atol=atol,
    )
    torch.testing.assert_close(
        ort_conv,
        reference_conv,
        rtol=rtol,
        atol=atol,
    )
    torch.testing.assert_close(
        ort_ssm,
        reference_ssm,
        rtol=rtol,
        atol=atol,
    )
    return {
        "max_logit_abs_error": float(
            (ort_logits - reference_logits).abs().max().item()
        ),
        "max_conv_state_abs_error": float(
            (ort_conv - reference_conv).abs().max().item()
        ),
        "max_ssm_state_abs_error": float(
            (ort_ssm - reference_ssm).abs().max().item()
        ),
    }
