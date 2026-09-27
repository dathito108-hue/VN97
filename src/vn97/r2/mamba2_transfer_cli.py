from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .mamba2_transfer import (
    Mamba2SourceSpec,
    build_transfer_manifest,
    convert_state_dict_1to1,
    expected_unique_parameter_count,
    save_g0_checkpoint,
    sha256_file,
    write_transfer_manifest,
)


def _load_config(root: Path) -> tuple[dict[str, object], Path]:
    path = root / "config.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("config.json must contain one JSON object")
    return data, path


def _load_source_weights(root: Path) -> tuple[dict[str, torch.Tensor], Path]:
    candidates = (
        root / "pytorch_model.bin",
        root / "model.pt",
    )
    existing = [path for path in candidates if path.is_file()]
    if len(existing) != 1:
        raise ValueError(
            "exact G0 conversion currently requires exactly one local "
            "pytorch_model.bin or model.pt; sharded/safetensors streaming "
            "support is a later transport optimization"
        )
    path = existing[0]
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("source checkpoint must be a state_dict mapping")
    if not all(
        isinstance(name, str) and isinstance(value, torch.Tensor)
        for name, value in payload.items()
    ):
        raise ValueError("source checkpoint contains non-tensor state entries")
    return payload, path


def _build_manifest(
    root: Path,
    *,
    source_revision: str,
    weight_path: Path | None,
) -> tuple[Mamba2SourceSpec, dict[str, object]]:
    config, config_path = _load_config(root)
    spec = Mamba2SourceSpec.from_config(config)
    spec.require_official_27b_contract()
    weight_sha = (
        sha256_file(weight_path)
        if weight_path is not None
        else "0" * 64
    )
    manifest = build_transfer_manifest(
        spec,
        source_revision=source_revision,
        source_config_sha256=sha256_file(config_path),
        source_weight_sha256=weight_sha,
    )
    return spec, manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "VN97 G0 exact Mamba-2 2.7B transfer. "
            "No slicing, tokenizer conversion, quantization or blending."
        )
    )
    sub = parser.add_subparsers(dest="command", required=True)

    assess = sub.add_parser("assess")
    assess.add_argument("--source-root", required=True, type=Path)
    assess.add_argument("--source-revision", required=True)

    convert = sub.add_parser("convert")
    convert.add_argument("--source-root", required=True, type=Path)
    convert.add_argument("--source-revision", required=True)
    convert.add_argument("--output", required=True, type=Path)
    convert.add_argument("--manifest-output", type=Path)

    args = parser.parse_args()
    root = args.source_root.resolve(strict=True)

    if args.command == "assess":
        config, _ = _load_config(root)
        spec = Mamba2SourceSpec.from_config(config)
        spec.require_official_27b_contract()
        print(
            json.dumps(
                {
                    "status": "STRUCTURALLY_EXACT_TRANSFER_COMPATIBLE",
                    "source": "state-spaces/mamba2-2.7b",
                    "unique_core_parameters": expected_unique_parameter_count(
                        spec
                    ),
                    "d_model": spec.d_model,
                    "n_layers": spec.n_layers,
                    "d_state": spec.d_state,
                    "n_heads": spec.n_heads,
                    "head_dim": spec.head_dim,
                    "ssm_state_per_layer": list(
                        spec.recurrent_ssm_state_shape
                    ),
                    "conv_state_per_layer": list(
                        spec.recurrent_conv_state_shape
                    ),
                    "transfer": "1to1_tensor_value_identity",
                    "training_required_for_g0": False,
                },
                sort_keys=True,
            )
        )
        return

    source_state, weight_path = _load_source_weights(root)
    spec, manifest = _build_manifest(
        root,
        source_revision=args.source_revision,
        weight_path=weight_path,
    )
    converted = convert_state_dict_1to1(source_state, spec)
    checkpoint_sha = save_g0_checkpoint(
        args.output,
        converted_state_dict=converted,
        manifest=manifest,
    )
    manifest_path = (
        args.manifest_output
        if args.manifest_output is not None
        else args.output.with_suffix(args.output.suffix + ".transfer.json")
    )
    write_transfer_manifest(manifest_path, manifest)
    print(
        json.dumps(
            {
                "status": "VN97_G0_DIRECT_TRANSFER_CREATED",
                "checkpoint": str(args.output),
                "checkpoint_sha256": checkpoint_sha,
                "transfer_manifest": str(manifest_path),
                "transfer_manifest_id": manifest["manifest_id"],
                "unique_core_parameters": expected_unique_parameter_count(spec),
                "augmentation_effect": "exact_zero",
                "promotion_authorized": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
