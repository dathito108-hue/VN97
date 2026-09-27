from __future__ import annotations

import argparse
import json
from pathlib import Path

from .mamba2_source_integrity import (
    PINNED_SOURCE_REVISION,
    inspect_pinned_source,
    load_source_config,
    require_pinned_revision,
    write_source_receipt,
)
from .mamba2_transfer import (
    build_transfer_manifest,
    convert_state_dict_1to1,
    expected_unique_parameter_count,
    save_g0_checkpoint,
    sha256_file,
    write_transfer_manifest,
)


def _assessment(root: Path, source_revision: str) -> dict[str, object]:
    require_pinned_revision(source_revision)
    spec, config_path = load_source_config(root)
    return {
        "status": "STRUCTURALLY_EXACT_TRANSFER_COMPATIBLE",
        "source": "state-spaces/mamba2-2.7b",
        "source_revision": source_revision,
        "config_sha256": sha256_file(config_path),
        "unique_core_parameters": expected_unique_parameter_count(spec),
        "d_model": spec.d_model,
        "n_layers": spec.n_layers,
        "d_state": spec.d_state,
        "n_heads": spec.n_heads,
        "head_dim": spec.head_dim,
        "ssm_state_per_layer": list(spec.recurrent_ssm_state_shape),
        "conv_state_per_layer": list(spec.recurrent_conv_state_shape),
        "transfer": "1to1_tensor_value_identity",
        "source_transport": "pytorch_model.bin_mmap",
        "training_required_for_g0": False,
        "real_weight_integrity_verified": False,
    }


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
    assess.add_argument(
        "--source-revision",
        default=PINNED_SOURCE_REVISION,
    )

    verify = sub.add_parser("verify-source")
    verify.add_argument("--source-root", required=True, type=Path)
    verify.add_argument(
        "--source-revision",
        default=PINNED_SOURCE_REVISION,
    )
    verify.add_argument("--receipt-output", type=Path)

    convert = sub.add_parser("convert")
    convert.add_argument("--source-root", required=True, type=Path)
    convert.add_argument(
        "--source-revision",
        default=PINNED_SOURCE_REVISION,
    )
    convert.add_argument("--output", required=True, type=Path)
    convert.add_argument("--manifest-output", type=Path)
    convert.add_argument("--source-receipt-output", type=Path)

    args = parser.parse_args()
    root = args.source_root.resolve(strict=True)

    if args.command == "assess":
        print(json.dumps(_assessment(root, args.source_revision), sort_keys=True))
        return

    spec, source_state, receipt = inspect_pinned_source(
        root,
        source_revision=args.source_revision,
        verify_weight_sha256=True,
    )

    if args.command == "verify-source":
        if args.receipt_output is not None:
            write_source_receipt(args.receipt_output, receipt)
        print(
            json.dumps(
                {
                    "status": "PINNED_SOURCE_VERIFIED",
                    "source": receipt.source_model_id,
                    "source_revision": receipt.source_revision,
                    "source_receipt_id": receipt.receipt_id(),
                    "weight_sha256": receipt.weight_sha256,
                    "weight_size_bytes": receipt.weight_size_bytes,
                    "tensor_count": receipt.tensor_count,
                    "unique_core_parameters": receipt.unique_core_parameters,
                    "mmap_used": receipt.mmap_used,
                    "receipt_output": (
                        str(args.receipt_output)
                        if args.receipt_output is not None
                        else None
                    ),
                },
                sort_keys=True,
            )
        )
        return

    manifest = build_transfer_manifest(
        spec,
        source_revision=receipt.source_revision,
        source_config_sha256=receipt.config_sha256,
        source_weight_sha256=receipt.weight_sha256,
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
    source_receipt_path = (
        args.source_receipt_output
        if args.source_receipt_output is not None
        else args.output.with_suffix(args.output.suffix + ".source.json")
    )
    write_transfer_manifest(manifest_path, manifest)
    write_source_receipt(source_receipt_path, receipt)
    print(
        json.dumps(
            {
                "status": "VN97_G0_DIRECT_TRANSFER_CREATED",
                "checkpoint": str(args.output),
                "checkpoint_sha256": checkpoint_sha,
                "transfer_manifest": str(manifest_path),
                "transfer_manifest_id": manifest["manifest_id"],
                "source_receipt": str(source_receipt_path),
                "source_receipt_id": receipt.receipt_id(),
                "source_weight_sha256": receipt.weight_sha256,
                "source_transport": "pytorch_model.bin_mmap",
                "unique_core_parameters": expected_unique_parameter_count(spec),
                "augmentation_effect": "exact_zero",
                "parity_required_before_production": True,
                "promotion_authorized": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
