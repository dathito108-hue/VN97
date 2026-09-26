from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import torch

from .deployment_checkpoint import save_deployment_checkpoint
from .model import VN97LanguageCore
from .model_image import build_model_image
from .p5b_mamba_transplant import (
    P5B_PROFILE_ID,
    P5B_SCHEMA,
    SOURCE_LICENSE,
    SOURCE_MODEL_SHA256,
    SOURCE_REPO,
    MambaSourceSpec,
    assess_compatibility,
    canonical_report_sha256,
    layer_mapping,
    load_source_config,
    target_config,
    validate_source_tensor_shapes,
)
from .tokenizer import (
    CONTROL_TOKENS,
    VN97Tokenizer,
    VN97TokenizerPackage,
)
from .training_cli import _atomic_write


class VN97P5BError(RuntimeError):
    pass


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Transplant a pinned Mamba-130M selective-SSM checkpoint into "
            "native VN97 weights. The conversion is approximate and requires "
            "post-transplant alignment before any promotion."
        )
    )
    parser.add_argument(
        "--source-dir",
        required=True,
    )
    parser.add_argument(
        "--vn97-tokenizer",
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        required=True,
    )
    parser.add_argument(
        "--skip-source-hash-check",
        action="store_true",
    )
    parser.add_argument(
        "--assess-only",
        action="store_true",
    )
    parser.add_argument(
        "--svd-device",
        default="cpu",
    )
    return parser


def _sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()
    with path.open(
        "rb"
    ) as handle:
        while True:
            block = handle.read(
                8 * 1024 * 1024
            )
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _source_dependencies():
    try:
        from safetensors.torch import load_file
    except Exception as exc:
        raise VN97P5BError(
            "P5B requires the safetensors package for conversion"
        ) from exc

    try:
        from tokenizers import Tokenizer
    except Exception as exc:
        raise VN97P5BError(
            "P5B requires the tokenizers package for lexical projection"
        ) from exc

    return (
        load_file,
        Tokenizer,
    )


def _decode_target_token(
    tokenizer: VN97Tokenizer,
    token_id: int,
) -> str:
    if (
        0 <= token_id
        < len(CONTROL_TOKENS)
    ):
        return CONTROL_TOKENS[
            token_id
        ]

    payload = tokenizer.decode_bytes(
        [token_id],
        skip_control=True,
    )
    try:
        return payload.decode(
            "utf-8"
        )
    except UnicodeDecodeError:
        return payload.decode(
            "latin-1"
        )


def _lexical_projection(
    *,
    source_embeddings: torch.Tensor,
    source_tokenizer: Any,
    target_tokenizer: VN97Tokenizer,
    rank: int,
    svd_device: str,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    dict[str, object],
]:
    source = (
        source_embeddings
        .detach()
        .float()
        .cpu()
    )
    rows: list[
        torch.Tensor
    ] = []
    empty = 0
    multi_piece = 0
    unique_sequences: set[
        tuple[int, ...]
    ] = set()

    for token_id in range(
        target_tokenizer.vocab_size
    ):
        text = _decode_target_token(
            target_tokenizer,
            token_id,
        )
        encoded = source_tokenizer.encode(
            text,
            add_special_tokens=False,
        )
        ids = tuple(
            int(value)
            for value in encoded.ids
            if (
                0
                <= int(value)
                < source.shape[0]
            )
        )
        if not ids:
            ids = (0,)
            empty += 1
        if len(ids) > 1:
            multi_piece += 1
        unique_sequences.add(ids)
        index = torch.tensor(
            ids,
            dtype=torch.long,
        )
        rows.append(
            source.index_select(
                0,
                index,
            ).mean(
                dim=0
            )
        )

    matrix = torch.stack(
        rows,
        dim=0,
    )
    if (
        rank <= 0
        or rank
        >= min(
            matrix.shape
        )
    ):
        raise VN97P5BError(
            "invalid P5B embedding factorization rank"
        )

    svd_input = matrix.to(
        svd_device
    )
    u, s, vh = torch.linalg.svd(
        svd_input,
        full_matrices=False,
    )
    u = u[
        :,
        :rank
    ]
    s = s[
        :rank
    ]
    vh = vh[
        :rank
    ]

    token_factors = (
        u
        * s.unsqueeze(0)
    )
    projection = vh

    reconstructed = (
        token_factors
        @ projection
    )
    residual = (
        reconstructed
        - svd_input
    )
    mse = float(
        residual.square()
        .mean()
        .item()
    )
    denom = float(
        svd_input.square()
        .mean()
        .clamp_min(1e-12)
        .item()
    )

    stats = {
        "empty_source_encodings":
            empty,
        "multi_piece_tokens":
            multi_piece,
        "relative_mse":
            mse / denom,
        "source_token_sequences":
            len(
                unique_sequences
            ),
        "target_vocab_size":
            target_tokenizer.vocab_size,
    }
    return (
        token_factors
        .detach()
        .cpu(),
        projection
        .detach()
        .cpu(),
        stats,
    )


def _copy_layer(
    target,
    mapped: dict[
        str,
        torch.Tensor,
    ],
) -> None:
    with torch.no_grad():
        target.norm.weight.copy_(
            mapped[
                "norm.weight"
            ]
        )
        target.core.a_log.copy_(
            mapped[
                "core.a_log"
            ]
        )
        target.core.in_proj.weight.copy_(
            mapped[
                "core.in_proj.weight"
            ]
        )
        target.core.dt_proj.weight.copy_(
            mapped[
                "core.dt_proj.weight"
            ]
        )
        if (
            target.core.dt_proj.bias
            is None
        ):
            raise VN97P5BError(
                "VN97 dt projection unexpectedly has no bias"
            )
        target.core.dt_proj.bias.copy_(
            mapped[
                "core.dt_proj.bias"
            ]
        )
        target.core.b_proj.weight.copy_(
            mapped[
                "core.b_proj.weight"
            ]
        )
        target.core.c_proj.weight.copy_(
            mapped[
                "core.c_proj.weight"
            ]
        )
        target.core.out_proj.weight.copy_(
            mapped[
                "core.out_proj.weight"
            ]
        )


def _finite_smoke(
    model: VN97LanguageCore,
    tokenizer: VN97Tokenizer,
) -> dict[str, object]:
    prompts = (
        "Hello VN97",
        "2 + 3 =",
        "Return JSON only",
    )
    max_abs = 0.0

    model.eval()
    with torch.inference_mode():
        for prompt in prompts:
            ids = tokenizer.encode(
                prompt,
                add_bos=True,
                add_text_tag=True,
            )
            input_ids = torch.tensor(
                [ids],
                dtype=torch.long,
            )
            logits, _ = (
                model.forward_sequential_reference(
                    input_ids
                )
            )
            if not bool(
                torch.isfinite(
                    logits
                ).all()
            ):
                raise VN97P5BError(
                    "transplanted model produced non-finite logits"
                )
            max_abs = max(
                max_abs,
                float(
                    logits.abs()
                    .max()
                    .item()
                ),
            )

    return {
        "finite":
            True,
        "max_abs_logit":
            max_abs,
        "prompts":
            len(prompts),
    }


def main(
    argv: list[str] | None = None,
) -> int:
    args = _parser().parse_args(
        argv
    )
    source_root = Path(
        args.source_dir
    ).resolve(
        strict=True
    )
    output = Path(
        args.output_dir
    )
    if (
        output.exists()
        and (
            output.is_symlink()
            or not output.is_dir()
            or any(
                output.iterdir()
            )
        )
    ):
        raise VN97P5BError(
            "P5B output-dir must be new or empty"
        )

    config = load_source_config(
        source_root
    )
    spec = (
        MambaSourceSpec
        .from_config(
            config
        )
    )
    compatibility = (
        assess_compatibility(
            spec
        )
    )
    print(
        "VN97 P5B COMPATIBILITY "
        f"feasible={str(compatibility.feasible).lower()} "
        f"lossless={str(compatibility.lossless).lower()} "
        f"alignment_required={str(compatibility.alignment_required).lower()} "
        f"hidden={spec.hidden_size} "
        f"inner={spec.intermediate_size} "
        f"layers={spec.num_hidden_layers} "
        f"state={spec.state_size}",
        flush=True,
    )
    if not compatibility.feasible:
        raise VN97P5BError(
            "source architecture does not satisfy the frozen P5B bridge contract"
        )

    if args.assess_only:
        print(
            "VN97P5B1 "
            "status=STRUCTURALLY_FEASIBLE "
            "lossless=false "
            "alignment_required=true",
            flush=True,
        )
        return 0

    model_path = (
        source_root
        / "model.safetensors"
    )
    tokenizer_path = (
        source_root
        / "tokenizer.json"
    )
    if (
        not model_path.is_file()
        or not tokenizer_path.is_file()
    ):
        raise VN97P5BError(
            "source-dir must contain model.safetensors and tokenizer.json"
        )

    source_sha256 = (
        _sha256_file(
            model_path
        )
    )
    if (
        not args.skip_source_hash_check
        and source_sha256
        != SOURCE_MODEL_SHA256
    ):
        raise VN97P5BError(
            "source model SHA-256 does not match the pinned Mamba-130M-HF checkpoint"
        )

    (
        load_file,
        SourceTokenizer,
    ) = _source_dependencies()

    print(
        "VN97 P5B SOURCE "
        f"repo={SOURCE_REPO} "
        f"license={SOURCE_LICENSE} "
        f"sha256={source_sha256}",
        flush=True,
    )

    source_tensors = load_file(
        str(
            model_path
        ),
        device="cpu",
    )
    validate_source_tensor_shapes(
        source_tensors,
        spec,
    )

    vn97_package = (
        VN97TokenizerPackage
        .from_bytes(
            Path(
                args.vn97_tokenizer
            ).read_bytes()
        )
    )
    vn97_tokenizer = (
        VN97Tokenizer(
            vn97_package
        )
    )
    source_tokenizer = (
        SourceTokenizer
        .from_file(
            str(
                tokenizer_path
            )
        )
    )

    config_target = target_config(
        vocab_size=
            vn97_tokenizer.vocab_size,
    )
    model = VN97LanguageCore(
        config_target
    )

    source_embedding = (
        source_tensors[
            "backbone.embeddings.weight"
        ]
    )
    (
        token_factors,
        projection,
        lexical_stats,
    ) = _lexical_projection(
        source_embeddings=
            source_embedding,
        source_tokenizer=
            source_tokenizer,
        target_tokenizer=
            vn97_tokenizer,
        rank=
            config_target.embedding_rank
            or 0,
        svd_device=
            args.svd_device,
    )
    with torch.no_grad():
        model.embedding.token_factors.copy_(
            token_factors
        )
        model.embedding.projection.copy_(
            projection
        )
        model.final_norm.weight.copy_(
            source_tensors[
                "backbone.norm_f.weight"
            ].float()
        )

    layer_channel_hashes: list[
        str
    ] = []
    for layer_index, layer in enumerate(
        model.layers
    ):
        mapped = layer_mapping(
            source_tensors,
            layer=layer_index,
            spec=spec,
            threshold=
                config_target.ternary_threshold,
        )
        _copy_layer(
            layer,
            mapped,
        )
        digest = hashlib.sha256()
        for key in sorted(
            mapped
        ):
            digest.update(
                key.encode(
                    "utf-8"
                )
            )
            digest.update(
                mapped[key]
                .detach()
                .cpu()
                .contiguous()
                .numpy()
                .tobytes()
            )
        layer_channel_hashes.append(
            digest.hexdigest()
        )

        if (
            (layer_index + 1)
            % 4
            == 0
            or layer_index + 1
            == len(
                model.layers
            )
        ):
            print(
                "VN97 P5B LAYER "
                f"{layer_index + 1}/{len(model.layers)} transplanted",
                flush=True,
            )

    smoke = _finite_smoke(
        model,
        vn97_tokenizer,
    )
    parameter_count = sum(
        int(
            parameter.numel()
        )
        for parameter
        in model.parameters()
    )

    output.mkdir(
        parents=True,
        exist_ok=True,
    )
    checkpoint_path = (
        output
        / "model.vn97ck1"
    )
    checkpoint_sha256 = (
        save_deployment_checkpoint(
            model,
            checkpoint_path,
        )
    )
    tokenizer_bytes = (
        vn97_package.to_bytes()
    )
    _atomic_write(
        output
        / "tokenizer.vn97tk1",
        tokenizer_bytes,
    )
    image_bytes = (
        build_model_image(
            model,
            tokenizer=
                vn97_package,
            tile_rows=16,
            tile_cols=16,
        ).data
    )
    _atomic_write(
        output
        / "model.vn97mi1",
        image_bytes,
    )

    report: dict[
        str,
        object,
    ] = {
        "alignment_required":
            True,
        "checkpoint_sha256":
            checkpoint_sha256,
        "compatibility":
            compatibility
            .canonical_object(),
        "layer_mapping_hashes":
            layer_channel_hashes,
        "lexical_projection":
            lexical_stats,
        "model_image_bytes":
            len(
                image_bytes
            ),
        "model_image_sha256":
            hashlib.sha256(
                image_bytes
            ).hexdigest(),
        "parameter_count":
            parameter_count,
        "profile_id":
            P5B_PROFILE_ID,
        "schema":
            P5B_SCHEMA,
        "smoke":
            smoke,
        "source": {
            "license":
                SOURCE_LICENSE,
            "model_sha256":
                source_sha256,
            "repo":
                SOURCE_REPO,
            "spec":
                spec.canonical_object(),
        },
        "status":
            "TRANSPLANT_READY_FOR_ALIGNMENT",
        "target_config": {
            "d_model":
                config_target.d_model,
            "d_state":
                config_target.d_state,
            "embedding_rank":
                config_target.embedding_rank,
            "n_layers":
                config_target.n_layers,
            "vocab_size":
                config_target.vocab_size,
        },
        "tokenizer_sha256":
            hashlib.sha256(
                tokenizer_bytes
            ).hexdigest(),
    }
    report[
        "report_sha256"
    ] = canonical_report_sha256(
        report
    )
    report_bytes = (
        json.dumps(
            report,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    _atomic_write(
        output
        / "p5b-report.json",
        report_bytes,
    )

    files = {
        "model.vn97ck1":
            checkpoint_path.read_bytes(),
        "model.vn97mi1":
            image_bytes,
        "p5b-report.json":
            report_bytes,
        "tokenizer.vn97tk1":
            tokenizer_bytes,
    }
    sums = b"".join(
        (
            hashlib.sha256(
                files[name]
            ).hexdigest()
            + "  "
            + name
            + "\n"
        ).encode("ascii")
        for name in sorted(
            files
        )
    )
    _atomic_write(
        output
        / "SHA256SUMS",
        sums,
    )

    print(
        "VN97P5B1 "
        "status=TRANSPLANT_READY_FOR_ALIGNMENT "
        f"parameters={parameter_count} "
        f"checkpoint={checkpoint_sha256} "
        f"lexical_relative_mse={float(lexical_stats['relative_mse']):.8f} "
        f"smoke_finite={str(bool(smoke['finite'])).lower()}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(
            main()
        )
    except Exception as exc:
        print(
            "vn97-p5b-mamba-transplant: "
            f"{exc}",
            file=sys.stderr,
        )
        raise
