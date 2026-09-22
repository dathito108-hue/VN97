from __future__ import annotations

from dataclasses import dataclass
import ctypes
import hashlib
import math
import struct
import sys
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .model import VN97LanguageCore
    from .tokenizer import VN97TokenizerPackage

MAGIC = b"VN97MI1\0"
VERSION = 1
HEADER_SIZE = 96
ENTRY_SIZE = 32
MAX_IMAGE_BYTES = 512 * 1024 * 1024
GLOBAL_LAYER = 0xFFFFFFFF

FLAG_FACTORIZED = 1 << 0
FLAG_TOKENIZER = 1 << 1

SECTION_EMBEDDING = 1
SECTION_TOKEN_FACTORS = 2
SECTION_EMBEDDING_PROJECTION = 3
SECTION_FINAL_NORM = 4
SECTION_TOKENIZER = 5
SECTION_LAYER_NORM = 16
SECTION_IN_PROJ = 17
SECTION_DT_PROJ = 18
SECTION_DT_BIAS = 19
SECTION_B_PROJ = 20
SECTION_C_PROJ = 21
SECTION_OUT_PROJ = 22
SECTION_A_LOG = 23

_HEADER = struct.Struct("<8s10I3fI4Q")
_ENTRY = struct.Struct("<IIQQII")


@dataclass(frozen=True)
class VN97ModelImage:
    data: bytes

    @property
    def model_id(self) -> bytes:
        return hashlib.sha256(self.data).digest()

    @property
    def model_id_hex(self) -> str:
        return self.model_id.hex()


def _f32_bytes(tensor: torch.Tensor, label: str) -> bytes:
    value = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{label} contains non-finite values")
    if sys.byteorder != "little":
        raise RuntimeError("VN97MI1 export requires a little-endian host")
    return ctypes.string_at(value.data_ptr(), value.numel() * 4)


def _packed_bytes(linear: object, *, tile_rows: int, tile_cols: int, label: str) -> bytes:
    export = getattr(linear, "export_packed", None)
    if export is None:
        raise TypeError(f"{label} is not a canonical VN97 ternary linear")
    packed = export(tile_rows=tile_rows, tile_cols=tile_cols)
    blob = packed.to_bytes()
    if not isinstance(blob, bytes) or not blob.startswith(b"VN97T2\0\0"):
        raise ValueError(f"{label} did not export canonical VN97T2")
    return blob


def _tokenizer_bytes(tokenizer: object | bytes | None) -> tuple[bytes, int] | None:
    if tokenizer is None:
        return None
    if isinstance(tokenizer, bytes):
        blob = tokenizer
    else:
        method = getattr(tokenizer, "to_bytes", None)
        if method is None:
            raise TypeError("tokenizer must be VN97TK1 bytes or expose to_bytes()")
        blob = method()
    if not isinstance(blob, bytes) or not blob.startswith(b"VN97TK1\0"):
        raise ValueError("tokenizer must serialize as VN97TK1")
    if len(blob) < 24:
        raise ValueError("VN97TK1 tokenizer is truncated")
    learned_count = struct.unpack_from("<I", blob, 20)[0]
    return blob, 264 + learned_count


def build_model_image(
    model: "VN97LanguageCore",
    *,
    tokenizer: "VN97TokenizerPackage | bytes | None" = None,
    tile_rows: int = 16,
    tile_cols: int = 16,
) -> VN97ModelImage:
    """Serialize the canonical VN97 language core into its mmap-ready runtime image.

    VN97MI1 is not a second model format or backend. It is a bounded, read-only
    image of the same Code-1 -> Code-2 VN97 parameters consumed by LanguageModelView.
    M9 activation is expected to publish these exact bytes as its committed model artifact.
    """
    config = model.config
    if config.n_layers != len(model.layers):
        raise ValueError("model layer count does not match config")
    if not (0 < tile_rows <= 256 and 0 < tile_cols <= 256):
        raise ValueError("VN97T2 tile dimensions must be in [1, 256]")

    tokenizer_value = _tokenizer_bytes(tokenizer)
    tokenizer_blob = None if tokenizer_value is None else tokenizer_value[0]
    if tokenizer_value is not None and tokenizer_value[1] != config.vocab_size:
        raise ValueError("VN97TK1 vocabulary does not match model vocab_size")

    factorized = config.embedding_rank is not None
    flags = (FLAG_FACTORIZED if factorized else 0) | (
        FLAG_TOKENIZER if tokenizer_blob is not None else 0
    )
    rank = 0 if config.embedding_rank is None else int(config.embedding_rank)

    sections: list[tuple[int, int, bytes]] = []
    if factorized:
        sections.append((
            SECTION_TOKEN_FACTORS,
            GLOBAL_LAYER,
            _f32_bytes(model.embedding.token_factors, "token_factors"),
        ))
        sections.append((
            SECTION_EMBEDDING_PROJECTION,
            GLOBAL_LAYER,
            _f32_bytes(model.embedding.projection, "embedding_projection"),
        ))
    else:
        sections.append((
            SECTION_EMBEDDING,
            GLOBAL_LAYER,
            _f32_bytes(model.embedding.weight, "embedding"),
        ))

    for layer_index, layer in enumerate(model.layers):
        core = layer.core
        sections.extend((
            (SECTION_LAYER_NORM, layer_index, _f32_bytes(layer.norm.weight, f"layer.{layer_index}.norm")),
            (SECTION_IN_PROJ, layer_index, _packed_bytes(core.in_proj, tile_rows=tile_rows, tile_cols=tile_cols, label=f"layer.{layer_index}.in_proj")),
            (SECTION_DT_PROJ, layer_index, _packed_bytes(core.dt_proj, tile_rows=tile_rows, tile_cols=tile_cols, label=f"layer.{layer_index}.dt_proj")),
            (SECTION_DT_BIAS, layer_index, _f32_bytes(core.dt_proj.bias, f"layer.{layer_index}.dt_bias")),
            (SECTION_B_PROJ, layer_index, _packed_bytes(core.b_proj, tile_rows=tile_rows, tile_cols=tile_cols, label=f"layer.{layer_index}.b_proj")),
            (SECTION_C_PROJ, layer_index, _packed_bytes(core.c_proj, tile_rows=tile_rows, tile_cols=tile_cols, label=f"layer.{layer_index}.c_proj")),
            (SECTION_OUT_PROJ, layer_index, _packed_bytes(core.out_proj, tile_rows=tile_rows, tile_cols=tile_cols, label=f"layer.{layer_index}.out_proj")),
            (SECTION_A_LOG, layer_index, _f32_bytes(core.a_log, f"layer.{layer_index}.a_log")),
        ))

    sections.append((
        SECTION_FINAL_NORM,
        GLOBAL_LAYER,
        _f32_bytes(model.final_norm.weight, "final_norm"),
    ))
    if tokenizer_blob is not None:
        sections.append((SECTION_TOKENIZER, GLOBAL_LAYER, tokenizer_blob))

    section_count = len(sections)
    payload_offset = HEADER_SIZE + section_count * ENTRY_SIZE
    cursor = payload_offset
    table = bytearray()
    payload = bytearray()

    for section_type, layer_index, section_data in sections:
        aligned = (cursor + 3) & ~3
        payload.extend(b"\0" * (aligned - cursor))
        table.extend(_ENTRY.pack(
            section_type,
            layer_index,
            aligned,
            len(section_data),
            0,
            0,
        ))
        payload.extend(section_data)
        cursor = aligned + len(section_data)
        if cursor > MAX_IMAGE_BYTES:
            raise ValueError("VN97MI1 exceeds the 512 MiB runtime bound")

    values = (
        MAGIC,
        VERSION,
        HEADER_SIZE,
        flags,
        section_count,
        ENTRY_SIZE,
        int(config.vocab_size),
        int(config.d_model),
        int(config.n_layers),
        int(config.d_state),
        rank,
        float(config.dt_min),
        float(config.dt_max),
        float(config.rms_eps),
        0,
        HEADER_SIZE,
        payload_offset,
        cursor,
        0,
    )
    if not all(math.isfinite(v) for v in (values[11], values[12], values[13])):
        raise ValueError("model runtime config contains non-finite values")
    header = _HEADER.pack(*values)
    data = header + bytes(table) + bytes(payload)
    if len(data) != cursor:
        raise AssertionError("internal VN97MI1 size mismatch")
    return VN97ModelImage(data)
