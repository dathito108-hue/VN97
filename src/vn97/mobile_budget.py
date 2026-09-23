from __future__ import annotations

from dataclasses import dataclass
import math

from .config import VN97Config
from .model_image import ENTRY_SIZE, HEADER_SIZE, MAX_IMAGE_BYTES


VN97_MAX_RECURRENT_STATE_BYTES = 512 * 1024 * 1024
_VN97T2_HEADER_BYTES = 40
_F32_BYTES = 4


def _round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def _validate_tile(tile_rows: int, tile_cols: int) -> None:
    if not 0 < tile_rows <= 256 or not 0 < tile_cols <= 256:
        raise ValueError("VN97 deployment tile dimensions must be in [1, 256]")


def _packed_ternary_section_nbytes(
    rows: int,
    cols: int,
    *,
    tile_rows: int,
    tile_cols: int,
) -> int:
    if rows <= 0 or cols <= 0:
        raise ValueError("packed VN97 projection dimensions must be positive")
    _validate_tile(tile_rows, tile_cols)
    padded_rows = _round_up(rows, tile_rows)
    padded_cols = _round_up(cols, tile_cols)
    packed_weight_bytes = (padded_rows * padded_cols + 3) // 4
    return _VN97T2_HEADER_BYTES + rows * _F32_BYTES + packed_weight_bytes


@dataclass(frozen=True)
class VN97MobileFootprint:
    model_image_bytes: int
    recurrent_state_bytes: int
    packed_ternary_bytes: int
    float_parameter_bytes: int
    tokenizer_bytes: int
    section_count: int

    def __post_init__(self) -> None:
        values = (
            self.model_image_bytes,
            self.recurrent_state_bytes,
            self.packed_ternary_bytes,
            self.float_parameter_bytes,
            self.tokenizer_bytes,
            self.section_count,
        )
        if any(type(value) is not int or value < 0 for value in values):
            raise ValueError("mobile footprint values must be non-negative integers")
        if self.model_image_bytes <= 0 or self.recurrent_state_bytes <= 0:
            raise ValueError("mobile model/state footprint must be positive")
        if self.section_count <= 0:
            raise ValueError("mobile footprint section_count must be positive")

    def canonical_object(self) -> dict[str, int]:
        return {
            "float_parameter_bytes": self.float_parameter_bytes,
            "model_image_bytes": self.model_image_bytes,
            "packed_ternary_bytes": self.packed_ternary_bytes,
            "recurrent_state_bytes": self.recurrent_state_bytes,
            "section_count": self.section_count,
            "tokenizer_bytes": self.tokenizer_bytes,
        }


@dataclass(frozen=True)
class VN97MobileBudget:
    max_model_image_bytes: int = MAX_IMAGE_BYTES
    max_recurrent_state_bytes: int = VN97_MAX_RECURRENT_STATE_BYTES

    def __post_init__(self) -> None:
        if (
            type(self.max_model_image_bytes) is not int
            or self.max_model_image_bytes <= 0
        ):
            raise ValueError("max_model_image_bytes must be a positive integer")
        if (
            type(self.max_recurrent_state_bytes) is not int
            or self.max_recurrent_state_bytes <= 0
        ):
            raise ValueError("max_recurrent_state_bytes must be a positive integer")

    def rejection_status(self, footprint: VN97MobileFootprint) -> str | None:
        if footprint.model_image_bytes > self.max_model_image_bytes:
            return "REJECTED_MODEL_IMAGE_BUDGET"
        if footprint.recurrent_state_bytes > self.max_recurrent_state_bytes:
            return "REJECTED_RECURRENT_STATE_BUDGET"
        return None


def estimate_vn97_mobile_footprint(
    config: VN97Config,
    *,
    tokenizer_nbytes: int | None = None,
    tile_rows: int = 16,
    tile_cols: int = 16,
    batch_size: int = 1,
) -> VN97MobileFootprint:
    """Estimate the exact VN97MI1 byte length from model geometry.

    The estimate mirrors model_image.build_model_image() section order, 4-byte
    alignment, VN97T2 tile padding/scales/header cost and tied embedding storage.
    Learned values are not needed, so campaign candidates can be rejected before
    real parameter allocation/training.
    """
    if not isinstance(config, VN97Config):
        raise TypeError("config must be VN97Config")
    _validate_tile(tile_rows, tile_cols)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if tokenizer_nbytes is not None:
        if type(tokenizer_nbytes) is not int or tokenizer_nbytes <= 0:
            raise ValueError("tokenizer_nbytes must be a positive integer or None")

    d_model = int(config.d_model)
    d_state = int(config.d_state)
    vocab = int(config.vocab_size)

    section_sizes: list[int] = []
    packed_ternary_bytes = 0
    float_parameter_bytes = 0

    def add_f32(elements: int) -> None:
        nonlocal float_parameter_bytes
        if elements <= 0:
            raise ValueError("VN97 float section element count must be positive")
        size = elements * _F32_BYTES
        section_sizes.append(size)
        float_parameter_bytes += size

    def add_packed(rows: int, cols: int) -> None:
        nonlocal packed_ternary_bytes
        size = _packed_ternary_section_nbytes(
            rows,
            cols,
            tile_rows=tile_rows,
            tile_cols=tile_cols,
        )
        section_sizes.append(size)
        packed_ternary_bytes += size

    if config.embedding_rank is None:
        add_f32(vocab * d_model)
    else:
        rank = int(config.embedding_rank)
        add_f32(vocab * rank)
        add_f32(rank * d_model)

    for _ in range(config.n_layers):
        add_f32(d_model)
        add_packed(2 * d_model, d_model)
        add_packed(d_model, d_model)
        add_f32(d_model)
        add_packed(d_state, d_model)
        add_packed(d_state, d_model)
        add_packed(d_model, d_model)
        add_f32(d_model * d_state)

    add_f32(d_model)

    tokenizer_bytes = 0
    if tokenizer_nbytes is not None:
        tokenizer_bytes = tokenizer_nbytes
        section_sizes.append(tokenizer_bytes)

    section_count = len(section_sizes)
    cursor = HEADER_SIZE + section_count * ENTRY_SIZE
    for size in section_sizes:
        cursor = (cursor + 3) & ~3
        cursor += size

    if not math.isfinite(float(cursor)):
        raise ValueError("VN97 mobile model image size is not finite")

    return VN97MobileFootprint(
        model_image_bytes=cursor,
        recurrent_state_bytes=config.recurrent_state_bytes(
            batch_size=batch_size,
            bytes_per_value=_F32_BYTES,
        ),
        packed_ternary_bytes=packed_ternary_bytes,
        float_parameter_bytes=float_parameter_bytes,
        tokenizer_bytes=tokenizer_bytes,
        section_count=section_count,
    )
