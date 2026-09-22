from __future__ import annotations

from dataclasses import dataclass
import struct

import torch
import torch.nn.functional as F

from .quantization import ternary_symbols_and_scales


_MAGIC = b"VN97T2\x00\x00"
_VERSION = 1
_HEADER = struct.Struct("<8s8I")
_CODE_POSITIVE = 1
_CODE_NEGATIVE = 2
_CODE_RESERVED = 3


def _round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def _validate_tile(tile_rows: int, tile_cols: int) -> None:
    if tile_rows <= 0 or tile_cols <= 0:
        raise ValueError("tile dimensions must be positive")
    if tile_rows > 256 or tile_cols > 256:
        raise ValueError("tile dimensions above 256 are not supported by format v1")


def _symbols_to_codes(symbols: torch.Tensor) -> torch.Tensor:
    if symbols.dtype != torch.int8:
        symbols = symbols.to(torch.int8)
    valid = (symbols == -1) | (symbols == 0) | (symbols == 1)
    if not bool(valid.all()):
        raise ValueError("ternary symbols must be only -1, 0 or +1")
    return torch.where(
        symbols > 0,
        torch.ones_like(symbols, dtype=torch.uint8),
        torch.where(
            symbols < 0,
            torch.full_like(symbols, _CODE_NEGATIVE, dtype=torch.uint8),
            torch.zeros_like(symbols, dtype=torch.uint8),
        ),
    )


def _pack_codes(codes: torch.Tensor) -> bytes:
    flat = codes.reshape(-1).to(torch.uint8).cpu()
    remainder = flat.numel() % 4
    if remainder:
        flat = torch.cat(
            [flat, torch.zeros(4 - remainder, dtype=torch.uint8)], dim=0
        )
    grouped = flat.view(-1, 4)
    packed = (
        grouped[:, 0]
        | (grouped[:, 1] << 2)
        | (grouped[:, 2] << 4)
        | (grouped[:, 3] << 6)
    )
    return bytes(packed.tolist())


def _unpack_codes(data: bytes, count: int) -> torch.Tensor:
    if count < 0:
        raise ValueError("count must be non-negative")
    required = (count + 3) // 4
    if len(data) != required:
        raise ValueError(f"expected {required} packed bytes, got {len(data)}")
    raw = torch.tensor(list(data), dtype=torch.uint8)
    if raw.numel() == 0:
        return torch.empty(0, dtype=torch.uint8)
    codes = torch.stack(
        [
            raw & 0x03,
            (raw >> 2) & 0x03,
            (raw >> 4) & 0x03,
            (raw >> 6) & 0x03,
        ],
        dim=1,
    ).reshape(-1)[:count]
    if bool((codes == _CODE_RESERVED).any()):
        raise ValueError("packed ternary stream contains reserved 2-bit code 0b11")
    return codes


@dataclass(frozen=True)
class PackedTernaryMatrix:
    rows: int
    cols: int
    tile_rows: int
    tile_cols: int
    padded_rows: int
    padded_cols: int
    scales: torch.Tensor
    data: bytes

    def __post_init__(self) -> None:
        _validate_tile(self.tile_rows, self.tile_cols)
        if self.rows <= 0 or self.cols <= 0:
            raise ValueError("matrix dimensions must be positive")
        if self.padded_rows != _round_up(self.rows, self.tile_rows):
            raise ValueError("padded_rows does not match tile geometry")
        if self.padded_cols != _round_up(self.cols, self.tile_cols):
            raise ValueError("padded_cols does not match tile geometry")
        if tuple(self.scales.shape) != (self.rows,):
            raise ValueError("scales must have shape [rows]")
        if not bool(torch.isfinite(self.scales).all()):
            raise ValueError("scales must be finite")
        if not bool((self.scales > 0).all()):
            raise ValueError("scales must be positive")
        expected = (self.padded_rows * self.padded_cols + 3) // 4
        if len(self.data) != expected:
            raise ValueError(
                f"packed byte length mismatch: expected {expected}, got {len(self.data)}"
            )

    @property
    def packed_weight_nbytes(self) -> int:
        return len(self.data)

    @property
    def serialized_nbytes(self) -> int:
        return _HEADER.size + self.rows * 4 + len(self.data)

    @property
    def fp32_weight_nbytes(self) -> int:
        return self.rows * self.cols * 4

    @property
    def compression_ratio_vs_fp32(self) -> float:
        return self.fp32_weight_nbytes / self.serialized_nbytes

    def to_bytes(self) -> bytes:
        scales = self.scales.detach().to(dtype=torch.float32, device="cpu")
        header = _HEADER.pack(
            _MAGIC,
            _VERSION,
            self.rows,
            self.cols,
            self.tile_rows,
            self.tile_cols,
            self.padded_rows,
            self.padded_cols,
            len(self.data),
        )
        scale_blob = struct.pack(f"<{self.rows}f", *scales.tolist())
        return header + scale_blob + self.data

    @classmethod
    def from_bytes(cls, blob: bytes) -> "PackedTernaryMatrix":
        if len(blob) < _HEADER.size:
            raise ValueError("blob is shorter than VN97T2 header")
        (
            magic,
            version,
            rows,
            cols,
            tile_rows,
            tile_cols,
            padded_rows,
            padded_cols,
            data_nbytes,
        ) = _HEADER.unpack_from(blob, 0)
        if magic != _MAGIC:
            raise ValueError("invalid VN97T2 magic")
        if version != _VERSION:
            raise ValueError(f"unsupported VN97T2 version {version}")
        scale_bytes = rows * 4
        expected = _HEADER.size + scale_bytes + data_nbytes
        if len(blob) != expected:
            raise ValueError(f"expected {expected} bytes, got {len(blob)}")
        scales = torch.tensor(
            struct.unpack_from(f"<{rows}f", blob, _HEADER.size),
            dtype=torch.float32,
        )
        data = blob[_HEADER.size + scale_bytes :]
        packed = cls(
            rows=rows,
            cols=cols,
            tile_rows=tile_rows,
            tile_cols=tile_cols,
            padded_rows=padded_rows,
            padded_cols=padded_cols,
            scales=scales,
            data=data,
        )
        _unpack_codes(data, padded_rows * padded_cols)
        return packed

    def ternary_symbols(self) -> torch.Tensor:
        count = self.padded_rows * self.padded_cols
        codes = _unpack_codes(self.data, count)
        symbols_flat = torch.where(
            codes == _CODE_POSITIVE,
            torch.ones_like(codes, dtype=torch.int8),
            torch.where(
                codes == _CODE_NEGATIVE,
                torch.full_like(codes, -1, dtype=torch.int8),
                torch.zeros_like(codes, dtype=torch.int8),
            ),
        )

        n_tile_rows = self.padded_rows // self.tile_rows
        n_tile_cols = self.padded_cols // self.tile_cols
        tiled = symbols_flat.view(
            n_tile_rows,
            n_tile_cols,
            self.tile_rows,
            self.tile_cols,
        )
        padded = tiled.permute(0, 2, 1, 3).contiguous().view(
            self.padded_rows, self.padded_cols
        )
        return padded[: self.rows, : self.cols]

    def dequantize(self, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        symbols = self.ternary_symbols().to(dtype=dtype)
        scales = self.scales.to(dtype=dtype).unsqueeze(1)
        return symbols * scales


def pack_ternary_symbols(
    symbols: torch.Tensor,
    scales: torch.Tensor,
    *,
    tile_rows: int = 16,
    tile_cols: int = 16,
) -> PackedTernaryMatrix:
    if symbols.ndim != 2:
        raise ValueError("symbols must have shape [rows, cols]")
    rows, cols = symbols.shape
    if rows <= 0 or cols <= 0:
        raise ValueError("matrix dimensions must be positive")
    _validate_tile(tile_rows, tile_cols)
    if tuple(scales.shape) != (rows,):
        raise ValueError("scales must have shape [rows]")

    padded_rows = _round_up(rows, tile_rows)
    padded_cols = _round_up(cols, tile_cols)
    padded = torch.zeros((padded_rows, padded_cols), dtype=torch.int8)
    padded[:rows, :cols] = symbols.to(torch.int8).cpu()

    n_tile_rows = padded_rows // tile_rows
    n_tile_cols = padded_cols // tile_cols
    tile_major = (
        padded.view(n_tile_rows, tile_rows, n_tile_cols, tile_cols)
        .permute(0, 2, 1, 3)
        .contiguous()
    )
    data = _pack_codes(_symbols_to_codes(tile_major))
    return PackedTernaryMatrix(
        rows=rows,
        cols=cols,
        tile_rows=tile_rows,
        tile_cols=tile_cols,
        padded_rows=padded_rows,
        padded_cols=padded_cols,
        scales=scales.detach().to(dtype=torch.float32, device="cpu"),
        data=data,
    )


def pack_ternary_weight(
    weight: torch.Tensor,
    *,
    threshold: float = 0.5,
    tile_rows: int = 16,
    tile_cols: int = 16,
) -> PackedTernaryMatrix:
    symbols, scales = ternary_symbols_and_scales(weight.detach().cpu(), threshold)
    return pack_ternary_symbols(
        symbols,
        scales,
        tile_rows=tile_rows,
        tile_cols=tile_cols,
    )


def packed_linear_reference(
    x: torch.Tensor,
    packed: PackedTernaryMatrix,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    if x.shape[-1] != packed.cols:
        raise ValueError(
            f"expected input last dimension {packed.cols}, got {x.shape[-1]}"
        )
    weight = packed.dequantize(dtype=x.dtype).to(device=x.device)
    if bias is not None:
        bias = bias.to(device=x.device, dtype=x.dtype)
    return F.linear(x, weight, bias)
