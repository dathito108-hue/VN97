from __future__ import annotations

from dataclasses import dataclass
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
import stat
import sys
import tempfile
import zlib

import torch

from .bootstrap_bundle import (
    VN97BootstrapBundle,
    VN97BootstrapSigner,
    build_bootstrap_bundle,
)
from .capability_package import CapabilitySource
from .config import VN97Config
from .model import VN97LanguageCore
from .modality import AudioAdapterConfig, AudioFrameAdapter
from .tokenizer import VN97TokenizerPackage


MAGIC = b"VN97CK1\0"
VERSION = 1
HEADER_SIZE = 80
FLAG_AUDIO_ADAPTER = 1 << 0
KNOWN_FLAGS = FLAG_AUDIO_ADAPTER
MAX_MANIFEST_BYTES = 4 * 1024 * 1024
MAX_TENSORS = 10_000
MAX_CHECKPOINT_BYTES = 2 * 1024 * 1024 * 1024
_HEADER = struct.Struct("<8sHHIIIQQ32sII")
_NAME_RE = re.compile(r"^[A-Za-z0-9_.]{1,256}$")

_CONFIG_KEYS = (
    "vocab_size",
    "d_model",
    "n_layers",
    "d_state",
    "ternary_threshold",
    "dt_min",
    "dt_max",
    "min_decay",
    "max_decay",
    "rms_eps",
    "embedding_rank",
)


class VN97DeploymentCheckpointError(RuntimeError):
    pass


class VN97DeploymentCheckpointFormatError(VN97DeploymentCheckpointError):
    pass


class VN97DeploymentCheckpointIntegrityError(VN97DeploymentCheckpointError):
    pass


@dataclass(frozen=True)
class VN97LoadedDeploymentCheckpoint:
    model: VN97LanguageCore
    config: VN97Config
    checkpoint_sha256: str
    tensor_count: int
    audio_adapter: AudioFrameAdapter | None = None


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise VN97DeploymentCheckpointFormatError(
            "VN97CK1 manifest is not canonicalizable JSON"
        ) from exc


def _strict_json_object(data: bytes) -> dict[str, object]:
    duplicates: list[str] = []

    def hook(pairs: list[tuple[str, object]]) -> dict[str, object]:
        out: dict[str, object] = {}
        for key, value in pairs:
            if key in out:
                duplicates.append(key)
            out[key] = value
        return out

    try:
        text = data.decode("utf-8", errors="strict")
        value = json.loads(
            text,
            object_pairs_hook=hook,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(value)
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise VN97DeploymentCheckpointFormatError(
            "VN97CK1 manifest is not strict UTF-8 JSON"
        ) from exc

    if duplicates or not isinstance(value, dict):
        raise VN97DeploymentCheckpointFormatError(
            "VN97CK1 manifest must be an object without duplicate keys"
        )
    if _canonical_json(value) != data:
        raise VN97DeploymentCheckpointFormatError(
            "VN97CK1 manifest must use canonical JSON"
        )
    return value


def _config_object(config: VN97Config) -> dict[str, object]:
    return {
        "d_model": config.d_model,
        "d_state": config.d_state,
        "dt_max": float(config.dt_max),
        "dt_min": float(config.dt_min),
        "embedding_rank": config.embedding_rank,
        "max_decay": float(config.max_decay),
        "min_decay": float(config.min_decay),
        "n_layers": config.n_layers,
        "rms_eps": float(config.rms_eps),
        "ternary_threshold": float(config.ternary_threshold),
        "vocab_size": config.vocab_size,
    }


def _audio_config_object(adapter: AudioFrameAdapter) -> dict[str, object]:
    config = adapter.config
    return {
        "eps": float(config.eps),
        "frame_size": int(config.frame_size),
        "hop_size": int(config.hop_size),
        "rms_eps": float(adapter.norm.eps),
        "schema": "VN97AUDIO1",
    }


def _parse_audio_config(raw: object, *, d_model: int) -> AudioFrameAdapter:
    if not isinstance(raw, dict) or set(raw) != {
        "eps",
        "frame_size",
        "hop_size",
        "rms_eps",
        "schema",
    }:
        raise VN97DeploymentCheckpointFormatError(
            "VN97CK1 audio config keys are not exact"
        )
    if raw["schema"] != "VN97AUDIO1":
        raise VN97DeploymentCheckpointFormatError(
            "VN97CK1 audio config schema mismatch"
        )
    frame_size = raw["frame_size"]
    hop_size = raw["hop_size"]
    if type(frame_size) is not int or type(hop_size) is not int:
        raise VN97DeploymentCheckpointFormatError(
            "VN97CK1 audio frame geometry must be integer"
        )
    values: dict[str, float] = {}
    for key in ("eps", "rms_eps"):
        value = raw[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise VN97DeploymentCheckpointFormatError(
                f"VN97CK1 audio {key} must be numeric"
            )
        numeric = float(value)
        if not math.isfinite(numeric) or numeric <= 0.0:
            raise VN97DeploymentCheckpointFormatError(
                f"VN97CK1 audio {key} must be finite and positive"
            )
        values[key] = numeric
    if frame_size != 320 or hop_size != 320 or values["eps"] != 1e-5:
        raise VN97DeploymentCheckpointFormatError(
            "VN97CK1 production audio config must use canonical 320/320 and eps=1e-5"
        )
    try:
        return AudioFrameAdapter(
            d_model,
            config=AudioAdapterConfig(
                frame_size=frame_size,
                hop_size=hop_size,
                eps=values["eps"],
            ),
            rms_eps=values["rms_eps"],
        )
    except ValueError as exc:
        raise VN97DeploymentCheckpointFormatError(
            f"VN97CK1 audio config is invalid: {exc}"
        ) from exc


def _parse_config(raw: object) -> VN97Config:
    if not isinstance(raw, dict) or set(raw) != set(_CONFIG_KEYS):
        raise VN97DeploymentCheckpointFormatError(
            "VN97CK1 config keys are not exact"
        )

    for key in ("vocab_size", "d_model", "n_layers", "d_state"):
        if type(raw[key]) is not int:
            raise VN97DeploymentCheckpointFormatError(
                f"VN97CK1 config {key} must be integer"
            )

    rank = raw["embedding_rank"]
    if rank is not None and type(rank) is not int:
        raise VN97DeploymentCheckpointFormatError(
            "VN97CK1 embedding_rank must be integer or null"
        )

    float_values: dict[str, float] = {}
    for key in (
        "ternary_threshold",
        "dt_min",
        "dt_max",
        "min_decay",
        "max_decay",
        "rms_eps",
    ):
        value = raw[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise VN97DeploymentCheckpointFormatError(
                f"VN97CK1 config {key} must be numeric"
            )
        numeric = float(value)
        if not math.isfinite(numeric):
            raise VN97DeploymentCheckpointFormatError(
                f"VN97CK1 config {key} must be finite"
            )
        float_values[key] = numeric

    try:
        return VN97Config(
            vocab_size=raw["vocab_size"],
            d_model=raw["d_model"],
            n_layers=raw["n_layers"],
            d_state=raw["d_state"],
            ternary_threshold=float_values["ternary_threshold"],
            dt_min=float_values["dt_min"],
            dt_max=float_values["dt_max"],
            min_decay=float_values["min_decay"],
            max_decay=float_values["max_decay"],
            rms_eps=float_values["rms_eps"],
            embedding_rank=rank,
        )
    except ValueError as exc:
        raise VN97DeploymentCheckpointFormatError(
            f"VN97CK1 config is invalid: {exc}"
        ) from exc


def _tensor_bytes(tensor: torch.Tensor, name: str) -> tuple[bytes, tuple[int, ...]]:
    if sys.byteorder != "little":
        raise VN97DeploymentCheckpointError(
            "VN97CK1 serialization requires a little-endian host"
        )
    value = tensor.detach().cpu().to(dtype=torch.float32).contiguous()
    if value.numel() <= 0:
        raise VN97DeploymentCheckpointFormatError(
            f"VN97CK1 tensor is empty: {name}"
        )
    if not bool(torch.isfinite(value).all()):
        raise VN97DeploymentCheckpointFormatError(
            f"VN97CK1 tensor contains non-finite values: {name}"
        )
    byte_count = value.numel() * 4
    data = ctypes.string_at(value.data_ptr(), byte_count)
    return data, tuple(int(v) for v in value.shape)


def build_deployment_checkpoint(
    model: VN97LanguageCore,
    *,
    audio_adapter: AudioFrameAdapter | None = None,
) -> bytes:
    if not isinstance(model, VN97LanguageCore):
        raise TypeError("model must be VN97LanguageCore")

    state = dict(model.state_dict())
    flags = 0
    audio_manifest: dict[str, object] | None = None
    if audio_adapter is not None:
        if not isinstance(audio_adapter, AudioFrameAdapter):
            raise TypeError("audio_adapter must be AudioFrameAdapter")
        if audio_adapter.projection.out_features != model.config.d_model:
            raise VN97DeploymentCheckpointFormatError(
                "VN97CK1 audio adapter d_model does not match language core"
            )
        if (
            audio_adapter.config.frame_size != 320
            or audio_adapter.config.hop_size != 320
            or float(audio_adapter.config.eps) != 1e-5
            or float(audio_adapter.norm.eps) != float(model.config.rms_eps)
        ):
            raise VN97DeploymentCheckpointFormatError(
                "VN97CK1 audio adapter must use canonical production geometry"
            )
        for name, tensor in audio_adapter.state_dict().items():
            state[f"audio_adapter.{name}"] = tensor
        flags |= FLAG_AUDIO_ADAPTER
        audio_manifest = _audio_config_object(audio_adapter)

    names = sorted(state)
    if not names or len(names) > MAX_TENSORS:
        raise VN97DeploymentCheckpointFormatError(
            "VN97CK1 tensor count is outside bounds"
        )
    if any(not _NAME_RE.fullmatch(name) for name in names):
        raise VN97DeploymentCheckpointFormatError(
            "VN97CK1 tensor name is invalid"
        )

    payload = bytearray()
    tensor_meta: list[dict[str, object]] = []
    for name in names:
        data, shape = _tensor_bytes(state[name], name)
        offset = len(payload)
        payload.extend(data)
        tensor_meta.append(
            {
                "dtype": "float32",
                "name": name,
                "offset": offset,
                "sha256": hashlib.sha256(data).hexdigest(),
                "shape": list(shape),
                "size": len(data),
            }
        )
        if HEADER_SIZE + len(payload) > MAX_CHECKPOINT_BYTES:
            raise VN97DeploymentCheckpointFormatError(
                "VN97CK1 checkpoint exceeds byte limit"
            )

    manifest_object: dict[str, object] = {
        "config": _config_object(model.config),
        "schema": "VN97CK1",
        "tensors": tensor_meta,
    }
    if audio_manifest is not None:
        manifest_object["audio"] = audio_manifest
    manifest = _canonical_json(manifest_object)
    if not 0 < len(manifest) <= MAX_MANIFEST_BYTES:
        raise VN97DeploymentCheckpointFormatError(
            "VN97CK1 manifest size is outside bounds"
        )

    total_size = HEADER_SIZE + len(manifest) + len(payload)
    if total_size > MAX_CHECKPOINT_BYTES:
        raise VN97DeploymentCheckpointFormatError(
            "VN97CK1 checkpoint exceeds byte limit"
        )

    content = manifest + bytes(payload)
    header = bytearray(
        _HEADER.pack(
            MAGIC,
            VERSION,
            HEADER_SIZE,
            flags,
            len(manifest),
            len(tensor_meta),
            len(payload),
            total_size,
            hashlib.sha256(content).digest(),
            0,
            0,
        )
    )
    crc = zlib.crc32(header[:72]) & 0xFFFFFFFF
    struct.pack_into("<I", header, 72, crc)
    return bytes(header) + content


def _parse_tensor_meta(
    raw: object,
    *,
    expected_count: int,
    payload_size: int,
) -> list[tuple[str, tuple[int, ...], int, int, str]]:
    if not isinstance(raw, list) or len(raw) != expected_count:
        raise VN97DeploymentCheckpointFormatError(
            "VN97CK1 tensor list/count mismatch"
        )

    result: list[tuple[str, tuple[int, ...], int, int, str]] = []
    previous_name: str | None = None
    cursor = 0
    for item in raw:
        if not isinstance(item, dict) or set(item) != {
            "dtype",
            "name",
            "offset",
            "sha256",
            "shape",
            "size",
        }:
            raise VN97DeploymentCheckpointFormatError(
                "VN97CK1 tensor metadata keys are invalid"
            )

        name = item["name"]
        if not isinstance(name, str) or not _NAME_RE.fullmatch(name):
            raise VN97DeploymentCheckpointFormatError(
                "VN97CK1 tensor name is invalid"
            )
        if previous_name is not None and name <= previous_name:
            raise VN97DeploymentCheckpointFormatError(
                "VN97CK1 tensor names must be sorted and unique"
            )
        previous_name = name

        if item["dtype"] != "float32":
            raise VN97DeploymentCheckpointFormatError(
                "VN97CK1 supports only float32 deployment tensors"
            )
        shape_raw = item["shape"]
        if (
            not isinstance(shape_raw, list)
            or not 1 <= len(shape_raw) <= 8
            or any(type(v) is not int or v <= 0 for v in shape_raw)
        ):
            raise VN97DeploymentCheckpointFormatError(
                "VN97CK1 tensor shape is invalid"
            )
        shape = tuple(shape_raw)
        numel = math.prod(shape)

        offset = item["offset"]
        size = item["size"]
        if type(offset) is not int or type(size) is not int:
            raise VN97DeploymentCheckpointFormatError(
                "VN97CK1 tensor offset/size must be integer"
            )
        if offset != cursor or size != numel * 4 or size <= 0:
            raise VN97DeploymentCheckpointFormatError(
                "VN97CK1 tensor layout is invalid"
            )
        if offset > payload_size or size > payload_size - offset:
            raise VN97DeploymentCheckpointFormatError(
                "VN97CK1 tensor range is outside payload"
            )

        digest = item["sha256"]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(ch not in "0123456789abcdef" for ch in digest)
        ):
            raise VN97DeploymentCheckpointFormatError(
                "VN97CK1 tensor SHA-256 encoding is invalid"
            )

        result.append((name, shape, offset, size, digest))
        cursor += size

    if cursor != payload_size:
        raise VN97DeploymentCheckpointFormatError(
            "VN97CK1 payload contains hidden/trailing bytes"
        )
    return result


def _require_tied_checkpoint_values(
    tensors: dict[str, torch.Tensor],
    config: VN97Config,
) -> None:
    pairs = (
        (
            ("embedding.weight", "lm_head.weight"),
        )
        if config.embedding_rank is None
        else (
            ("embedding.token_factors", "lm_head.token_factors"),
            ("embedding.projection", "lm_head.projection"),
        )
    )
    for first, second in pairs:
        if first not in tensors or second not in tensors:
            raise VN97DeploymentCheckpointFormatError(
                "VN97CK1 tied parameter keys are missing"
            )
        if not torch.equal(tensors[first], tensors[second]):
            raise VN97DeploymentCheckpointIntegrityError(
                f"VN97CK1 tied parameter values diverge: {first} vs {second}"
            )


def load_deployment_checkpoint(
    blob: bytes,
) -> VN97LoadedDeploymentCheckpoint:
    if not isinstance(blob, bytes):
        raise TypeError("VN97CK1 blob must be bytes")
    if not HEADER_SIZE <= len(blob) <= MAX_CHECKPOINT_BYTES:
        raise VN97DeploymentCheckpointFormatError(
            "VN97CK1 byte size is outside bounds"
        )
    if sys.byteorder != "little":
        raise VN97DeploymentCheckpointError(
            "VN97CK1 loading requires a little-endian host"
        )

    (
        magic,
        version,
        header_size,
        flags,
        manifest_size,
        tensor_count,
        payload_size,
        total_size,
        content_digest,
        header_crc,
        reserved,
    ) = _HEADER.unpack_from(blob, 0)

    if magic != MAGIC or version != VERSION or header_size != HEADER_SIZE:
        raise VN97DeploymentCheckpointFormatError(
            "VN97CK1 magic/version/header mismatch"
        )
    if flags & ~KNOWN_FLAGS or reserved != 0:
        raise VN97DeploymentCheckpointFormatError(
            "VN97CK1 header flags/reserved fields are invalid"
        )
    if not 0 < manifest_size <= MAX_MANIFEST_BYTES:
        raise VN97DeploymentCheckpointFormatError(
            "VN97CK1 manifest size is outside bounds"
        )
    if not 0 < tensor_count <= MAX_TENSORS:
        raise VN97DeploymentCheckpointFormatError(
            "VN97CK1 tensor count is outside bounds"
        )
    if payload_size <= 0:
        raise VN97DeploymentCheckpointFormatError(
            "VN97CK1 payload is empty"
        )
    if total_size != len(blob):
        raise VN97DeploymentCheckpointFormatError(
            "VN97CK1 total size mismatch"
        )
    expected_total = HEADER_SIZE + manifest_size + payload_size
    if expected_total != total_size:
        raise VN97DeploymentCheckpointFormatError(
            "VN97CK1 manifest/payload sizes do not match total"
        )
    if header_crc != (zlib.crc32(blob[:72]) & 0xFFFFFFFF):
        raise VN97DeploymentCheckpointIntegrityError(
            "VN97CK1 header CRC mismatch"
        )

    content = blob[HEADER_SIZE:]
    if hashlib.sha256(content).digest() != content_digest:
        raise VN97DeploymentCheckpointIntegrityError(
            "VN97CK1 content SHA-256 mismatch"
        )

    manifest_bytes = blob[HEADER_SIZE : HEADER_SIZE + manifest_size]
    payload = blob[HEADER_SIZE + manifest_size :]
    manifest = _strict_json_object(manifest_bytes)
    has_audio = bool(flags & FLAG_AUDIO_ADAPTER)
    expected_manifest_keys = (
        {"audio", "config", "schema", "tensors"}
        if has_audio
        else {"config", "schema", "tensors"}
    )
    if set(manifest) != expected_manifest_keys:
        raise VN97DeploymentCheckpointFormatError(
            "VN97CK1 manifest keys are not exact for header flags"
        )
    if manifest["schema"] != "VN97CK1":
        raise VN97DeploymentCheckpointFormatError(
            "VN97CK1 manifest schema mismatch"
        )

    config = _parse_config(manifest["config"])
    metadata = _parse_tensor_meta(
        manifest["tensors"],
        expected_count=tensor_count,
        payload_size=payload_size,
    )

    model = VN97LanguageCore(config)
    audio_adapter = (
        _parse_audio_config(manifest["audio"], d_model=config.d_model)
        if has_audio
        else None
    )
    expected_state = dict(model.state_dict())
    if audio_adapter is not None:
        for name, tensor in audio_adapter.state_dict().items():
            expected_state[f"audio_adapter.{name}"] = tensor
    expected_names = sorted(expected_state)
    actual_names = [entry[0] for entry in metadata]
    if actual_names != expected_names:
        raise VN97DeploymentCheckpointFormatError(
            "VN97CK1 tensor names do not match canonical VN97 model state"
        )

    tensors: dict[str, torch.Tensor] = {}
    for name, shape, offset, size, digest in metadata:
        raw = payload[offset : offset + size]
        if hashlib.sha256(raw).hexdigest() != digest:
            raise VN97DeploymentCheckpointIntegrityError(
                f"VN97CK1 tensor SHA-256 mismatch: {name}"
            )
        expected_shape = tuple(int(v) for v in expected_state[name].shape)
        if shape != expected_shape:
            raise VN97DeploymentCheckpointFormatError(
                f"VN97CK1 tensor shape mismatch: {name}"
            )
        buffer = bytearray(raw)
        tensor = torch.frombuffer(
            buffer,
            dtype=torch.float32,
        ).clone().reshape(shape)
        tensors[name] = tensor

    language_tensors = {
        name: tensor
        for name, tensor in tensors.items()
        if not name.startswith("audio_adapter.")
    }
    _require_tied_checkpoint_values(language_tensors, config)

    try:
        model.load_state_dict(language_tensors, strict=True)
        if audio_adapter is not None:
            audio_adapter.load_state_dict(
                {
                    name.removeprefix("audio_adapter."): tensor
                    for name, tensor in tensors.items()
                    if name.startswith("audio_adapter.")
                },
                strict=True,
            )
    except RuntimeError as exc:
        raise VN97DeploymentCheckpointFormatError(
            "VN97CK1 state_dict is incompatible with canonical VN97 model"
        ) from exc

    if config.embedding_rank is None:
        if model.embedding.weight is not model.lm_head.weight:
            raise VN97DeploymentCheckpointIntegrityError(
                "VN97 tied embedding/head identity was not preserved"
            )
    else:
        if (
            model.embedding.token_factors is not model.lm_head.token_factors
            or model.embedding.projection is not model.lm_head.projection
        ):
            raise VN97DeploymentCheckpointIntegrityError(
                "VN97 factorized tied embedding/head identity was not preserved"
            )

    model.eval()
    if audio_adapter is not None:
        audio_adapter.eval()
    return VN97LoadedDeploymentCheckpoint(
        model=model,
        config=config,
        checkpoint_sha256=hashlib.sha256(blob).hexdigest(),
        tensor_count=tensor_count,
        audio_adapter=audio_adapter,
    )


def save_deployment_checkpoint(
    model: VN97LanguageCore,
    path: str | os.PathLike[str],
    *,
    audio_adapter: AudioFrameAdapter | None = None,
) -> str:
    blob = build_deployment_checkpoint(
        model,
        audio_adapter=audio_adapter,
    )
    target = Path(path)
    if target.is_symlink():
        raise VN97DeploymentCheckpointError(
            "VN97CK1 target must not be a symlink"
        )
    parent = target.parent if str(target.parent) else Path(".")
    parent.mkdir(parents=True, exist_ok=True)
    if parent.is_symlink():
        raise VN97DeploymentCheckpointError(
            "VN97CK1 parent must not be a symlink"
        )

    fd, temp_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=parent,
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "wb", closefd=True) as output:
            output.write(blob)
            output.flush()
            os.fsync(output.fileno())
        if target.is_symlink():
            raise VN97DeploymentCheckpointError(
                "VN97CK1 target became a symlink"
            )
        os.replace(temp_path, target)
        dir_fd = os.open(
            parent.resolve(strict=True),
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    finally:
        temp_path.unlink(missing_ok=True)

    actual = target.read_bytes()
    if actual != blob:
        raise VN97DeploymentCheckpointIntegrityError(
            "VN97CK1 post-write verification failed"
        )
    return hashlib.sha256(blob).hexdigest()


def load_deployment_checkpoint_file(
    path: str | os.PathLike[str],
) -> VN97LoadedDeploymentCheckpoint:
    target = Path(path)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(target, flags)
    except OSError as exc:
        raise VN97DeploymentCheckpointError(
            "VN97CK1 source could not be opened safely"
        ) from exc

    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise VN97DeploymentCheckpointError(
                "VN97CK1 source must be a regular file"
            )
        size = info.st_size
        if not HEADER_SIZE <= size <= MAX_CHECKPOINT_BYTES:
            raise VN97DeploymentCheckpointFormatError(
                "VN97CK1 file size is outside bounds"
            )

        out = bytearray()
        while len(out) < size:
            chunk = os.read(fd, min(1024 * 1024, size - len(out)))
            if not chunk:
                break
            out.extend(chunk)

        after = os.fstat(fd)
        if (
            len(out) != size
            or after.st_size != size
            or after.st_ino != info.st_ino
            or after.st_dev != info.st_dev
        ):
            raise VN97DeploymentCheckpointIntegrityError(
                "VN97CK1 file changed while being read"
            )
        return load_deployment_checkpoint(bytes(out))
    finally:
        os.close(fd)


def build_bootstrap_bundle_from_checkpoint(
    checkpoint: bytes,
    *,
    tokenizer: VN97TokenizerPackage,
    capability_version: int,
    signer: VN97BootstrapSigner,
    source_origin: str,
    source_license: str,
    tile_rows: int = 16,
    tile_cols: int = 16,
) -> VN97BootstrapBundle:
    loaded = load_deployment_checkpoint(checkpoint)
    if tokenizer.vocab_size != loaded.config.vocab_size:
        raise VN97DeploymentCheckpointFormatError(
            "VN97CK1 vocabulary does not match VN97TK1 tokenizer"
        )
    source = CapabilitySource(
        source_origin,
        loaded.checkpoint_sha256,
        source_license,
    )
    return build_bootstrap_bundle(
        loaded.model,
        tokenizer=tokenizer,
        audio_adapter=loaded.audio_adapter,
        source=source,
        capability_version=capability_version,
        signer=signer,
        tile_rows=tile_rows,
        tile_cols=tile_cols,
    )
