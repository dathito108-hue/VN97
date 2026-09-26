from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping

import torch

from .config import VN97Config
from .quantization import ternary_symbols_and_scales


P5B_SCHEMA = "VN97P5B1"
P5B_PROFILE_ID = "vn97-p5b-mamba130m-transplant-v1"

SOURCE_REPO = "state-spaces/mamba-130m-hf"
SOURCE_LICENSE = "apache-2.0"
SOURCE_MODEL_SHA256 = (
    "1a5ed29c492ef4d485df3b7c2c8109771696589855b2162ad1ba618b6067cbea"
)

SOURCE_HIDDEN = 768
SOURCE_INNER = 1536
SOURCE_LAYERS = 24
SOURCE_STATE = 16
SOURCE_DT_RANK = 48
SOURCE_VOCAB = 50280
SOURCE_CONV_KERNEL = 4

TARGET_D_MODEL = 768
TARGET_LAYERS = 24
TARGET_D_STATE = 16
TARGET_EMBEDDING_RANK = 384
TARGET_DT_MIN = 1e-3
TARGET_DT_MAX = 1e-1


@dataclass(frozen=True)
class MambaSourceSpec:
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    state_size: int
    time_step_rank: int
    vocab_size: int
    conv_kernel: int
    expand: int
    rms_norm: bool
    use_bias: bool

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, object],
    ) -> "MambaSourceSpec":
        hidden = int(
            config.get(
                "hidden_size",
                config.get(
                    "d_model",
                    0,
                ),
            )
        )
        intermediate = int(
            config.get(
                "intermediate_size",
                config.get(
                    "d_inner",
                    0,
                ),
            )
        )
        layers = int(
            config.get(
                "num_hidden_layers",
                config.get(
                    "n_layer",
                    0,
                ),
            )
        )
        return cls(
            hidden_size=hidden,
            intermediate_size=intermediate,
            num_hidden_layers=layers,
            state_size=int(
                config.get(
                    "state_size",
                    16,
                )
            ),
            time_step_rank=int(
                config.get(
                    "time_step_rank",
                    math.ceil(
                        hidden
                        / 16
                    ),
                )
            ),
            vocab_size=int(
                config.get(
                    "vocab_size",
                    0,
                )
            ),
            conv_kernel=int(
                config.get(
                    "conv_kernel",
                    4,
                )
            ),
            expand=int(
                config.get(
                    "expand",
                    2,
                )
            ),
            rms_norm=bool(
                config.get(
                    "rms_norm",
                    True,
                )
            ),
            use_bias=bool(
                config.get(
                    "use_bias",
                    False,
                )
            ),
        )

    def canonical_object(
        self,
    ) -> dict[str, object]:
        return {
            "conv_kernel":
                self.conv_kernel,
            "expand":
                self.expand,
            "hidden_size":
                self.hidden_size,
            "intermediate_size":
                self.intermediate_size,
            "num_hidden_layers":
                self.num_hidden_layers,
            "rms_norm":
                self.rms_norm,
            "state_size":
                self.state_size,
            "time_step_rank":
                self.time_step_rank,
            "use_bias":
                self.use_bias,
            "vocab_size":
                self.vocab_size,
        }


@dataclass(frozen=True)
class P5BCompatibility:
    feasible: bool
    lossless: bool
    alignment_required: bool
    exact_components: tuple[str, ...]
    compressed_components: tuple[str, ...]
    lexical_components: tuple[str, ...]
    unsupported_components: tuple[str, ...]
    notes: tuple[str, ...]

    def canonical_object(
        self,
    ) -> dict[str, object]:
        return {
            "alignment_required":
                self.alignment_required,
            "compressed_components":
                list(
                    self.compressed_components
                ),
            "exact_components":
                list(
                    self.exact_components
                ),
            "feasible":
                self.feasible,
            "lexical_components":
                list(
                    self.lexical_components
                ),
            "lossless":
                self.lossless,
            "notes":
                list(
                    self.notes
                ),
            "unsupported_components":
                list(
                    self.unsupported_components
                ),
        }


def target_config(
    *,
    vocab_size: int,
) -> VN97Config:
    return VN97Config(
        vocab_size=vocab_size,
        d_model=TARGET_D_MODEL,
        n_layers=TARGET_LAYERS,
        d_state=TARGET_D_STATE,
        dt_min=TARGET_DT_MIN,
        dt_max=TARGET_DT_MAX,
        embedding_rank=
            TARGET_EMBEDDING_RANK,
    )


def assess_compatibility(
    spec: MambaSourceSpec,
) -> P5BCompatibility:
    exact = (
        "24x RMSNorm scale",
        "final RMSNorm scale",
        "diagonal A_log after channel selection",
        "dt bias after channel selection",
        "layer count",
        "external hidden width",
        "SSM state width",
    )
    compressed = (
        "in_proj signal/gate 1536->768 channel selection",
        "out_proj 1536->768 channel selection",
        "B projection 1536->768 channel selection",
        "C projection 1536->768 channel selection",
        "dt projection composite 1536x1536->768x768",
    )
    lexical = (
        "source embedding -> VN97 byte/BPE token semantic projection",
        "mapped embedding -> rank-384 VN97 factorization",
        "VN97 tied LM head from transplanted embedding",
    )
    unsupported = (
        "Mamba depthwise causal conv1d",
        "Mamba inner D skip parameter",
    )

    compatible = (
        spec.hidden_size
        == SOURCE_HIDDEN
        and spec.intermediate_size
        == SOURCE_INNER
        and spec.num_hidden_layers
        == SOURCE_LAYERS
        and spec.state_size
        == SOURCE_STATE
        and spec.time_step_rank
        == SOURCE_DT_RANK
        and spec.expand == 2
        and spec.rms_norm
        and not spec.use_bias
    )

    notes = (
        "VN97 and Mamba are both selective diagonal SSMs with input-dependent dt/B/C.",
        "The bridge is intentionally not a lossless format conversion.",
        "Mamba's 2x inner expansion is compressed to one VN97 channel per hidden dimension.",
        "Conv1d and D-skip have no native VN97 equivalents and are not copied.",
        "A short post-transplant alignment/calibration stage is required before promotion.",
        "The source model is used only during conversion; runtime remains a single native VN97 model.",
    )
    return P5BCompatibility(
        feasible=compatible,
        lossless=False,
        alignment_required=True,
        exact_components=exact,
        compressed_components=compressed,
        lexical_components=lexical,
        unsupported_components=unsupported,
        notes=notes,
    )


def expected_source_shapes(
    spec: MambaSourceSpec,
) -> dict[str, tuple[int, ...]]:
    shapes: dict[
        str,
        tuple[int, ...],
    ] = {
        "backbone.embeddings.weight":
            (
                spec.vocab_size,
                spec.hidden_size,
            ),
        "backbone.norm_f.weight":
            (
                spec.hidden_size,
            ),
    }
    for layer in range(
        spec.num_hidden_layers
    ):
        prefix = (
            f"backbone.layers.{layer}"
        )
        mixer = (
            prefix
            + ".mixer"
        )
        shapes.update(
            {
                prefix
                + ".norm.weight":
                    (
                        spec.hidden_size,
                    ),
                mixer
                + ".A_log":
                    (
                        spec.intermediate_size,
                        spec.state_size,
                    ),
                mixer
                + ".D":
                    (
                        spec.intermediate_size,
                    ),
                mixer
                + ".conv1d.weight":
                    (
                        spec.intermediate_size,
                        1,
                        spec.conv_kernel,
                    ),
                mixer
                + ".conv1d.bias":
                    (
                        spec.intermediate_size,
                    ),
                mixer
                + ".dt_proj.bias":
                    (
                        spec.intermediate_size,
                    ),
                mixer
                + ".dt_proj.weight":
                    (
                        spec.intermediate_size,
                        spec.time_step_rank,
                    ),
                mixer
                + ".in_proj.weight":
                    (
                        spec.intermediate_size
                        * 2,
                        spec.hidden_size,
                    ),
                mixer
                + ".out_proj.weight":
                    (
                        spec.hidden_size,
                        spec.intermediate_size,
                    ),
                mixer
                + ".x_proj.weight":
                    (
                        spec.time_step_rank
                        + spec.state_size
                        * 2,
                        spec.intermediate_size,
                    ),
            }
        )
    return shapes


def validate_source_tensor_shapes(
    tensors: Mapping[
        str,
        torch.Tensor,
    ],
    spec: MambaSourceSpec,
) -> None:
    shapes = expected_source_shapes(
        spec
    )
    missing = [
        name
        for name in shapes
        if name not in tensors
    ]
    if missing:
        raise ValueError(
            "Mamba source is missing required tensors: "
            + ", ".join(
                missing[:8]
            )
        )

    bad = []
    for name, shape in (
        shapes.items()
    ):
        actual = tuple(
            int(value)
            for value in tensors[
                name
            ].shape
        )
        if actual != shape:
            bad.append(
                (
                    name,
                    actual,
                    shape,
                )
            )
    if bad:
        name, actual, expected = bad[0]
        raise ValueError(
            "Mamba tensor shape mismatch for "
            f"{name}: got {actual}, expected {expected}"
        )


def _channel_energy(
    tensors: Mapping[
        str,
        torch.Tensor,
    ],
    *,
    layer: int,
    spec: MambaSourceSpec,
) -> torch.Tensor:
    prefix = (
        f"backbone.layers.{layer}.mixer"
    )
    in_proj = (
        tensors[
            prefix
            + ".in_proj.weight"
        ].float()
    )
    signal = in_proj[
        :spec.intermediate_size
    ]
    gate = in_proj[
        spec.intermediate_size:
    ]
    out_proj = (
        tensors[
            prefix
            + ".out_proj.weight"
        ].float()
    )
    x_proj = (
        tensors[
            prefix
            + ".x_proj.weight"
        ].float()
    )
    dt_proj = (
        tensors[
            prefix
            + ".dt_proj.weight"
        ].float()
    )
    conv = (
        tensors[
            prefix
            + ".conv1d.weight"
        ].float()
        .squeeze(1)
    )
    d_skip = (
        tensors[
            prefix
            + ".D"
        ].float()
    )

    score = (
        signal.square().mean(
            dim=1
        )
        + gate.square().mean(
            dim=1
        )
        + out_proj.square().mean(
            dim=0
        )
        + x_proj.square().mean(
            dim=0
        )
        + dt_proj.square().mean(
            dim=1
        )
        + conv.square().mean(
            dim=1
        )
        + d_skip.square()
    )
    return score


def selected_channels(
    tensors: Mapping[
        str,
        torch.Tensor,
    ],
    *,
    layer: int,
    spec: MambaSourceSpec,
    target_width: int =
        TARGET_D_MODEL,
) -> torch.Tensor:
    if (
        target_width <= 0
        or target_width
        > spec.intermediate_size
    ):
        raise ValueError(
            "invalid P5B target channel width"
        )
    energy = _channel_energy(
        tensors,
        layer=layer,
        spec=spec,
    )
    chosen = torch.topk(
        energy,
        k=target_width,
        largest=True,
        sorted=False,
    ).indices
    return torch.sort(
        chosen
    ).values


def snap_ternary_parameter(
    weight: torch.Tensor,
    *,
    threshold: float,
) -> torch.Tensor:
    source = weight.float()
    symbols, scales = (
        ternary_symbols_and_scales(
            source,
            threshold,
        )
    )
    density = (
        symbols.ne(0)
        .float()
        .mean(dim=1)
        .clamp_min(1.0 / source.shape[1])
    )
    stored_scale = (
        scales
        / density
    )
    return (
        symbols.float()
        * stored_scale.unsqueeze(1)
    ).to(
        dtype=weight.dtype
    )


def layer_mapping(
    tensors: Mapping[
        str,
        torch.Tensor,
    ],
    *,
    layer: int,
    spec: MambaSourceSpec,
    threshold: float,
) -> dict[str, torch.Tensor]:
    idx = selected_channels(
        tensors,
        layer=layer,
        spec=spec,
    )
    prefix = (
        f"backbone.layers.{layer}"
    )
    mixer = (
        prefix
        + ".mixer"
    )

    in_proj = (
        tensors[
            mixer
            + ".in_proj.weight"
        ].float()
    )
    source_signal = in_proj[
        :spec.intermediate_size
    ]
    source_gate = in_proj[
        spec.intermediate_size:
    ]

    signal = source_signal[
        idx
    ]
    gate = source_gate[
        idx
    ]
    target_in = torch.cat(
        [
            signal,
            gate,
        ],
        dim=0,
    )

    x_proj = (
        tensors[
            mixer
            + ".x_proj.weight"
        ].float()
    )
    dt_rank = (
        spec.time_step_rank
    )
    b_start = dt_rank
    c_start = (
        b_start
        + spec.state_size
    )
    source_dt_low = x_proj[
        :dt_rank
    ]
    source_b = x_proj[
        b_start:c_start
    ]
    source_c = x_proj[
        c_start:
            c_start
            + spec.state_size
    ]

    source_dt_high = (
        tensors[
            mixer
            + ".dt_proj.weight"
        ].float()
    )
    full_dt = (
        source_dt_high
        @ source_dt_low
    )
    target_dt = full_dt[
        idx
    ][
        :,
        idx
    ]

    target_b = source_b[
        :,
        idx
    ]
    target_c = source_c[
        :,
        idx
    ]
    target_out = (
        tensors[
            mixer
            + ".out_proj.weight"
        ].float()[
            :,
            idx
        ]
    )

    mapped = {
        "norm.weight":
            tensors[
                prefix
                + ".norm.weight"
            ].float(),
        "core.a_log":
            tensors[
                mixer
                + ".A_log"
            ].float()[
                idx
            ],
        "core.in_proj.weight":
            snap_ternary_parameter(
                target_in,
                threshold=
                    threshold,
            ),
        "core.dt_proj.weight":
            snap_ternary_parameter(
                target_dt,
                threshold=
                    threshold,
            ),
        "core.dt_proj.bias":
            tensors[
                mixer
                + ".dt_proj.bias"
            ].float()[
                idx
            ],
        "core.b_proj.weight":
            snap_ternary_parameter(
                target_b,
                threshold=
                    threshold,
            ),
        "core.c_proj.weight":
            snap_ternary_parameter(
                target_c,
                threshold=
                    threshold,
            ),
        "core.out_proj.weight":
            snap_ternary_parameter(
                target_out,
                threshold=
                    threshold,
            ),
    }
    return mapped


def canonical_report_sha256(
    report: Mapping[
        str,
        object,
    ],
) -> str:
    payload = json.dumps(
        report,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(
        b"VN97P5B1\0"
        + payload
    ).hexdigest()


def load_source_config(
    root: Path,
) -> dict[str, object]:
    path = (
        root
        / "config.json"
    )
    data = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )
    if not isinstance(
        data,
        dict,
    ):
        raise ValueError(
            "Mamba config must be a JSON object"
        )
    return data
