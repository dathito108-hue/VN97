"""Small research adapter reusing the real VN97R2Model outer backbone."""
from dataclasses import asdict, dataclass
import hashlib
import importlib
import json
from pathlib import Path
import sys
import types
import uuid

import torch
from torch import nn
from adaptive_state import StateConfig
from trainable_state import TensorState, TrainableAdaptiveState

# Import actual repository modules in a private namespace; avoid the broad
# training-package __init__ facade. No source module or dependency is mocked.
ROOT = Path(__file__).resolve().parents[2]
for name, path in (("_vn97_candidate_core", ROOT / "src/vn97"),
                   ("_vn97_candidate_core.r2", ROOT / "src/vn97/r2")):
    if name not in sys.modules:
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        sys.modules[name] = module
VN97R2Config = importlib.import_module("_vn97_candidate_core.r2.config").VN97R2Config
VN97R2Model = importlib.import_module("_vn97_candidate_core.r2.model").VN97R2Model
ssm = importlib.import_module("_vn97_candidate_core.r2.ssm")


@dataclass(frozen=True)
class CandidateConfig:
    d_model: int = 24
    d_inner: int = 32
    n_layers: int = 2
    vocab_size: int = 256
    schema: str = "VN97ASC-LM1"

    def __post_init__(self):
        for value in (self.d_model, self.d_inner, self.n_layers, self.vocab_size):
            if type(value) is not int or value <= 0:
                raise ValueError("dimensions must be positive integers")
        if not self.d_model <= self.d_inner <= 1024 or self.n_layers > 8 or self.parameter_count() > 10_500_000:
            raise ValueError("candidate exceeds the bounded 10M research budget")
        if self.vocab_size != 256 or self.schema != "VN97ASC-LM1":
            raise ValueError("candidate requires the experimental byte vocabulary")

    def parameter_count(self):
        return (self.vocab_size + 1) * self.d_model + self.n_layers * (
            5 * self.d_model * self.d_inner + 4 * self.d_inner + self.d_model + 2
        )

    def fingerprint(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class CandidateBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm = ssm.R2RMSNorm(config.d_model, 1e-5)
        self.cell = TrainableAdaptiveState(config.d_model, StateConfig(config.d_inner))
        self.out_proj = nn.Linear(config.d_inner, config.d_model, bias=False)

    def initial_state(self, batch_size, *, device, dtype):
        width = self.cell.config.width
        return ssm.R2LayerState(torch.zeros(batch_size, width, 0, device=device, dtype=dtype),
                               torch.zeros(batch_size, width, 2, device=device, dtype=dtype))

    def forward(self, x, state):
        current = TensorState(self.cell.architecture, state.ssm[..., 0], state.ssm[..., 1])
        y, next_state = self.cell(self.norm(x), current)
        return x + self.out_proj(y), ssm.R2LayerState(
            state.conv, torch.stack((next_state.fast, next_state.slow), dim=-1))

    def step(self, x, state):
        y, state = self.forward(x.unsqueeze(1), state)
        return y[:, 0], state


@dataclass(frozen=True)
class CandidateStream:
    owner: str
    architecture: str
    versions: tuple
    recurrent: object


class NativeLanguageCandidate(nn.Module):
    def __init__(self, config=CandidateConfig()):
        super().__init__()
        self.config = config
        self.owner = uuid.uuid4().hex
        outer = VN97R2Config(vocab_size=config.vocab_size, d_model=config.d_model,
                            d_inner=config.d_inner, n_layers=config.n_layers,
                            d_state=2, d_conv=1, dt_rank=1, fast_layers=config.n_layers)
        self.backbone = VN97R2Model(outer)
        # Reuse embedding, tied output, normalization, layer loop and state carrier.
        # Only the recurrent blocks are experimental; old blocks are discarded.
        self.backbone.layers = nn.ModuleList(CandidateBlock(config) for _ in range(config.n_layers))

    def _versions(self):
        return tuple(p._version for p in self.parameters())

    def forward(self, ids, state=None, *, token=False):
        if ids.dtype != torch.long or ids.ndim != (1 if token else 2):
            raise ValueError("expected int64 token vector or sequence matrix")
        if ids.numel() == 0 or bool(((ids < 0) | (ids >= 256)).any()):
            raise ValueError("tokens must be nonempty bytes")
        current = None
        if state is not None:
            if not isinstance(state, CandidateStream) or (
                state.owner != self.owner or state.architecture != self.config.fingerprint()
                or state.versions != self._versions()
            ):
                raise ValueError("foreign or stale stream; restart after weight updates")
            current = state.recurrent
        fn = self.backbone.step if token else self.backbone.forward
        logits, recurrent = fn(ids, current)
        return logits, CandidateStream(self.owner, self.config.fingerprint(), self._versions(), recurrent)


def save_candidate(model, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    path = directory / "weights.pt"
    torch.save(model.state_dict(), path)
    manifest = {"schema": "VN97ASC-LMCP1", "architecture": model.config.fingerprint(),
                "config": asdict(model.config), "weights_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "production_activation_authorized": False, "tokenizer": "utf8-bytes-256"}
    (directory / "manifest.json").write_text(json.dumps(manifest, sort_keys=True) + "\n")
    return manifest


def load_candidate(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if set(manifest) != {"schema", "architecture", "config", "weights_sha256", "production_activation_authorized", "tokenizer"}:
        raise ValueError("candidate manifest fields mismatch")
    if (manifest["schema"] != "VN97ASC-LMCP1" or manifest["production_activation_authorized"] is not False
            or manifest["tokenizer"] != "utf8-bytes-256"):
        raise ValueError("invalid candidate identity or activation")
    config = CandidateConfig(**manifest["config"])
    if config.fingerprint() != manifest["architecture"]:
        raise ValueError("architecture fingerprint mismatch")
    path = directory / "weights.pt"
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["weights_sha256"]:
        raise ValueError("weights hash mismatch")
    model = NativeLanguageCandidate(config)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
    if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):
        raise ValueError("non-finite weights")
    return model
