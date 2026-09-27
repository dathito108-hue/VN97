from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import hashlib
import json


class VN97EvolutionStage(str, Enum):
    G0_DIRECT_TRANSFER = "G0_DIRECT_TRANSFER"
    G1_AUGMENTED = "G1_AUGMENTED"
    G2_ADAPTIVE_CORE = "G2_ADAPTIVE_CORE"
    G3_NATIVE_BLOCKS = "G3_NATIVE_BLOCKS"
    G4_NATIVE_PRETRAIN = "G4_NATIVE_PRETRAIN"


_STAGE_ORDER = {
    VN97EvolutionStage.G0_DIRECT_TRANSFER: 0,
    VN97EvolutionStage.G1_AUGMENTED: 1,
    VN97EvolutionStage.G2_ADAPTIVE_CORE: 2,
    VN97EvolutionStage.G3_NATIVE_BLOCKS: 3,
    VN97EvolutionStage.G4_NATIVE_PRETRAIN: 4,
}


def _require_sha256(value: str, label: str) -> None:
    if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
        raise ValueError(f"{label} must be lowercase SHA-256")


@dataclass(frozen=True)
class VN97EvolutionCandidate:
    """Training-side candidate contract; never an activation bypass.

    Promotion still has to pass the existing M17A/M17B/M17C review,
    held-out evaluation, explicit human approval and rollback ledger.
    """

    stage: VN97EvolutionStage
    parent_checkpoint_sha256: str
    candidate_checkpoint_sha256: str
    core_layers_total: int
    core_layers_unfrozen: int
    native_blocks: int
    inherited_source_tensors: int
    fresh_initialization: bool
    held_out_evidence_sha256: str
    architecture_evidence_sha256: str

    def validate(self) -> None:
        _require_sha256(
            self.parent_checkpoint_sha256,
            "parent_checkpoint_sha256",
        )
        _require_sha256(
            self.candidate_checkpoint_sha256,
            "candidate_checkpoint_sha256",
        )
        _require_sha256(
            self.held_out_evidence_sha256,
            "held_out_evidence_sha256",
        )
        _require_sha256(
            self.architecture_evidence_sha256,
            "architecture_evidence_sha256",
        )
        if self.parent_checkpoint_sha256 == self.candidate_checkpoint_sha256:
            raise ValueError("candidate must differ from parent checkpoint")
        if self.core_layers_total != 64:
            raise ValueError("Mamba-2 2.7B lineage starts from exactly 64 layers")
        if not 0 <= self.core_layers_unfrozen <= self.core_layers_total:
            raise ValueError("core_layers_unfrozen is outside the layer range")
        if not 0 <= self.native_blocks <= self.core_layers_total:
            raise ValueError("native_blocks is outside the layer range")
        if self.inherited_source_tensors < 0:
            raise ValueError("inherited_source_tensors cannot be negative")

        if self.stage == VN97EvolutionStage.G0_DIRECT_TRANSFER:
            if (
                self.core_layers_unfrozen != 0
                or self.native_blocks != 0
                or self.inherited_source_tensors <= 0
                or self.fresh_initialization
            ):
                raise ValueError("G0 must be a frozen direct source transfer")
        elif self.stage == VN97EvolutionStage.G1_AUGMENTED:
            if (
                self.core_layers_unfrozen != 0
                or self.native_blocks != 0
                or self.inherited_source_tensors <= 0
                or self.fresh_initialization
            ):
                raise ValueError(
                    "G1 trains augmentation while preserving the frozen core"
                )
        elif self.stage == VN97EvolutionStage.G2_ADAPTIVE_CORE:
            if (
                self.core_layers_unfrozen <= 0
                or self.inherited_source_tensors <= 0
                or self.fresh_initialization
            ):
                raise ValueError(
                    "G2 must selectively adapt an inherited core"
                )
        elif self.stage == VN97EvolutionStage.G3_NATIVE_BLOCKS:
            if (
                self.native_blocks <= 0
                or self.inherited_source_tensors <= 0
                or self.fresh_initialization
            ):
                raise ValueError(
                    "G3 must replace inherited blocks incrementally"
                )
        elif self.stage == VN97EvolutionStage.G4_NATIVE_PRETRAIN:
            if (
                self.native_blocks != self.core_layers_total
                or self.inherited_source_tensors != 0
                or not self.fresh_initialization
            ):
                raise ValueError(
                    "G4 is the first Mamba-weight-independent generation"
                )

    @property
    def mamba_weight_independent(self) -> bool:
        self.validate()
        return (
            self.stage == VN97EvolutionStage.G4_NATIVE_PRETRAIN
            and self.fresh_initialization
            and self.inherited_source_tensors == 0
            and self.native_blocks == self.core_layers_total
        )

    def manifest_id(self) -> str:
        self.validate()
        body = asdict(self)
        body["stage"] = self.stage.value
        encoded = json.dumps(
            body,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        return hashlib.sha256(
            b"VN97EVOLM2G1\0" + encoded
        ).hexdigest()


def require_evolution_transition(
    parent: VN97EvolutionCandidate,
    candidate: VN97EvolutionCandidate,
) -> None:
    parent.validate()
    candidate.validate()
    if candidate.parent_checkpoint_sha256 != parent.candidate_checkpoint_sha256:
        raise ValueError("evolution candidate is not bound to exact parent")
    parent_order = _STAGE_ORDER[parent.stage]
    candidate_order = _STAGE_ORDER[candidate.stage]
    if candidate_order < parent_order:
        raise ValueError("evolution cannot move backward in stage")
    if candidate_order > parent_order + 1:
        raise ValueError("evolution cannot skip a generation stage")
    if candidate.stage == parent.stage:
        if candidate.stage == VN97EvolutionStage.G2_ADAPTIVE_CORE:
            if candidate.core_layers_unfrozen < parent.core_layers_unfrozen:
                raise ValueError("G2 cannot reduce adapted layer coverage")
        elif candidate.stage == VN97EvolutionStage.G3_NATIVE_BLOCKS:
            if candidate.native_blocks <= parent.native_blocks:
                raise ValueError("G3 must increase native block coverage")
        else:
            raise ValueError("same-stage evolution is allowed only in G2/G3")


@dataclass(frozen=True)
class VN97EvolutionPromotionEvidence:
    held_out_passed: bool
    authority_passed: bool
    rollback_ready: bool
    explicit_human_approval: bool

    def require_promotable(self) -> None:
        missing = []
        if not self.held_out_passed:
            missing.append("held-out evaluation")
        if not self.authority_passed:
            missing.append("authority regression")
        if not self.rollback_ready:
            missing.append("rollback evidence")
        if not self.explicit_human_approval:
            missing.append("explicit human approval")
        if missing:
            raise ValueError(
                "candidate cannot enter M17C promotion: "
                + ", ".join(missing)
            )
