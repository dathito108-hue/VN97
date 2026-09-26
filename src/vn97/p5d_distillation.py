from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math

from .config import VN97Config


P5D0_SCHEMA = "VN97P5D0"
P5D0_PROFILE_ID = "vn97-p5d0-falcon3-mamba-distillation-v1"

TEACHER_REPO = "tiiuae/Falcon3-Mamba-7B-Instruct"
TEACHER_ARCHITECTURE = "Mamba1 causal decoder"
TEACHER_LICENSE = "TII Falcon-LLM License 2.0"
TEACHER_LAYERS = 64
TEACHER_D_MODEL = 4096
TEACHER_D_STATE = 16
TEACHER_CONTEXT = 32_768
TEACHER_VOCAB_APPROX = 65_000

TARGET_D_MODEL = 1536
TARGET_LAYERS = 32
TARGET_D_STATE = 16
TARGET_EMBEDDING_RANK = 768
TARGET_DT_MIN = 1e-4
TARGET_DT_MAX = 1.0

DISTILL_SEQUENCE_LENGTH = 256
DISTILL_LOGICAL_BATCH = 8
DISTILL_MICROBATCH = 1

# Distillation is deliberately staged. Dense float-shadow weights are trained
# first; ternary QAT is a later milestone only after held-out capability moves.
FLOAT_SHADOW_REQUIRED = True
TERNARY_DURING_INITIAL_DISTILLATION = False


@dataclass(frozen=True)
class P5DTeacherContract:
    repo: str
    architecture: str
    license_name: str
    layers: int
    d_model: int
    d_state: int
    context_length: int
    vocab_approx: int

    def __post_init__(self) -> None:
        if (
            not self.repo
            or not self.architecture
            or not self.license_name
        ):
            raise ValueError(
                "teacher identity fields must be non-empty"
            )
        if (
            self.layers <= 0
            or self.d_model <= 0
            or self.d_state <= 0
            or self.context_length <= 0
            or self.vocab_approx <= 1
        ):
            raise ValueError(
                "teacher dimensions must be positive"
            )

    def canonical_object(
        self,
    ) -> dict[str, object]:
        return {
            "architecture":
                self.architecture,
            "context_length":
                self.context_length,
            "d_model":
                self.d_model,
            "d_state":
                self.d_state,
            "layers":
                self.layers,
            "license":
                self.license_name,
            "repo":
                self.repo,
            "vocab_approx":
                self.vocab_approx,
        }


TEACHER = P5DTeacherContract(
    repo=TEACHER_REPO,
    architecture=TEACHER_ARCHITECTURE,
    license_name=TEACHER_LICENSE,
    layers=TEACHER_LAYERS,
    d_model=TEACHER_D_MODEL,
    d_state=TEACHER_D_STATE,
    context_length=TEACHER_CONTEXT,
    vocab_approx=TEACHER_VOCAB_APPROX,
)


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
        embedding_rank=TARGET_EMBEDDING_RANK,
    )


def estimate_target_parameters(
    *,
    vocab_size: int,
) -> int:
    if vocab_size <= 1:
        raise ValueError(
            "vocab_size must be > 1"
        )

    d_model = TARGET_D_MODEL
    d_state = TARGET_D_STATE
    rank = TARGET_EMBEDDING_RANK

    embedding = (
        vocab_size * rank
        + rank * d_model
    )

    # Per VN97 block:
    # in_proj: d -> 2d
    # dt_proj: d -> d + bias
    # b_proj/c_proj: d -> d_state
    # out_proj: d -> d
    # a_log: d x d_state
    # RMSNorm: d
    per_layer = (
        2 * d_model * d_model
        + d_model * d_model
        + d_model
        + d_state * d_model
        + d_state * d_model
        + d_model * d_model
        + d_model * d_state
        + d_model
    )

    return (
        embedding
        + TARGET_LAYERS * per_layer
        + d_model
    )


def recurrent_state_bytes(
    *,
    batch_size: int = 1,
    bytes_per_value: int = 2,
) -> int:
    if batch_size <= 0:
        raise ValueError(
            "batch_size must be positive"
        )
    if bytes_per_value <= 0:
        raise ValueError(
            "bytes_per_value must be positive"
        )
    return (
        batch_size
        * TARGET_LAYERS
        * TARGET_D_MODEL
        * TARGET_D_STATE
        * bytes_per_value
    )


def approximate_packed_ternary_bytes(
    *,
    vocab_size: int,
) -> int:
    """Lower-bound planning estimate for 2-bit packed model parameters.

    Embedding factors and non-matrix scalars are not all physically ternary in
    the current runtime, so this is a planning number rather than an artifact
    size guarantee.
    """
    parameters = estimate_target_parameters(
        vocab_size=vocab_size
    )
    return math.ceil(
        parameters * 2 / 8
    )


@dataclass(frozen=True)
class P5DDistillationRecord:
    record_id: str
    category: str
    prompt: str
    teacher_response: str
    source: str

    def __post_init__(self) -> None:
        if (
            not self.record_id
            or not self.category
            or not self.prompt
            or not self.teacher_response
            or not self.source
        ):
            raise ValueError(
                "distillation record fields must be non-empty"
            )

    def canonical_object(
        self,
    ) -> dict[str, str]:
        return {
            "category":
                self.category,
            "prompt":
                self.prompt,
            "record_id":
                self.record_id,
            "source":
                self.source,
            "teacher_response":
                self.teacher_response,
        }

    @property
    def sha256(
        self,
    ) -> str:
        payload = json.dumps(
            self.canonical_object(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(
            b"VN97P5DREC1\0"
            + payload
        ).hexdigest()


def profile_object(
    *,
    vocab_size: int,
) -> dict[str, object]:
    return {
        "distillation": {
            "float_shadow_required":
                FLOAT_SHADOW_REQUIRED,
            "logical_batch":
                DISTILL_LOGICAL_BATCH,
            "microbatch":
                DISTILL_MICROBATCH,
            "sequence_length":
                DISTILL_SEQUENCE_LENGTH,
            "ternary_during_initial_distillation":
                TERNARY_DURING_INITIAL_DISTILLATION,
        },
        "profile_id":
            P5D0_PROFILE_ID,
        "schema":
            P5D0_SCHEMA,
        "target": {
            "approx_packed_ternary_bytes":
                approximate_packed_ternary_bytes(
                    vocab_size=vocab_size
                ),
            "d_model":
                TARGET_D_MODEL,
            "d_state":
                TARGET_D_STATE,
            "embedding_rank":
                TARGET_EMBEDDING_RANK,
            "estimated_parameters":
                estimate_target_parameters(
                    vocab_size=vocab_size
                ),
            "layers":
                TARGET_LAYERS,
            "recurrent_state_bytes_fp16_batch1":
                recurrent_state_bytes(),
            "vocab_size":
                vocab_size,
        },
        "teacher":
            TEACHER.canonical_object(),
    }


def profile_sha256(
    *,
    vocab_size: int,
) -> str:
    payload = json.dumps(
        profile_object(
            vocab_size=vocab_size
        ),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(
        b"VN97P5D0\0"
        + payload
    ).hexdigest()
