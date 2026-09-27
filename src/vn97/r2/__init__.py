from .data_bridge import (
    assert_r2_data_compatible,
    build_causal_windows,
    build_completion_windows,
    load_vn97tk1,
    windows_to_tensors,
)
from .dense_training import (
    R2DenseTrainingConfig,
    R2DenseTrainingResult,
    evaluate_dense_loss,
    train_dense,
)
from .fast_path_training import (
    R2FastPathConfig,
    R2FastPathResult,
    align_fast_path,
)
from .bridge import VN97R2InferenceView, assert_tokenizer_compatible
from .evaluation import (
    EvaluationDomain,
    R2EvaluationDecision,
    R2EvaluationMetrics,
    R2ValidationGate,
    R2ValidationThresholds,
)
from .capability import (
    CAPABILITY_PACK_SCHEMA,
    CapabilityManifest,
    capability_ledger_entry,
    merge_capability_delta,
    validate_delta,
)
from .checkpoint import (
    R2_CHECKPOINT_SCHEMA,
    VN97R2CheckpointError,
    load_r2_checkpoint,
    save_r2_checkpoint,
    sha256_file,
)
from .config import (
    R2_ARCHITECTURE_ID,
    VN97R2Config,
    r2_cpu_pilot_config,
    r2_mobile_1b_config,
    r2_smoke_config,
)
from .model import VN97R2Model, VN97R2State
from .runtime import (
    CognitionDecision,
    CognitionMode,
    CognitionRequest,
    RealtimeBudget,
    ReflexPlanContract,
    VN97R2ExecutionPolicy,
)
from .training_plan import (
    R2_CANONICAL_TRAINING_SEQUENCE,
    R2PromotionPolicy,
    R2TrainingStage,
    StageEvidence,
)

__all__ = [
    "CAPABILITY_PACK_SCHEMA",
    "R2DenseTrainingConfig",
    "R2DenseTrainingResult",
    "R2FastPathConfig",
    "R2FastPathResult",
    "EvaluationDomain",
    "R2EvaluationDecision",
    "R2EvaluationMetrics",
    "R2ValidationGate",
    "R2ValidationThresholds",
    "CapabilityManifest",
    "CognitionDecision",
    "CognitionMode",
    "CognitionRequest",
    "R2_ARCHITECTURE_ID",
    "R2_CANONICAL_TRAINING_SEQUENCE",
    "R2_CHECKPOINT_SCHEMA",
    "R2PromotionPolicy",
    "R2TrainingStage",
    "RealtimeBudget",
    "ReflexPlanContract",
    "StageEvidence",
    "VN97R2CheckpointError",
    "VN97R2Config",
    "VN97R2ExecutionPolicy",
    "VN97R2Model",
    "VN97R2State",
    "capability_ledger_entry",
    "load_r2_checkpoint",
    "merge_capability_delta",
    "r2_cpu_pilot_config",
    "r2_mobile_1b_config",
    "r2_smoke_config",
    "save_r2_checkpoint",
    "sha256_file",
    "validate_delta",
    "VN97R2InferenceView",
    "assert_tokenizer_compatible",
    "assert_r2_data_compatible",
    "build_causal_windows",
    "build_completion_windows",
    "load_vn97tk1",
    "windows_to_tensors",
    "evaluate_dense_loss",
    "train_dense",
    "align_fast_path",
]
