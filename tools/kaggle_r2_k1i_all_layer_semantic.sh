#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

WORK_ROOT="${VN97_KAGGLE_WORK_ROOT:-/kaggle/working/vn97-r2-k1}"
SOURCE_ROOT="$WORK_ROOT/mamba2-source"
MAMBA_ORACLE="$WORK_ROOT/mamba-oracle"
OUTPUT_JSON="/kaggle/working/vn97-k1i-semantic-parity.json"
MAMBA_COMMIT="e9594ce1c732d97440f0332fdc43170a2294dbfa"

python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is required")
name = torch.cuda.get_device_name(0)
print("gpu=", name)
if "T4" not in name.upper():
    raise SystemExit(f"Expected Tesla T4, got {name!r}")
print("torch=", torch.__version__, "cuda=", torch.version.cuda)
PY

if [[ ! -f "$SOURCE_ROOT/pytorch_model.bin" ]]; then
  echo "Existing K1 source is missing; downloading pinned source."
  bash tools/r2_g02_fetch_mamba2.sh "$SOURCE_ROOT"
else
  echo "Reusing existing pinned Mamba-2 source at $SOURCE_ROOT"
fi

if [[ ! -d "$MAMBA_ORACLE/.git" ]]; then
  git clone --filter=blob:none https://github.com/state-spaces/mamba.git "$MAMBA_ORACLE"
fi
git -C "$MAMBA_ORACLE" fetch --depth 1 origin "$MAMBA_COMMIT"
git -C "$MAMBA_ORACLE" checkout --detach "$MAMBA_COMMIT"

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$MAMBA_ORACLE:$REPO_ROOT/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

python -m vn97.r2.mamba2_kaggle_all_layer_semantic   --source-root "$SOURCE_ROOT"   --evidence "$REPO_ROOT/evidence/r2-g03-real-transfer.json"   --output "$OUTPUT_JSON"   --max-prompt-tokens "${VN97_K1I_MAX_PROMPT_TOKENS:-4}"   --max-epsilon-units "${VN97_K1I_MAX_EPSILON_UNITS:-2.0}"

python - <<PY
import json
from pathlib import Path
p = Path("$OUTPUT_JSON")
r = json.loads(p.read_text())
print("K1I status:", r["status"])
print("K1I probe_cases:", r["probe_cases"])
print("K1I probe_tokens:", r["probe_tokens"])
print("K1I layer_samples:", r["layer_samples"])
print("K1I allowed epsilon units:", r["max_epsilon_units_allowed"])
print("K1I worst same-input result:", r["worst_same_input_result"])
print("K1I violation count:", len(r["violations"]))
print("K1I first violations:", r["violations"][:10])
print("K1I receipt_id:", r["receipt_id"])
PY
