#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

WORK_ROOT="${VN97_KAGGLE_WORK_ROOT:-/kaggle/working/vn97-r2-k1}"
SOURCE_ROOT="$WORK_ROOT/mamba2-source"
MAMBA_ORACLE="$WORK_ROOT/mamba-oracle"
OUTPUT_JSON="/kaggle/working/vn97-k1h-backend-envelope.json"
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

python -m vn97.r2.mamba2_kaggle_backend_envelope   --source-root "$SOURCE_ROOT"   --evidence "$REPO_ROOT/evidence/r2-g03-real-transfer.json"   --output "$OUTPUT_JSON"   --max-prompt-tokens "${VN97_K1H_MAX_PROMPT_TOKENS:-12}"   --continuation-tokens "${VN97_K1H_CONTINUATION_TOKENS:-4}"

python - <<PY
import json
from pathlib import Path
p = Path("$OUTPUT_JSON")
r = json.loads(p.read_text())
print("K1H status:", r["status"])
print("K1H all_argmax_equal:", r["all_argmax_equal"])
print("K1H envelope_pass:", r["envelope_pass"])
print("K1H envelope:")
for name, value in r["envelope"].items():
    print(name, value)
print("K1H receipt_id:", r["receipt_id"])
PY
