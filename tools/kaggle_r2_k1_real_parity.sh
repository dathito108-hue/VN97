#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

WORK_ROOT="${VN97_KAGGLE_WORK_ROOT:-/kaggle/working/vn97-r2-k1}"
SOURCE_ROOT="$WORK_ROOT/mamba2-source"
MAMBA_ORACLE="$WORK_ROOT/mamba-oracle"
OUTPUT_JSON="$WORK_ROOT/vn97-k1-real-parity.json"
MAMBA_COMMIT="e9594ce1c732d97440f0332fdc43170a2294dbfa"

mkdir -p "$WORK_ROOT"

python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is required for VN97 K1 parity")
name = torch.cuda.get_device_name(0)
print("torch=", torch.__version__)
print("cuda=", torch.version.cuda)
print("gpu=", name)
if "T4" not in name.upper():
    raise SystemExit(f"Expected Kaggle T4 on cuda:0, got {name!r}")
props = torch.cuda.get_device_properties(0)
print("vram_gib=", round(props.total_memory / 1024**3, 2))
PY

echo "VN97 K1: checking storage"
df -h /kaggle/working || true
AVAILABLE_KB="$(df -Pk /kaggle/working | awk 'NR==2 {print $4}')"
REQUIRED_KB="$((16 * 1024 * 1024))"
if [[ -z "$AVAILABLE_KB" || "$AVAILABLE_KB" -lt "$REQUIRED_KB" ]]; then
  echo "K1 needs at least 16 GiB free in /kaggle/working." >&2
  exit 2
fi

python -m pip install --disable-pip-version-check -q   "transformers>=4.45,<5"   "huggingface_hub>=0.24,<2"   "einops>=0.7"   "packaging>=23"   "ninja>=1.11"

python -m pip install --disable-pip-version-check --no-deps -e .

if [[ ! -d "$MAMBA_ORACLE/.git" ]]; then
  rm -rf "$MAMBA_ORACLE"
  git clone --filter=blob:none https://github.com/state-spaces/mamba.git "$MAMBA_ORACLE"
fi

git -C "$MAMBA_ORACLE" fetch --depth 1 origin "$MAMBA_COMMIT"
git -C "$MAMBA_ORACLE" checkout --detach "$MAMBA_COMMIT"
ACTUAL_MAMBA_COMMIT="$(git -C "$MAMBA_ORACLE" rev-parse HEAD)"
if [[ "$ACTUAL_MAMBA_COMMIT" != "$MAMBA_COMMIT" ]]; then
  echo "Pinned Mamba oracle commit mismatch." >&2
  exit 3
fi

if [[ ! -f "$SOURCE_ROOT/pytorch_model.bin" ]]; then
  bash tools/r2_g02_fetch_mamba2.sh "$SOURCE_ROOT"
else
  echo "Reusing existing pinned Mamba-2 source at $SOURCE_ROOT"
fi

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$MAMBA_ORACLE:$REPO_ROOT/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

python -m vn97.r2.mamba2_kaggle_real_parity   --source-root "$SOURCE_ROOT"   --evidence "$REPO_ROOT/evidence/r2-g03-real-transfer.json"   --output "$OUTPUT_JSON"   --max-prompt-tokens "${VN97_K1_MAX_PROMPT_TOKENS:-12}"   --generation-tokens "${VN97_K1_GENERATION_TOKENS:-4}"   --max-logit-error "${VN97_K1_MAX_LOGIT_ERROR:-0.005}"   --max-hidden-error "${VN97_K1_MAX_HIDDEN_ERROR:-0.005}"   --max-state-error "${VN97_K1_MAX_STATE_ERROR:-0.005}"

python - <<PY
import hashlib, json
from pathlib import Path
p = Path("$OUTPUT_JSON")
data = p.read_bytes()
body = json.loads(data)
print("K1 receipt:", p)
print("K1 sha256:", hashlib.sha256(data).hexdigest())
print("K1 status:", body["status"])
print("K1 receipt_id:", body["receipt_id"])
print("K1 metrics:", json.dumps(body["metrics"], sort_keys=True))
print("K1 generation_exact:", body["generation_exact"])
PY

(
  cd "$WORK_ROOT"
  sha256sum     vn97-k1-real-parity.json     mamba2-source/config.json     mamba2-source/pytorch_model.bin     > K1_SHA256SUMS
)

ZIP="/kaggle/working/VN97-R2-K1-real-parity.zip"
rm -f "$ZIP"
python - <<PY
from pathlib import Path
import zipfile
root = Path("$WORK_ROOT")
out = Path("$ZIP")
with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    for name in ("vn97-k1-real-parity.json", "K1_SHA256SUMS"):
        z.write(root / name, arcname=name)
print(out)
PY

echo "VN97 K1 real parity complete"
echo "output=$OUTPUT_JSON"
echo "bundle=$ZIP"
