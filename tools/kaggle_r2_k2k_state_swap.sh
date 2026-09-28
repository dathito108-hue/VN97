#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

K2_ROOT="/kaggle/working/vn97-r2-k2"
CAPSULE_ROOT="$K2_ROOT/g03-capsule"
ONNX_ROOT="$K2_ROOT/g04-step-onnx"
K2J_RECEIPT="/kaggle/working/vn97-k2j-closed-loop.json"
OUTPUT_JSON="/kaggle/working/vn97-k2k-state-swap.json"

python - <<'PY'
import onnxruntime as ort
import torch

print("gpu=", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
print("torch=", torch.__version__, "cuda=", torch.version.cuda)
print("onnxruntime=", ort.__version__)
print("providers=", ort.get_available_providers())

if not torch.cuda.is_available():
    raise SystemExit("K2K requires CUDA")
if "T4" not in torch.cuda.get_device_name(0).upper():
    raise SystemExit("K2K requires Tesla T4")
if not torch.__version__.startswith("2.10.0"):
    raise SystemExit(
        f"K2K requires Torch 2.10.0 baseline; got {torch.__version__}"
    )
if ort.__version__ != "1.26.0":
    raise SystemExit(
        f"K2K requires ONNX Runtime 1.26.0; got {ort.__version__}"
    )
ort.preload_dlls()
if "CUDAExecutionProvider" not in ort.get_available_providers():
    raise SystemExit("K2K requires CUDAExecutionProvider")
PY

for required in   "$K2J_RECEIPT"   "$CAPSULE_ROOT/capsule.vn97m2g03.json"   "$ONNX_ROOT/manifest.vn97m2g04.json"   "$ONNX_ROOT/step.onnx"; do
  if [[ ! -f "$required" ]]; then
    echo "K2K missing prerequisite: $required" >&2
    echo "Run K2J successfully before K2K." >&2
    exit 2
  fi
done

AVAILABLE_KB="$(df -Pk /kaggle/working | awk 'NR==2 {print $4}')"
REQUIRED_KB="$((3 * 1024 * 1024))"
df -h /kaggle/working || true
if [[ -z "$AVAILABLE_KB" || "$AVAILABLE_KB" -lt "$REQUIRED_KB" ]]; then
  echo "K2K needs at least 3 GiB free working storage" >&2
  exit 3
fi

export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -m vn97.r2.mamba2_kaggle_k2k_state_swap   --capsule-root "$CAPSULE_ROOT"   --bundle-dir "$ONNX_ROOT"   --k2j-receipt "$K2J_RECEIPT"   --output "$OUTPUT_JSON"

python - <<PY
import hashlib
import json
from pathlib import Path

p = Path("$OUTPUT_JSON")
data = p.read_bytes()
r = json.loads(data)

print("K2K status:", r["status"])
print("K2K prompt_index:", r["prompt_index"])
print("K2K divergence_step:", r["divergence_step"])
print("K2K current_input_token:", r["current_input_token"])
print("K2K reference_next_token:", r["reference_next_token"])
print("K2K ort_next_token:", r["ort_next_token"])
print("K2K diagnosis:", r["diagnosis"])
print(
    "K2K prestate conv max_eps:",
    r["prestate_metrics"]["conv_state"]["max_epsilon_units"],
)
print(
    "K2K prestate ssm max_eps:",
    r["prestate_metrics"]["ssm_state"]["max_epsilon_units"],
)
for key in sorted(r["matrix"]):
    block = r["matrix"][key]
    print(
        "K2K",
        key,
        "top1=",
        block["top1"],
        "class=",
        block["top1_class"],
        "ref_minus_ort=",
        block["reference_minus_ort_logit"],
        "top5=",
        block["top5_ids"],
        "vs_py_pp_max_eps=",
        block["vs_pytorch_pp"]["max_epsilon_units"],
    )
print("K2K receipt_id:", r["receipt_id"])
print("K2K file_sha256:", hashlib.sha256(data).hexdigest())
PY

ZIP="/kaggle/working/VN97-R2-K2K-state-swap.zip"
rm -f "$ZIP"
python - <<PY
from pathlib import Path
import zipfile

out = Path("$ZIP")
with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    z.write(Path("$OUTPUT_JSON"), arcname="vn97-k2k-state-swap.json")
print(out)
PY

echo "VN97 K2K state-swap divergence attribution complete"
echo "receipt=$OUTPUT_JSON"
echo "bundle=$ZIP"
