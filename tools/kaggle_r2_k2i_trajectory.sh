#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

K2_ROOT="/kaggle/working/vn97-r2-k2"
CAPSULE_ROOT="$K2_ROOT/g03-capsule"
ONNX_ROOT="$K2_ROOT/g04-step-onnx"
K2H_RECEIPT="/kaggle/working/vn97-k2h-micro-isolation.json"
OUTPUT_JSON="/kaggle/working/vn97-k2i-trajectory.json"

python - <<'PY'
import onnxruntime as ort
import torch

print("gpu=", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
print("torch=", torch.__version__, "cuda=", torch.version.cuda)
print("onnxruntime=", ort.__version__)
print("providers=", ort.get_available_providers())

if not torch.cuda.is_available():
    raise SystemExit("K2I requires CUDA")
if "T4" not in torch.cuda.get_device_name(0).upper():
    raise SystemExit("K2I requires Tesla T4")
if not torch.__version__.startswith("2.10.0"):
    raise SystemExit(
        f"K2I requires Torch 2.10.0 baseline; got {torch.__version__}"
    )
if ort.__version__ != "1.26.0":
    raise SystemExit(
        f"K2I requires ONNX Runtime 1.26.0; got {ort.__version__}"
    )
ort.preload_dlls()
if "CUDAExecutionProvider" not in ort.get_available_providers():
    raise SystemExit("K2I requires CUDAExecutionProvider")
PY

for required in   "$K2H_RECEIPT"   "$CAPSULE_ROOT/capsule.vn97m2g03.json"   "$ONNX_ROOT/manifest.vn97m2g04.json"   "$ONNX_ROOT/step.onnx"; do
  if [[ ! -f "$required" ]]; then
    echo "K2I missing prerequisite: $required" >&2
    echo "Run K2H successfully before K2I." >&2
    exit 2
  fi
done

AVAILABLE_KB="$(df -Pk /kaggle/working | awk 'NR==2 {print $4}')"
REQUIRED_KB="$((3 * 1024 * 1024))"
df -h /kaggle/working || true
if [[ -z "$AVAILABLE_KB" || "$AVAILABLE_KB" -lt "$REQUIRED_KB" ]]; then
  echo "K2I needs at least 3 GiB free working storage" >&2
  exit 3
fi

export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -m vn97.r2.mamba2_kaggle_k2i_trajectory   --capsule-root "$CAPSULE_ROOT"   --bundle-dir "$ONNX_ROOT"   --k2h-receipt "$K2H_RECEIPT"   --output "$OUTPUT_JSON"

python - <<PY
import hashlib
import json
from pathlib import Path

p = Path("$OUTPUT_JSON")
data = p.read_bytes()
r = json.loads(data)

print("K2I status:", r["status"])
print("K2I token_count:", r["token_count"])
print("K2I argmax_exact_all_steps:", r["argmax_exact_all_steps"])
print(
    "K2I argmax_certified_by_margin_all_steps:",
    r["argmax_certified_by_margin_all_steps"],
)
print(
    "K2I logits max_eps:",
    r["aggregate_metrics"]["logits"]["max_epsilon_units"],
)
print(
    "K2I conv_state max_eps:",
    r["aggregate_metrics"]["conv_state"]["max_epsilon_units"],
)
print(
    "K2I ssm_state max_eps:",
    r["aggregate_metrics"]["ssm_state"]["max_epsilon_units"],
)
print(
    "K2I conv growth:",
    json.dumps(
        r["state_growth"]["conv_state_max_epsilon_units"],
        sort_keys=True,
    ),
)
print(
    "K2I ssm growth:",
    json.dumps(
        r["state_growth"]["ssm_state_max_epsilon_units"],
        sort_keys=True,
    ),
)
print("K2I receipt_id:", r["receipt_id"])
print("K2I file_sha256:", hashlib.sha256(data).hexdigest())
PY

ZIP="/kaggle/working/VN97-R2-K2I-trajectory.zip"
rm -f "$ZIP"
python - <<PY
from pathlib import Path
import zipfile

out = Path("$ZIP")
with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    z.write(Path("$OUTPUT_JSON"), arcname="vn97-k2i-trajectory.json")
print(out)
PY

echo "VN97 K2I 32-step recurrent trajectory measurement complete"
echo "receipt=$OUTPUT_JSON"
echo "bundle=$ZIP"
