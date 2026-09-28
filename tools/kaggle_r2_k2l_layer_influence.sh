#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

K2_ROOT="/kaggle/working/vn97-r2-k2"
ONNX_ROOT="$K2_ROOT/g04-step-onnx"
K2J_RECEIPT="/kaggle/working/vn97-k2j-closed-loop.json"
K2K_RECEIPT="/kaggle/working/vn97-k2k-state-swap.json"
OUTPUT_JSON="/kaggle/working/vn97-k2l-layer-influence.json"

python - <<'PY'
import onnxruntime as ort
import torch

print("gpu=", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
print("torch=", torch.__version__, "cuda=", torch.version.cuda)
print("onnxruntime=", ort.__version__)
print("providers=", ort.get_available_providers())

if not torch.cuda.is_available():
    raise SystemExit("K2L requires CUDA")
if "T4" not in torch.cuda.get_device_name(0).upper():
    raise SystemExit("K2L requires Tesla T4")
if not torch.__version__.startswith("2.10.0"):
    raise SystemExit(
        f"K2L requires Torch 2.10.0 baseline; got {torch.__version__}"
    )
if ort.__version__ != "1.26.0":
    raise SystemExit(
        f"K2L requires ONNX Runtime 1.26.0; got {ort.__version__}"
    )
ort.preload_dlls()
if "CUDAExecutionProvider" not in ort.get_available_providers():
    raise SystemExit("K2L requires CUDAExecutionProvider")
PY

for required in   "$K2J_RECEIPT"   "$K2K_RECEIPT"   "$K2_ROOT/g03-capsule/capsule.vn97m2g03.json"   "$ONNX_ROOT/manifest.vn97m2g04.json"   "$ONNX_ROOT/step.onnx"; do
  if [[ ! -f "$required" ]]; then
    echo "K2L missing prerequisite: $required" >&2
    echo "Run K2K successfully before K2L." >&2
    exit 2
  fi
done

AVAILABLE_KB="$(df -Pk /kaggle/working | awk 'NR==2 {print $4}')"
REQUIRED_KB="$((3 * 1024 * 1024))"
df -h /kaggle/working || true
if [[ -z "$AVAILABLE_KB" || "$AVAILABLE_KB" -lt "$REQUIRED_KB" ]]; then
  echo "K2L needs at least 3 GiB free working storage" >&2
  exit 3
fi

export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -m vn97.r2.mamba2_kaggle_k2l_layer_influence   --bundle-dir "$ONNX_ROOT"   --k2j-receipt "$K2J_RECEIPT"   --k2k-receipt "$K2K_RECEIPT"   --output "$OUTPUT_JSON"

python - <<PY
import hashlib
import json
from pathlib import Path

p = Path("$OUTPUT_JSON")
data = p.read_bytes()
r = json.loads(data)

print("K2L status:", r["status"])
print("K2L diagnosis:", r["diagnosis"])
print("K2L layer_count:", r["layer_count"])
print("K2L selected_group_starts:", r["selected_group_starts"])
print(
    "K2L baseline:",
    "top1=", r["baseline"]["top1"],
    "gap=", r["baseline"]["reference_minus_ort_logit"],
)

print("K2L repairing_groups:", json.dumps(r["repairing_groups"], sort_keys=True))
print("K2L repairing_layers:", json.dumps(r["repairing_layers"], sort_keys=True))

for mode in ("conv", "ssm", "both"):
    print("K2L top", mode)
    for item in r["top_by_mode"][mode][:5]:
        print(
            "  layer=", item["layer"],
            "influence=", item["gap_influence"],
            "gap=", item["reference_minus_ort_logit"],
            "top1_class=", item["top1_class"],
        )

print("K2L receipt_id:", r["receipt_id"])
print("K2L file_sha256:", hashlib.sha256(data).hexdigest())
PY

ZIP="/kaggle/working/VN97-R2-K2L-layer-influence.zip"
rm -f "$ZIP"
python - <<PY
from pathlib import Path
import zipfile

out = Path("$ZIP")
with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    z.write(Path("$OUTPUT_JSON"), arcname="vn97-k2l-layer-influence.json")
print(out)
PY

echo "VN97 K2L layerwise state influence scan complete"
echo "receipt=$OUTPUT_JSON"
echo "bundle=$ZIP"
