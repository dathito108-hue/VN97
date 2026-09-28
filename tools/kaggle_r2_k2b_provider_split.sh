#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

K2_ROOT="/kaggle/working/vn97-r2-k2"
CAPSULE_ROOT="$K2_ROOT/g03-capsule"
ONNX_ROOT="$K2_ROOT/g04-step-onnx"
K2A_RECEIPT="/kaggle/working/vn97-k2a-ort-parity.json"
OUTPUT_JSON="/kaggle/working/vn97-k2b-provider-split.json"

python - <<'PY'
import torch
import onnxruntime as ort

if not torch.cuda.is_available():
    raise SystemExit("CUDA is required for K2B")
name = torch.cuda.get_device_name(0)
if "T4" not in name.upper():
    raise SystemExit(f"Expected Tesla T4, got {name!r}")
if ort.__version__ != "1.26.0":
    raise SystemExit(
        f"K2B requires onnxruntime-gpu 1.26.0; got {ort.__version__}"
    )
if not str(torch.version.cuda).startswith("12."):
    raise SystemExit(
        f"K2B expects CUDA 12.x; torch CUDA={torch.version.cuda}"
    )
ort.preload_dlls()
providers = ort.get_available_providers()
for required in ("CPUExecutionProvider", "CUDAExecutionProvider"):
    if required not in providers:
        raise SystemExit(f"K2B provider missing: {required}; {providers!r}")
print("gpu=", name)
print("torch=", torch.__version__, "cuda=", torch.version.cuda)
print("onnxruntime=", ort.__version__)
print("providers=", providers)
PY

for required in   "$K2A_RECEIPT"   "$CAPSULE_ROOT/capsule.vn97m2g03.json"   "$ONNX_ROOT/manifest.vn97m2g04.json"   "$ONNX_ROOT/step.onnx"; do
  if [[ ! -f "$required" ]]; then
    echo "K2B missing prerequisite: $required" >&2
    echo "Run K2A successfully before K2B." >&2
    exit 2
  fi
done

AVAILABLE_KB="$(df -Pk /kaggle/working | awk 'NR==2 {print $4}')"
REQUIRED_KB="$((2 * 1024 * 1024))"
df -h /kaggle/working || true
if [[ -z "$AVAILABLE_KB" || "$AVAILABLE_KB" -lt "$REQUIRED_KB" ]]; then
  echo "K2B needs at least 2 GiB free working storage" >&2
  exit 3
fi

export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -m vn97.r2.mamba2_kaggle_k2b_provider_split   --capsule-root "$CAPSULE_ROOT"   --bundle-dir "$ONNX_ROOT"   --k2a-receipt "$K2A_RECEIPT"   --output "$OUTPUT_JSON"

python - <<PY
import hashlib
import json
from pathlib import Path

p = Path("$OUTPUT_JSON")
data = p.read_bytes()
r = json.loads(data)
print("K2B status:", r["status"])
print("K2B PyTorch->CPU argmax_exact:", r["pytorch_vs_cpu"]["argmax_exact"])
print("K2B CPU->CUDA argmax_exact:", r["cpu_vs_cuda"]["argmax_exact"])
print("K2B PyTorch->CUDA argmax_exact:", r["pytorch_vs_cuda"]["argmax_exact"])
print("K2B PyTorch->CPU within 2 eps:", r["pytorch_vs_cpu_within_reference"])
print("K2B CPU->CUDA within 2 eps:", r["cpu_vs_cuda_within_reference"])
print("K2B PyTorch->CPU metrics:", json.dumps(r["pytorch_vs_cpu"]["metrics"], sort_keys=True))
print("K2B CPU->CUDA metrics:", json.dumps(r["cpu_vs_cuda"]["metrics"], sort_keys=True))
print("K2B PyTorch->CUDA metrics:", json.dumps(r["pytorch_vs_cuda"]["metrics"], sort_keys=True))
print("K2B receipt_id:", r["receipt_id"])
print("K2B file_sha256:", hashlib.sha256(data).hexdigest())
PY

ZIP="/kaggle/working/VN97-R2-K2B-provider-split.zip"
rm -f "$ZIP"
python - <<PY
from pathlib import Path
import zipfile

out = Path("$ZIP")
with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    z.write(Path("$OUTPUT_JSON"), arcname="vn97-k2b-provider-split.json")
print(out)
PY

echo "VN97 K2B provider-split measurement complete"
echo "receipt=$OUTPUT_JSON"
echo "bundle=$ZIP"
