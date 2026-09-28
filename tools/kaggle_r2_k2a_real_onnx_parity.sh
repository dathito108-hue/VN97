#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

WORK_ROOT="${VN97_KAGGLE_WORK_ROOT:-/kaggle/working/vn97-r2-k1}"
SOURCE_ROOT="$WORK_ROOT/mamba2-source"
K2_ROOT="/kaggle/working/vn97-r2-k2"
CAPSULE_ROOT="$K2_ROOT/g03-capsule"
ONNX_ROOT="$K2_ROOT/g04-step-onnx"
K1I_RECEIPT="/kaggle/working/vn97-k1i-semantic-parity.json"
OUTPUT_JSON="/kaggle/working/vn97-k2a-ort-parity.json"

python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is required for K2A")
name = torch.cuda.get_device_name(0)
print("gpu=", name)
if "T4" not in name.upper():
    raise SystemExit(f"Expected Tesla T4, got {name!r}")
print("torch=", torch.__version__, "cuda=", torch.version.cuda)
free, total = torch.cuda.mem_get_info(0)
print("gpu_free_gib=", round(free / 1024**3, 2))
print("gpu_total_gib=", round(total / 1024**3, 2))
PY

if [[ ! -f "$K1I_RECEIPT" ]]; then
  echo "K2A requires the passing K1I receipt at $K1I_RECEIPT" >&2
  exit 2
fi

python - <<PY
import json
from pathlib import Path
p = Path("$K1I_RECEIPT")
r = json.loads(p.read_text())
assert r["schema"] == "VN97M2K1ISEMANTIC1"
assert r["status"] == "PASS"
assert r["same_input_all_layers_passed"] is True
assert len(r["violations"]) == 0
print("K1I receipt_id=", r["receipt_id"])
print("K1I layer_samples=", r["layer_samples"])
PY

if [[ ! -f "$SOURCE_ROOT/pytorch_model.bin" ]]; then
  echo "Existing pinned source missing; downloading."
  bash tools/r2_g02_fetch_mamba2.sh "$SOURCE_ROOT"
else
  echo "Reusing existing pinned Mamba-2 source at $SOURCE_ROOT"
fi

mkdir -p "$K2_ROOT"

if [[ ! -f "$CAPSULE_ROOT/capsule.vn97m2g03.json" ]]; then
  rm -rf "$CAPSULE_ROOT"
  vn97-r2-mamba2-g03 materialize     --source-root "$SOURCE_ROOT"     --output-root "$CAPSULE_ROOT"
else
  echo "Reusing existing zero-copy G0.3 capsule"
  vn97-r2-mamba2-g03 inspect --capsule-root "$CAPSULE_ROOT"
fi

python -m pip uninstall -y -q onnxruntime onnxruntime-gpu || true
python -m pip install --disable-pip-version-check -q   "onnx>=1.17,<2"   "onnxscript>=0.2,<1"   "onnxruntime-gpu>=1.20,<2"

python - <<'PY'
import onnxruntime as ort
print("onnxruntime=", ort.__version__)
print("providers=", ort.get_available_providers())
if "CUDAExecutionProvider" not in ort.get_available_providers():
    raise SystemExit("CUDAExecutionProvider is unavailable")
PY

AVAILABLE_KB="$(df -Pk /kaggle/working | awk 'NR==2 {print $4}')"
if [[ -f "$ONNX_ROOT/manifest.vn97m2g04.json" ]]; then
  REQUIRED_GIB=2
  STORAGE_MODE="reuse-existing-onnx"
else
  REQUIRED_GIB=8
  STORAGE_MODE="real-onnx-export"
fi
REQUIRED_KB="$((REQUIRED_GIB * 1024 * 1024))"
echo "storage_mode=$STORAGE_MODE required_gib=$REQUIRED_GIB"
df -h /kaggle/working || true
if [[ -z "$AVAILABLE_KB" || "$AVAILABLE_KB" -lt "$REQUIRED_KB" ]]; then
  echo "K2A needs at least $REQUIRED_GIB GiB free for $STORAGE_MODE" >&2
  exit 3
fi

if [[ ! -f "$ONNX_ROOT/manifest.vn97m2g04.json" ]]; then
  rm -rf "$ONNX_ROOT"
  echo "K2A: exporting real dense G0.4 step ONNX"
  vn97-r2-mamba2-g04 export-step     --capsule-root "$CAPSULE_ROOT"     --output-dir "$ONNX_ROOT"
else
  echo "Reusing existing G0.4 step ONNX bundle"
  vn97-r2-mamba2-g04 verify --bundle-dir "$ONNX_ROOT"
fi

export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -m vn97.r2.mamba2_kaggle_k2a_ort   --capsule-root "$CAPSULE_ROOT"   --bundle-dir "$ONNX_ROOT"   --k1i-receipt "$K1I_RECEIPT"   --output "$OUTPUT_JSON"

python - <<PY
import hashlib
import json
from pathlib import Path
p = Path("$OUTPUT_JSON")
data = p.read_bytes()
r = json.loads(data)
print("K2A status:", r["status"])
print("K2A ORT provider:", r["ort_provider"])
print("K2A argmax_exact:", r["argmax_exact"])
print("K2A probe_cases:", r["probe_cases"])
print("K2A steps_compared:", r["steps_compared"])
print("K2A metrics:", json.dumps(r["metrics"], sort_keys=True))
print("K2A receipt_id:", r["receipt_id"])
print("K2A file_sha256:", hashlib.sha256(data).hexdigest())
PY

ZIP="/kaggle/working/VN97-R2-K2A-ORT-parity.zip"
rm -f "$ZIP"
python - <<PY
from pathlib import Path
import zipfile
root = Path("$K2_ROOT")
out = Path("$ZIP")
with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    z.write(Path("$OUTPUT_JSON"), arcname="vn97-k2a-ort-parity.json")
    z.write(
        root / "g04-step-onnx/manifest.vn97m2g04.json",
        arcname="manifest.vn97m2g04.json",
    )
print(out)
PY

echo "VN97 K2A real ONNX Runtime parity measurement complete"
echo "receipt=$OUTPUT_JSON"
echo "bundle=$ZIP"
