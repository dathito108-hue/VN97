#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

K2_ROOT="/kaggle/working/vn97-r2-k2"
CAPSULE_ROOT="$K2_ROOT/g03-capsule"
ONNX_ROOT="$K2_ROOT/g04-step-onnx"
K2C_RECEIPT="/kaggle/working/vn97-k2c-backend-matrix.json"
TRACE_ROOT="/kaggle/working/vn97-r2-k2d-trace"
OUTPUT_JSON="/kaggle/working/vn97-k2d-stage-trace.json"

python - <<'PY'
import onnxruntime as ort
import torch

if ort.__version__ != "1.26.0":
    raise SystemExit(
        f"K2D requires onnxruntime 1.26.0 in the current K2 environment; "
        f"got {ort.__version__}"
    )
print("torch=", torch.__version__)
print("onnxruntime=", ort.__version__)
print("providers=", ort.get_available_providers())
if "CPUExecutionProvider" not in ort.get_available_providers():
    raise SystemExit("K2D requires CPUExecutionProvider")
PY

for required in   "$K2C_RECEIPT"   "$CAPSULE_ROOT/capsule.vn97m2g03.json"   "$ONNX_ROOT/manifest.vn97m2g04.json"   "$ONNX_ROOT/step.onnx"; do
  if [[ ! -f "$required" ]]; then
    echo "K2D missing prerequisite: $required" >&2
    echo "Run K2C successfully before K2D." >&2
    exit 2
  fi
done

AVAILABLE_KB="$(df -Pk /kaggle/working | awk 'NR==2 {print $4}')"
REQUIRED_KB="$((2 * 1024 * 1024))"
df -h /kaggle/working || true
if [[ -z "$AVAILABLE_KB" || "$AVAILABLE_KB" -lt "$REQUIRED_KB" ]]; then
  echo "K2D needs at least 2 GiB free working storage" >&2
  exit 3
fi

rm -rf "$TRACE_ROOT"
export TOKENIZERS_PARALLELISM=false

python -m vn97.r2.mamba2_kaggle_k2d_stage_trace   --capsule-root "$CAPSULE_ROOT"   --bundle-dir "$ONNX_ROOT"   --k2c-receipt "$K2C_RECEIPT"   --trace-output-dir "$TRACE_ROOT"   --output "$OUTPUT_JSON"

python - <<PY
import hashlib
import json
from pathlib import Path

p = Path("$OUTPUT_JSON")
data = p.read_bytes()
r = json.loads(data)
print("K2D status:", r["status"])
print("K2D trace_layer:", r["trace_layer"])
print("K2D provider:", r["provider"])
print("K2D first_stage_over_reference:", json.dumps(r["first_stage_over_reference"], sort_keys=True))
for name in r["stage_order"]:
    m = r["stage_metrics"][name]
    print(
        "K2D stage",
        name,
        "max_abs=",
        m["max_abs_error"],
        "max_eps=",
        m["max_epsilon_units"],
    )
print("K2D receipt_id:", r["receipt_id"])
print("K2D file_sha256:", hashlib.sha256(data).hexdigest())
PY

ZIP="/kaggle/working/VN97-R2-K2D-stage-trace.zip"
rm -f "$ZIP"
python - <<PY
from pathlib import Path
import zipfile

out = Path("$ZIP")
root = Path("$TRACE_ROOT")
with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    z.write(Path("$OUTPUT_JSON"), arcname="vn97-k2d-stage-trace.json")
    for path in sorted(root.iterdir()):
        if path.is_file():
            z.write(path, arcname=path.name)
print(out)
PY

echo "VN97 K2D same-input ONNX CPU stage trace complete"
echo "receipt=$OUTPUT_JSON"
echo "bundle=$ZIP"
