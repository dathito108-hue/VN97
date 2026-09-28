#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

K2_ROOT="/kaggle/working/vn97-r2-k2"
CAPSULE_ROOT="$K2_ROOT/g03-capsule"
ONNX_ROOT="$K2_ROOT/g04-step-onnx"
K2C_RECEIPT="/kaggle/working/vn97-k2c-backend-matrix.json"
K2D_RECEIPT="/kaggle/working/vn97-k2d-stage-trace.json"
TRACE_ROOT="/kaggle/working/vn97-r2-k2e-trace"
POST_TRACE="/kaggle/working/vn97-k2e-post-trace.json"
OUTPUT_JSON="/kaggle/working/vn97-k2e-dbx-repair.json"

python - <<'PY'
import onnxruntime as ort
import torch

print("torch=", torch.__version__)
print("onnxruntime=", ort.__version__)
if not torch.__version__.startswith("2.10.0"):
    raise SystemExit(
        f"K2E source-semantic baseline requires Torch 2.10.0; "
        f"got {torch.__version__}"
    )
print("providers=", ort.get_available_providers())
if "CPUExecutionProvider" not in ort.get_available_providers():
    raise SystemExit("K2E requires CPUExecutionProvider")
PY

for required in   "$K2C_RECEIPT"   "$K2D_RECEIPT"   "$CAPSULE_ROOT/capsule.vn97m2g03.json"   "$ONNX_ROOT/manifest.vn97m2g04.json"   "$ONNX_ROOT/step.onnx"; do
  if [[ ! -f "$required" ]]; then
    echo "K2E missing prerequisite: $required" >&2
    echo "Run K2D successfully before K2E." >&2
    exit 2
  fi
done

rm -rf "$TRACE_ROOT"
rm -f "$POST_TRACE" "$OUTPUT_JSON"

export TOKENIZERS_PARALLELISM=false

python -m vn97.r2.mamba2_kaggle_k2e_dbx_repair   --capsule-root "$CAPSULE_ROOT"   --bundle-dir "$ONNX_ROOT"   --k2c-receipt "$K2C_RECEIPT"   --k2d-receipt "$K2D_RECEIPT"   --trace-output-dir "$TRACE_ROOT"   --post-trace-output "$POST_TRACE"   --output "$OUTPUT_JSON"

python - <<PY
import hashlib
import json
from pathlib import Path

p = Path("$OUTPUT_JSON")
data = p.read_bytes()
r = json.loads(data)
print("K2E status:", r["status"])
print("K2E repair:", r["repair"])
print("K2E operation_order:", r["operation_order"])
print("K2E before d_b_x max_eps:", r["before_d_b_x"]["max_epsilon_units"])
print("K2E after d_b_x max_eps:", r["after_d_b_x"]["max_epsilon_units"])
print("K2E d_b_x within 2 eps:", r["d_b_x_within_reference_after"])
print("K2E after first stage over reference:", json.dumps(r["after_first_stage_over_reference"], sort_keys=True))
print("K2E receipt_id:", r["receipt_id"])
print("K2E file_sha256:", hashlib.sha256(data).hexdigest())
PY

ZIP="/kaggle/working/VN97-R2-K2E-dBx-repair.zip"
rm -f "$ZIP"
python - <<PY
from pathlib import Path
import zipfile

out = Path("$ZIP")
with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    z.write(Path("$OUTPUT_JSON"), arcname="vn97-k2e-dbx-repair.json")
    z.write(Path("$POST_TRACE"), arcname="vn97-k2e-post-trace.json")
print(out)
PY

echo "VN97 K2E dBx lowering repair measurement complete"
echo "receipt=$OUTPUT_JSON"
echo "post_trace=$POST_TRACE"
echo "bundle=$ZIP"
