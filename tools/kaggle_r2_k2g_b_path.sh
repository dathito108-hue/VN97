#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

K2_ROOT="/kaggle/working/vn97-r2-k2"
CAPSULE_ROOT="$K2_ROOT/g03-capsule"
ONNX_ROOT="$K2_ROOT/g04-step-onnx"
K2F_RECEIPT="/kaggle/working/vn97-k2f-dbx-operands.json"
TRACE_ROOT="/kaggle/working/vn97-r2-k2g-b-path"
OUTPUT_JSON="/kaggle/working/vn97-k2g-b-path.json"

python - <<'PY'
import onnxruntime as ort
import torch

print("torch=", torch.__version__)
print("onnxruntime=", ort.__version__)
print("providers=", ort.get_available_providers())
if not torch.__version__.startswith("2.10.0"):
    raise SystemExit(
        f"K2G requires the measured Torch 2.10.0 baseline; "
        f"got {torch.__version__}"
    )
if ort.__version__ != "1.26.0":
    raise SystemExit(
        f"K2G requires ONNX Runtime 1.26.0; got {ort.__version__}"
    )
if "CPUExecutionProvider" not in ort.get_available_providers():
    raise SystemExit("K2G requires CPUExecutionProvider")
PY

for required in   "$K2F_RECEIPT"   "$CAPSULE_ROOT/capsule.vn97m2g03.json"   "$ONNX_ROOT/manifest.vn97m2g04.json"   "$ONNX_ROOT/step.onnx"; do
  if [[ ! -f "$required" ]]; then
    echo "K2G missing prerequisite: $required" >&2
    echo "Run K2F successfully before K2G." >&2
    exit 2
  fi
done

rm -rf "$TRACE_ROOT"
rm -f "$OUTPUT_JSON"
export TOKENIZERS_PARALLELISM=false

python -m vn97.r2.mamba2_kaggle_k2g_b_path   --capsule-root "$CAPSULE_ROOT"   --bundle-dir "$ONNX_ROOT"   --k2f-receipt "$K2F_RECEIPT"   --output-dir "$TRACE_ROOT"   --output "$OUTPUT_JSON"

python - <<PY
import hashlib
import json
from pathlib import Path

p = Path("$OUTPUT_JSON")
data = p.read_bytes()
r = json.loads(data)
print("K2G status:", r["status"])
print("K2G diagnosis:", r["diagnosis"])
print("K2G first_sensitive_boundary:", r["first_sensitive_boundary"])
for name in r["stage_order"]:
    m = r["stage_metrics"][name]
    cf = r["counterfactuals"][name]
    print(
        "K2G stage",
        name,
        "stage_max_eps=",
        m["max_epsilon_units"],
        "B_max_eps=",
        cf["b_value"]["max_epsilon_units"],
        "dBx_max_eps=",
        cf["d_b_x"]["max_epsilon_units"],
    )
print("K2G receipt_id:", r["receipt_id"])
print("K2G file_sha256:", hashlib.sha256(data).hexdigest())
PY

ZIP="/kaggle/working/VN97-R2-K2G-B-path.zip"
rm -f "$ZIP"
python - <<PY
from pathlib import Path
import zipfile

out = Path("$ZIP")
root = Path("$TRACE_ROOT")
with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    z.write(Path("$OUTPUT_JSON"), arcname="vn97-k2g-b-path.json")
    for path in sorted(root.iterdir()):
        if path.is_file():
            z.write(path, arcname=path.name)
print(out)
PY

echo "VN97 K2G B-value path trace complete"
echo "receipt=$OUTPUT_JSON"
echo "bundle=$ZIP"
