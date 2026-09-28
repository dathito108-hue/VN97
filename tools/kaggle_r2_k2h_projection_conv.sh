#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

K2_ROOT="/kaggle/working/vn97-r2-k2"
CAPSULE_ROOT="$K2_ROOT/g03-capsule"
ONNX_ROOT="$K2_ROOT/g04-step-onnx"
K2G_RECEIPT="/kaggle/working/vn97-k2g-b-path.json"
MICRO_ROOT="/kaggle/working/vn97-r2-k2h-micro"
OUTPUT_JSON="/kaggle/working/vn97-k2h-micro-isolation.json"

python - <<'PY'
import onnxruntime as ort
import torch

print("torch=", torch.__version__)
print("onnxruntime=", ort.__version__)
print("providers=", ort.get_available_providers())
if not torch.__version__.startswith("2.10.0"):
    raise SystemExit(
        f"K2H requires Torch 2.10.0 baseline; got {torch.__version__}"
    )
if ort.__version__ != "1.26.0":
    raise SystemExit(
        f"K2H requires ONNX Runtime 1.26.0; got {ort.__version__}"
    )
if "CPUExecutionProvider" not in ort.get_available_providers():
    raise SystemExit("K2H requires CPUExecutionProvider")
PY

for required in   "$K2G_RECEIPT"   "$CAPSULE_ROOT/capsule.vn97m2g03.json"   "$ONNX_ROOT/manifest.vn97m2g04.json"   "$ONNX_ROOT/step.onnx"; do
  if [[ ! -f "$required" ]]; then
    echo "K2H missing prerequisite: $required" >&2
    echo "Run K2G successfully before K2H." >&2
    exit 2
  fi
done

rm -rf "$MICRO_ROOT"
rm -f "$OUTPUT_JSON"
export TOKENIZERS_PARALLELISM=false

python -m vn97.r2.mamba2_kaggle_k2h_projection_conv   --capsule-root "$CAPSULE_ROOT"   --bundle-dir "$ONNX_ROOT"   --k2g-receipt "$K2G_RECEIPT"   --output-dir "$MICRO_ROOT"   --output "$OUTPUT_JSON"

python - <<PY
import hashlib
import json
from pathlib import Path

p = Path("$OUTPUT_JSON")
data = p.read_bytes()
r = json.loads(data)
print("K2H status:", r["status"])
print("K2H diagnosis:", r["diagnosis"])
print("K2H in_proj_candidate:", r["in_proj_candidate"])
print("K2H conv_affine_candidate:", r["conv_affine_candidate"])

for mode, block in r["in_proj_metrics"].items():
    print(
        "K2H in_proj",
        mode,
        "xbc_eps=",
        block["xbc_projected"]["max_epsilon_units"],
        "B_eps=",
        block["b_value"]["max_epsilon_units"],
        "dBx_eps=",
        block["d_b_x"]["max_epsilon_units"],
    )

for mode, block in r["conv_affine_metrics"].items():
    print(
        "K2H conv",
        mode,
        "affine_eps=",
        block["conv_affine"]["max_epsilon_units"],
        "B_eps=",
        block["b_value"]["max_epsilon_units"],
        "dBx_eps=",
        block["d_b_x"]["max_epsilon_units"],
    )

print("K2H receipt_id:", r["receipt_id"])
print("K2H file_sha256:", hashlib.sha256(data).hexdigest())
PY

ZIP="/kaggle/working/VN97-R2-K2H-micro-isolation.zip"
rm -f "$ZIP"
python - <<PY
from pathlib import Path
import zipfile

out = Path("$ZIP")
root = Path("$MICRO_ROOT")
with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    z.write(Path("$OUTPUT_JSON"), arcname="vn97-k2h-micro-isolation.json")
    for path in sorted(root.iterdir()):
        if path.is_file():
            z.write(path, arcname=path.name)
print(out)
PY

echo "VN97 K2H projection/conv micro-isolation complete"
echo "receipt=$OUTPUT_JSON"
echo "bundle=$ZIP"
