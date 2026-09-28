#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

K2_ROOT="/kaggle/working/vn97-r2-k2"
CAPSULE_ROOT="$K2_ROOT/g03-capsule"
ONNX_ROOT="$K2_ROOT/g04-step-onnx"
K2D_RECEIPT="/kaggle/working/vn97-k2d-stage-trace.json"
K2E_RECEIPT="/kaggle/working/vn97-k2e-dbx-repair.json"
MICRO_ROOT="/kaggle/working/vn97-r2-k2f-dbx-operands"
OUTPUT_JSON="/kaggle/working/vn97-k2f-dbx-operands.json"

python - <<'PY'
import onnxruntime as ort
import torch

print("torch=", torch.__version__)
print("onnxruntime=", ort.__version__)
print("providers=", ort.get_available_providers())
if not torch.__version__.startswith("2.10.0"):
    raise SystemExit(
        f"K2F requires the measured Torch 2.10.0 baseline; "
        f"got {torch.__version__}"
    )
if ort.__version__ != "1.26.0":
    raise SystemExit(
        f"K2F requires ONNX Runtime 1.26.0; got {ort.__version__}"
    )
if "CPUExecutionProvider" not in ort.get_available_providers():
    raise SystemExit("K2F requires CPUExecutionProvider")
PY

for required in   "$K2D_RECEIPT"   "$K2E_RECEIPT"   "$CAPSULE_ROOT/capsule.vn97m2g03.json"   "$ONNX_ROOT/manifest.vn97m2g04.json"   "$ONNX_ROOT/step.onnx"; do
  if [[ ! -f "$required" ]]; then
    echo "K2F missing prerequisite: $required" >&2
    echo "Run K2E successfully before K2F." >&2
    exit 2
  fi
done

rm -rf "$MICRO_ROOT"
rm -f "$OUTPUT_JSON"

export TOKENIZERS_PARALLELISM=false

python -m vn97.r2.mamba2_kaggle_k2f_dbx_operands   --capsule-root "$CAPSULE_ROOT"   --bundle-dir "$ONNX_ROOT"   --k2d-receipt "$K2D_RECEIPT"   --k2e-receipt "$K2E_RECEIPT"   --output-dir "$MICRO_ROOT"   --output "$OUTPUT_JSON"

python - <<PY
import hashlib
import json
from pathlib import Path

p = Path("$OUTPUT_JSON")
data = p.read_bytes()
r = json.loads(data)
print("K2F status:", r["status"])
print("K2F diagnosis:", r["diagnosis"])
for name, metrics in r["operand_metrics"].items():
    print(
        "K2F operand",
        name,
        "max_abs=",
        metrics["max_abs_error"],
        "max_eps=",
        metrics["max_epsilon_units"],
    )
for name, metrics in r["operand_sensitivity"].items():
    print(
        "K2F sensitivity",
        name,
        "max_abs=",
        metrics["max_abs_error"],
        "max_eps=",
        metrics["max_epsilon_units"],
    )
for mode, block in r["micro_operator_metrics"].items():
    print(
        "K2F micro",
        mode,
        "ORT-vs-same-PyTorch max_eps=",
        block["ort_vs_same_mode_pytorch"]["max_epsilon_units"],
        "ORT-vs-source max_eps=",
        block["ort_vs_source_einsum"]["max_epsilon_units"],
    )
print("K2F receipt_id:", r["receipt_id"])
print("K2F file_sha256:", hashlib.sha256(data).hexdigest())
PY

ZIP="/kaggle/working/VN97-R2-K2F-dBx-operands.zip"
rm -f "$ZIP"
python - <<PY
from pathlib import Path
import zipfile

out = Path("$ZIP")
root = Path("$MICRO_ROOT")
with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    z.write(Path("$OUTPUT_JSON"), arcname="vn97-k2f-dbx-operands.json")
    for path in sorted(root.iterdir()):
        if path.is_file():
            z.write(path, arcname=path.name)
print(out)
PY

echo "VN97 K2F dBx operand-isolation measurement complete"
echo "receipt=$OUTPUT_JSON"
echo "bundle=$ZIP"
