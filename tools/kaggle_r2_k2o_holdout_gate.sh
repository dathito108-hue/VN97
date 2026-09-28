#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

K2_ROOT="/kaggle/working/vn97-r2-k2"
CAPSULE_ROOT="$K2_ROOT/g03-capsule"
ONNX_ROOT="$K2_ROOT/g04-step-onnx"
K2N_RECEIPT="/kaggle/working/vn97-k2n-ulp-audit.json"
OUTPUT_JSON="/kaggle/working/vn97-k2o-holdout-gate.json"

python - <<'PY'
import onnxruntime as ort
import torch

print("gpu=", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
print("torch=", torch.__version__, "cuda=", torch.version.cuda)
print("onnxruntime=", ort.__version__)
print("providers=", ort.get_available_providers())

if not torch.cuda.is_available():
    raise SystemExit("K2O requires CUDA")
if "T4" not in torch.cuda.get_device_name(0).upper():
    raise SystemExit("K2O requires Tesla T4")
if not torch.__version__.startswith("2.10.0"):
    raise SystemExit(
        f"K2O requires Torch 2.10.0 baseline; got {torch.__version__}"
    )
if ort.__version__ != "1.26.0":
    raise SystemExit(
        f"K2O requires ONNX Runtime 1.26.0; got {ort.__version__}"
    )
ort.preload_dlls()
if "CUDAExecutionProvider" not in ort.get_available_providers():
    raise SystemExit("K2O requires CUDAExecutionProvider")
PY

for required in   "$K2N_RECEIPT"   "$CAPSULE_ROOT/capsule.vn97m2g03.json"   "$ONNX_ROOT/manifest.vn97m2g04.json"   "$ONNX_ROOT/step.onnx"; do
  if [[ ! -f "$required" ]]; then
    echo "K2O missing prerequisite: $required" >&2
    echo "Run K2N successfully before K2O." >&2
    exit 2
  fi
done

df -h /kaggle/working || true

export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -m vn97.r2.mamba2_kaggle_k2o_holdout_gate   --capsule-root "$CAPSULE_ROOT"   --bundle-dir "$ONNX_ROOT"   --k2n-receipt "$K2N_RECEIPT"   --output "$OUTPUT_JSON"

python - <<PY
import hashlib
import json
from pathlib import Path

p = Path("$OUTPUT_JSON")
data = p.read_bytes()
r = json.loads(data)

print("K2O status:", r["status"])
print("K2O gate_passed:", r["gate_passed"])
print("K2O gate_failures:", json.dumps(r["gate_failures"], sort_keys=True))
print("K2O prompt_count:", r["prompt_count"])
print("K2O total_same_input_decisions:", r["total_same_input_decisions"])
print("K2O exact_decisions:", r["exact_decisions"])
print("K2O exact_decision_fraction:", r["exact_decision_fraction"])
print("K2O mismatch_count:", r["mismatch_count"])
print("K2O nonfinite_count:", r["nonfinite_count"])
print("K2O top5_overlap_min:", r["top5_overlap_min"])
print("K2O top5_overlap_mean:", r["top5_overlap_mean"])
print(
    "K2O logits max_eps:",
    r["aggregate_logit_metrics"]["max_epsilon_units"],
)

for item in r["mismatches"]:
    print(
        "K2O mismatch",
        "prompt=", item["prompt_index"],
        "step=", item["step"],
        "ref=", item["reference_top1"],
        "ort=", item["ort_top1"],
        "gap_ulp=", item["reference_candidate_gap_ulp"],
        "top5_overlap=", item["top5_overlap"],
    )

print("K2O receipt_id:", r["receipt_id"])
print("K2O file_sha256:", hashlib.sha256(data).hexdigest())
PY

ZIP="/kaggle/working/VN97-R2-K2O-holdout-gate.zip"
rm -f "$ZIP"
python - <<PY
from pathlib import Path
import zipfile

out = Path("$ZIP")
with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    z.write(Path("$OUTPUT_JSON"), arcname="vn97-k2o-holdout-gate.json")
print(out)
PY

echo "VN97 K2O predeclared holdout parity gate complete"
echo "receipt=$OUTPUT_JSON"
echo "bundle=$ZIP"
