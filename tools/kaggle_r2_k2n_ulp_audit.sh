#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

K2_ROOT="/kaggle/working/vn97-r2-k2"
CAPSULE_ROOT="$K2_ROOT/g03-capsule"
ONNX_ROOT="$K2_ROOT/g04-step-onnx"
K2M_RECEIPT="/kaggle/working/vn97-k2m-tie-sensitivity.json"
OUTPUT_JSON="/kaggle/working/vn97-k2n-ulp-audit.json"

python - <<'PY'
import onnxruntime as ort
import torch

print("gpu=", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
print("torch=", torch.__version__, "cuda=", torch.version.cuda)
print("onnxruntime=", ort.__version__)
print("providers=", ort.get_available_providers())

if not torch.cuda.is_available():
    raise SystemExit("K2N requires CUDA")
if "T4" not in torch.cuda.get_device_name(0).upper():
    raise SystemExit("K2N requires Tesla T4")
if not torch.__version__.startswith("2.10.0"):
    raise SystemExit(
        f"K2N requires Torch 2.10.0 baseline; got {torch.__version__}"
    )
if ort.__version__ != "1.26.0":
    raise SystemExit(
        f"K2N requires ONNX Runtime 1.26.0; got {ort.__version__}"
    )
ort.preload_dlls()
if "CUDAExecutionProvider" not in ort.get_available_providers():
    raise SystemExit("K2N requires CUDAExecutionProvider")
PY

for required in   "$K2M_RECEIPT"   "$CAPSULE_ROOT/capsule.vn97m2g03.json"   "$ONNX_ROOT/manifest.vn97m2g04.json"   "$ONNX_ROOT/step.onnx"; do
  if [[ ! -f "$required" ]]; then
    echo "K2N missing prerequisite: $required" >&2
    echo "Run K2M successfully before K2N." >&2
    exit 2
  fi
done

df -h /kaggle/working || true

export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -m vn97.r2.mamba2_kaggle_k2n_ulp_audit   --capsule-root "$CAPSULE_ROOT"   --bundle-dir "$ONNX_ROOT"   --k2m-receipt "$K2M_RECEIPT"   --output "$OUTPUT_JSON"

python - <<PY
import hashlib
import json
from pathlib import Path

p = Path("$OUTPUT_JSON")
data = p.read_bytes()
r = json.loads(data)

print("K2N status:", r["status"])
print("K2N diagnosis:", r["diagnosis"])
print("K2N prompt_count:", r["prompt_count"])
print("K2N total_same_input_decisions:", r["total_same_input_decisions"])
print("K2N exact_decisions:", r["exact_decisions"])
print("K2N exact_decision_fraction:", r["exact_decision_fraction"])
print("K2N mismatch_count:", r["mismatch_count"])
print(
    "K2N mismatches_outside_one_reference_ulp:",
    r["mismatches_outside_one_reference_ulp"],
)
print("K2N top5_overlap_min:", r["top5_overlap_min"])
print("K2N top5_overlap_mean:", r["top5_overlap_mean"])
print(
    "K2N logits max_eps:",
    r["aggregate_logit_metrics"]["max_epsilon_units"],
)

for item in r["mismatches"]:
    print(
        "K2N mismatch",
        "prompt=", item["prompt_index"],
        "step=", item["step"],
        "ref=", item["reference_top1"],
        "ort=", item["ort_top1"],
        "gap_ulp=", item["reference_candidate_gap_ulp"],
        "within_1ulp=", item["within_one_reference_ulp"],
        "top5_overlap=", item["top5_overlap"],
    )

print("K2N receipt_id:", r["receipt_id"])
print("K2N file_sha256:", hashlib.sha256(data).hexdigest())
PY

ZIP="/kaggle/working/VN97-R2-K2N-ULP-audit.zip"
rm -f "$ZIP"
python - <<PY
from pathlib import Path
import zipfile

out = Path("$ZIP")
with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    z.write(Path("$OUTPUT_JSON"), arcname="vn97-k2n-ulp-audit.json")
print(out)
PY

echo "VN97 K2N ULP-aware same-input decision audit complete"
echo "receipt=$OUTPUT_JSON"
echo "bundle=$ZIP"
