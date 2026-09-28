#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

K2_ROOT="/kaggle/working/vn97-r2-k2"
CAPSULE_ROOT="$K2_ROOT/g03-capsule"
G05_ROOT="$K2_ROOT/g05-recurrent-8"
K2O_RECEIPT="/kaggle/working/vn97-k2o-holdout-gate.json"
OUTPUT_JSON="/kaggle/working/vn97-k2p-g05-bridge.json"

python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("K2P requires CUDA")
name = torch.cuda.get_device_name(0)
print("gpu=", name)
if "T4" not in name.upper():
    raise SystemExit(f"K2P expects Tesla T4, got {name!r}")
print("torch=", torch.__version__, "cuda=", torch.version.cuda)
PY

for required in   "$K2O_RECEIPT"   "$CAPSULE_ROOT/capsule.vn97m2g03.json"; do
  if [[ ! -f "$required" ]]; then
    echo "K2P missing prerequisite: $required" >&2
    exit 2
  fi
done

python - <<PY
import json
from pathlib import Path
r = json.loads(Path("$K2O_RECEIPT").read_text())
assert r["schema"] == "VN97M2K2OHOLDOUT1"
assert r["status"] == "PASS"
assert r["gate_passed"] is True
assert r["gate_failures"] == []
print("K2O receipt_id=", r["receipt_id"])
print("K2O decisions=", r["total_same_input_decisions"])
PY

python -m pip uninstall -y -q onnxruntime onnxruntime-gpu || true
python -m pip install --disable-pip-version-check -q   "onnx>=1.17,<2"   "onnxscript>=0.2,<1"   "onnxruntime-gpu==1.26.0"

python - <<'PY'
import onnxruntime as ort
import torch
if ort.__version__ != "1.26.0":
    raise SystemExit(f"K2P requires ORT 1.26.0, got {ort.__version__}")
if not str(torch.version.cuda).startswith("12."):
    raise SystemExit(f"K2P requires CUDA 12.x, got {torch.version.cuda}")
ort.preload_dlls()
print("onnxruntime=", ort.__version__)
print("providers=", ort.get_available_providers())
if "CUDAExecutionProvider" not in ort.get_available_providers():
    raise SystemExit("K2P requires CUDAExecutionProvider")
PY

if [[ -f "$G05_ROOT/manifest.vn97m2g05.json" ]]; then
  REQUIRED_GIB=2
  STORAGE_MODE="reuse-existing-g05"
else
  REQUIRED_GIB=7
  STORAGE_MODE="real-g05-export"
fi

AVAILABLE_KB="$(df -Pk /kaggle/working | awk 'NR==2 {print $4}')"
REQUIRED_KB="$((REQUIRED_GIB * 1024 * 1024))"
echo "storage_mode=$STORAGE_MODE required_gib=$REQUIRED_GIB"
df -h /kaggle/working || true
if [[ -z "$AVAILABLE_KB" || "$AVAILABLE_KB" -lt "$REQUIRED_KB" ]]; then
  echo "K2P needs at least $REQUIRED_GIB GiB free for $STORAGE_MODE" >&2
  exit 3
fi

if [[ ! -f "$G05_ROOT/manifest.vn97m2g05.json" ]]; then
  rm -rf "$G05_ROOT"
  echo "K2P: exporting real G0.5 recurrent-8 shared-weight graph"
  python -m vn97.r2.mamba2_parallel_onnx_cli export     --capsule-root "$CAPSULE_ROOT"     --output-dir "$G05_ROOT"     --chunk-size 8
else
  echo "K2P: reusing existing G0.5 recurrent-8 bundle"
  python -m vn97.r2.mamba2_parallel_onnx_cli verify     --bundle-dir "$G05_ROOT"
fi

export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -m vn97.r2.mamba2_kaggle_k2p_g05_bridge   --capsule-root "$CAPSULE_ROOT"   --g05-bundle-dir "$G05_ROOT"   --k2o-receipt "$K2O_RECEIPT"   --output "$OUTPUT_JSON"

python - <<PY
import hashlib
import json
from pathlib import Path
p = Path("$OUTPUT_JSON")
data = p.read_bytes()
r = json.loads(data)

print("K2P status:", r["status"])
print("K2P gate_passed:", r["gate_passed"])
print("K2P gate_failures:", json.dumps(r["gate_failures"], sort_keys=True))
print("K2P g05_manifest_id:", r["g05_manifest_id"])
print("K2P total_same_input_decisions:", r["total_same_input_decisions"])
print("K2P exact_decisions:", r["exact_decisions"])
print("K2P exact_decision_fraction:", r["exact_decision_fraction"])
print("K2P mismatch_count:", r["mismatch_count"])
print("K2P nonfinite_count:", r["nonfinite_count"])
print("K2P top5_overlap_min:", r["top5_overlap_min"])
print("K2P top5_overlap_mean:", r["top5_overlap_mean"])
print("K2P logits max_eps:", r["aggregate_logit_metrics"]["max_epsilon_units"])
print(
    "K2P prefill conv max_eps:",
    r["prefill_state_metrics_diagnostic_only"]["conv_state"]["max_epsilon_units"],
)
print(
    "K2P prefill ssm max_eps:",
    r["prefill_state_metrics_diagnostic_only"]["ssm_state"]["max_epsilon_units"],
)
for item in r["mismatches"]:
    print(
        "K2P mismatch",
        "phase=", item["phase"],
        "prompt=", item["prompt_index"],
        "step=", item["step"],
        "ref=", item["reference_top1"],
        "ort=", item["ort_top1"],
        "gap_ulp=", item["reference_candidate_gap_ulp"],
        "top5_overlap=", item["top5_overlap"],
    )
print("K2P receipt_id:", r["receipt_id"])
print("K2P file_sha256:", hashlib.sha256(data).hexdigest())
PY

ZIP="/kaggle/working/VN97-R2-K2P-G05-mobile-bridge.zip"
rm -f "$ZIP"
python - <<PY
from pathlib import Path
import zipfile

out = Path("$ZIP")
with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    z.write(Path("$OUTPUT_JSON"), arcname="vn97-k2p-g05-bridge.json")
    z.write(
        Path("$G05_ROOT/manifest.vn97m2g05.json"),
        arcname="manifest.vn97m2g05.json",
    )
print(out)
PY

echo "VN97 K2P G0.4 -> G0.5 mobile-graph parity bridge complete"
echo "receipt=$OUTPUT_JSON"
echo "bundle=$ZIP"
