#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

K2_ROOT="/kaggle/working/vn97-r2-k2"
ONNX_ROOT="$K2_ROOT/g04-step-onnx"
K2J_RECEIPT="/kaggle/working/vn97-k2j-closed-loop.json"
K2K_RECEIPT="/kaggle/working/vn97-k2k-state-swap.json"
K2L_RECEIPT="/kaggle/working/vn97-k2l-layer-influence.json"
OUTPUT_JSON="/kaggle/working/vn97-k2m-tie-sensitivity.json"

python - <<'PY'
import onnxruntime as ort
import torch

print("gpu=", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
print("torch=", torch.__version__, "cuda=", torch.version.cuda)
print("onnxruntime=", ort.__version__)
print("providers=", ort.get_available_providers())

if not torch.cuda.is_available():
    raise SystemExit("K2M requires CUDA")
if "T4" not in torch.cuda.get_device_name(0).upper():
    raise SystemExit("K2M requires Tesla T4")
if not torch.__version__.startswith("2.10.0"):
    raise SystemExit(
        f"K2M requires Torch 2.10.0 baseline; got {torch.__version__}"
    )
if ort.__version__ != "1.26.0":
    raise SystemExit(
        f"K2M requires ONNX Runtime 1.26.0; got {ort.__version__}"
    )
ort.preload_dlls()
if "CUDAExecutionProvider" not in ort.get_available_providers():
    raise SystemExit("K2M requires CUDAExecutionProvider")
PY

for required in   "$K2J_RECEIPT"   "$K2K_RECEIPT"   "$K2L_RECEIPT"   "$K2_ROOT/g03-capsule/capsule.vn97m2g03.json"   "$ONNX_ROOT/manifest.vn97m2g04.json"   "$ONNX_ROOT/step.onnx"; do
  if [[ ! -f "$required" ]]; then
    echo "K2M missing prerequisite: $required" >&2
    echo "Run K2L successfully before K2M." >&2
    exit 2
  fi
done

export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -m vn97.r2.mamba2_kaggle_k2m_tie_sensitivity   --bundle-dir "$ONNX_ROOT"   --k2j-receipt "$K2J_RECEIPT"   --k2k-receipt "$K2K_RECEIPT"   --k2l-receipt "$K2L_RECEIPT"   --output "$OUTPUT_JSON"

python - <<PY
import hashlib
import json
from pathlib import Path

p = Path("$OUTPUT_JSON")
data = p.read_bytes()
r = json.loads(data)

print("K2M status:", r["status"])
print("K2M diagnosis:", r["diagnosis"])
print(
    "K2M baseline:",
    "top1=", r["baseline"]["top1"],
    "gap=", r["baseline"]["reference_minus_ort_logit"],
)
print("K2M ranked thresholds:")
for item in r["ranked_thresholds"]:
    print(
        "  layer=", item["layer"],
        "mode=", item["mode"],
        "min_alpha=", item["minimal_alpha_to_reference"],
        "k2l_influence=", item["k2l_gap_influence"],
    )

for candidate in r["candidates"]:
    print(
        "K2M candidate",
        "layer=", candidate["layer"],
        "mode=", candidate["mode"],
        "min_alpha=", candidate["minimal_alpha_to_reference"],
    )
    for point in candidate["curve"]:
        print(
            "  alpha=", point["alpha"],
            "gap=", point["reference_minus_ort_logit"],
            "class=", point["top1_class"],
            "conv_changed=", point["conv_perturbation"]["changed_fraction"],
            "ssm_changed=", point["ssm_perturbation"]["changed_fraction"],
        )

print("K2M receipt_id:", r["receipt_id"])
print("K2M file_sha256:", hashlib.sha256(data).hexdigest())
PY

ZIP="/kaggle/working/VN97-R2-K2M-tie-sensitivity.zip"
rm -f "$ZIP"
python - <<PY
from pathlib import Path
import zipfile

out = Path("$ZIP")
with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    z.write(Path("$OUTPUT_JSON"), arcname="vn97-k2m-tie-sensitivity.json")
print(out)
PY

echo "VN97 K2M near-tie sensitivity measurement complete"
echo "receipt=$OUTPUT_JSON"
echo "bundle=$ZIP"
