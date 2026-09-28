#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

WORK_ROOT="/kaggle/working/vn97-k2t"
SOURCE_ROOT="$WORK_ROOT/mamba2-source"
CAPSULE_ROOT="$WORK_ROOT/g03-capsule"
OUTPUT="/kaggle/working/vn97-k2t-multiplane-ternary.json"

python - <<'PY'
import torch
print("gpu=", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")
print("torch=", torch.__version__)
print("cuda=", torch.version.cuda)
assert torch.cuda.is_available()
assert "T4" in torch.cuda.get_device_name(0).upper()
assert torch.__version__ == "2.10.0+cu128"
assert torch.version.cuda == "12.8"
PY

python -m pip install --disable-pip-version-check -q --no-deps -e .

if [[ ! -f "$CAPSULE_ROOT/capsule.vn97m2g03.json" ]]; then
  rm -rf "$WORK_ROOT"
  mkdir -p "$WORK_ROOT"

  echo "K2T: downloading exact pinned Mamba-2 2.7B source"
  bash tools/r2_g02_fetch_mamba2.sh "$SOURCE_ROOT"

  echo "K2T: materializing zero-copy G0.3 capsule"
  python -m vn97.r2.mamba2_g03_capsule_cli materialize     --source-root "$SOURCE_ROOT"     --output-root "$CAPSULE_ROOT"

  # The capsule hardlinks the verified weight inode. Remove only the source
  # directory so the 5.4 GB payload remains once, under the capsule path.
  rm -rf "$SOURCE_ROOT"
else
  echo "K2T: reusing existing G0.3 capsule"
fi

python -m vn97.r2.mamba2_g03_capsule_cli inspect   --capsule-root "$CAPSULE_ROOT"   | tee "$WORK_ROOT/g03-inspect.json"

python -m vn97.r2.mamba2_kaggle_k2t_multiplane_ternary   --capsule-root "$CAPSULE_ROOT"   --output "$OUTPUT"   --planes 4   --threshold 0.5

python - <<PY
import json
from pathlib import Path

r = json.loads(Path("$OUTPUT").read_text())
assert r["schema"] == "VN97M2K2TMULTIPLANE1"
assert r["status"] == "MEASURED"
assert r["projection_matrix_count"] == 128
assert r["behavior_gate_required"] is True
assert r["production_activation_authorized"] is False

print("K2T status:", r["status"])
print("K2T receipt_id:", r["receipt_id"])
print("K2T projection_matrix_count:", r["projection_matrix_count"])
for item in r["global_planes"]:
    print(
        "K2T plane",
        item["plane_count"],
        "relative_rmse=",
        item["relative_rmse"],
        "cosine=",
        item["cosine_similarity"],
        "max_abs_error=",
        item["max_abs_error"],
    )
for item in r["candidate_sizes"]:
    print(
        "K2T size plane",
        item["plane_count"],
        "estimated_total_gib=",
        item["estimated_total_gib"],
        "compression_ratio=",
        item["compression_ratio_vs_fp16_unique"],
    )
print("K2T output:", "$OUTPUT")
PY
