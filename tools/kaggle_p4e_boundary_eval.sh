#!/usr/bin/env bash
set -euo pipefail

[[ $# -ge 1 && $# -le 3 ]] || {
  echo "usage: tools/kaggle_p4e_boundary_eval.sh <p4d-dir> [dev-suite] [output-json]" >&2
  exit 2
}

P4D_DIR="$1"
DEV_SUITE=""
OUTPUT_JSON="/kaggle/working/VN97-P4E-B-boundary.json"
[[ $# -ge 2 ]] && DEV_SUITE="$2"
[[ $# -ge 3 ]] && OUTPUT_JSON="$3"

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available")
print(
    "CUDA READY "
    f"torch={torch.__version__} "
    f"runtime={torch.version.cuda} "
    f"gpu={torch.cuda.get_device_name(0)}"
)
PY

python -m pip install --disable-pip-version-check --no-deps -e .

if [[ -n "$DEV_SUITE" ]]; then
  vn97-p4e-boundary-eval \
    --p4d-dir "$P4D_DIR" \
    --dev-suite "$DEV_SUITE" \
    --device cuda:0 \
    --samples-per-category 3 \
    --output-json "$OUTPUT_JSON"
else
  vn97-p4e-boundary-eval \
    --p4d-dir "$P4D_DIR" \
    --device cuda:0 \
    --samples-per-category 3 \
    --output-json "$OUTPUT_JSON"
fi
