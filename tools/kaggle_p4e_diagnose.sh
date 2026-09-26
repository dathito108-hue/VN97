#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
usage:
  tools/kaggle_p4e_diagnose.sh <p4d-dir> [dev-suite] [output-json]
EOF
  exit 2
}

[[ $# -ge 1 && $# -le 3 ]] || usage

P4D_DIR="$1"
DEV_SUITE=""
OUTPUT_JSON="/kaggle/working/VN97-P4E-A-diagnostic.json"

if [[ $# -ge 2 ]]; then
  DEV_SUITE="$2"
fi
if [[ $# -ge 3 ]]; then
  OUTPUT_JSON="$3"
fi

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available in this notebook session")
print(
    "CUDA READY "
    f"torch={torch.__version__} "
    f"runtime={torch.version.cuda} "
    f"gpu={torch.cuda.get_device_name(0)}"
)
PY

python -m pip install \
  --disable-pip-version-check \
  --no-deps \
  -e .

if [[ -n "$DEV_SUITE" ]]; then
  vn97-p4e-diagnose \
    --p4d-dir "$P4D_DIR" \
    --device cuda:0 \
    --samples-per-category 3 \
    --max-new-tokens 96 \
    --dev-suite "$DEV_SUITE" \
    --output-json "$OUTPUT_JSON"
else
  vn97-p4e-diagnose \
    --p4d-dir "$P4D_DIR" \
    --device cuda:0 \
    --samples-per-category 3 \
    --max-new-tokens 96 \
    --output-json "$OUTPUT_JSON"
fi

echo "P4E-A diagnostic complete"
echo "report: $OUTPUT_JSON"
