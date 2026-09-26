#!/usr/bin/env bash
set -euo pipefail

[[ $# -ge 1 && $# -le 2 ]] || {
  echo "usage: tools/kaggle_p4e_arithmetic_mechanism_audit.sh <p4e-k-final-dir> [output-json]" >&2
  exit 2
}

ARTIFACT_DIR="$1"
OUTPUT_JSON=""
if [[ $# -ge 2 ]]; then OUTPUT_JSON="$2"; fi

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available")
print("CUDA READY", torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))
PY

python -m pip install --disable-pip-version-check --no-deps -e .

if [[ -n "$OUTPUT_JSON" ]]; then
  vn97-p4e-arithmetic-mechanism-audit --artifact-dir "$ARTIFACT_DIR" --device cuda:0 --output-json "$OUTPUT_JSON"
else
  vn97-p4e-arithmetic-mechanism-audit --artifact-dir "$ARTIFACT_DIR" --device cuda:0
fi
