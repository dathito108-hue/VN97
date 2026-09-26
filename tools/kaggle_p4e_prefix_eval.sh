#!/usr/bin/env bash
set -euo pipefail

[[ $# -ge 1 && $# -le 2 ]] || {
  echo "usage: tools/kaggle_p4e_prefix_eval.sh <p4e-c-final-dir> [dev-suite]" >&2
  exit 2
}

ARTIFACT_DIR="$1"
DEV_SUITE=""
if [[ $# -ge 2 ]]; then DEV_SUITE="$2"; fi

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available")
print("CUDA READY", torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))
PY

python -m pip install --disable-pip-version-check --no-deps -e .

if [[ -n "$DEV_SUITE" ]]; then
  vn97-p4e-prefix-eval --artifact-dir "$ARTIFACT_DIR" --dev-suite "$DEV_SUITE" --device cuda:0
else
  vn97-p4e-prefix-eval --artifact-dir "$ARTIFACT_DIR" --device cuda:0
fi
