#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
usage:
  tools/kaggle_p4e_numeric_arithmetic_repair.sh fresh|resume <p4e-g-final-dir> <p3-corpus-dir> [dev-suite]

Outputs:
  /kaggle/working/p4e-i-work
  /kaggle/working/p4e-i-final
EOF
  exit 2
}

[[ $# -ge 3 && $# -le 4 ]] || usage

MODE="$1"
P4E_G_DIR="$2"
P3_DIR="$3"
DEV_SUITE=""
if [[ $# -ge 4 ]]; then DEV_SUITE="$4"; fi
WORK_DIR="/kaggle/working/p4e-i-work"
OUTPUT_DIR="/kaggle/working/p4e-i-final"

case "$MODE" in
  fresh)
    rm -rf "$WORK_DIR" "$OUTPUT_DIR"
    ;;
  resume)
    ;;
  *)
    usage
    ;;
esac

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available in this Kaggle session")
print(
    "CUDA READY "
    f"torch={torch.__version__} "
    f"runtime={torch.version.cuda} "
    f"gpu={torch.cuda.get_device_name(0)}"
)
PY

python -m pip install --disable-pip-version-check --no-deps -e .

ARGS=(
  --p4e-g-dir "$P4E_G_DIR"
  --p3-corpus-dir "$P3_DIR"
  --work-dir "$WORK_DIR"
  --output-dir "$OUTPUT_DIR"
  --device cuda:0
)

if [[ -n "$DEV_SUITE" ]]; then
  ARGS+=(--dev-suite "$DEV_SUITE")
fi

vn97-p4e-numeric-arithmetic-repair "${ARGS[@]}"

echo "P4E-I complete"
echo "work: $WORK_DIR"
echo "final: $OUTPUT_DIR"
