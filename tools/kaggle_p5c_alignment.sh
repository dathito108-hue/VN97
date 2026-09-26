#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
usage:
  tools/kaggle_p5c_alignment.sh fresh|resume <p5b-final-dir> <p3-corpus-dir> [dev-suite] [device]

default device: cuda:0

Outputs:
  /kaggle/working/p5c-work
  /kaggle/working/p5c-final
EOF
  exit 2
}

[[ $# -ge 3 && $# -le 5 ]] || usage

MODE="$1"
P5B_DIR="$2"
P3_DIR="$3"
DEV_SUITE=""
DEVICE="cuda:0"

if [[ $# -ge 4 ]]; then DEV_SUITE="$4"; fi
if [[ $# -ge 5 ]]; then DEVICE="$5"; fi

WORK_DIR="/kaggle/working/p5c-work"
OUTPUT_DIR="/kaggle/working/p5c-final"

case "$MODE" in
  fresh)
    rm -rf "$WORK_DIR" "$OUTPUT_DIR"
    ;;
  resume)
    if [[ -f "$OUTPUT_DIR/p5c-report.json" ]]; then
      echo "P5C already complete: $OUTPUT_DIR"
      exit 0
    fi
    if [[ -d "$OUTPUT_DIR" ]]; then
      rm -rf "$OUTPUT_DIR"
    fi
    ;;
  *)
    usage
    ;;
esac

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

python - <<PY
import torch
device = "$DEVICE"
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available")
index = int(device.split(":")[1]) if ":" in device else 0
print(
    "CUDA READY "
    f"torch={torch.__version__} "
    f"runtime={torch.version.cuda} "
    f"device={device} "
    f"gpu={torch.cuda.get_device_name(index)}"
)
PY

python -m pip install --disable-pip-version-check --no-deps -e .

ARGS=(
  --p5b-dir "$P5B_DIR"
  --p3-corpus-dir "$P3_DIR"
  --work-dir "$WORK_DIR"
  --output-dir "$OUTPUT_DIR"
  --device "$DEVICE"
)

if [[ -n "$DEV_SUITE" ]]; then
  ARGS+=(--dev-suite "$DEV_SUITE")
fi

vn97-p5c-align "${ARGS[@]}"

echo "P5C complete"
echo "work: $WORK_DIR"
echo "final: $OUTPUT_DIR"
