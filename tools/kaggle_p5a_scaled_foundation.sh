#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
usage:
  tools/kaggle_p5a_scaled_foundation.sh fresh|resume <candidate-index> <p4e-k-final-dir> <p3-corpus-dir> [dev-suite] [device]

candidate-index:
  0 = ~26M
  1 = ~39M
  2 = ~50M

default device: cuda:0
EOF
  exit 2
}

[[ $# -ge 4 && $# -le 6 ]] || usage

MODE="$1"
INDEX="$2"
PARENT="$3"
P3_DIR="$4"
DEV_SUITE=""
DEVICE="cuda:0"

if [[ $# -ge 5 ]]; then DEV_SUITE="$5"; fi
if [[ $# -ge 6 ]]; then DEVICE="$6"; fi

case "$INDEX" in
  0|1|2) ;;
  *) usage ;;
esac

WORK_DIR="/kaggle/working/p5a-candidate-${INDEX}-work"
OUTPUT_DIR="/kaggle/working/p5a-candidate-${INDEX}-final"

case "$MODE" in
  fresh)
    rm -rf "$WORK_DIR" "$OUTPUT_DIR"
    ;;
  resume)
    if [[ -f "$OUTPUT_DIR/p5a-report.json" ]]; then
      echo "P5A candidate already complete: $OUTPUT_DIR"
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
  --parent-p4e-k-dir "$PARENT"
  --p3-corpus-dir "$P3_DIR"
  --candidate-index "$INDEX"
  --work-dir "$WORK_DIR"
  --output-dir "$OUTPUT_DIR"
  --device "$DEVICE"
)

if [[ -n "$DEV_SUITE" ]]; then
  ARGS+=(--dev-suite "$DEV_SUITE")
fi

vn97-p5a-scaled-foundation "${ARGS[@]}"

echo "P5A candidate complete"
echo "work: $WORK_DIR"
echo "final: $OUTPUT_DIR"
