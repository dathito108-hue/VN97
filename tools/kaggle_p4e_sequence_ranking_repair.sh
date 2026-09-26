#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
usage:
  tools/kaggle_p4e_sequence_ranking_repair.sh fresh|resume <p4e-k-final-dir> <p3-corpus-dir> [dev-suite]

Outputs:
  /kaggle/working/p4e-n-work
  /kaggle/working/p4e-n-final
EOF
  exit 2
}

[[ $# -ge 3 && $# -le 4 ]] || usage

MODE="$1"
P4E_K_DIR="$2"
P3_DIR="$3"
DEV_SUITE=""
if [[ $# -ge 4 ]]; then DEV_SUITE="$4"; fi

WORK_DIR="/kaggle/working/p4e-n-work"
OUTPUT_DIR="/kaggle/working/p4e-n-final"

case "$MODE" in
  fresh)
    rm -rf "$WORK_DIR" "$OUTPUT_DIR"
    ;;
  resume)
    if [[ -f "$OUTPUT_DIR/p4n-report.json" ]]; then
      echo "P4E-N already complete: $OUTPUT_DIR"
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

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

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
  --p4e-k-dir "$P4E_K_DIR"
  --p3-corpus-dir "$P3_DIR"
  --work-dir "$WORK_DIR"
  --output-dir "$OUTPUT_DIR"
  --device cuda:0
)

if [[ -n "$DEV_SUITE" ]]; then
  ARGS+=(--dev-suite "$DEV_SUITE")
fi

vn97-p4e-sequence-ranking-repair "${ARGS[@]}"

echo "P4E-N complete"
echo "work: $WORK_DIR"
echo "final: $OUTPUT_DIR"
