#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
usage:
  tools/kaggle_p5d3a_teacher_features.sh fresh|resume <p5d1-final-dir>

Requires:
  2 CUDA GPUs
  >=15 GiB free in /kaggle/working before teacher download

Outputs:
  /kaggle/working/p5d3a-work
  /kaggle/working/p5d3a-final

The temporary teacher cache is removed automatically after successful
completion, but retained after failure so resume does not need to re-download.
EOF
  exit 2
}

[[ $# -eq 2 ]] || usage

MODE="$1"
P5D1_DIR="$2"

WORK="/kaggle/working/p5d3a-work"
FINAL="/kaggle/working/p5d3a-final"
HF_CACHE="/kaggle/working/hf-cache-p5d3a"

case "$MODE" in
  fresh)
    rm -rf "$WORK" "$FINAL" "$HF_CACHE"
    ;;
  resume)
    if [[ -f "$FINAL/p5d3a-report.json" ]]; then
      echo "P5D3A already complete: $FINAL"
      exit 0
    fi
    ;;
  *)
    usage
    ;;
esac

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

python - <<'PY'
import torch
if torch.cuda.device_count() < 2:
    raise SystemExit(
        f"P5D3A needs 2 CUDA GPUs; found {torch.cuda.device_count()}"
    )
print(
    "CUDA READY "
    f"torch={torch.__version__} "
    f"runtime={torch.version.cuda} "
    f"gpu0={torch.cuda.get_device_name(0)} "
    f"gpu1={torch.cuda.get_device_name(1)}"
)
PY

MIN_FREE_BYTES=$((15 * 1024 * 1024 * 1024))
FREE_BYTES="$(df -PB1 /kaggle/working | awk 'NR==2 {print $4}')"
if [[ -z "$FREE_BYTES" || "$FREE_BYTES" -lt "$MIN_FREE_BYTES" ]]; then
  echo "P5D3A needs at least 15 GiB free before downloading the 7B teacher."
  df -h /kaggle/working || true
  echo "Safe cleanup candidates before P5D3A:"
  echo "  /kaggle/working/p5d2-c1-final"
  echo "  /kaggle/working/p5d2-c0-work"
  echo "  /kaggle/working/p5d2-c1-work"
  echo "  /kaggle/working/p5d2-c0.log"
  echo "  /kaggle/working/p5d2-c1.log"
  echo "Keep:"
  echo "  /kaggle/working/p5d1-final"
  echo "  /kaggle/working/p5d2-c0-final"
  exit 3
fi

export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"
export HF_HOME="$HF_CACHE"
export TOKENIZERS_PARALLELISM=false
export PIP_NO_CACHE_DIR=1

python -m pip install --disable-pip-version-check --no-cache-dir -q \
  "transformers==4.46.1" \
  "tokenizers==0.20.3" \
  "accelerate>=0.34,<2" \
  "huggingface_hub>=0.24,<1" \
  safetensors \
  sentencepiece

python - <<'PY'
import tokenizers
import transformers
print(
    "P5D3A HF STACK "
    f"transformers={transformers.__version__} "
    f"tokenizers={tokenizers.__version__}"
)
if transformers.__version__ != "4.46.1":
    raise SystemExit("P5D3A transformers compatibility pin was not applied")
if tokenizers.__version__ != "0.20.3":
    raise SystemExit("P5D3A tokenizers compatibility pin was not applied")
PY

python -m vn97.p5d3_teacher_features_cli   --p5d1-dir "$P5D1_DIR"   --work-dir "$WORK"   --output-dir "$FINAL"   --progress-interval 10   --accept-teacher-license TII-FALCON-LLM-2.0

rm -rf "$HF_CACHE"

echo "P5D3A complete"
echo "work: $WORK"
echo "final: $FINAL"
df -h /kaggle/working || true
