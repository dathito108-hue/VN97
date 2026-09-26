#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
usage:
  tools/kaggle_p5d1_teacher_corpus.sh fresh|resume <p3-corpus-dir> [dev-suite]

Outputs:
  /kaggle/working/p5d1-work
  /kaggle/working/p5d1-final

Teacher:
  tiiuae/Falcon3-Mamba-7B-Instruct
  TII Falcon-LLM License 2.0
EOF
  exit 2
}

[[ $# -ge 2 && $# -le 3 ]] || usage

MODE="$1"
P3_DIR="$2"
DEV_SUITE="${3:-}"

WORK_DIR="/kaggle/working/p5d1-work"
OUTPUT_DIR="/kaggle/working/p5d1-final"

case "$MODE" in
  fresh)
    rm -rf "$WORK_DIR" "$OUTPUT_DIR"
    ;;
  resume)
    if [[ -f "$OUTPUT_DIR/p5d1-report.json" ]]; then
      echo "P5D1 already complete: $OUTPUT_DIR"
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
        f"P5D1 needs 2 CUDA GPUs; found {torch.cuda.device_count()}"
    )
print(
    "CUDA READY "
    f"torch={torch.__version__} "
    f"runtime={torch.version.cuda} "
    f"gpu0={torch.cuda.get_device_name(0)} "
    f"gpu1={torch.cuda.get_device_name(1)}"
)
PY

python -m pip install --disable-pip-version-check -q   "transformers>=4.46,<5"   "accelerate>=0.34"   "huggingface_hub>=0.24"   safetensors   sentencepiece
python -m pip install --disable-pip-version-check --no-deps -e .

export HF_HOME="/kaggle/working/hf-cache"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$HF_HOME"

ARGS=(
  --p3-corpus-dir "$P3_DIR"
  --work-dir "$WORK_DIR"
  --output-dir "$OUTPUT_DIR"
  --accept-teacher-license TII-FALCON-LLM-2.0
  --progress-interval 10
)

if [[ -n "$DEV_SUITE" ]]; then
  ARGS+=(--dev-suite "$DEV_SUITE")
fi

vn97-p5d1-teacher-corpus "${ARGS[@]}"

echo "P5D1 complete"
echo "work: $WORK_DIR"
echo "final: $OUTPUT_DIR"
