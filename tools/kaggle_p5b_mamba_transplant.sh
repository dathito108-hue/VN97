#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
usage:
  tools/kaggle_p5b_mamba_transplant.sh assess|convert <vn97-tokenizer> [source-dir]

Defaults:
  source-dir=/kaggle/working/mamba-130m-hf
  output=/kaggle/working/p5b-mamba130m-final

The source is pinned to state-spaces/mamba-130m-hf revision:
5708daa364c50b880e7bd92eab456e0d34492ee9
EOF
  exit 2
}

[[ $# -ge 2 && $# -le 3 ]] || usage

MODE="$1"
VN97_TOKENIZER="$2"
SOURCE_DIR="${3:-/kaggle/working/mamba-130m-hf}"
OUTPUT_DIR="/kaggle/working/p5b-mamba130m-final"
REVISION="5708daa364c50b880e7bd92eab456e0d34492ee9"
REPO_ID="state-spaces/mamba-130m-hf"

case "$MODE" in
  assess|convert) ;;
  *) usage ;;
esac

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

python -m pip install --disable-pip-version-check -q safetensors tokenizers huggingface_hub

mkdir -p "$SOURCE_DIR"

if [[ "$MODE" == "assess" ]]; then
  python - <<PY
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id="$REPO_ID",
    filename="config.json",
    revision="$REVISION",
    local_dir="$SOURCE_DIR",
)
print("P5B source config ready")
PY

  vn97-p5b-mamba-transplant     --source-dir "$SOURCE_DIR"     --vn97-tokenizer "$VN97_TOKENIZER"     --output-dir "$OUTPUT_DIR"     --assess-only
  exit 0
fi

rm -rf "$OUTPUT_DIR"

python - <<PY
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="$REPO_ID",
    revision="$REVISION",
    allow_patterns=[
        "config.json",
        "model.safetensors",
        "tokenizer.json",
    ],
    local_dir="$SOURCE_DIR",
)
print("P5B pinned source checkpoint ready")
PY

vn97-p5b-mamba-transplant   --source-dir "$SOURCE_DIR"   --vn97-tokenizer "$VN97_TOKENIZER"   --output-dir "$OUTPUT_DIR"   --svd-device cpu

echo "P5B transplant complete"
echo "final: $OUTPUT_DIR"
