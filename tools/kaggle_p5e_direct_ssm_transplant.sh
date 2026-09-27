#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
usage:
  tools/kaggle_p5e_direct_ssm_transplant.sh assess|convert <base-p5d3b-dir> [hf-cache-root] [output-dir]

Defaults:
  hf-cache-root=/kaggle/working/hf-cache-p5d4a
  output-dir=/kaggle/working/p5e1-direct-final

The script never runs Falcon3-Mamba autoregressive generation. It reads the
already-cached pinned teacher tensors directly and grafts selective-SSM
operators into the canonical 309M VN97 float-shadow student.
EOF
  exit 2
}

[[ $# -ge 2 && $# -le 4 ]] || usage

MODE="$1"
BASE_DIR="$2"
HF_CACHE="${3:-/kaggle/working/hf-cache-p5d4a}"
OUTPUT_DIR="${4:-/kaggle/working/p5e1-direct-final}"
REVISION="79268d5c8e650ec0ec24aad2729bfc906f569580"
SNAPSHOT="$HF_CACHE/hub/models--tiiuae--Falcon3-Mamba-7B-Instruct/snapshots/$REVISION"

case "$MODE" in
  assess|convert) ;;
  *) usage ;;
esac

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

python -m pip install --disable-pip-version-check --no-cache-dir -q \
  safetensors \
  tokenizers
python -m pip install --disable-pip-version-check --no-deps -e .

[[ -f "$BASE_DIR/student-float.pt" ]] || {
  echo "Missing base VN97 float student: $BASE_DIR/student-float.pt" >&2
  exit 3
}
[[ -f "$BASE_DIR/tokenizer.vn97tk1" ]] || {
  echo "Missing base VN97 tokenizer: $BASE_DIR/tokenizer.vn97tk1" >&2
  exit 3
}
[[ -f "$SNAPSHOT/config.json" ]] || {
  echo "Missing cached Falcon3-Mamba config: $SNAPSHOT/config.json" >&2
  exit 3
}
[[ -f "$SNAPSHOT/tokenizer.json" ]] || {
  echo "Missing cached Falcon3-Mamba tokenizer.json" >&2
  exit 3
}
if [[ ! -f "$SNAPSHOT/model.safetensors.index.json" && ! -f "$SNAPSHOT/model.safetensors" ]]; then
  echo "Missing cached Falcon3-Mamba safetensors weights" >&2
  exit 3
fi

echo "VN97 P5E1 SOURCE snapshot=$SNAPSHOT"
echo "VN97 P5E1 BASE student=$BASE_DIR/student-float.pt"
df -h /kaggle/working || true

if [[ "$MODE" == "assess" ]]; then
  python -m vn97.p5e_direct_ssm_transplant_cli \
    --teacher-snapshot "$SNAPSHOT" \
    --base-student-dir "$BASE_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --blend 0.35 \
    --assess-only
  exit 0
fi

FREE_BYTES="$(df -PB1 /kaggle/working | awk 'NR==2 {print $4}')"
MIN_FREE_BYTES=$((1400 * 1024 * 1024))
if [[ -z "$FREE_BYTES" || "$FREE_BYTES" -lt "$MIN_FREE_BYTES" ]]; then
  echo "P5E1 needs at least 1.4 GiB free to seal a 309M float-shadow artifact." >&2
  echo "Safe cleanup: stop P5D4A and remove only obsolete P5D4A record/work outputs." >&2
  echo "Keep the teacher cache: $HF_CACHE" >&2
  echo "Keep the base student: $BASE_DIR" >&2
  exit 4
fi

rm -rf "$OUTPUT_DIR"

python -m vn97.p5e_direct_ssm_transplant_cli \
  --teacher-snapshot "$SNAPSHOT" \
  --base-student-dir "$BASE_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --blend 0.35 \
  --svd-device cuda:0 \
  --smoke-device cuda:0

echo "P5E1 direct SSM transplant complete"
echo "final: $OUTPUT_DIR"
