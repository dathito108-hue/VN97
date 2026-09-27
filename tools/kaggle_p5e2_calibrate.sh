#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
usage:
  tools/kaggle_p5e2_calibrate.sh <p5e1-dir> <p3-corpus-dir> [output-dir]

Defaults:
  output-dir=/kaggle/working/p5e2-calibrated-final

P5E2 does not load Falcon3-Mamba. It runs a short native VN97 calibration on
the transplanted 309M float-shadow student, training only the final eight VN97 layers and final norm. The transplanted
embedding and the first 24 layers stay frozen to reduce VRAM and preserve the graft.
EOF
  exit 2
}

[[ $# -ge 2 && $# -le 3 ]] || usage

P5E1_DIR="$1"
P3_CORPUS="$2"
OUTPUT_DIR="${3:-/kaggle/working/p5e2-calibrated-final}"

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

python -m pip install --disable-pip-version-check --no-deps -e .

[[ -f "$P5E1_DIR/student-float.pt" ]] || {
  echo "Missing P5E1 student-float.pt" >&2
  exit 3
}
[[ -f "$P3_CORPUS/train.jsonl" ]] || {
  echo "Missing P3 train.jsonl" >&2
  exit 3
}
[[ -f "$P3_CORPUS/validation.jsonl" ]] || {
  echo "Missing P3 validation.jsonl" >&2
  exit 3
}

FREE_BYTES="$(df -PB1 /kaggle/working | awk 'NR==2 {print $4}')"
MIN_FREE_BYTES=$((1800 * 1024 * 1024))
if [[ -z "$FREE_BYTES" || "$FREE_BYTES" -lt "$MIN_FREE_BYTES" ]]; then
  echo "P5E2 needs at least 1.8 GiB free to save the calibrated float artifact." >&2
  echo "After successful P5E1, the Falcon3 teacher cache can be removed before P5E2." >&2
  echo "Safe large cleanup candidate: /kaggle/working/hf-cache-p5d4a" >&2
  exit 4
fi

rm -rf "$OUTPUT_DIR"

echo "VN97 P5E2 launch"
echo "source=$P5E1_DIR"
echo "corpus=$P3_CORPUS"
echo "teacher_loaded=false"
df -h /kaggle/working || true

python -m vn97.p5e2_calibration_cli \
  --p5e1-dir "$P5E1_DIR" \
  --p3-corpus-dir "$P3_CORPUS" \
  --output-dir "$OUTPUT_DIR" \
  --device cuda:0

echo "P5E2 short calibration complete"
echo "final: $OUTPUT_DIR"
