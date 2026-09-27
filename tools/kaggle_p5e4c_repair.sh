#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
usage:
  tools/kaggle_p5e4c_repair.sh <p5e4a-candidate-dir> <p3-corpus-dir> [work-dir] [output-dir]

Defaults:
  work-dir   /kaggle/working/p5e4c-work
  output-dir /kaggle/working/p5e4c-decoder-final

P5E4C freezes all 32 SSM layers and the tied lexical interface. It trains
only final_norm.weight on completion-aligned repair + P3 replay records.
It never authorizes QAT directly and requires a new fresh validation next.
EOF
  exit 2
}

[[ $# -ge 2 && $# -le 4 ]] || usage

CANDIDATE_DIR="$1"
P3_CORPUS_DIR="$2"
WORK_DIR="${3:-/kaggle/working/p5e4c-work}"
OUTPUT_DIR="${4:-/kaggle/working/p5e4c-decoder-final}"

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

python -m pip install --disable-pip-version-check --no-deps -e .

[[ -f "$CANDIDATE_DIR/student-float.pt" ]] || {
  echo "Missing P5E4A student-float.pt" >&2
  exit 3
}
[[ -f "$CANDIDATE_DIR/p5e4a-report.json" ]] || {
  echo "Missing P5E4A report" >&2
  exit 3
}
[[ -f "$P3_CORPUS_DIR/training.jsonl" ]] || {
  echo "Missing P3 training.jsonl" >&2
  exit 3
}
[[ -f "$P3_CORPUS_DIR/validation.jsonl" ]] || {
  echo "Missing P3 validation.jsonl" >&2
  exit 3
}

rm -rf "$OUTPUT_DIR"

echo "VN97 P5E4C launch"
echo "mode=final_norm_decoder_alignment_repair"
echo "ssm_layers_trainable=false"
echo "embedding_trainable=false"
echo "final_norm_trainable=true"
echo "candidate=$CANDIDATE_DIR"
echo "p3_corpus=$P3_CORPUS_DIR"

python -m vn97.p5e4c_decoder_repair_cli \
  --candidate-dir "$CANDIDATE_DIR" \
  --p3-corpus-dir "$P3_CORPUS_DIR" \
  --work-dir "$WORK_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --device cuda:0

echo "P5E4C decoder alignment repair complete"
echo "final: $OUTPUT_DIR"
