#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
usage:
  tools/kaggle_p5e4a_reblend.sh <p5d3b-base-dir> <p5e1-dir> <held-out-p4.jsonl> [output-dir]

Default output-dir:
  /kaggle/working/p5e4a-selective-final

P5E4A does not load Falcon3-Mamba and does not train. It reconstructs lower-
alpha selective SSM deltas from the existing P5E1 alpha=0.35 artifact, keeps
the base embedding/norms intact, searches conservative dynamics-only profiles,
and saves only a candidate that passes strict base-preservation guards.
EOF
  exit 2
}

[[ $# -ge 3 && $# -le 4 ]] || usage

BASE_DIR="$1"
P5E1_DIR="$2"
SUITE="$3"
OUTPUT_DIR="${4:-/kaggle/working/p5e4a-selective-final}"

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

python -m pip install --disable-pip-version-check --no-deps -e .

for dir in "$BASE_DIR" "$P5E1_DIR"; do
  [[ -f "$dir/student-float.pt" ]] || {
    echo "Missing student-float.pt in $dir" >&2
    exit 3
  }
  [[ -f "$dir/tokenizer.vn97tk1" ]] || {
    echo "Missing tokenizer.vn97tk1 in $dir" >&2
    exit 3
  }
done

[[ -f "$P5E1_DIR/p5e1-report.json" ]] || {
  echo "Missing P5E1 report" >&2
  exit 3
}
[[ -f "$SUITE" ]] || {
  echo "Missing diagnostic P4 suite: $SUITE" >&2
  exit 3
}

rm -rf "$OUTPUT_DIR"

echo "VN97 P5E4A launch"
echo "mode=selective_delta_reblend_search"
echo "teacher_loaded=false"
echo "training=false"
echo "base=$BASE_DIR"
echo "p5e1=$P5E1_DIR"
echo "suite=$SUITE"

python -m vn97.p5e4a_selective_reblend_cli \
  --base-dir "$BASE_DIR" \
  --p5e1-dir "$P5E1_DIR" \
  --suite "$SUITE" \
  --output-dir "$OUTPUT_DIR" \
  --device cuda:0

echo "P5E4A selective reblend search complete"
echo "final: $OUTPUT_DIR"
