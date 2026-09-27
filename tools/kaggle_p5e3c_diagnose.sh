#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
usage:
  tools/kaggle_p5e3c_diagnose.sh <p5d3b-base-dir> <p5e1-dir> <p5e2-dir> <held-out-p4.jsonl> [output]

Default output:
  /kaggle/working/p5e3c-target-rank-report.json

P5E3C does not train and does not alter checkpoints. It measures where the
held-out expected tokens rank under teacher forcing for base, P5E1 and P5E2.
Use it only to diagnose the P5E3/P5E3B 0/12 generation signal.
EOF
  exit 2
}

[[ $# -ge 4 && $# -le 5 ]] || usage

BASE_DIR="$1"
P5E1_DIR="$2"
P5E2_DIR="$3"
SUITE="$4"
OUTPUT="${5:-/kaggle/working/p5e3c-target-rank-report.json}"

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

python -m pip install --disable-pip-version-check --no-deps -e .

for dir in "$BASE_DIR" "$P5E1_DIR" "$P5E2_DIR"; do
  [[ -f "$dir/student-float.pt" ]] || {
    echo "Missing student-float.pt in $dir" >&2
    exit 3
  }
  [[ -f "$dir/tokenizer.vn97tk1" ]] || {
    echo "Missing tokenizer.vn97tk1 in $dir" >&2
    exit 3
  }
done

[[ -f "$SUITE" ]] || {
  echo "Missing held-out P4 suite: $SUITE" >&2
  exit 3
}

rm -f "$OUTPUT"

echo "VN97 P5E3C launch"
echo "mode=teacher_forced_target_rank_diagnostic"
echo "weights_changed=false"

python -m vn97.p5e3c_target_rank_diagnostic_cli \
  --base-dir "$BASE_DIR" \
  --p5e1-dir "$P5E1_DIR" \
  --p5e2-dir "$P5E2_DIR" \
  --suite "$SUITE" \
  --output "$OUTPUT" \
  --device cuda:0

echo "P5E3C target-rank diagnostic complete"
echo "report: $OUTPUT"
