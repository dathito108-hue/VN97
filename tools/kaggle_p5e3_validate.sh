#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
usage:
  tools/kaggle_p5e3_validate.sh <p5d3b-base-dir> <p5e1-dir> <p5e2-dir> <held-out-p4.jsonl> [output]

Default output:
  /kaggle/working/p5e3-heldout-report.json

P5E3 loads one VN97 checkpoint at a time, evaluates the same held-out P4
suite, compares base -> direct transplant -> calibrated transplant, and
rejects any overall/tool/authority regression against the base.
EOF
  exit 2
}

[[ $# -ge 4 && $# -le 5 ]] || usage

BASE_DIR="$1"
P5E1_DIR="$2"
P5E2_DIR="$3"
SUITE="$4"
OUTPUT="${5:-/kaggle/working/p5e3-heldout-report.json}"

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

echo "VN97 P5E3 launch"
echo "base=$BASE_DIR"
echo "p5e1=$P5E1_DIR"
echo "p5e2=$P5E2_DIR"
echo "suite=$SUITE"

python -m vn97.p5e3_heldout_validation_cli \
  --base-dir "$BASE_DIR" \
  --p5e1-dir "$P5E1_DIR" \
  --p5e2-dir "$P5E2_DIR" \
  --suite "$SUITE" \
  --output "$OUTPUT" \
  --device cuda:0

echo "P5E3 held-out capability validation complete"
echo "report: $OUTPUT"
