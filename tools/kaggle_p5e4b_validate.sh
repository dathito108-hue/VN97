#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
usage:
  tools/kaggle_p5e4b_validate.sh <p5d3b-base-dir> <p5e4a-candidate-dir> [output]

Default output:
  /kaggle/working/p5e4b-fresh-report.json

P5E4B is a one-shot fresh validation. It generates a new deterministic
60-task probe set from immutable artifact hashes and does not reuse held-out-p4.
It compares the frozen base and P5E4A candidate, records latent target-rank
metrics plus exact generation, and never authorizes QAT directly.
EOF
  exit 2
}

[[ $# -ge 2 && $# -le 3 ]] || usage

BASE_DIR="$1"
CANDIDATE_DIR="$2"
OUTPUT="${3:-/kaggle/working/p5e4b-fresh-report.json}"

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

python -m pip install --disable-pip-version-check --no-deps -e .

for dir in "$BASE_DIR" "$CANDIDATE_DIR"; do
  [[ -f "$dir/student-float.pt" ]] || {
    echo "Missing student-float.pt in $dir" >&2
    exit 3
  }
  [[ -f "$dir/tokenizer.vn97tk1" ]] || {
    echo "Missing tokenizer.vn97tk1 in $dir" >&2
    exit 3
  }
done

[[ -f "$CANDIDATE_DIR/p5e4a-report.json" ]] || {
  echo "Missing P5E4A report" >&2
  exit 3
}

[[ ! -e "$OUTPUT" ]] || {
  echo "Refusing to overwrite one-shot P5E4B report: $OUTPUT" >&2
  exit 4
}

echo "VN97 P5E4B launch"
echo "mode=fresh_one_shot_validation"
echo "reused_p4_suite=false"
echo "base=$BASE_DIR"
echo "candidate=$CANDIDATE_DIR"

python -m vn97.p5e4b_fresh_validation_cli \
  --base-dir "$BASE_DIR" \
  --candidate-dir "$CANDIDATE_DIR" \
  --output "$OUTPUT" \
  --device cuda:0 \
  --per-category 10

echo "P5E4B fresh one-shot validation complete"
echo "report: $OUTPUT"
