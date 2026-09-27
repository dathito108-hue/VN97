#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
usage:
  tools/kaggle_p5e4d_validate.sh <p5e4a-parent-dir> <p5e4c-repaired-dir> [output]

Default output:
  /kaggle/working/p5e4d-fresh-report.json

P5E4D is a new one-shot validation. It does not reuse held-out-p4.jsonl or
the P5E4B manifest. It compares the fixed P5E4A parent with P5E4C repaired
weights and verifies all frozen SSM/embedding state still matches exactly.
EOF
  exit 2
}

[[ $# -ge 2 && $# -le 3 ]] || usage

PARENT_DIR="$1"
REPAIRED_DIR="$2"
OUTPUT="${3:-/kaggle/working/p5e4d-fresh-report.json}"

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

python -m pip install --disable-pip-version-check --no-deps -e .

[[ -f "$PARENT_DIR/student-float.pt" ]] || {
  echo "Missing P5E4A parent student-float.pt" >&2
  exit 3
}
[[ -f "$PARENT_DIR/p5e4a-report.json" ]] || {
  echo "Missing P5E4A parent report" >&2
  exit 3
}
[[ -f "$REPAIRED_DIR/student-float.pt" ]] || {
  echo "Missing P5E4C repaired student-float.pt" >&2
  exit 3
}
[[ -f "$REPAIRED_DIR/p5e4c-report.json" ]] || {
  echo "Missing P5E4C repaired report" >&2
  exit 3
}

[[ ! -e "$OUTPUT" ]] || {
  echo "Refusing to overwrite one-shot P5E4D report: $OUTPUT" >&2
  exit 4
}

echo "VN97 P5E4D launch"
echo "mode=fresh_decoder_validation"
echo "reused_p4_suite=false"
echo "reused_p5e4b_manifest=false"
echo "parent=$PARENT_DIR"
echo "repaired=$REPAIRED_DIR"

python -m vn97.p5e4d_fresh_decoder_validation_cli \
  --parent-dir "$PARENT_DIR" \
  --repaired-dir "$REPAIRED_DIR" \
  --output "$OUTPUT" \
  --device cuda:0

echo "P5E4D fresh decoder validation complete"
echo "report: $OUTPUT"
