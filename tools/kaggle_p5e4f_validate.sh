#!/usr/bin/env bash
set -euo pipefail

REPAIRED_DIR="${1:-/kaggle/working/p5e4c-decoder-final}"
EXPANDED_DIR="${2:-/kaggle/working/p5e4e-output-final}"
OUTPUT="${3:-/kaggle/working/p5e4f-fresh-output-report.json}"

cd /kaggle/working/VN97
python -m pip install -e .

[[ -f "${REPAIRED_DIR}/student-float.pt" ]] || {
  echo "missing P5E4C student: ${REPAIRED_DIR}/student-float.pt" >&2
  exit 2
}
[[ -f "${EXPANDED_DIR}/output-adapter.pt" ]] || {
  echo "missing P5E4E adapter: ${EXPANDED_DIR}/output-adapter.pt" >&2
  exit 2
}
[[ -f "${EXPANDED_DIR}/p5e4e-report.json" ]] || {
  echo "missing P5E4E report: ${EXPANDED_DIR}/p5e4e-report.json" >&2
  exit 2
}

python -m vn97.p5e4f_fresh_output_validation_cli \
  --repaired-dir "${REPAIRED_DIR}" \
  --expanded-dir "${EXPANDED_DIR}" \
  --output "${OUTPUT}" \
  --device cuda:0
