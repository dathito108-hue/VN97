#!/usr/bin/env bash
set -euo pipefail

REPAIRED_DIR="${1:-/kaggle/working/p5e4c-decoder-final}"
EXPANDED_DIR="${2:-/kaggle/working/p5e4e-output-final}"
PREFIX_REPAIRED_DIR="${3:-/kaggle/working/p5e4g-decoding-final}"
P5E4H_REPORT="${4:-/kaggle/working/p5e4h-fresh-prefix-report.json}"
OUTPUT="${5:-/kaggle/working/p5e4j-autoregressive-diagnostic.json}"

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
[[ -f "${PREFIX_REPAIRED_DIR}/output-adapter.pt" ]] || {
  echo "missing accepted P5E4G adapter: ${PREFIX_REPAIRED_DIR}/output-adapter.pt" >&2
  exit 2
}
[[ -f "${P5E4H_REPORT}" ]] || {
  echo "missing P5E4H report: ${P5E4H_REPORT}" >&2
  exit 2
}

python -m vn97.p5e4j_autoregressive_parity_diagnostic_cli \
  --repaired-dir "${REPAIRED_DIR}" \
  --expanded-dir "${EXPANDED_DIR}" \
  --prefix-repaired-dir "${PREFIX_REPAIRED_DIR}" \
  --p5e4h-report "${P5E4H_REPORT}" \
  --output "${OUTPUT}" \
  --device cuda:0
