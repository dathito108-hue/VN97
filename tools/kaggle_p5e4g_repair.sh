#!/usr/bin/env bash
set -euo pipefail

REPAIRED_DIR="${1:-/kaggle/working/p5e4c-decoder-final}"
EXPANDED_DIR="${2:-/kaggle/working/p5e4e-output-final}"
P5E4F_REPORT="${3:-/kaggle/working/p5e4f-fresh-output-report.json}"
P3_CORPUS_DIR="${4:-/kaggle/working/p3-corpus}"
WORK_DIR="${5:-/kaggle/working/p5e4g-work}"
OUTPUT_DIR="${6:-/kaggle/working/p5e4g-decoding-final}"

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
[[ -f "${P5E4F_REPORT}" ]] || {
  echo "missing P5E4F report: ${P5E4F_REPORT}" >&2
  exit 2
}
[[ -f "${P3_CORPUS_DIR}/training.jsonl" ]] || {
  echo "missing P3 training split: ${P3_CORPUS_DIR}/training.jsonl" >&2
  exit 2
}
[[ -f "${P3_CORPUS_DIR}/validation.jsonl" ]] || {
  echo "missing P3 validation split: ${P3_CORPUS_DIR}/validation.jsonl" >&2
  exit 2
}

python -m vn97.p5e4g_prefix_decoding_repair_cli \
  --repaired-dir "${REPAIRED_DIR}" \
  --expanded-dir "${EXPANDED_DIR}" \
  --p5e4f-report "${P5E4F_REPORT}" \
  --p3-corpus-dir "${P3_CORPUS_DIR}" \
  --work-dir "${WORK_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --device cuda:0
