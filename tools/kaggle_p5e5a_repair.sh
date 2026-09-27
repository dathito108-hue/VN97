#!/usr/bin/env bash
set -euo pipefail

REPAIRED_DIR="${1:-/kaggle/working/p5e4c-decoder-final}"
P5E4L_DIR="${2:-/kaggle/working/p5e4l-sequence-final}"
P3_CORPUS_DIR="${3:-/kaggle/working/p3-corpus}"
WORK_DIR="${4:-/kaggle/working/p5e5a-work}"
OUTPUT_DIR="${5:-/kaggle/working/p5e5a-late-core-final}"
MAX_EPOCHS="${6:-2}"

cd /kaggle/working/VN97
python -m pip install -e .

[[ -f "${REPAIRED_DIR}/student-float.pt" ]] || {
  echo "missing P5E4C student: ${REPAIRED_DIR}/student-float.pt" >&2
  exit 2
}
[[ -f "${P5E4L_DIR}/p5e4l-report.json" ]] || {
  echo "missing P5E4L report: ${P5E4L_DIR}/p5e4l-report.json" >&2
  exit 2
}
[[ -f "${P5E4L_DIR}/output-adapter.pt" ]] || {
  echo "missing P5E4L rank-96 seed adapter: ${P5E4L_DIR}/output-adapter.pt" >&2
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

python -m vn97.p5e5a_late_core_sequence_repair_cli \
  --repaired-dir "${REPAIRED_DIR}" \
  --p5e4l-dir "${P5E4L_DIR}" \
  --p3-corpus-dir "${P3_CORPUS_DIR}" \
  --work-dir "${WORK_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --device cuda:0 \
  --max-epochs "${MAX_EPOCHS}"
