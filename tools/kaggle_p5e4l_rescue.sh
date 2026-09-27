#!/usr/bin/env bash
set -euo pipefail

REPAIRED_DIR="${1:-/kaggle/working/p5e4c-decoder-final}"
EXPANDED_DIR="${2:-/kaggle/working/p5e4e-output-final}"
PREFIX_REPAIRED_DIR="${3:-/kaggle/working/p5e4g-decoding-final}"
P5E4K_REPORT="${4:-/kaggle/working/p5e4k-generation-constraint-ablation.json}"
P3_CORPUS_DIR="${5:-/kaggle/working/p3-corpus}"
WORK_DIR="${6:-/kaggle/working/p5e4l-work}"
OUTPUT_DIR="${7:-/kaggle/working/p5e4l-sequence-final}"
MAX_EPOCHS="${8:-4}"

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
[[ -f "${P5E4K_REPORT}" ]] || {
  echo "missing P5E4K report: ${P5E4K_REPORT}" >&2
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

python -m vn97.p5e4l_sequence_rescue_cli \
  --repaired-dir "${REPAIRED_DIR}" \
  --expanded-dir "${EXPANDED_DIR}" \
  --prefix-repaired-dir "${PREFIX_REPAIRED_DIR}" \
  --p5e4k-report "${P5E4K_REPORT}" \
  --p3-corpus-dir "${P3_CORPUS_DIR}" \
  --work-dir "${WORK_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --device cuda:0 \
  --max-epochs "${MAX_EPOCHS}"
