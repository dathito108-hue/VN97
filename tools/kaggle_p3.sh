#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
usage:
  tools/kaggle_p3.sh candidate <0|1|2|3> <corpus-dir> [output-dir]
  tools/kaggle_p3.sh finalize <corpus-dir> <candidate-root> [output-dir]

candidate-root must contain:
  candidate-0/
  candidate-1/
  candidate-2/
  candidate-3/
EOF
  exit 2
}

[[ $# -ge 1 ]] || usage
MODE="$1"
shift

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available in this notebook session")
print(
    "CUDA READY "
    f"torch={torch.__version__} "
    f"runtime={torch.version.cuda} "
    f"gpu={torch.cuda.get_device_name(0)}"
)
PY

python -m pip install   --disable-pip-version-check   --no-deps   -e .

zip_dir() {
  local source_dir="$1"
  local archive_path="$2"
  python - "$source_dir" "$archive_path" <<'PY'
from pathlib import Path
import shutil
import sys

source = Path(sys.argv[1]).resolve(strict=True)
archive = Path(sys.argv[2]).resolve().with_suffix("")
created = shutil.make_archive(
    str(archive),
    "zip",
    root_dir=source.parent,
    base_dir=source.name,
)
print(f"ARCHIVE {created}")
PY
}

if [[ "$MODE" == "candidate" ]]; then
  [[ $# -ge 2 && $# -le 3 ]] || usage
  INDEX="$1"
  CORPUS_DIR="$2"
  OUTPUT_DIR="${3:-/kaggle/working/candidate-$INDEX}"

  [[ "$INDEX" =~ ^[0-3]$ ]] || {
    echo "candidate index must be 0, 1, 2 or 3" >&2
    exit 3
  }

  vn97-p3-kaggle-candidate     --corpus-dir "$CORPUS_DIR"     --candidate-index "$INDEX"     --output-dir "$OUTPUT_DIR"     --device cuda:0

  zip_dir     "$OUTPUT_DIR"     "/kaggle/working/VN97-P3-candidate-$INDEX.zip"

  echo "P3 candidate $INDEX complete"
  echo "save this file before ending the session:"
  echo "/kaggle/working/VN97-P3-candidate-$INDEX.zip"
  exit 0
fi

if [[ "$MODE" == "finalize" ]]; then
  [[ $# -ge 2 && $# -le 3 ]] || usage
  CORPUS_DIR="$1"
  CANDIDATE_ROOT="$2"
  OUTPUT_DIR="${3:-/kaggle/working/p3-language-output}"

  for index in 0 1 2 3; do
    [[ -d "$CANDIDATE_ROOT/candidate-$index" ]] || {
      echo "missing candidate directory: $CANDIDATE_ROOT/candidate-$index" >&2
      exit 4
    }
  done

  vn97-p3-kaggle-finalize     --corpus-dir "$CORPUS_DIR"     --candidate-dir "$CANDIDATE_ROOT/candidate-0"     --candidate-dir "$CANDIDATE_ROOT/candidate-1"     --candidate-dir "$CANDIDATE_ROOT/candidate-2"     --candidate-dir "$CANDIDATE_ROOT/candidate-3"     --output-dir "$OUTPUT_DIR"     --device cuda:0

  (
    cd "$OUTPUT_DIR"
    find . -maxdepth 1 -type f       ! -name SHA256SUMS       -printf '%f\0'       | sort -z       | xargs -0 sha256sum       > SHA256SUMS
  )

  zip_dir     "$OUTPUT_DIR"     "/kaggle/working/VN97-P3-final.zip"

  echo "P3 finalization complete"
  echo "save this file:"
  echo "/kaggle/working/VN97-P3-final.zip"
  exit 0
fi

usage
