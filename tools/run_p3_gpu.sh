#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: tools/run_p3_gpu.sh <p3-corpus-dir> [output-dir]" >&2
  exit 2
fi

CORPUS_DIR="$1"
OUTPUT_DIR="${2:-p3-language-output}"
DEVICE="${VN97_CUDA_DEVICE:-cuda:0}"
MIN_TOTAL_VRAM="${VN97_MIN_TOTAL_VRAM_BYTES:-0}"
MIN_FREE_VRAM="${VN97_MIN_FREE_VRAM_BYTES:-0}"

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"
REPOSITORY_COMMIT="$(git rev-parse HEAD)"

if [[ -n "${VN97_EXPECTED_REPOSITORY_COMMIT:-}" ]]   && [[ "$REPOSITORY_COMMIT" != "$VN97_EXPECTED_REPOSITORY_COMMIT" ]]; then
  echo "repository commit mismatch" >&2
  exit 3
fi

if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "tracked repository files must be clean before P3 training" >&2
  exit 4
fi

if [[ -e "$OUTPUT_DIR" ]] && [[ -n "$(find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
  echo "P3 output directory must be new or empty" >&2
  exit 5
fi

python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("PyTorch CUDA runtime is not available")
print(
    "CUDA READY "
    f"torch={torch.__version__} "
    f"runtime={torch.version.cuda} "
    f"devices={torch.cuda.device_count()}"
)
PY

python -m pip install   --disable-pip-version-check   --no-deps   -e .

PREFLIGHT_TMP="$(mktemp -t vn97-p3-gpuenv.XXXXXX.json)"
cleanup() {
  rm -f "$PREFLIGHT_TMP"
}
trap cleanup EXIT

vn97-p3-gpu-preflight   --corpus-dir "$CORPUS_DIR"   --repository-commit "$REPOSITORY_COMMIT"   --device "$DEVICE"   --min-total-vram-bytes "$MIN_TOTAL_VRAM"   --min-free-vram-bytes "$MIN_FREE_VRAM"   --output "$PREFLIGHT_TMP"

vn97-p3-language-campaign   --corpus-dir "$CORPUS_DIR"   --output-dir "$OUTPUT_DIR"   --device "$DEVICE"

cp "$PREFLIGHT_TMP"   "$OUTPUT_DIR/gpu-environment.vn97gpuenv1.json"

(
  cd "$OUTPUT_DIR"
  find . -maxdepth 1 -type f     ! -name SHA256SUMS     -printf '%f\0'     | sort -z     | xargs -0 sha256sum     > SHA256SUMS
)

python - "$OUTPUT_DIR" <<'PY'
from pathlib import Path
import hashlib
import json
import sys

root = Path(sys.argv[1])
required = {
    "campaign-report.json",
    "gpu-environment.vn97gpuenv1.json",
    "model.vn97ck1",
    "model.vn97mi1",
    "p3-run.vn97p3run1.json",
    "tokenizer.vn97tk1",
    "SHA256SUMS",
}
actual = {item.name for item in root.iterdir()}
if actual != required:
    raise SystemExit(
        f"unexpected P3 output set: {sorted(actual)}"
    )
run = json.loads(
    (root / "p3-run.vn97p3run1.json").read_text(
        encoding="utf-8"
    )
)
env = json.loads(
    (root / "gpu-environment.vn97gpuenv1.json").read_text(
        encoding="utf-8"
    )
)
print(
    "VN97 P3 GPU CAMPAIGN COMPLETE "
    f"candidate={run['selected_candidate_id']} "
    f"checkpoint={run['checkpoint_sha256']} "
    f"gpu_env={env['environment_id']}"
)
PY
