#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
usage:
  tools/kaggle_p3_dual_gpu.sh prepare <corpus-dir> [work-root]
  tools/kaggle_p3_dual_gpu.sh pair <first-index> <second-index> [work-root]
  tools/kaggle_p3_dual_gpu.sh all <corpus-dir> [work-root]
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
    raise SystemExit("CUDA is not available")
if torch.cuda.device_count() < 2:
    raise SystemExit("P3 dual-GPU launcher requires at least two CUDA devices")
print("CUDA READY", torch.__version__, torch.version.cuda)
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
PY

python -m pip install --disable-pip-version-check --no-deps -e .

prepare_cache() {
  local corpus="$1"
  local root="$2"
  local cache="$root/cache"
  if [[ -f "$cache/cache.vn97p3cache1.json" ]]; then
    echo "Reusing existing VN97P3CACHE1: $cache"
    return
  fi
  rm -rf "$cache"
  vn97-p3-kaggle-prepare \
    --corpus-dir "$corpus" \
    --output-dir "$cache"
}

run_one() {
  local index="$1"
  local physical_gpu="$2"
  local root="$3"
  local out="$root/candidate-$index"
  if [[ -f "$out/candidate-report.vn97p3cand1.json" ]]; then
    echo "Reusing completed candidate $index"
    return 0
  fi
  rm -rf "$out"

  # Pin one concurrent candidate to one physical T4. The pair launcher starts
  # two processes at the same time, so physical GPU 0 and GPU 1 are both busy.
  # CPU threads concurrently stage/prefetch batches for each GPU.
  CUDA_VISIBLE_DEVICES="$physical_gpu" \
  OMP_NUM_THREADS="${VN97_CPU_THREADS_PER_GPU:-2}" \
  MKL_NUM_THREADS="${VN97_CPU_THREADS_PER_GPU:-2}" \
  vn97-p3-kaggle-cached-candidate \
    --cache-dir "$root/cache" \
    --candidate-index "$index" \
    --output-dir "$out" \
    --device cuda:0 \
    --cpu-prefetch-workers "${VN97_CPU_PREFETCH_WORKERS:-1}"
}

run_pair() {
  local first="$1"
  local second="$2"
  local root="$3"

  run_one "$first" 0 "$root" >"$root/candidate-$first.log" 2>&1 &
  local p1=$!
  run_one "$second" 1 "$root" >"$root/candidate-$second.log" 2>&1 &
  local p2=$!

  echo "candidate $first -> physical GPU 0 pid=$p1"
  echo "candidate $second -> physical GPU 1 pid=$p2"
  echo "CPU support threads per GPU process: ${VN97_CPU_THREADS_PER_GPU:-2}"
  echo "CPU prefetch workers per GPU process: ${VN97_CPU_PREFETCH_WORKERS:-1}"

  local monitor_pid=""
  if command -v nvidia-smi >/dev/null 2>&1; then
    (
      while kill -0 "$p1" 2>/dev/null || kill -0 "$p2" 2>/dev/null; do
        echo "---- GPU MONITOR $(date -u +%Y-%m-%dT%H:%M:%SZ) ----"
        nvidia-smi \
          --query-gpu=index,name,utilization.gpu,memory.used,memory.total \
          --format=csv,noheader,nounits || true
        sleep 20
      done
    ) &
    monitor_pid=$!
  fi

  local r1=0
  local r2=0
  wait "$p1" || r1=$?
  wait "$p2" || r2=$?

  if [[ -n "$monitor_pid" ]]; then
    kill "$monitor_pid" 2>/dev/null || true
    wait "$monitor_pid" 2>/dev/null || true
  fi

  cat "$root/candidate-$first.log"
  cat "$root/candidate-$second.log"

  [[ "$r1" -eq 0 && "$r2" -eq 0 ]] || {
    echo "candidate pair failed: first=$r1 second=$r2" >&2
    exit 20
  }
}

if [[ "$MODE" == "prepare" ]]; then
  [[ $# -ge 1 && $# -le 2 ]] || usage
  CORPUS="$1"
  ROOT="${2:-/kaggle/working/p3-fast}"
  mkdir -p "$ROOT"
  prepare_cache "$CORPUS" "$ROOT"
  exit 0
fi

if [[ "$MODE" == "pair" ]]; then
  [[ $# -ge 2 && $# -le 3 ]] || usage
  FIRST="$1"
  SECOND="$2"
  ROOT="${3:-/kaggle/working/p3-fast}"
  [[ "$FIRST" =~ ^[0-3]$ && "$SECOND" =~ ^[0-3]$ && "$FIRST" != "$SECOND" ]] || usage
  [[ -f "$ROOT/cache/cache.vn97p3cache1.json" ]] || {
    echo "cache missing; run prepare first" >&2
    exit 21
  }
  run_pair "$FIRST" "$SECOND" "$ROOT"
  exit 0
fi

if [[ "$MODE" == "all" ]]; then
  [[ $# -ge 1 && $# -le 2 ]] || usage
  CORPUS="$1"
  ROOT="${2:-/kaggle/working/p3-fast}"
  mkdir -p "$ROOT"
  prepare_cache "$CORPUS" "$ROOT"
  run_pair 0 1 "$ROOT"
  run_pair 2 3 "$ROOT"

  FINAL="$ROOT/final"
  if [[ ! -f "$FINAL/p3-run.vn97p3run1.json" ]]; then
    rm -rf "$FINAL"
    vn97-p3-kaggle-finalize \
      --corpus-dir "$CORPUS" \
      --candidate-dir "$ROOT/candidate-0" \
      --candidate-dir "$ROOT/candidate-1" \
      --candidate-dir "$ROOT/candidate-2" \
      --candidate-dir "$ROOT/candidate-3" \
      --output-dir "$FINAL" \
      --device cuda:0
  fi

  (
    cd "$FINAL"
    find . -maxdepth 1 -type f ! -name SHA256SUMS -printf '%f\0'       | sort -z | xargs -0 sha256sum > SHA256SUMS
  )

  python - "$FINAL" <<'PY'
from pathlib import Path
import shutil
import sys
root = Path(sys.argv[1]).resolve(strict=True)
archive = shutil.make_archive(
    "/kaggle/working/VN97-P3-final",
    "zip",
    root_dir=root.parent,
    base_dir=root.name,
)
print("P3 COMPLETE", archive)
PY
  exit 0
fi

usage
