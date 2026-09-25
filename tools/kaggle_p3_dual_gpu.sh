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

micro_batch_for() {
  local index="$1"
  local specific_var="VN97_MICRO_BATCH_SIZE_$index"
  local specific="${!specific_var:-}"

  if [[ -n "${VN97_MICRO_BATCH_SIZE:-}" ]]; then
    echo "$VN97_MICRO_BATCH_SIZE"
    return
  fi
  if [[ -n "$specific" ]]; then
    echo "$specific"
    return
  fi

  case "$index" in
    0) echo 2 ;;
    1|2|3) echo 1 ;;
    *) echo 1 ;;
  esac
}

run_one() {
  local index="$1"
  local physical_gpu="$2"
  local root="$3"
  local out="$root/candidate-$index"
  local micro_batch
  micro_batch="$(micro_batch_for "$index")"

  if [[ -f "$out/candidate-report.vn97p3cand1.json" ]]; then
    echo "Reusing completed candidate $index"
    return 0
  fi
  rm -rf "$out"

  echo "candidate $index runtime: physical_gpu=$physical_gpu micro_batch=$micro_batch" >&2

  # One process is pinned to each physical T4. CPU threads concurrently
  # stage/prefetch the next logical batch for that GPU.
  CUDA_VISIBLE_DEVICES="$physical_gpu" \
  OMP_NUM_THREADS="${VN97_CPU_THREADS_PER_GPU:-2}" \
  MKL_NUM_THREADS="${VN97_CPU_THREADS_PER_GPU:-2}" \
  vn97-p3-kaggle-cached-candidate \
    --cache-dir "$root/cache" \
    --candidate-index "$index" \
    --output-dir "$out" \
    --device cuda:0 \
    --cpu-prefetch-workers "${VN97_CPU_PREFETCH_WORKERS:-1}" \
    --micro-batch-size "$micro_batch" \
    --progress-interval-steps "${VN97_PROGRESS_INTERVAL_STEPS:-50}"
}

run_pair() {
  local first="$1"
  local second="$2"
  local root="$3"

  run_one "$first" 0 "$root" >"$root/candidate-$first.log" 2>&1 &
  local p1=$!
  run_one "$second" 1 "$root" >"$root/candidate-$second.log" 2>&1 &
  local p2=$!

  echo "candidate $first -> physical GPU 0 pid=$p1 micro_batch=$(micro_batch_for "$first")"
  echo "candidate $second -> physical GPU 1 pid=$p2 micro_batch=$(micro_batch_for "$second")"
  echo "CPU support threads per GPU process: ${VN97_CPU_THREADS_PER_GPU:-2}"
  echo "CPU prefetch workers per GPU process: ${VN97_CPU_PREFETCH_WORKERS:-1}"
  echo "logical batch remains 8"
  echo "progress interval: ${VN97_PROGRESS_INTERVAL_STEPS:-50} logical steps"

  local monitor_pid=""
  if command -v nvidia-smi >/dev/null 2>&1; then
    (
      while kill -0 "$p1" 2>/dev/null || kill -0 "$p2" 2>/dev/null; do
        echo "---- GPU MONITOR $(date -u +%Y-%m-%dT%H:%M:%SZ) ----"
        nvidia-smi \
          --query-gpu=index,name,utilization.gpu,memory.used,memory.total \
          --format=csv,noheader,nounits || true

        for idx in "$first" "$second"; do
          log="$root/candidate-$idx.log"
          if [[ -f "$log" ]]; then
            line="$(grep 'VN97 P3 PROGRESS' "$log" | tail -n 1 || true)"
            if [[ -n "$line" ]]; then
              echo "$line"
            fi
          fi
        done
        sleep 20
      done
    ) &
    monitor_pid=$!
  fi

  local first_done=""
  local first_status=0
  set +e
  wait -n -p first_done "$p1" "$p2"
  first_status=$?
  set -e

  if [[ "$first_status" -ne 0 ]]; then
    echo "candidate process failed early: pid=$first_done status=$first_status" >&2
    kill "$p1" "$p2" 2>/dev/null || true
    wait "$p1" 2>/dev/null || true
    wait "$p2" 2>/dev/null || true
    if [[ -n "$monitor_pid" ]]; then
      kill "$monitor_pid" 2>/dev/null || true
      wait "$monitor_pid" 2>/dev/null || true
    fi
    echo "===== candidate $first log =====" >&2
    cat "$root/candidate-$first.log" >&2 || true
    echo "===== candidate $second log =====" >&2
    cat "$root/candidate-$second.log" >&2 || true
    exit 20
  fi

  local remaining_pid=""
  local remaining_index=""
  if [[ "$first_done" == "$p1" ]]; then
    remaining_pid="$p2"
    remaining_index="$second"
  else
    remaining_pid="$p1"
    remaining_index="$first"
  fi

  echo "candidate process pid=$first_done completed successfully; waiting for candidate $remaining_index"

  local remaining_status=0
  set +e
  wait "$remaining_pid"
  remaining_status=$?
  set -e

  if [[ -n "$monitor_pid" ]]; then
    kill "$monitor_pid" 2>/dev/null || true
    wait "$monitor_pid" 2>/dev/null || true
  fi

  cat "$root/candidate-$first.log"
  cat "$root/candidate-$second.log"

  if [[ "$remaining_status" -ne 0 ]]; then
    echo "candidate $remaining_index failed: status=$remaining_status" >&2
    exit 20
  fi
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
