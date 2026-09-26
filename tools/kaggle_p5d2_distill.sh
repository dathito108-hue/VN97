#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
usage:
  tools/kaggle_p5d2_distill.sh fresh|resume <p5d1-final-dir> <vn97-tokenizer> <p3-corpus-dir>

Requires:
  2 CUDA GPUs

Outputs:
  /kaggle/working/p5d2-c0-work
  /kaggle/working/p5d2-c0-final
  /kaggle/working/p5d2-c1-work
  /kaggle/working/p5d2-c1-final
  /kaggle/working/p5d2-selection.json
EOF
  exit 2
}

[[ $# -eq 4 ]] || usage

MODE="$1"
P5D1_DIR="$2"
TOKENIZER="$3"
P3_DIR="$4"

case "$MODE" in
  fresh|resume) ;;
  *) usage ;;
esac

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

python - <<'PY'
import torch
if torch.cuda.device_count() < 2:
    raise SystemExit(
        f"P5D2 dual pilot needs 2 CUDA GPUs; found {torch.cuda.device_count()}"
    )
print(
    "CUDA READY "
    f"torch={torch.__version__} "
    f"runtime={torch.version.cuda} "
    f"gpu0={torch.cuda.get_device_name(0)} "
    f"gpu1={torch.cuda.get_device_name(1)}"
)
PY

export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

MIN_FREE_BYTES=$((5 * 1024 * 1024 * 1024))
FREE_BYTES="$(df -PB1 /kaggle/working | awk 'NR==2 {print $4}')"
if [[ -z "$FREE_BYTES" || "$FREE_BYTES" -lt "$MIN_FREE_BYTES" ]]; then
  echo "P5D2 requires at least 5 GiB free in /kaggle/working."
  echo "Current disk status:"
  df -h /kaggle/working || true
  echo "Safe cleanup candidates after P5D1:"
  echo "  /kaggle/working/hf-cache"
  echo "  /kaggle/working/p5d2-c0-work"
  echo "  /kaggle/working/p5d2-c1-work"
  echo "  /kaggle/working/p5d2-c0-final"
  echo "  /kaggle/working/p5d2-c1-final"
  echo "  /kaggle/working/p5d2-c0.log"
  echo "  /kaggle/working/p5d2-c1.log"
  exit 3
fi

for IDX in 0 1; do
  WORK="/kaggle/working/p5d2-c${IDX}-work"
  FINAL="/kaggle/working/p5d2-c${IDX}-final"
  if [[ "$MODE" == "fresh" ]]; then
    rm -rf "$WORK" "$FINAL"
  else
    if [[ -d "$FINAL" && ! -f "$FINAL/p5d2-report.json" ]]; then
      rm -rf "$FINAL"
    fi
    # A failed torch.save can leave a multi-gigabyte partial temp file behind.
    # It is never a valid resume checkpoint and must not consume the next run's
    # disk budget.
    rm -f "$WORK/state.p5d2.pt.tmp"
  fi
done
rm -f /kaggle/working/p5d2-selection.json

echo "P5D2 disk status before launch:"
df -h /kaggle/working || true

run_candidate() {
  local IDX="$1"
  local PHYSICAL_GPU="$2"
  local WORK="/kaggle/working/p5d2-c${IDX}-work"
  local FINAL="/kaggle/working/p5d2-c${IDX}-final"
  local LOG="/kaggle/working/p5d2-c${IDX}.log"

  if [[ "$MODE" == "resume" && -f "$FINAL/p5d2-report.json" ]]; then
    echo "P5D2 candidate ${IDX} already complete"
    return 0
  fi

  CUDA_VISIBLE_DEVICES="$PHYSICAL_GPU" \
    python -m vn97.p5d_behavioral_distillation_cli \
      --teacher-corpus-dir "$P5D1_DIR" \
      --vn97-tokenizer "$TOKENIZER" \
      --p3-corpus-dir "$P3_DIR" \
      --candidate-index "$IDX" \
      --work-dir "$WORK" \
      --output-dir "$FINAL" \
      --device cuda:0 \
      2>&1 | tee "$LOG"
}

run_candidate 0 0 &
PID0=$!
run_candidate 1 1 &
PID1=$!

FAIL=0
wait "$PID0" || FAIL=1
wait "$PID1" || FAIL=1

if [[ "$FAIL" -ne 0 ]]; then
  echo "One or more P5D2 candidates failed. Use resume after fixing the error."
  exit 1
fi

python - <<'PY'
import json
from pathlib import Path

rows = []
for idx in (0, 1):
    root = Path(f"/kaggle/working/p5d2-c{idx}-final")
    report = json.loads((root / "p5d2-report.json").read_text())
    before = report["teacher_holdout"]["before"]
    after = report["teacher_holdout"]["after"]
    improvement = float(before["mean_loss"]) - float(after["mean_loss"])
    top1_gain = float(after["top1_accuracy"]) - float(before["top1_accuracy"])
    probe = int(report["p4_probe_after"]["passed"])
    rank = {
        "BEHAVIORAL_SIGNAL": 2,
        "WEAK_SIGNAL": 1,
        "NO_SIGNAL": 0,
        "REJECTED_NONFINITE": -1,
    }.get(report["status"], -2)
    rows.append(
        {
            "candidate_index": idx,
            "candidate_id": report["candidate_id"],
            "status": report["status"],
            "teacher_loss_improvement": improvement,
            "teacher_top1_gain": top1_gain,
            "teacher_after_loss": float(after["mean_loss"]),
            "p3_after_loss": float(report["p3"]["after"]["mean_loss"]),
            "probe_passed": probe,
            "directory": str(root),
            "_key": [
                rank,
                improvement,
                top1_gain,
                probe,
                -float(after["mean_loss"]),
            ],
        }
    )

selected = max(rows, key=lambda row: tuple(row["_key"]))
for row in rows:
    row.pop("_key", None)

payload = {
    "schema": "VN97P5D2SELECT1",
    "selected_candidate_index": selected["candidate_index"],
    "selected_candidate_id": selected["candidate_id"],
    "selected_directory": selected["directory"],
    "candidates": rows,
}
path = Path("/kaggle/working/p5d2-selection.json")
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(
    "VN97P5D2SELECT "
    f"candidate={selected['candidate_index']} "
    f"candidate_id={selected['candidate_id']} "
    f"status={selected['status']} "
    f"teacher_loss_improvement={selected['teacher_loss_improvement']:.6f} "
    f"probe_passed={selected['probe_passed']}/12 "
    f"directory={selected['directory']}"
)
print(f"selection: {path}")
PY
