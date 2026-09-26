#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
usage:
  tools/kaggle_p5d3b_distill.sh fresh|resume <p5d1-final-dir> <selected-p5d2-dir> <p5d3a-final-dir> <p3-corpus-dir>

Requires:
  2 CUDA GPUs
  >=8 GiB free in /kaggle/working

Outputs:
  /kaggle/working/p5d3b-c0-work
  /kaggle/working/p5d3b-c0-final
  /kaggle/working/p5d3b-c1-work
  /kaggle/working/p5d3b-c1-final
  /kaggle/working/p5d3b-selection.json
EOF
  exit 2
}

[[ $# -eq 5 ]] || usage

MODE="$1"
P5D1_DIR="$2"
P5D2_DIR="$3"
P5D3A_DIR="$4"
P3_DIR="$5"

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
        f"P5D3B dual pilot needs 2 CUDA GPUs; found {torch.cuda.device_count()}"
    )
print(
    "CUDA READY "
    f"torch={torch.__version__} "
    f"runtime={torch.version.cuda} "
    f"gpu0={torch.cuda.get_device_name(0)} "
    f"gpu1={torch.cuda.get_device_name(1)}"
)
PY

MIN_FREE_BYTES=$((8 * 1024 * 1024 * 1024))
FREE_BYTES="$(df -PB1 /kaggle/working | awk 'NR==2 {print $4}')"
if [[ -z "$FREE_BYTES" || "$FREE_BYTES" -lt "$MIN_FREE_BYTES" ]]; then
  echo "P5D3B requires at least 8 GiB free in /kaggle/working."
  df -h /kaggle/working || true
  echo "Safe cleanup candidates:"
  echo "  /kaggle/working/p5d3a-work"
  echo "  /kaggle/working/p5d2-c1-final"
  echo "  /kaggle/working/p5d2-c0-work"
  echo "  /kaggle/working/p5d2-c1-work"
  echo "Keep:"
  echo "  $P5D1_DIR"
  echo "  $P5D2_DIR"
  echo "  $P5D3A_DIR"
  exit 3
fi

export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

for IDX in 0 1; do
  WORK="/kaggle/working/p5d3b-c${IDX}-work"
  FINAL="/kaggle/working/p5d3b-c${IDX}-final"
  if [[ "$MODE" == "fresh" ]]; then
    rm -rf "$WORK" "$FINAL"
  else
    if [[ -d "$FINAL" && ! -f "$FINAL/p5d3b-report.json" ]]; then
      rm -rf "$FINAL"
    fi
    rm -f "$WORK/state.p5d3b.pt.tmp"
  fi
done
rm -f /kaggle/working/p5d3b-selection.json

echo "P5D3B disk status before launch:"
df -h /kaggle/working || true

run_candidate() {
  local IDX="$1"
  local PHYSICAL_GPU="$2"
  local WORK="/kaggle/working/p5d3b-c${IDX}-work"
  local FINAL="/kaggle/working/p5d3b-c${IDX}-final"
  local LOG="/kaggle/working/p5d3b-c${IDX}.log"

  if [[ "$MODE" == "resume" && -f "$FINAL/p5d3b-report.json" ]]; then
    echo "P5D3B candidate ${IDX} already complete"
    return 0
  fi

  CUDA_VISIBLE_DEVICES="$PHYSICAL_GPU" \
    python -m vn97.p5d3_relational_distillation_cli \
      --p5d1-dir "$P5D1_DIR" \
      --p5d2-student-dir "$P5D2_DIR" \
      --p5d3a-dir "$P5D3A_DIR" \
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
  echo "One or more P5D3B candidates failed. Use resume after fixing the error."
  exit 1
fi

python - <<'PY'
import json
from pathlib import Path

rows = []
rank = {
    "RELATIONAL_SIGNAL": 3,
    "WEAK_RELATIONAL_SIGNAL": 2,
    "REJECTED_REGRESSION": 0,
    "REJECTED_NONFINITE": -1,
}

for idx in (0, 1):
    root = Path(f"/kaggle/working/p5d3b-c{idx}-final")
    report = json.loads((root / "p5d3b-report.json").read_text())

    rel_before = float(
        report["relational_holdout"]["before"]["mean_total_loss"]
    )
    rel_after = float(
        report["relational_holdout"]["after"]["mean_total_loss"]
    )
    rel_gain = (rel_before - rel_after) / max(rel_before, 1e-9)

    teacher_before = float(
        report["teacher_holdout"]["before"]["mean_loss"]
    )
    teacher_after = float(
        report["teacher_holdout"]["after"]["mean_loss"]
    )
    p3_before = float(
        report["p3"]["before"]["mean_loss"]
    )
    p3_after = float(
        report["p3"]["after"]["mean_loss"]
    )
    probe = int(
        report["p4_probe_after"]["passed"]
    )

    row = {
        "candidate_index": idx,
        "candidate_id": report["candidate_id"],
        "status": report["status"],
        "relational_gain": rel_gain,
        "relational_after": rel_after,
        "teacher_loss_ratio": teacher_after / max(teacher_before, 1e-9),
        "p3_loss_ratio": p3_after / max(p3_before, 1e-9),
        "probe_passed": probe,
        "directory": str(root),
    }
    row["_key"] = (
        rank.get(report["status"], -2),
        rel_gain,
        -row["teacher_loss_ratio"],
        -row["p3_loss_ratio"],
        probe,
    )
    rows.append(row)

selected = max(rows, key=lambda row: row["_key"])
for row in rows:
    row.pop("_key", None)

payload = {
    "schema": "VN97P5D3BSELECT1",
    "selected_candidate_index": selected["candidate_index"],
    "selected_candidate_id": selected["candidate_id"],
    "selected_directory": selected["directory"],
    "candidates": rows,
}
path = Path("/kaggle/working/p5d3b-selection.json")
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

print(
    "VN97P5D3BSELECT "
    f"candidate={selected['candidate_index']} "
    f"candidate_id={selected['candidate_id']} "
    f"status={selected['status']} "
    f"relational_gain={selected['relational_gain']:.6f} "
    f"teacher_ratio={selected['teacher_loss_ratio']:.6f} "
    f"p3_ratio={selected['p3_loss_ratio']:.6f} "
    f"probe_passed={selected['probe_passed']}/12 "
    f"directory={selected['directory']}"
)
print(f"selection: {path}")
PY
