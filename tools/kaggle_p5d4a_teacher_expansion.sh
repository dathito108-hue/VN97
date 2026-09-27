#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
usage:
  tools/kaggle_p5d4a_teacher_expansion.sh fresh|resume <p5d1-parent-dir> <p3-corpus-dir> <held-out-p4.jsonl>

Purpose:
  Expand the sealed 600-record P5D1 Falcon3-Mamba teacher corpus to a
  2400-record P5D1-compatible corpus while reusing the original 600 teacher
  generations exactly.

Expansion target:
  P4: 240 prompts/category x 6 categories = 1440
  P3: 960 prompts
  Total: 2400 records

Only 1800 new teacher generations are required because the parent 600 records
are seeded into the new work directory before generation.

Requires:
  2 CUDA GPUs
  >=15 GiB free in /kaggle/working before teacher download

Outputs:
  /kaggle/working/p5d4a-work
  /kaggle/working/p5d4a-final

The final corpus intentionally retains the VN97P5D1 schema so existing sealed
teacher-corpus verification code can consume it without a parallel format.
EOF
  exit 2
}

[[ $# -eq 4 ]] || usage

MODE="$1"
PARENT="$2"
P3_DIR="$3"
DEV_SUITE="$4"

WORK="/kaggle/working/p5d4a-work"
FINAL="/kaggle/working/p5d4a-final"
HF_CACHE="/kaggle/working/hf-cache-p5d4a"

case "$MODE" in
  fresh)
    rm -rf "$WORK" "$FINAL" "$HF_CACHE"
    ;;
  resume)
    if [[ -f "$FINAL/p5d1-report.json" ]]; then
      echo "P5D4A already complete: $FINAL"
      exit 0
    fi
    ;;
  *)
    usage
    ;;
esac

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

python - <<'PY'
import torch
if torch.cuda.device_count() < 2:
    raise SystemExit(
        f"P5D4A teacher expansion needs 2 CUDA GPUs; found {torch.cuda.device_count()}"
    )
print(
    "CUDA READY "
    f"torch={torch.__version__} "
    f"runtime={torch.version.cuda} "
    f"gpu0={torch.cuda.get_device_name(0)} "
    f"gpu1={torch.cuda.get_device_name(1)}"
)
PY

FULL_DOWNLOAD_FREE_BYTES=$((15 * 1024 * 1024 * 1024))
CACHED_RESUME_FREE_BYTES=$((1 * 1024 * 1024 * 1024))
CACHE_READY_BYTES=$((8 * 1024 * 1024 * 1024))

FREE_BYTES="$(df -PB1 /kaggle/working | awk 'NR==2 {print $4}')"
CACHE_BYTES=0
if [[ -d "$HF_CACHE" ]]; then
  CACHE_BYTES="$(du -sb "$HF_CACHE" 2>/dev/null | awk '{print $1}')"
  CACHE_BYTES="${CACHE_BYTES:-0}"
fi

MIN_FREE_BYTES="$FULL_DOWNLOAD_FREE_BYTES"
DISK_MODE="fresh_teacher_download"
if [[ "$MODE" == "resume" && "$CACHE_BYTES" -ge "$CACHE_READY_BYTES" ]]; then
  MIN_FREE_BYTES="$CACHED_RESUME_FREE_BYTES"
  DISK_MODE="reuse_existing_teacher_cache"
fi

echo "P5D4A disk gate mode=$DISK_MODE free_bytes=$FREE_BYTES cache_bytes=$CACHE_BYTES"

if [[ -z "$FREE_BYTES" || "$FREE_BYTES" -lt "$MIN_FREE_BYTES" ]]; then
  if [[ "$DISK_MODE" == "reuse_existing_teacher_cache" ]]; then
    echo "P5D4A resume has a reusable teacher cache but still needs at least 1 GiB free."
  else
    echo "P5D4A needs at least 15 GiB free before downloading the 7B teacher."
  fi
  df -h /kaggle/working || true
  echo "Safe cleanup candidates before P5D4A:"
  echo "  /kaggle/working/p5d3a-work"
  echo "  /kaggle/working/p5d3b-c1-final"
  echo "  /kaggle/working/p5d3b-c0-work"
  echo "  /kaggle/working/p5d3b-c1-work"
  echo "  /kaggle/working/p5d3b-c0.log"
  echo "  /kaggle/working/p5d3b-c1.log"
  echo "Keep:"
  echo "  $PARENT"
  echo "  /kaggle/working/p5d3b-c0-final"
  echo "  $HF_CACHE  # keep on resume if it already contains the teacher"
  exit 3
fi

mkdir -p "$WORK/records"

# Seed the exact original 600 records into the expanded work directory.
# This avoids regenerating already-sealed teacher outputs.
PARENT="$PARENT" WORK="$WORK" python - <<'PY'
from pathlib import Path
import hashlib
import json
import os

parent = Path(os.environ["PARENT"]).resolve(strict=True)
work = Path(os.environ["WORK"]).resolve()

required = {
    "SHA256SUMS",
    "p5d1-report.json",
    "teacher-corpus.jsonl",
}
names = {item.name for item in parent.iterdir()}
if names != required:
    raise SystemExit("P5D4A parent P5D1 file set mismatch")

sums = {}
for line in (parent / "SHA256SUMS").read_text(encoding="ascii").splitlines():
    if len(line) < 67 or line[64:66] != "  ":
        raise SystemExit("P5D4A parent SHA256SUMS malformed")
    sums[line[66:]] = line[:64]

for name, expected in sums.items():
    actual = hashlib.sha256((parent / name).read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit(f"P5D4A parent SHA256 mismatch: {name}")

report = json.loads((parent / "p5d1-report.json").read_text())
if (
    report.get("schema") != "VN97P5D1"
    or report.get("completed_records") != 600
    or report.get("total_records") != 600
):
    raise SystemExit("P5D4A requires the sealed 600-record P5D1 parent")

revision = str(report["teacher"]["revision"])
records_dir = work / "records"
records_dir.mkdir(parents=True, exist_ok=True)

seeded = 0
for line in (parent / "teacher-corpus.jsonl").read_text(
    encoding="utf-8"
).splitlines():
    if not line.strip():
        continue
    row = json.loads(line)
    if str(row.get("teacher_revision", "")) != revision:
        raise SystemExit("P5D4A parent teacher revision drift")
    record_id = str(row["record_id"])
    payload = (
        json.dumps(
            row,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    path = records_dir / f"{record_id}.json"
    if path.exists() and path.read_bytes() != payload:
        raise SystemExit(f"P5D4A seeded record mismatch: {record_id}")
    path.write_bytes(payload)
    seeded += 1

if seeded != 600:
    raise SystemExit(f"P5D4A expected 600 parent records, got {seeded}")

print(
    "VN97 P5D4A SEED "
    f"parent_records={seeded} "
    f"teacher_revision={revision}"
)
print(revision)
PY

TEACHER_REVISION="$(
  PARENT="$PARENT" python - <<'PY'
from pathlib import Path
import json
import os
report = json.loads(
    (Path(os.environ["PARENT"]) / "p5d1-report.json").read_text()
)
print(report["teacher"]["revision"])
PY
)"

export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"
export HF_HOME="$HF_CACHE"
export TOKENIZERS_PARALLELISM=false
export PIP_NO_CACHE_DIR=1

python -m pip install --disable-pip-version-check --no-cache-dir -q \
  "transformers==4.46.1" \
  "tokenizers==0.20.3" \
  "accelerate>=0.34,<2" \
  "huggingface_hub>=0.24,<1" \
  safetensors \
  sentencepiece

python - <<'PY'
import tokenizers
import transformers
print(
    "P5D4A HF STACK "
    f"transformers={transformers.__version__} "
    f"tokenizers={tokenizers.__version__}"
)
if transformers.__version__ != "4.46.1":
    raise SystemExit("P5D4A transformers compatibility pin was not applied")
if tokenizers.__version__ != "0.20.3":
    raise SystemExit("P5D4A tokenizers compatibility pin was not applied")
PY

python -m vn97.p5d_teacher_corpus_cli \
  --p3-corpus-dir "$P3_DIR" \
  --dev-suite "$DEV_SUITE" \
  --work-dir "$WORK" \
  --output-dir "$FINAL" \
  --teacher-revision "$TEACHER_REVISION" \
  --p3-records 960 \
  --p4-per-category 240 \
  --accept-teacher-license TII-FALCON-LLM-2.0 \
  --progress-interval 20

PARENT="$PARENT" FINAL="$FINAL" python - <<'PY'
from pathlib import Path
import json
import os

parent = json.loads(
    (Path(os.environ["PARENT"]) / "p5d1-report.json").read_text()
)
expanded = json.loads(
    (Path(os.environ["FINAL"]) / "p5d1-report.json").read_text()
)

if expanded.get("schema") != "VN97P5D1":
    raise SystemExit("P5D4A expanded corpus schema mismatch")
if expanded.get("completed_records") != 2400:
    raise SystemExit("P5D4A expanded corpus did not reach 2400 records")
if expanded.get("total_records") != 2400:
    raise SystemExit("P5D4A expanded corpus total mismatch")
if expanded["teacher"]["revision"] != parent["teacher"]["revision"]:
    raise SystemExit("P5D4A teacher revision changed")

print(
    "VN97P5D4A status=EXPANDED_TEACHER_CORPUS_READY "
    f"records={expanded['completed_records']} "
    "reused_parent=600 "
    "new_teacher_generations=1800 "
    f"teacher_revision={expanded['teacher']['revision']} "
    f"corpus_sha256={expanded['corpus_sha256']}"
)
PY

rm -rf "$HF_CACHE"

echo "P5D4A complete"
echo "work: $WORK"
echo "final: $FINAL"
df -h /kaggle/working || true
