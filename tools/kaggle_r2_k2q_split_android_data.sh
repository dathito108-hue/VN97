#!/usr/bin/env bash
set -euo pipefail

SOURCE="${1:-/kaggle/working/recurrent-8.onnx.data}"
OUT_DIR="${2:-/kaggle/working/K2Q-PARTS}"
CHUNK_MIB="${VN97_K2Q_CHUNK_MIB:-512}"
EXPECTED_SHA="eb253a78b3145704616e61ae90e4f944acaef672b9958d17ed3578181452d890"
EXPECTED_BYTES="5412077568"

if [[ ! -f "$SOURCE" ]]; then
  echo "K2Q split source not found: $SOURCE" >&2
  exit 2
fi

actual_bytes="$(wc -c < "$SOURCE" | tr -d ' ')"
if [[ "$actual_bytes" != "$EXPECTED_BYTES" ]]; then
  echo "K2Q data size mismatch: $actual_bytes != $EXPECTED_BYTES" >&2
  exit 3
fi

actual_sha="$(sha256sum "$SOURCE" | awk '{print $1}')"
if [[ "$actual_sha" != "$EXPECTED_SHA" ]]; then
  echo "K2Q data SHA-256 mismatch." >&2
  exit 4
fi

rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"

echo "K2Q: splitting recurrent data into $CHUNK_MIB MiB parts"
split \
  --bytes="$CHUNK_MIB"M \
  --numeric-suffixes=0 \
  --suffix-length=3 \
  "$SOURCE" \
  "$OUT_DIR/recurrent-8.onnx.data.part"

cp /kaggle/working/runtime.vn97m2g06.json "$OUT_DIR/"
cp /kaggle/working/recurrent-8.onnx "$OUT_DIR/"

(
  cd "$OUT_DIR"
  sha256sum \
    runtime.vn97m2g06.json \
    recurrent-8.onnx \
    recurrent-8.onnx.data.part* \
    > K2Q_SPLIT_SHA256SUMS.txt
)

part_count="$(find "$OUT_DIR" -maxdepth 1 -type f -name 'recurrent-8.onnx.data.part*' | wc -l | tr -d ' ')"
part_bytes="$(OUT_DIR="$OUT_DIR" python - <<'PY'
import os
from pathlib import Path

root = Path(os.environ["OUT_DIR"])
total = sum(
    path.stat().st_size
    for path in root.glob("recurrent-8.onnx.data.part*")
    if path.is_file()
)
print(total)
PY
)"

if [[ "$part_bytes" != "$EXPECTED_BYTES" ]]; then
  echo "K2Q split byte total mismatch: $part_bytes != $EXPECTED_BYTES" >&2
  exit 5
fi

echo "K2Q SPLIT PASS"
echo "part_count=$part_count"
echo "total_bytes=$part_bytes"
echo "source_sha256=$EXPECTED_SHA"
echo "folder=$OUT_DIR"
ls -lh "$OUT_DIR"
