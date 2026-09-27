#!/usr/bin/env bash
set -euo pipefail

WORK_ROOT="${1:-${RUNNER_TEMP:-/tmp}/vn97-g03-real}"
SOURCE_ROOT="$WORK_ROOT/mamba2-source"
CAPSULE_ROOT="$WORK_ROOT/vn97-g03-capsule"
REPORT="$WORK_ROOT/g03-campaign-report.json"

if [[ -e "$WORK_ROOT" ]]; then
  echo "G0.3 work root already exists: $WORK_ROOT" >&2
  exit 2
fi
mkdir -p "$WORK_ROOT"

bash tools/r2_g02_fetch_mamba2.sh "$SOURCE_ROOT"

vn97-r2-mamba2-transfer verify-source   --source-root "$SOURCE_ROOT"   --receipt-output "$WORK_ROOT/source-verified.json"

source_inode="$(stat -c '%d:%i' "$SOURCE_ROOT/pytorch_model.bin")"

vn97-r2-mamba2-g03 materialize   --source-root "$SOURCE_ROOT"   --output-root "$CAPSULE_ROOT"   > "$WORK_ROOT/materialize.json"

capsule_weight="$CAPSULE_ROOT/weights/pytorch_model.bin"
capsule_inode="$(stat -c '%d:%i' "$capsule_weight")"
if [[ "$source_inode" != "$capsule_inode" ]]; then
  echo "G0.3 zero-copy invariant failed: source/capsule inode differs" >&2
  exit 3
fi

# Remove all acquisition files. The hardlinked weight inode must stay alive
# solely through the VN97 capsule path after this point.
rm -rf "$SOURCE_ROOT"
test ! -e "$SOURCE_ROOT"
test -s "$capsule_weight"

vn97-r2-mamba2-g03 inspect   --capsule-root "$CAPSULE_ROOT"   > "$WORK_ROOT/post-source-removal-inspect.json"

CAPSULE_ROOT="$CAPSULE_ROOT" REPORT="$REPORT" python - <<'PY'
import hashlib
import json
import os
from pathlib import Path

root = Path(os.environ["CAPSULE_ROOT"])
report_path = Path(os.environ["REPORT"])

manifest_path = root / "capsule.vn97m2g03.json"
manifest = json.loads(manifest_path.read_text(encoding="ascii"))
weight = root / "weights/pytorch_model.bin"

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()

payload = {
    "schema": "VN97M2G03RUN1",
    "status": "REAL_CAPSULE_MATERIALIZED",
    "capsule_id": manifest["capsule_id"],
    "capsule_manifest_sha256": sha256_file(manifest_path),
    "source_weight_sha256": manifest["source_weight_sha256"],
    "weight_bytes": weight.stat().st_size,
    "source_directory_removed": True,
    "zero_copy_materialization": manifest["zero_copy_materialization"],
    "source_runtime_required": manifest["source_runtime_required"],
    "production_parity_required": True,
    "production_activation_authorized": False,
}
report_path.write_text(
    json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
    encoding="ascii",
)
PY

(
  cd "$CAPSULE_ROOT"
  find . -type f -print0     | sort -z     | xargs -0 sha256sum     > "$WORK_ROOT/CAPSULE_SHA256SUMS"
)

echo "VN97 R2-G0.3 real capsule campaign completed."
echo "capsule_root=$CAPSULE_ROOT"
echo "report=$REPORT"
du -sh "$CAPSULE_ROOT"
df -h "$WORK_ROOT"
