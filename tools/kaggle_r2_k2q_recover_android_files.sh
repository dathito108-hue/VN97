#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

WORK_ROOT="/kaggle/working/vn97-k2q-recovery"
SOURCE_ROOT="$WORK_ROOT/mamba2-source"
CAPSULE_ROOT="$WORK_ROOT/g03-capsule"
FINAL_ROOT="/kaggle/working/VN97-K2Q-RECOVERED"

EXPECTED_CAPSULE_ID="8e1aff6c6b6d2947cb25111e48febae6850100fd782a2f112adcab2e047f674e"
EXPECTED_SOURCE_SHA="254d89bfef6dd9f44c8e3f54e0730af3f10c482969117cb37fb33486348b00be"
EXPECTED_G05_MANIFEST_ID="ae782f452368b77b0c17dd7f9a2fd284b98bc0b24bc58e4c8e7d82fba046d2ce"
EXPECTED_G06_RUNTIME_ID="a00feb6b05be1e9219fc9d6bdd6c6ecd9a14fc930202e9689d8c416c6d153762"

python - <<'PY'
import torch
print("torch=", torch.__version__)
print("cuda=", torch.version.cuda)
if not torch.__version__.startswith("2.10.0"):
    raise SystemExit(
        "Recovery requires the same Torch 2.10.0 exporter baseline; "
        f"got {torch.__version__}"
    )
PY

AVAILABLE_KB="$(df -Pk /kaggle/working | awk 'NR==2 {print $4}')"
REQUIRED_KB="$((13 * 1024 * 1024))"
echo "K2Q recovery free_kb=$AVAILABLE_KB required_kb=$REQUIRED_KB"
df -h /kaggle/working || true
if [[ -z "$AVAILABLE_KB" || "$AVAILABLE_KB" -lt "$REQUIRED_KB" ]]; then
  echo "K2Q recovery needs at least 13 GiB free in /kaggle/working." >&2
  echo "Use a fresh Kaggle session/notebook if necessary." >&2
  exit 2
fi

rm -rf "$WORK_ROOT" "$FINAL_ROOT"
mkdir -p "$WORK_ROOT" "$FINAL_ROOT"

python -m pip install --disable-pip-version-check -q   "onnx>=1.17,<2"   "onnxscript>=0.2,<1"
python -m pip install --disable-pip-version-check -q --no-deps -e .

echo "K2Q recovery: downloading exact pinned Mamba-2 2.7B source"
bash tools/r2_g02_fetch_mamba2.sh "$SOURCE_ROOT"

echo "K2Q recovery: materializing zero-copy G0.3 capsule"
python -m vn97.r2.mamba2_g03_capsule_cli materialize   --source-root "$SOURCE_ROOT"   --output-root "$CAPSULE_ROOT"   | tee "$WORK_ROOT/g03-materialize.json"

python - <<PY
import json
from pathlib import Path
line = Path("$WORK_ROOT/g03-materialize.json").read_text().strip().splitlines()[-1]
r = json.loads(line)
assert r["capsule_id"] == "$EXPECTED_CAPSULE_ID", r["capsule_id"]
assert r["source_weight_sha256"] == "$EXPECTED_SOURCE_SHA", r["source_weight_sha256"]
assert r["zero_copy_materialization"] is True
print("K2Q recovery capsule_id=", r["capsule_id"])
PY

# Capsule files are hardlinks to the verified source payload. Remove only the
# acquisition path so the 5.4 GB inode remains once, under the capsule.
rm -rf "$SOURCE_ROOT"

echo "K2Q recovery: exporting exact recurrent-8 G0.5 graph"
python -m vn97.r2.mamba2_parallel_onnx_cli export   --capsule-root "$CAPSULE_ROOT"   --output-dir "$FINAL_ROOT"   --chunk-size 8

python -m vn97.r2.mamba2_parallel_onnx_cli verify   --bundle-dir "$FINAL_ROOT"

echo "K2Q recovery: creating Android G0.6 runtime descriptor"
python -m vn97.r2.mamba2_android_runtime_cli   --bundle-dir "$FINAL_ROOT"

python - <<PY
import json
from pathlib import Path

root = Path("$FINAL_ROOT")
g05 = json.loads((root / "manifest.vn97m2g05.json").read_text())
g06 = json.loads((root / "runtime.vn97m2g06.json").read_text())

assert g05["manifest_id"] == "$EXPECTED_G05_MANIFEST_ID", g05["manifest_id"]
assert g05["capsule_id"] == "$EXPECTED_CAPSULE_ID", g05["capsule_id"]
assert g05["source_weight_sha256"] == "$EXPECTED_SOURCE_SHA"
assert g05["max_chunk_size"] == 8
assert g05["state_contract"]["dtype"] == "float16"

assert g06["runtime_id"] == "$EXPECTED_G06_RUNTIME_ID", g06["runtime_id"]
assert g06["g05_manifest_id"] == "$EXPECTED_G05_MANIFEST_ID"
assert g06["state_dtype"] == "float16"
assert g06["graph_filename"] == "recurrent-8.onnx"
assert g06["production_activation_authorized"] is False

required = [
    root / "runtime.vn97m2g06.json",
    root / "recurrent-8.onnx",
    root / "recurrent-8.onnx.data",
]
for path in required:
    assert path.is_file() and path.stat().st_size > 0, path

print("K2Q RECOVERY PASS")
print("g05_manifest_id=", g05["manifest_id"])
print("g06_runtime_id=", g06["runtime_id"])
print("state_dtype=", g06["state_dtype"])
print("Recovered Android files:")
for path in required:
    print(f"  {path}  {path.stat().st_size} bytes")
PY

# The external-data ONNX graph is now self-contained. Release the temporary
# 5.4 GB capsule inode so only the recovered Android runtime remains.
rm -rf "$CAPSULE_ROOT" "$WORK_ROOT"
sync

echo
echo "VN97 K2Q lost-runtime recovery complete"
echo "folder=$FINAL_ROOT"
ls -lh   "$FINAL_ROOT/runtime.vn97m2g06.json"   "$FINAL_ROOT/recurrent-8.onnx"   "$FINAL_ROOT/recurrent-8.onnx.data"
df -h /kaggle/working || true
