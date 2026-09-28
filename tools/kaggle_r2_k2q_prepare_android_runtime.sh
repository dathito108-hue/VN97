#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

K2_ROOT="/kaggle/working/vn97-r2-k2"
G05_ROOT="$K2_ROOT/g05-recurrent-8"
K2P_RECEIPT="/kaggle/working/vn97-k2p-g05-bridge.json"
G06_DESCRIPTOR="$G05_ROOT/runtime.vn97m2g06.json"
TRANSFER_MANIFEST="/kaggle/working/vn97-k2q-android-transfer.json"

for required in   "$K2P_RECEIPT"   "$G05_ROOT/manifest.vn97m2g05.json"   "$G05_ROOT/recurrent-8.onnx"; do
  if [[ ! -f "$required" ]]; then
    echo "K2Q missing prerequisite: $required" >&2
    exit 2
  fi
done

python - <<PY
import json
from pathlib import Path
r = json.loads(Path("$K2P_RECEIPT").read_text())
assert r["schema"] == "VN97M2K2PG05BRIDGE1"
assert r["status"] == "PASS"
assert r["gate_passed"] is True
assert r["gate_failures"] == []
assert r["exact_decisions"] == r["total_same_input_decisions"]
assert r["mismatch_count"] == 0
print("K2P receipt_id=", r["receipt_id"])
print("K2P g05_manifest_id=", r["g05_manifest_id"])
PY

if [[ ! -f "$G06_DESCRIPTOR" ]]; then
  python -m vn97.r2.mamba2_android_runtime_cli     --bundle-dir "$G05_ROOT"
else
  echo "K2Q: reusing existing G0.6 descriptor"
fi

python - <<PY
import hashlib
import json
from pathlib import Path

g05_root = Path("$G05_ROOT")
k2p = json.loads(Path("$K2P_RECEIPT").read_text())
g05 = json.loads((g05_root / "manifest.vn97m2g05.json").read_text())
g06 = json.loads((g05_root / "runtime.vn97m2g06.json").read_text())

assert g05["manifest_id"] == k2p["g05_manifest_id"]
assert g06["g05_manifest_id"] == g05["manifest_id"]
assert g06["capsule_id"] == k2p["capsule_id"]
assert g06["source_weight_sha256"] == k2p["source_weight_sha256"]
assert g06["state_dtype"] == "float16"
assert g06["max_chunk_size"] == 8
assert g06["graph_filename"] == "recurrent-8.onnx"
assert g06["production_activation_authorized"] is False

def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()

files = ["runtime.vn97m2g06.json"]
for item in g06["graph_files"]:
    path = g05_root / item["filename"]
    assert path.is_file()
    actual = sha256_file(path)
    assert actual == item["sha256"]
    assert path.stat().st_size == item["bytes"]
    files.append(item["filename"])

payload = {
    "schema": "VN97M2K2QTRANSFER1",
    "k2p_receipt_id": k2p["receipt_id"],
    "g05_manifest_id": g05["manifest_id"],
    "g06_runtime_id": g06["runtime_id"],
    "state_dtype": g06["state_dtype"],
    "graph_filename": g06["graph_filename"],
    "files_to_select_on_android": files,
    "graph_files": g06["graph_files"],
    "weights_changed": False,
    "graph_changed": False,
    "production_activation_authorized": False,
}
raw = json.dumps(
    payload,
    sort_keys=True,
    separators=(",", ":"),
    ensure_ascii=True,
).encode("ascii")
receipt_id = hashlib.sha256(b"VN97M2K2QTRANSFER1\0" + raw).hexdigest()
payload["receipt_id"] = receipt_id
Path("$TRANSFER_MANIFEST").write_text(
    json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ) + "\n",
    encoding="ascii",
)

print("K2Q g06_runtime_id:", g06["runtime_id"])
print("K2Q state_dtype:", g06["state_dtype"])
print("K2Q files to download/select on Android:")
for name in files:
    path = g05_root / name
    print("  ", path, path.stat().st_size, "bytes")
print("K2Q transfer_manifest:", "$TRANSFER_MANIFEST")
PY

ZIP="/kaggle/working/VN97-R2-K2Q-android-metadata.zip"
rm -f "$ZIP"
python - <<PY
from pathlib import Path
import zipfile

out = Path("$ZIP")
with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as z:
    z.write(
        Path("$G06_DESCRIPTOR"),
        arcname="runtime.vn97m2g06.json",
    )
    z.write(
        Path("$TRANSFER_MANIFEST"),
        arcname="vn97-k2q-android-transfer.json",
    )
    z.write(
        Path("$K2P_RECEIPT"),
        arcname="vn97-k2p-g05-bridge.json",
    )
print(out)
PY

echo "VN97 K2Q Android transfer preparation complete"
echo "metadata_bundle=$ZIP"
echo "NOTE: graph payloads remain in $G05_ROOT and must be downloaded separately."
