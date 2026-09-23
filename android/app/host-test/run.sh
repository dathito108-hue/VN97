#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

KOTLINC="${KOTLINC:-kotlinc}"

"$KOTLINC" \
    "$ROOT/android/app/src/main/java/ai/vn97/app/VN97AppState.kt" \
    "$HERE/M10AAppStateTest.kt" \
    -Werror \
    -include-runtime \
    -d "$WORK/m10a-app-state-test.jar"

java -jar "$WORK/m10a-app-state-test.jar"


"$KOTLINC" \
    "$ROOT/android/runtime/src/main/java/ai/vn97/runtime/StrictJson.kt" \
    "$ROOT/android/runtime/src/main/java/ai/vn97/runtime/ActivatedInventoryEvidence.kt" \
    "$HERE/M10BStrictJsonStub.kt" \
    "$HERE/M10BInventoryEvidenceTest.kt" \
    -Werror \
    -include-runtime \
    -d "$WORK/m10b-inventory-evidence-test.jar"

java -jar "$WORK/m10b-inventory-evidence-test.jar"


"$KOTLINC" \
    "$ROOT/android/app/src/main/java/ai/vn97/app/VN97AppState.kt" \
    "$HERE/M10CApprovalStateTest.kt" \
    -Werror \
    -include-runtime \
    -d "$WORK/m10c-approval-state-test.jar"

java -jar "$WORK/m10c-approval-state-test.jar"


"$KOTLINC" \
    "$ROOT/android/app/src/main/java/ai/vn97/app/VN97BootstrapAssetContract.kt" \
    "$HERE/M10JBootstrapAssetContractTest.kt" \
    -Werror \
    -include-runtime \
    -d "$WORK/m10j-bootstrap-asset-contract-test.jar"

java -jar "$WORK/m10j-bootstrap-asset-contract-test.jar"
