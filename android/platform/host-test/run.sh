#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

KOTLINC="${KOTLINC:-kotlinc}"

"$KOTLINC" \
    "$ROOT/android/platform/src/main/java/ai/vn97/platform/M6Approval.kt" \
    "$ROOT/android/platform/src/main/java/ai/vn97/platform/ComputePolicy.kt" \
    "$HERE/ApprovalCompatibility.kt" \
    "$HERE/ComputePolicy.kt" \
    -Werror \
    -include-runtime \
    -d "$WORK/m7b2-approval-test.jar"

java -jar "$WORK/m7b2-approval-test.jar"
