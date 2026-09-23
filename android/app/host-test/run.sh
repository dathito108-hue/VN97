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
