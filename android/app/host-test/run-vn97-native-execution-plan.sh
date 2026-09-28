#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
"${KOTLINC:-kotlinc}" \
    "$ROOT/android/runtime/src/main/java/ai/vn97/runtime/VN97NativeExecutionPlan.kt" \
    "$HERE/VN97NativeExecutionPlanTest.kt" \
    -Werror -include-runtime -d "$WORK/vn97-native-execution-plan.jar"
java -jar "$WORK/vn97-native-execution-plan.jar"
