#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
"${KOTLINC:-kotlinc}" \
    "$ROOT/android/app/src/main/java/ai/vn97/app/VN97G06BundledRuntime.kt" \
    "$HERE/R2BundledAssetManagerStub.kt" \
    "$HERE/R2BundledApplicationStub.kt" \
    "$HERE/R2BundledRuntimeTest.kt" \
    -Werror -include-runtime -d "$WORK/r2-bundled.jar"
java -jar "$WORK/r2-bundled.jar"
