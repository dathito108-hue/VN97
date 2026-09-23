#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
OUT="$ROOT/android/platform/host-test/.m14d-out"
rm -rf "$OUT"
mkdir -p "$OUT"

kotlinc   "$ROOT/android/platform/src/main/java/ai/vn97/platform/VN97GameMultiTouch.kt"   "$ROOT/android/platform/host-test/M14DGameMultiTouchContractTest.kt"   -include-runtime   -d "$OUT/m14d.jar"

java -cp "$OUT/m14d.jar" ai.vn97.platform.M14DGameMultiTouchContractTestKt
