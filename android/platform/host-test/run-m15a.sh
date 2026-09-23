#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
OUT="$ROOT/android/platform/host-test/.m15a-out"
rm -rf "$OUT"
mkdir -p "$OUT"

kotlinc   "$ROOT/android/platform/src/main/java/ai/vn97/platform/VN97PaperTrading.kt"   "$ROOT/android/platform/host-test/M15APaperTradingTest.kt"   -include-runtime   -d "$OUT/m15a.jar"

java -cp "$OUT/m15a.jar" ai.vn97.platform.M15APaperTradingTestKt
