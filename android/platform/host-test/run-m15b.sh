#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
OUT="$ROOT/android/platform/host-test/.m15b-out"
rm -rf "$OUT"
mkdir -p "$OUT"

kotlinc \
  "$ROOT/android/platform/src/main/java/ai/vn97/platform/VN97PaperTrading.kt" \
  "$ROOT/android/platform/src/main/java/ai/vn97/platform/VN97MarketObservation.kt" \
  "$ROOT/android/platform/host-test/M15BMarketObservationTest.kt" \
  -include-runtime \
  -d "$OUT/m15b.jar"

java -cp "$OUT/m15b.jar" ai.vn97.platform.M15BMarketObservationTestKt
