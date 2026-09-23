#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
OUT="$(mktemp -d)"
trap 'rm -rf "$OUT"' EXIT

kotlinc \
  "$ROOT/android/platform/src/main/java/ai/vn97/platform/VN97PaperTrading.kt" \
  "$ROOT/android/platform/src/main/java/ai/vn97/platform/VN97MarketObservation.kt" \
  "$ROOT/android/platform/src/main/java/ai/vn97/platform/VN97PaperPerformance.kt" \
  "$ROOT/android/app/src/main/java/ai/vn97/app/VN97PaperPerformanceEvidenceStore.kt" \
  "$ROOT/android/app/host-test/M15FPaperPerformanceEvidenceTest.kt" \
  -include-runtime \
  -d "$OUT/m15f.jar"

java -cp "$OUT/m15f.jar" ai.vn97.app.M15FPaperPerformanceEvidenceTestKt
