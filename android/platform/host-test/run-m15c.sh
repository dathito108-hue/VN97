#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
OUT="$(mktemp -d)"
trap 'rm -rf "$OUT"' EXIT

kotlinc \
  "$ROOT/android/platform/src/main/java/ai/vn97/platform/VN97PaperTrading.kt" \
  "$ROOT/android/platform/src/main/java/ai/vn97/platform/VN97MarketObservation.kt" \
  "$ROOT/android/platform/src/main/java/ai/vn97/platform/VN97MarketDataAcquisition.kt" \
  "$ROOT/android/platform/src/main/java/ai/vn97/platform/VN97PaperTradingEpisodes.kt" \
  "$ROOT/android/platform/host-test/M15CMarketDataEpisodesTest.kt" \
  -include-runtime \
  -d "$OUT/m15c.jar"

java -cp "$OUT/m15c.jar" ai.vn97.platform.M15CMarketDataEpisodesTestKt
