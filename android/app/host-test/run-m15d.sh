#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
OUT="$(mktemp -d)"
trap 'rm -rf "$OUT"' EXIT

kotlinc \
  "$ROOT/android/app/src/main/java/ai/vn97/app/VN97PaperTradingSessionStore.kt" \
  "$ROOT/android/app/host-test/M15DPaperTradingSessionStoreTest.kt" \
  -include-runtime \
  -d "$OUT/m15d.jar"

java -cp "$OUT/m15d.jar" ai.vn97.app.M15DPaperTradingSessionStoreTestKt
