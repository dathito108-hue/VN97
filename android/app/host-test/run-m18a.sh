#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
OUT="$(mktemp -d)"
trap 'rm -rf "$OUT"' EXIT

kotlinc \
  "$ROOT/android/app/src/main/java/ai/vn97/app/VN97MobileRecoveryPolicy.kt" \
  "$ROOT/android/app/host-test/M18AMobileRecoveryTest.kt" \
  -Werror \
  -include-runtime \
  -d "$OUT/m18a.jar"

java -cp "$OUT/m18a.jar" ai.vn97.app.M18AMobileRecoveryTestKt
