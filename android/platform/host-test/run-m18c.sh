#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
OUT="$(mktemp -d)"
trap 'rm -rf "$OUT"' EXIT

kotlinc \
  "$ROOT/android/platform/src/main/java/ai/vn97/platform/VN97ExecutionHealth.kt" \
  "$ROOT/android/platform/host-test/M18CExecutionHealthTest.kt" \
  -Werror \
  -include-runtime \
  -d "$OUT/m18c.jar"

java -cp "$OUT/m18c.jar" ai.vn97.platform.M18CExecutionHealthTestKt
