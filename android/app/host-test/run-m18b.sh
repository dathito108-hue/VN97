#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
OUT="$(mktemp -d)"
trap 'rm -rf "$OUT"' EXIT

kotlinc \
  "$ROOT/android/platform/src/main/java/ai/vn97/platform/ComputePolicy.kt" \
  "$ROOT/android/platform/host-test/ComputePolicy.kt" \
  -Werror \
  -include-runtime \
  -d "$OUT/m7c-compute.jar"

java -jar "$OUT/m7c-compute.jar"

kotlinc \
  "$ROOT/android/platform/src/main/java/ai/vn97/platform/ComputePolicy.kt" \
  "$ROOT/android/app/src/main/java/ai/vn97/app/VN97RuntimeResourcePolicy.kt" \
  "$ROOT/android/app/host-test/M18BRuntimeResourcePolicyTest.kt" \
  -Werror \
  -include-runtime \
  -d "$OUT/m18b.jar"

java -cp "$OUT/m18b.jar" ai.vn97.app.M18BRuntimeResourcePolicyTestKt
