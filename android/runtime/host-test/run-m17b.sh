#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
OUT="$(mktemp -d)"
trap 'rm -rf "$OUT"' EXIT

kotlinc \
  "$ROOT/android/runtime/host-test/M10DStrictJsonStub.kt" \
  "$ROOT/android/runtime/src/main/java/ai/vn97/runtime/StrictJson.kt" \
  "$ROOT/android/runtime/src/main/java/ai/vn97/runtime/SelfImprovementEvaluation.kt" \
  "$ROOT/android/runtime/host-test/M17BHeldOutEvaluationTest.kt" \
  -include-runtime \
  -d "$OUT/m17b.jar"

java -cp "$OUT/m17b.jar" ai.vn97.runtime.M17BHeldOutEvaluationTestKt
