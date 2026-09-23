#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
OUT="$(mktemp -d)"
trap 'rm -rf "$OUT"' EXIT

kotlinc \
  "$ROOT/android/runtime/host-test/M10DStrictJsonStub.kt" \
  "$ROOT/android/runtime/src/main/java/ai/vn97/runtime/StrictJson.kt" \
  "$ROOT/android/runtime/src/main/java/ai/vn97/runtime/CapabilityPackage.kt" \
  "$ROOT/android/platform/src/main/java/ai/vn97/platform/VN97RemoteCapabilityArtifact.kt" \
  "$ROOT/android/platform/host-test/M16CRemoteCapabilityArtifactTest.kt" \
  -include-runtime \
  -d "$OUT/m16c.jar"

java -cp "$OUT/m16c.jar" ai.vn97.platform.M16CRemoteCapabilityArtifactTestKt
