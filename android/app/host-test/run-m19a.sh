#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
OUT="$(mktemp -d)"
trap 'rm -rf "$OUT"' EXIT

kotlinc \
  "$ROOT/android/app/host-test/M19ABuildConfigStub.kt" \
  "$ROOT/android/app/src/main/java/ai/vn97/app/VN97TurnkeyReleaseManifest.kt" \
  "$ROOT/android/app/host-test/M19ATurnkeyReleaseManifestTest.kt" \
  -Werror \
  -include-runtime \
  -d "$OUT/m19a.jar"

java -cp "$OUT/m19a.jar" ai.vn97.app.M19ATurnkeyReleaseManifestTestKt
