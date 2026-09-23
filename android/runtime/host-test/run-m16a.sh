#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
OUT="$(mktemp -d)"
trap 'rm -rf "$OUT"' EXIT

kotlinc \
  "$ROOT/android/runtime/host-test/M10DStrictJsonStub.kt" \
  "$ROOT/android/runtime/host-test/M16AKnowledgeRuntimeStubs.kt" \
  "$ROOT/android/runtime/src/main/java/ai/vn97/runtime/StrictJson.kt" \
  "$ROOT/android/runtime/src/main/java/ai/vn97/runtime/CapabilityPackage.kt" \
  "$ROOT/android/runtime/src/main/java/ai/vn97/runtime/CapabilitySignature.kt" \
  "$ROOT/android/runtime/src/main/java/ai/vn97/runtime/CapabilityStage.kt" \
  "$ROOT/android/runtime/src/main/java/ai/vn97/runtime/CapabilityTrust.kt" \
  "$ROOT/android/runtime/src/main/java/ai/vn97/runtime/KnowledgePublisherTrustRegistry.kt" \
  "$ROOT/android/runtime/src/main/java/ai/vn97/runtime/KnowledgeAcquisitionLedger.kt" \
  "$ROOT/android/runtime/src/main/java/ai/vn97/runtime/KnowledgeAcquisition.kt" \
  "$ROOT/android/runtime/host-test/M16ASignedKnowledgeAcquisitionTest.kt" \
  -include-runtime \
  -d "$OUT/m16a.jar"

java -cp "$OUT/m16a.jar" ai.vn97.runtime.M16ASignedKnowledgeAcquisitionTestKt
