#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
OUT="$(mktemp -d)"
trap 'rm -rf "$OUT"' EXIT

kotlinc \
  "$ROOT/android/runtime/host-test/M10DStrictJsonStub.kt" \
  "$ROOT/android/runtime/src/main/java/ai/vn97/runtime/StrictJson.kt" \
  "$ROOT/android/runtime/src/main/java/ai/vn97/runtime/KnowledgeAcquisitionProposal.kt" \
  "$ROOT/android/runtime/host-test/M16DBoundedAcquisitionProposalTest.kt" \
  -include-runtime \
  -d "$OUT/m16d.jar"

java -cp "$OUT/m16d.jar" ai.vn97.runtime.M16DBoundedAcquisitionProposalTestKt
