#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

KOTLINC="${KOTLINC:-kotlinc}"

"$KOTLINC" \
    "$ROOT/android/platform/src/main/java/ai/vn97/platform/M6Approval.kt" \
    "$ROOT/android/platform/src/main/java/ai/vn97/platform/ComputePolicy.kt" \
    "$HERE/ApprovalCompatibility.kt" \
    "$HERE/ComputePolicy.kt" \
    -Werror \
    -include-runtime \
    -d "$WORK/m7b2-approval-test.jar"

java -jar "$WORK/m7b2-approval-test.jar"

"$KOTLINC" \
    "$ROOT/android/platform/src/main/java/ai/vn97/platform/M6Approval.kt" \
    "$ROOT/android/platform/src/main/java/ai/vn97/platform/M6ExternalHandoff.kt" \
    "$HERE/M7ORuntimeStubs.kt" \
    "$HERE/M7OExternalHandoffTest.kt" \
    -Werror \
    -include-runtime \
    -d "$WORK/m7o-external-handoff-test.jar"

java -jar "$WORK/m7o-external-handoff-test.jar"

"$KOTLINC" \
    "$ROOT/android/platform/src/main/java/ai/vn97/platform/M6Approval.kt" \
    "$ROOT/android/platform/src/main/java/ai/vn97/platform/M6ExternalHandoff.kt" \
    "$ROOT/android/platform/src/main/java/ai/vn97/platform/M6ExecutionFabric.kt" \
    "$HERE/M7PRuntimeStubs.kt" \
    "$HERE/M7PExecutionFabricTest.kt" \
    -Werror \
    -include-runtime \
    -d "$WORK/m7p-execution-fabric-test.jar"

java -jar "$WORK/m7p-execution-fabric-test.jar"

"$KOTLINC" \
    "$ROOT/android/platform/src/main/java/ai/vn97/platform/M6Approval.kt" \
    "$ROOT/android/platform/src/main/java/ai/vn97/platform/M6ExternalHandoff.kt" \
    "$ROOT/android/platform/src/main/java/ai/vn97/platform/M6ExecutionFabric.kt" \
    "$ROOT/android/platform/src/main/java/ai/vn97/platform/M6DurableActionAudit.kt" \
    "$HERE/M7PRuntimeStubs.kt" \
    "$HERE/M7QDurableAuditTest.kt" \
    -Werror \
    -include-runtime \
    -d "$WORK/m7q-durable-audit-test.jar"

java -jar "$WORK/m7q-durable-audit-test.jar"

"$KOTLINC" \
    "$ROOT/android/platform/src/main/java/ai/vn97/platform/M6Approval.kt" \
    "$ROOT/android/platform/src/main/java/ai/vn97/platform/M6ExternalHandoff.kt" \
    "$ROOT/android/platform/src/main/java/ai/vn97/platform/M6ExecutionFabric.kt" \
    "$ROOT/android/platform/src/main/java/ai/vn97/platform/M6AndroidProductionCapabilities.kt" \
    "$HERE/M7PRuntimeStubs.kt" \
    "$HERE/M7RProductionCapabilitiesHostTest.kt" \
    -Werror \
    -include-runtime \
    -d "$WORK/m7r-production-capability-assembly-test.jar"

java -jar "$WORK/m7r-production-capability-assembly-test.jar"

"$KOTLINC" \
    "$ROOT"/android/runtime/src/main/java/ai/vn97/runtime/*.kt \
    "$ROOT/android/platform/src/main/java/ai/vn97/platform/M6Approval.kt" \
    "$ROOT/android/platform/src/main/java/ai/vn97/platform/M6ExternalHandoff.kt" \
    "$ROOT/android/platform/src/main/java/ai/vn97/platform/M6ExecutionFabric.kt" \
    "$ROOT/android/platform/src/main/java/ai/vn97/platform/M6EndToEndExternalCoordinator.kt" \
    "$HERE/M7SEndToEndExternalCoordinatorTest.kt" \
    -Werror \
    -include-runtime \
    -d "$WORK/m7s-end-to-end-external-coordinator-test.jar"

java -jar "$WORK/m7s-end-to-end-external-coordinator-test.jar"

"$KOTLINC" \
    "$ROOT"/android/runtime/src/main/java/ai/vn97/runtime/*.kt \
    "$ROOT/android/platform/src/main/java/ai/vn97/platform/M6Approval.kt" \
    "$ROOT/android/platform/src/main/java/ai/vn97/platform/M6ExternalHandoff.kt" \
    "$ROOT/android/platform/src/main/java/ai/vn97/platform/M6ExecutionFabric.kt" \
    "$ROOT/android/platform/src/main/java/ai/vn97/platform/M6EndToEndExternalCoordinator.kt" \
    "$ROOT/android/platform/src/main/java/ai/vn97/platform/VN97AssistantSession.kt" \
    "$HERE/M7TProductionAssistantSessionTest.kt" \
    -Werror \
    -include-runtime \
    -d "$WORK/m7t-production-assistant-session-test.jar"

java -jar "$WORK/m7t-production-assistant-session-test.jar"
