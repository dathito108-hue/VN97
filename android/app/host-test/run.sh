#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

KOTLINC="${KOTLINC:-kotlinc}"

"$KOTLINC" \
    "$ROOT/android/app/src/main/java/ai/vn97/app/VN97AppState.kt" \
    "$HERE/M10AAppStateTest.kt" \
    -Werror \
    -include-runtime \
    -d "$WORK/m10a-app-state-test.jar"

java -jar "$WORK/m10a-app-state-test.jar"


"$KOTLINC" \
    "$ROOT/android/runtime/src/main/java/ai/vn97/runtime/StrictJson.kt" \
    "$ROOT/android/runtime/src/main/java/ai/vn97/runtime/ActivatedInventoryEvidence.kt" \
    "$HERE/M10BStrictJsonStub.kt" \
    "$HERE/M10BInventoryEvidenceTest.kt" \
    -Werror \
    -include-runtime \
    -d "$WORK/m10b-inventory-evidence-test.jar"

java -jar "$WORK/m10b-inventory-evidence-test.jar"


"$KOTLINC" \
    "$ROOT/android/app/src/main/java/ai/vn97/app/VN97AppState.kt" \
    "$HERE/M10CApprovalStateTest.kt" \
    -Werror \
    -include-runtime \
    -d "$WORK/m10c-approval-state-test.jar"

java -jar "$WORK/m10c-approval-state-test.jar"


"$KOTLINC" \
    "$ROOT/android/app/src/main/java/ai/vn97/app/VN97BootstrapAssetContract.kt" \
    "$HERE/M10JBootstrapAssetContractTest.kt" \
    -Werror \
    -include-runtime \
    -d "$WORK/m10j-bootstrap-asset-contract-test.jar"

java -jar "$WORK/m10j-bootstrap-asset-contract-test.jar"


"$KOTLINC" \
    "$ROOT/android/app/src/main/java/ai/vn97/app/VN97AutonomousGoalStore.kt" \
    "$HERE/M13AAutonomousGoalStoreTest.kt" \
    -Werror \
    -include-runtime \
    -d "$WORK/m13a-autonomous-goal-store-test.jar"

java -jar "$WORK/m13a-autonomous-goal-store-test.jar"


"$KOTLINC" \
    "$ROOT/android/app/src/main/java/ai/vn97/app/VN97RevenueQualification.kt" \
    "$HERE/M20ARevenueEvidenceStub.kt" \
    "$HERE/M20ARevenueQualificationTest.kt" \
    -Werror \
    -include-runtime \
    -d "$WORK/m20a-revenue-qualification-test.jar"

java -jar "$WORK/m20a-revenue-qualification-test.jar"


"$KOTLINC" \
    "$ROOT/android/app/src/main/java/ai/vn97/app/VN97RevenueQualification.kt" \
    "$ROOT/android/app/src/main/java/ai/vn97/app/VN97RevenueCampaign.kt" \
    "$HERE/M20ARevenueEvidenceStub.kt" \
    "$HERE/M20BRevenueCampaignTest.kt" \
    -Werror \
    -include-runtime \
    -d "$WORK/m20b-revenue-campaign-test.jar"

java -jar "$WORK/m20b-revenue-campaign-test.jar"


"$KOTLINC" \
    "$ROOT/android/app/src/main/java/ai/vn97/app/VN97RevenueQualification.kt" \
    "$ROOT/android/app/src/main/java/ai/vn97/app/VN97RevenueCampaign.kt" \
    "$ROOT/android/app/src/main/java/ai/vn97/app/VN97RevenueLiveReadiness.kt" \
    "$HERE/M20ARevenueEvidenceStub.kt" \
    "$HERE/M20CLiveRevenueReadinessTest.kt" \
    -Werror \
    -include-runtime \
    -d "$WORK/m20c-live-revenue-readiness-test.jar"

java -jar "$WORK/m20c-live-revenue-readiness-test.jar"


"$KOTLINC" \
    "$ROOT/android/platform/src/main/java/ai/vn97/platform/VN97ExnessApiSigning.kt" \
    "$HERE/M20DExnessSigningTest.kt" \
    -Werror \
    -include-runtime \
    -d "$WORK/m20d-exness-signing-test.jar"

java -jar "$WORK/m20d-exness-signing-test.jar"


"$KOTLINC" \
    "$ROOT/android/platform/src/main/java/ai/vn97/platform/VN97ExnessCredentialPayload.kt" \
    "$HERE/M20EExnessCredentialPayloadTest.kt" \
    -Werror \
    -include-runtime \
    -d "$WORK/m20e-exness-credential-payload-test.jar"

java -jar "$WORK/m20e-exness-credential-payload-test.jar"


"$KOTLINC" \
    "$ROOT/android/platform/src/main/java/ai/vn97/platform/VN97ExnessPrivateKeyCodec.kt" \
    "$ROOT/android/platform/src/main/java/ai/vn97/platform/VN97ExnessEndpointPolicy.kt" \
    "$HERE/M20FExnessKeyEndpointTest.kt" \
    -Werror \
    -include-runtime \
    -d "$WORK/m20f-exness-key-endpoint-test.jar"

java -jar "$WORK/m20f-exness-key-endpoint-test.jar"
