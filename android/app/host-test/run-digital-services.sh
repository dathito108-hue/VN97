#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
"${KOTLINC:-kotlinc}" \
    "$ROOT/android/platform/src/main/java/ai/vn97/platform/VN97DigitalServiceCore.kt" \
    "$ROOT/android/app/src/main/java/ai/vn97/app/VN97DigitalServices.kt" \
    "$HERE/DigitalServicesTest.kt" \
    -Werror -include-runtime -d "$WORK/digital-services.jar"
java -jar "$WORK/digital-services.jar"
