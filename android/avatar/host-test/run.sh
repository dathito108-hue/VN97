#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

KOTLINC="${KOTLINC:-kotlinc}"

"$KOTLINC" \
    "$ROOT/android/avatar/src/main/java/ai/vn97/avatar/AvatarState.kt" \
    "$ROOT/android/avatar/src/main/java/ai/vn97/avatar/AvatarInteraction.kt" \
    "$ROOT/android/avatar/src/main/java/ai/vn97/avatar/AvatarFramePolicy.kt" \
    "$HERE/M8AStateTest.kt" \
    -Werror \
    -include-runtime \
    -d "$WORK/m8a-avatar-state.jar"

java -jar "$WORK/m8a-avatar-state.jar"
