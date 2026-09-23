#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

KOTLINC="${KOTLINC:-kotlinc}"

"$KOTLINC" \
    "$HERE/M14CGameMemoryRuntimeStubs.kt" \
    "$ROOT/android/platform/src/main/java/ai/vn97/platform/VN97GameEpisodeMemory.kt" \
    "$HERE/M14CGameEpisodeMemoryTest.kt" \
    -Werror \
    -include-runtime \
    -d "$WORK/m14c-game-episode-memory-test.jar"

java -cp "$WORK/m14c-game-episode-memory-test.jar" \
    ai.vn97.platform.M14CGameEpisodeMemoryTestKt
