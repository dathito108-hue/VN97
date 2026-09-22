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

"$KOTLINC" \
    "$ROOT/android/avatar/src/main/java/ai/vn97/avatar/AvatarState.kt" \
    "$ROOT/android/avatar/src/main/java/ai/vn97/avatar/SpeechSync.kt" \
    "$HERE/M8BSpeechSyncTest.kt" \
    -Werror \
    -include-runtime \
    -d "$WORK/m8b-speech-sync.jar"

java -jar "$WORK/m8b-speech-sync.jar"

"$KOTLINC" \
    "$ROOT/android/avatar/src/main/java/ai/vn97/avatar/AvatarState.kt" \
    "$ROOT/android/avatar/src/main/java/ai/vn97/avatar/RigPose.kt" \
    "$HERE/M8CRigPoseTest.kt" \
    -Werror \
    -include-runtime \
    -d "$WORK/m8c-rig-pose.jar"

java -jar "$WORK/m8c-rig-pose.jar"

JAVA_HOME="${JAVA_HOME:-$(dirname "$(dirname "$(readlink -f "$(command -v javac)")")")}"
CXX="${CXX:-g++}"

"$CXX" -std=c++17 -Wall -Wextra -Werror -pedantic -fPIC -shared \
    -I"$ROOT/native/include" \
    -I"$JAVA_HOME/include" \
    -I"$JAVA_HOME/include/linux" \
    "$ROOT/native/src/avatar_asset.cpp" \
    "$ROOT/android/avatar/src/main/cpp/vn97_avatar_jni.cpp" \
    -o "$WORK/libvn97_avatar_jni.so"

"$KOTLINC" \
    "$ROOT/android/avatar/src/main/java/ai/vn97/avatar/NativeAvatarAsset.kt" \
    "$HERE/M8CNativeAssetHostTest.kt" \
    -Werror \
    -include-runtime \
    -d "$WORK/m8c-native-asset.jar"

java -Djava.library.path="$WORK" -jar "$WORK/m8c-native-asset.jar"
