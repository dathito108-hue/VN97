#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

JAVA_HOME="${JAVA_HOME:-$(dirname "$(dirname "$(readlink -f "$(command -v javac)")")")}" 
CXX="${CXX:-g++}"
KOTLINC="${KOTLINC:-kotlinc}"

"$CXX" -std=c++17 -Wall -Wextra -Werror -pedantic -fPIC -shared \
    -I"$ROOT/native/include" \
    -I"$JAVA_HOME/include" \
    -I"$JAVA_HOME/include/linux" \
    "$ROOT/native/src/runtime.cpp" \
    "$ROOT/native/src/recurrent.cpp" \
    "$ROOT/native/src/packed_ternary.cpp" \
    "$ROOT/android/runtime/src/main/cpp/vn97_jni.cpp" \
    -pthread \
    -o "$WORK/libvn97_jni.so"

"$KOTLINC" \
    "$ROOT"/android/runtime/src/main/java/ai/vn97/runtime/*.kt \
    "$HERE/M7BIntegration.kt" \
    -Werror \
    -include-runtime \
    -d "$WORK/m7b-host-test.jar"

java -Djava.library.path="$WORK" -jar "$WORK/m7b-host-test.jar"
