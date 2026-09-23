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
    "$ROOT/native/src/language.cpp" \
    "$ROOT/native/src/selective.cpp" \
    "$ROOT/native/src/recurrent.cpp" \
    "$ROOT/native/src/packed_ternary.cpp" \
    "$ROOT/native/src/tokenizer.cpp" \
    "$ROOT/native/src/model_image.cpp" \
    "$ROOT/native/src/sampler.cpp" \
    "$ROOT/native/src/generation.cpp" \
    "$ROOT/native/src/memory.cpp" \
    "$ROOT/native/src/memory_crypto.cpp" \
    "$ROOT/native/src/memory_index.cpp" \
    "$ROOT/native/src/memory_store.cpp" \
    "$ROOT/android/runtime/src/main/cpp/vn97_jni.cpp" \
    "$ROOT/android/runtime/src/main/cpp/vn97_model_jni.cpp" \
    "$ROOT/android/runtime/src/main/cpp/vn97_generation_jni.cpp" \
    "$ROOT/android/runtime/src/main/cpp/vn97_memory_jni.cpp" \
    -pthread \
    -o "$WORK/libvn97_jni.so"

"$KOTLINC" \
    "$ROOT"/android/runtime/src/main/java/ai/vn97/runtime/*.kt \
    "$HERE/M7BIntegration.kt" \
    -Werror \
    -include-runtime \
    -d "$WORK/m7b-host-test.jar"

java -Djava.library.path="$WORK" -jar "$WORK/m7b-host-test.jar"

"$KOTLINC" \
    "$ROOT"/android/runtime/src/main/java/ai/vn97/runtime/*.kt \
    "$HERE/M7GGenerationTest.kt" \
    -Werror \
    -include-runtime \
    -d "$WORK/m7g-host-test.jar"

java -Djava.library.path="$WORK" -jar "$WORK/m7g-host-test.jar"

"$KOTLINC" \
    "$ROOT"/android/runtime/src/main/java/ai/vn97/runtime/*.kt \
    "$HERE/M7HChatTest.kt" \
    -Werror \
    -include-runtime \
    -d "$WORK/m7h-host-test.jar"

java -Djava.library.path="$WORK" -jar "$WORK/m7h-host-test.jar"

"$KOTLINC" \
    "$ROOT"/android/runtime/src/main/java/ai/vn97/runtime/*.kt \
    "$HERE/M7ICognitionTest.kt" \
    -Werror \
    -include-runtime \
    -d "$WORK/m7i-host-test.jar"

java -Djava.library.path="$WORK" -jar "$WORK/m7i-host-test.jar"

"$KOTLINC" \
    "$ROOT"/android/runtime/src/main/java/ai/vn97/runtime/*.kt \
    "$HERE/M7JTypedCognitionTest.kt" \
    -Werror \
    -include-runtime \
    -d "$WORK/m7j-host-test.jar"

java -Djava.library.path="$WORK" -jar "$WORK/m7j-host-test.jar"

"$KOTLINC" \
    "$ROOT"/android/runtime/src/main/java/ai/vn97/runtime/*.kt \
    "$HERE/M7KPlannerTest.kt" \
    -Werror \
    -include-runtime \
    -d "$WORK/m7k-host-test.jar"

java -Djava.library.path="$WORK" -jar "$WORK/m7k-host-test.jar"

"$KOTLINC" \
    "$ROOT"/android/runtime/src/main/java/ai/vn97/runtime/*.kt \
    "$HERE/M7LPlannerCheckpointTest.kt" \
    -Werror \
    -include-runtime \
    -d "$WORK/m7l-host-test.jar"

java -Djava.library.path="$WORK" -jar "$WORK/m7l-host-test.jar"

"$KOTLINC" \
    "$ROOT"/android/runtime/src/main/java/ai/vn97/runtime/*.kt \
    "$HERE/M7MCompositeContinuityTest.kt" \
    -Werror \
    -include-runtime \
    -d "$WORK/m7m-host-test.jar"

java -Djava.library.path="$WORK" -jar "$WORK/m7m-host-test.jar"

"$KOTLINC" \
    "$ROOT"/android/runtime/src/main/java/ai/vn97/runtime/*.kt \
    "$HERE/M7NCognitionLoopTest.kt" \
    -Werror \
    -include-runtime \
    -d "$WORK/m7n-host-test.jar"

java -Djava.library.path="$WORK" -jar "$WORK/m7n-host-test.jar"

"$KOTLINC" \
    "$ROOT"/android/runtime/src/main/java/ai/vn97/runtime/*.kt \
    "$HERE/M7UNativeMemoryBridgeHostTest.kt" \
    -Werror \
    -include-runtime \
    -d "$WORK/m7u-native-memory-bridge-test.jar"

java -Djava.library.path="$WORK" -jar "$WORK/m7u-native-memory-bridge-test.jar"


"$KOTLINC" \
    "$ROOT/android/runtime/src/main/java/ai/vn97/runtime/StrictJson.kt" \
    "$ROOT/android/runtime/src/main/java/ai/vn97/runtime/CapabilityPackage.kt" \
    "$ROOT/android/runtime/src/main/java/ai/vn97/runtime/CapabilitySignature.kt" \
    "$ROOT/android/runtime/src/main/java/ai/vn97/runtime/CapabilityStage.kt" \
    "$HERE/M10DStrictJsonStub.kt" \
    "$HERE/M10DCapabilityStagingTest.kt" \
    -Werror \
    -include-runtime \
    -d "$WORK/m10d-capability-staging-test.jar"

java -jar "$WORK/m10d-capability-staging-test.jar"


"$KOTLINC" \
    "$ROOT/android/runtime/src/main/java/ai/vn97/runtime/StrictJson.kt" \
    "$ROOT/android/runtime/src/main/java/ai/vn97/runtime/CapabilityPackage.kt" \
    "$ROOT/android/runtime/src/main/java/ai/vn97/runtime/CapabilitySignature.kt" \
    "$ROOT/android/runtime/src/main/java/ai/vn97/runtime/CapabilityStage.kt" \
    "$ROOT/android/runtime/src/main/java/ai/vn97/runtime/CapabilityTrust.kt" \
    "$HERE/M10DStrictJsonStub.kt" \
    "$HERE/M10EPublisherTrustTest.kt" \
    -Werror \
    -include-runtime \
    -d "$WORK/m10e-publisher-trust-test.jar"

java -jar "$WORK/m10e-publisher-trust-test.jar"
