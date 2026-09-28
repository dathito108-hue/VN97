# VN97 R2-K2Q — Android FP16 physical-device evidence

K2P proved that the real G0.5 recurrent-8 graph preserves the inherited
Mamba-2 2.7B decision behavior on CUDA: all 224 same-input decisions matched
exactly, including parallel-prefill continuations.

K2Q moves that exact graph lineage onto Android without changing weights or
graph arithmetic.

## Why K2Q exists

The verified K2P G0.5 artifact carries FP16 explicit recurrent state. The
older Android G0.6 profiler assumed FloatBuffer/float32 tensors. K2Q removes
that host/runtime mismatch by using ONNX Runtime Java's FP16 tensor path with
ShortBuffer-backed tensors.

The graph itself is not converted or requantized.

## Kaggle preparation

Run once while the existing K2P files are still available:

    %cd /kaggle/working/VN97
    !git pull --ff-only
    !bash tools/kaggle_r2_k2q_prepare_android_runtime.sh

This creates:

    /kaggle/working/vn97-r2-k2/g05-recurrent-8/runtime.vn97m2g06.json
    /kaggle/working/vn97-k2q-android-transfer.json
    /kaggle/working/VN97-R2-K2Q-android-metadata.zip

The script also prints every graph payload that must be downloaded to the
phone. Graph payloads are intentionally not duplicated into another multi-GB
archive.

## Android developer APK flow

The K2Q debug APK exposes:

    VN97 -> R2 Android ORT evidence

1. Download the printed G0.6 descriptor and every recurrent-8 graph payload
   to the phone.
2. Open the K2Q activity.
3. Tap **Import verified G0.6 runtime files** and multi-select the descriptor
   plus every graph payload.
4. The app copies them into app-private storage and verifies byte length,
   SHA-256, G0.5 lineage, geometry, and runtime identity.
5. Tap **Run Android ORT profile**.
6. The profiler attempts available Android providers and always requires a
   direct CPU baseline.
7. Tap **Copy evidence JSON** and return the JSON receipt.

The output schema remains the existing device-measured G0.7 profile receipt
and binds the exact G0.6 runtime ID.

## Evidence measured on the phone

For each available provider and valid length supported by recurrent-8, the
profile records:

- actual provider and fallback use;
- session creation time;
- p50/p95 execution latency;
- available memory before/after;
- thermal status before/after.

The activity verifies finite logits after execution.

## Honest boundary

K2Q repository CI can prove that the Android APK compiles and that the FP16
buffer path uses ONNX Runtime's typed FLOAT16 API. CI cannot claim a real-phone
PASS. That exists only after the profile is run on the physical device.

K2Q does not authorize production and does not modify the inherited weights.
