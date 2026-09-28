# VN97 R2-K2B — ONNX provider-split diagnosis

K2A successfully measured the real T4 boundary:

    VN97 native PyTorch -> ONNX Runtime CUDA

The observed K2A continuation kept exact greedy argmax on all 9 measured
steps, but the numerical envelope was materially wider than the K1I
same-input semantic envelope. K2B therefore does not manufacture a passing
threshold from one observation.

K2B decomposes the boundary on the exact same G0.4 graph, capsule, tokenizer,
weights and prompts:

    PyTorch -> ORT CPU
    ORT CPU -> ORT CUDA
    PyTorch -> ORT CUDA

This separates graph/export behavior from CUDA Execution Provider behavior.

The K1I two-FP16-epsilon semantic envelope is recorded only as a diagnostic
reference. K2B does not authorize production and does not define the eventual
K2 acceptance threshold.

## Prerequisites

K2B requires the successful K2A artifacts already present in the same Kaggle
session:

    /kaggle/working/vn97-k2a-ort-parity.json
    /kaggle/working/vn97-r2-k2/g03-capsule
    /kaggle/working/vn97-r2-k2/g04-step-onnx

It requires ONNX Runtime GPU 1.26.0 on the current CUDA-12 Kaggle T4 runtime.

## Run

    %cd /kaggle/working/VN97
    !git pull --ff-only
    !bash tools/kaggle_r2_k2b_provider_split.sh

Expected evidence:

    /kaggle/working/vn97-k2b-provider-split.json
    /kaggle/working/VN97-R2-K2B-provider-split.zip

Interpretation:

- PyTorch->CPU tight, CPU->CUDA wide: isolate drift to CUDA EP.
- PyTorch->CPU wide: inspect ONNX lowering/export before setting a gate.
- Both tight: K2C can define a fail-closed acceptance gate from repeatable
  evidence.
