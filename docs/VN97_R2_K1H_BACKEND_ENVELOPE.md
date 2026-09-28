# VN97 R2-K1H — official-backend numerical envelope

K1G established that the layer-2 operator semantics are effectively aligned
when official Mamba-2 fallback and VN97 receive the same input and recurrent
state:

- block RMSNorm: 0
- in-projection: 0
- conv state: 0
- SSD state: 0
- gated RMSNorm: about 1.19e-7
- full mixer output: about 1.91e-6
- out projection: 0

The remaining full-model discrepancy is therefore trajectory drift accumulated
through 64 FP16 layers, not a one-to-one tensor transfer or operator-formula
mismatch.

K1H calibrates that drift against the official implementation itself.

## Method

For every probe token K1H runs three trajectories against the same inherited
Mamba-2 2.7B weights:

1. official Mamba-2 optimized CUDA kernels;
2. official Mamba-2 unfused fallback kernels;
3. VN97 recurrent reference.

All three receive identical input tokens.

The official optimized-vs-fallback delta is treated as the empirical backend
numerical envelope for FP16 execution.

VN97 is then compared against the official fallback path.

For each metric:

    allowed_limit =
        official_optimized_vs_fallback_max_abs_error
        + fp16_epsilon * max(1, observed_fallback_scale)

This adds only one FP16 machine-epsilon unit at the observed tensor scale as
rounding/order slack; no manually enlarged parity threshold is introduced.

Metrics:

- logits
- layer-0 hidden state
- layer-0 convolution state
- layer-0 SSD state

K1H also requires the greedy argmax token from optimized official, fallback
official and VN97 to be identical at every continuation step.

## Run on Kaggle T4

On the existing session:

    %cd /kaggle/working/VN97
    !git pull
    !bash tools/kaggle_r2_k1h_backend_envelope.sh

The existing 5.4 GB checkpoint is reused.

Output:

    /kaggle/working/vn97-k1h-backend-envelope.json

The receipt schema is VN97M2K1HENVELOPE1.

## Interpretation

PASS means the VN97-vs-official fallback drift is no worse than the official
optimized-vs-fallback backend variability plus one FP16 epsilon at the observed
scale, while greedy token decisions remain identical.

FAIL means VN97 still exceeds the official backend envelope and further
operator/trajectory investigation is required.

K1H does not prove ONNX Runtime parity. K2 remains required before production
promotion.
