# VN97 R2-K2D — same-input ONNX CPU stage trace

K2C separated four backend paths and showed two independent numerical
envelopes.

For the first real one-token probe:

- PyTorch CUDA -> CPU first crossed the two-epsilon diagnostic reference at
  layer 2;
- PyTorch CPU -> ORT CPU first crossed it at layer 1;
- ORT CPU -> CUDA crossed it already at layer 0.

K2D targets only the ONNX-CPU question. It does not alter the inherited
weights, model geometry, graph semantics or numerical threshold.

## Same-input isolation

K2D reads the K2C receipt and automatically selects the first
PyTorch-CPU-vs-ORT-CPU layer over the diagnostic reference. For the current
evidence that is layer 1.

The preceding layers are executed in PyTorch CPU only. The exact resulting
hidden/residual input and exact recurrent state are then fed to both:

1. the original G0.4 PyTorch layer; and
2. an ONNX trace graph containing that same layer and the same weights.

The trace exports ordered intermediate stages:

    residual_next
    block_norm
    in_proj
    next_conv
    conv_affine
    activated_xbc
    dt_value
    d_a
    d_b_x
    next_ssm
    y_pre_gate
    gated_norm
    output

This identifies the first operation where ORT CPU diverges from PyTorch CPU
without contamination from earlier trajectory drift.

The existing two-FP16-epsilon value remains a diagnostic reference only.
K2D remains MEASURED and cannot authorize production.

## Run

In the existing Kaggle session:

    %cd /kaggle/working/VN97
    !git pull --ff-only
    !bash tools/kaggle_r2_k2d_stage_trace.sh

Expected outputs:

    /kaggle/working/vn97-k2d-stage-trace.json
    /kaggle/working/VN97-R2-K2D-stage-trace.zip

The key line is:

    K2D first_stage_over_reference: ...

If early normalization/projection stages are tight and the first crossing is
at d_b_x/next_ssm, the next repair should target recurrent mixed-precision
lowering. If the crossing appears at in_proj or conv_affine, the issue is an
ORT CPU FP16 GEMM/reduction envelope rather than recurrent-state semantics.
