# VN97 R2-K1G — layer-2 same-input stage trace

K1F identified layer 2 as the first layer crossing the locked 0.005 parity
threshold on the first token. K1G narrows the diagnosis to operator stages.

The trace deliberately feeds the **same official normalized input and the same
initial recurrent state** into the official Mamba-2 fallback mixer and the
VN97 reference mixer. This prevents accumulated trajectory drift from earlier
layers from contaminating the comparison.

K1G reports maximum absolute error for:

- block RMSNorm on the same residual input;
- in-projection on the same normalized input;
- mixer output on the same normalized input and recurrent state;
- convolution state;
- SSD state;
- gated RMSNorm on the same captured x/z inputs;
- output projection on the same captured input.

Default trace layer is 2 because the real K1F run reported
`first_layer_over_threshold=2`.

Run on the existing Kaggle T4 session:

    %cd /kaggle/working/VN97
    !git pull
    !bash tools/kaggle_r2_k1g_stage_trace.sh

To trace another layer:

    %env VN97_K1G_TRACE_LAYER=3
    !bash tools/kaggle_r2_k1g_stage_trace.sh

The output is written to:

    /kaggle/working/vn97-k1g-stage-trace.json

Interpretation:

- block norm error dominates while same-input mixer/state errors stay small:
  reduction/kernel numerical drift begins before the mixer;
- conv-state error dominates: causal-convolution fallback semantics differ;
- SSD-state error dominates with matching conv state: discretization/recurrent
  update precision still differs;
- gated-norm error dominates with matching state: normalization/gating kernel
  numerics are the first substantive divergence;
- out-proj error with matching input indicates GEMM/backend precision drift.

No threshold is widened and no production evidence is promoted by K1G.
