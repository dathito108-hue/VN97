# VN97 R2-K2H — projection and conv-affine micro isolation

K2G identified the first sensitive B-path boundary as xbc_projected and
diagnosed in_proj_xbc_rounding_amplified. The same trace also showed a second
amplification step at conv_affine, where downstream dBx rose from roughly
3.06 to 9.5 FP16 epsilon units.

K2H isolates both arithmetic boundaries before changing the production graph.

## In-proj micrographs

Each graph receives the exact PyTorch normalized activation and the inherited
layer-1 in_proj weight. It emits xbc_projected and tests:

    fp16_linear
    fp16_matmul
    fp32_widened -> cast FP16

The ORT result is injected into the remaining B path executed in PyTorch, then
B and dBx are measured.

## Conv-affine micrographs

Each graph receives the exact PyTorch next_conv state and inherited conv
weight/bias. It tests:

    fp16_reduce
    fp32_widened -> cast FP16

Again, the ORT result is fed through PyTorch SiLU/split and finally into dBx
with exact PyTorch dt and x operands.

A candidate is merely diagnostic if its downstream dBx stays within the
existing two-FP16-epsilon reference. K2H does not mutate G0.4, define a final
acceptance threshold, or authorize production.

## Run

    %cd /kaggle/working/VN97
    !git pull --ff-only
    !bash tools/kaggle_r2_k2h_projection_conv.sh

Expected evidence:

    /kaggle/working/vn97-k2h-micro-isolation.json
    /kaggle/working/VN97-R2-K2H-micro-isolation.zip
