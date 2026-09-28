# VN97 R2-K2E — dBx ordered-broadcast lowering repair

K2D isolated the first PyTorch-CPU to ORT-CPU divergence at layer 1 to:

    d_b_x = einsum("bh,bn,bhp->bhpn", dt, B, x)

The stages before dBx were within roughly one FP16 epsilon, while dBx and
next_ssm jumped to nine epsilon units.

This equation contains no reduction. It is only an elementwise product with
broadcasting. Allowing ONNX Runtime to execute it as a generic Einsum gives
the backend freedom to choose a different contraction/multiplication order.

K2E replaces only this lowering with the explicit activation-dtype order:

    (dt_value[:, :, None, None] * x_heads[:, :, :, None])
        * b_value[:, None, None, :]

For FP16 CPU tensors this is the same left-associated operation order observed
from PyTorch's source-side Einsum for the no-reduction equation. No weight,
model geometry, recurrent equation, tokenizer or authority policy changes.

## Measurement boundary

K2E does not re-export the full 2.7B production graph yet. It first reruns the
same layer-1 same-input trace used by K2D with the repaired lowering.

This is deliberately cheaper and falsifiable:

- if dBx drops back into the existing two-epsilon diagnostic envelope, K2F can
  re-export the full G0.4 graph and repeat the backend matrix;
- if it does not, the hypothesis is rejected before spending time and disk on
  a new 2.7B ONNX export.

The two-epsilon value remains diagnostic only. K2E does not define a production
acceptance threshold and cannot authorize activation.

## Run

In the existing Kaggle session:

    %cd /kaggle/working/VN97
    !git pull --ff-only
    !bash tools/kaggle_r2_k2e_dbx_repair.sh

Expected outputs:

    /kaggle/working/vn97-k2e-dbx-repair.json
    /kaggle/working/vn97-k2e-post-trace.json
    /kaggle/working/VN97-R2-K2E-dBx-repair.zip
