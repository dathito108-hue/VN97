# VN97 R2-K2F — dBx operand isolation

K2E falsified the simple "generic Einsum contraction order" repair:
the measured dBx error remained exactly 9 FP16 epsilon units before and after
the ordered-broadcast rewrite.

That means the next step must separate **operator error** from **operand
perturbation amplification**. K2F also restores the canonical G0.4 dBx
expression; the ineffective K2E lowering is not kept in the production path.

## What K2F measures

K2F uses the exact K2D layer, token, hidden state and recurrent state.

First it exports only the prefix that produces the three dBx operands:

    dt_value
    b_value
    x_heads

The same hidden/residual/conv inputs are run through PyTorch CPU and ORT CPU,
so each operand has a direct parity measurement.

Then K2F computes counterfactual sensitivity in PyTorch:

    ORT dt + PyTorch B + PyTorch x
    PyTorch dt + ORT B + PyTorch x
    PyTorch dt + PyTorch B + ORT x
    all ORT operands through PyTorch Einsum

Finally, four tiny ONNX graphs receive the **exact PyTorch operands** directly:

    Einsum
    (dt*x)*B
    (B*x)*dt
    FP32 widened multiply -> FP16

This removes all preceding layer operations from the operator test.

## Interpretation

If the isolated ORT Einsum is itself above the two-epsilon diagnostic
reference, the operator/runtime lowering remains the primary target.

If the isolated operator is tight but "all ORT operands through PyTorch
Einsum" exceeds the reference, the nine-epsilon K2D jump is amplification of
small upstream operand drift. The next step should trace the operand-producing
prefix rather than changing dBx algebra.

The two-epsilon value is still diagnostic only. K2F cannot authorize
production or define the final K2 acceptance threshold.

## Run

    %cd /kaggle/working/VN97
    !git pull --ff-only
    !bash tools/kaggle_r2_k2f_dbx_operands.sh

Expected outputs:

    /kaggle/working/vn97-k2f-dbx-operands.json
    /kaggle/working/VN97-R2-K2F-dBx-operands.zip
