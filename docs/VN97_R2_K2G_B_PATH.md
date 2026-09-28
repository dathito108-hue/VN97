# VN97 R2-K2G — B-value path counterfactual trace

K2F proved that the isolated dBx operator is not the main problem:

- exact-operand ORT micrographs stayed at about one FP16 epsilon;
- all ORT operands propagated through PyTorch dBx reached nine epsilon;
- replacing only B with the ORT B operand reached about 9.5 epsilon;
- replacing only dt or x stayed below two epsilon.

K2G therefore traces only the B-producing path.

## Trace boundaries

Using the exact same K2F layer, token, hidden state and zero-start recurrent
state, K2G exposes:

    normalized
    xbc_projected
    next_conv
    conv_affine
    activated_xbc
    b_value

Each boundary is measured directly between PyTorch CPU and ORT CPU.

Then, for every boundary, the ORT value is injected back into the remaining
B path executed entirely by PyTorch. The resulting B is inserted into dBx
while dt and x remain the exact PyTorch reference operands.

This answers a stricter question than raw stage error:

> At which first boundary does the tiny backend perturbation become capable of
> producing more than two FP16 epsilon units at dBx?

Possible diagnoses include:

    in_proj_xbc_rounding_amplified
    conv_state_assembly_rounding_amplified
    conv_affine_rounding_amplified
    silu_rounding_amplified

No repair is applied in K2G. The production G0.4 expression remains canonical,
weights and architecture stay unchanged, and the two-epsilon level remains a
diagnostic reference only.

## Run

    %cd /kaggle/working/VN97
    !git pull --ff-only
    !bash tools/kaggle_r2_k2g_b_path.sh

Expected evidence:

    /kaggle/working/vn97-k2g-b-path.json
    /kaggle/working/VN97-R2-K2G-B-path.zip
