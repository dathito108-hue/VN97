# VN97 M2A parallel affine scan

The selective SSM recurrence has the form:

    h_t = a_t * h_(t-1) + b_t

where the multiplication is elementwise over the recurrent state. Each token therefore
defines an affine state transform (a_t, b_t). Two transforms compose associatively:

    (a2, b2) o (a1, b1) = (a2*a1, b2 + a2*b1)

That associativity lets VN97 compute all sequence prefixes with a parallel scan rather than a
Python loop over every token.

## M2A contract

- B, C and timestep projections are computed for the whole sequence in tensor form.
- exact diagonal ZOH discretization from M0 is unchanged.
- the full-sequence path uses iterative-doubling affine prefix composition.
- a 4096-token sequence needs 12 Python composition rounds instead of 4096 token iterations.
- token-by-token inference remains recurrent and constant-state.
- forward_sequential_reference() remains available for numerical equivalence tests.

## Training/reference versus deployment

M2A remains the differentiable PyTorch/reference path. Its iterative-doubling implementation
performs O(L log L) tensor work and materializes sequence state, which is acceptable as the
semantic oracle but is not the mobile deployment kernel.

M2B provides native in-place recurrence from already discretized decay/drive. M2C moves the
token-dependent softplus/clamp and exact ZOH discretization into native execution and consumes
compact signal/dt/B/C/gate/A inputs directly. The deployment path therefore does not require
materializing the expanded [B,L,D,N] decay and drive tensors.

## Canonical invariant

For the same parameters, input and initial recurrent state, all three paths must match within
floating-point tolerance:

1. M2A associative scan;
2. sequential recurrent reference;
3. native M2B/M2C prefill and repeated token-step execution.

No optimized backend may weaken this invariant.
