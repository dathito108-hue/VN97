# VN97 M2A parallel affine scan

The selective SSM recurrence has the form:

    h_t = a_t * h_(t-1) + b_t

where the multiplication is elementwise over the recurrent state. Each token therefore
defines an affine state transform `(a_t, b_t)`. Two transforms compose associatively:

    (a2, b2) o (a1, b1) = (a2*a1, b2 + a2*b1)

That associativity lets VN97 compute all sequence prefixes with a parallel scan rather than a
Python loop over every token.

## M2A contract

- B, C and timestep projections are computed for the whole sequence in tensor form.
- exact diagonal ZOH discretization from M0 is unchanged.
- the full-sequence path uses iterative-doubling affine prefix composition.
- a 4096-token sequence needs 12 Python composition rounds instead of 4096 token iterations.
- token-by-token inference remains recurrent and constant-state.
- `forward_sequential_reference()` remains available for numerical equivalence tests.

## Important limitation

M2A reduces Python control-flow depth, but the iterative-doubling reference performs O(L log L)
tensor work and materializes sequence state. It is not the final mobile kernel. M2B must fuse
the scan/recurrent path in native code (or another backend primitive) so training/prefill can
retain parallelism without unnecessary intermediate allocations.

## Canonical invariant

For the same parameters, input and initial recurrent state, the parallel full-sequence path
must match the sequential recurrence within floating-point tolerance. Existing model tests also
require full-sequence logits/state to match token-by-token execution.
