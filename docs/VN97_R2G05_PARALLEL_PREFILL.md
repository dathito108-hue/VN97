# VN97 G06 lineage — Single-graph chunk execution

G0.5 introduced chunk execution for the Mamba-2-derived VN97 core. After
PR323/324, G06 retains batched projections and convolution, with a token-ordered
SSM update that preserves the canonical FP16 rounding boundaries.

The central constraint is simple:

`parallel prefill final state/logits ~= repeated exact G0.4 steps`

No second model and no alternate intelligence backend is introduced.

## One graph, one copy of 2.7B weights

Exporting separate step and chunk graphs would duplicate the external ONNX
weights. For a 2.7B model that can add roughly another full multi-gigabyte copy.

G0.5 therefore uses one fixed-maximum recurrent graph:

- `input_ids`: `[1, max_chunk]`;
- `valid_length`: `[1]` in the range `1..max_chunk`;
- explicit convolution state;
- explicit SSD state.

The same graph serves both modes:

- `valid_length=1` -> recurrent decode;
- `valid_length=max_chunk` -> batched prefill with ordered state updates;
- intermediate values -> prompt tail/remainder.

Only one ONNX graph owns the inherited weight initializers/external data.

## Current precision-preserving execution

Each layer uses the following operations in the same graph:

1. project token positions together;
2. evaluate causal convolution windows in parallel with activation-dtype
   multiplication, reduction and bias addition, matching the step kernel;
3. compute softplus dt;
4. update the SSM in token order using the shared step operation: FP32 decay,
   activation-dtype dBx, then rounding of each updated state;
5. read each token output from that rounded state;
6. add D skip, gated RMSNorm and the batched output projection.

The manifest reports
`parallel_projection_conv_token_rounded_state_scan`.
This is one model and one graph, not separate fast/slow intelligence branches.
The convolution state is selected after exactly `valid_length` updates.
The SSM explicitly retains the previous state and zeroes its readout for invalid
suffixes. The fixed graph still evaluates candidate updates at padded positions;
skipping that computation remains a decode performance task.

Before PR323/324, convolution used fused Conv and the SSM used one-chunk SSD
factorization. That formulation rounded state only at chunk end, which differed
from FP16 token recurrence. The old exported candidate must not be relabeled
as the repaired graph; re-export changes its identity and requires new checks.

## Measured scope of the repair

Run `36508779240` on head `05beba0a9f058e0a51c5363cd03a487b842ca59e` compared
verified original 2.7B weights against frozen sequential PyTorch references.
All logits, convolution states and SSM states matched exactly for reset length 1,
reset length 8 and continuation length 1. Eighteen small-fixture tests also passed,
including ONNX FP16 continuation at the unchanged 0.002 absolute-error gate.

These are finite test cases, not a proof of all-input equivalence. Full-weight
ONNX backend parity and physical Android qualification remain separate gates.
Single hosted-CPU timings do not establish mobile speed, and no production
activation is authorized by these results.

## Decode is the same graph

For `valid_length=1`, the graph must match one G0.4 step:

- first-token logits match;
- suffix logits are zero/ignored;
- final conv state matches;
- final SSD state matches.

This removes the need for separate step weights.

## Supported maximum chunks

The initial contract supports maximum chunks:

- 8;
- 16;
- 32.

The production default is 32 until physical S21 FE profiling identifies a
better RAM/latency point.

## Numerical parity

G0.5 has two parity layers:

1. native parallel factorization vs repeated G0.4 steps;
2. ONNX Runtime parallel graph vs repeated G0.4 steps.

Tests cover zero and non-zero incoming recurrent state, `valid_length=1`, prompt
tails, and full chunks.

For floating point, equality is tolerance-based because the parallel
factorization changes accumulation order. The inherited tensors themselves are
unchanged.

## Bundle contract

`VN97M2G05ONNX1` records:

- exact G0.3 capsule identity;
- exact source-weight identity;
- maximum chunk and valid-length bounds;
- explicit recurrent-state contract;
- graph/external-data hashes;
- `single_weight_graph=true`;
- `decode_via_valid_length_one=true`;
- `parallel_prefill_ready=true`;
- `same_weights_semantics=true`;
- `quantization_used=false`;
- `production_activation_authorized=false`.

## Remaining production gates

G0.5 does not claim mobile production readiness. The remaining path is:

real G0.3 capsule -> real source/VN97 parity -> real G0.5 ONNX export/parity ->
Android runtime integration -> physical S21 FE provider/RAM/latency/thermal
qualification -> only then consider quantization and release packaging.
