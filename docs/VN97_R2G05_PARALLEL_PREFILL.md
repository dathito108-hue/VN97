# VN97 R2-G0.5 — Single-graph parallel SSD prefill

R2-G0.5 adds the first parallel prompt-processing path for the Mamba-2-derived
VN97 core while preserving the same weights and recurrent-state semantics that
were locked in G0.4.

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
- `valid_length=max_chunk` -> full parallel prefill;
- intermediate values -> prompt tail/remainder.

Only one ONNX graph owns the inherited weight initializers/external data.

## Parallel SSD factorization

Inside each Mamba-2 layer G0.5 computes the whole valid prefix using the
one-chunk SSD factorization from the Mamba-2 formulation:

1. project all token positions;
2. compute the depthwise causal convolution in one grouped Conv operation;
3. apply softplus dt and diagonal A discretization;
4. build the lower-triangular segment transition matrix;
5. compute intra-chunk B/X -> C output contributions in parallel;
6. compute the initial recurrent-state contribution to all positions;
7. compute the exact chunk-boundary final SSD state;
8. add D skip, gated RMSNorm and output projection.

The convolution final state is selected after exactly `valid_length` updates.
Invalid suffix positions are masked and receive `dt=0`, so they do not advance
the recurrent SSD state.

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
