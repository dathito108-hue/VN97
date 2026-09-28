# VN97 R2-K2L — layerwise runtime/state influence scan

K2K proved that the step-29 rank flip is an interaction effect:

- ORT runtime + full ORT conv + full ORT SSM selects token 322;
- every other runtime/state combination selects the PyTorch reference token
  1583.

The ORT full-state case is also an extremely narrow boundary: the two
candidate logits are tied at the observed FP16 output precision.

K2L therefore localizes **which recurrent layers make ORT sit on that rank
boundary**.

## Efficient two-stage scan

The full model has 64 recurrent layers. K2L keeps the ORT runtime and the
exact same current token from K2K.

Starting from the divergent full ORT state, it performs state repair swaps:

    ORT state -> PyTorch state

for three modes:

    conv only
    SSM only
    conv + SSM

Stage 1 scans eight groups of eight layers each. The three groups with the
largest positive reference-vs-ORT logit influence are expanded.

Stage 2 scans every individual layer inside those selected groups, again for
conv, SSM, and combined state.

For every intervention K2L records:

- top-1 token and whether it is the reference/ORT candidate;
- reference-token minus ORT-token logit gap;
- change in that gap relative to the divergent baseline;
- top-5 IDs.

This uses the actual behavioral boundary as the objective rather than raw
state epsilon magnitude.

## Possible outcomes

- single_layer_state_interaction_localized:
  replacing one layer's combined state repairs the rank flip.
- component_layer_interaction_localized:
  a single layer's conv or SSM component is enough to repair the flip.
- multi_layer_state_interaction_localized:
  a group repair works but no scanned individual layer is sufficient.
- distributed_state_runtime_interaction:
  no localized intervention repairs the flip.

K2L is diagnostic only. It changes no weights, architecture, production
graph, acceptance threshold, or activation policy.

## Run

    %cd /kaggle/working/VN97
    !git pull --ff-only
    !bash tools/kaggle_r2_k2l_layer_influence.sh

Expected evidence:

    /kaggle/working/vn97-k2l-layer-influence.json
    /kaggle/working/VN97-R2-K2L-layer-influence.zip
