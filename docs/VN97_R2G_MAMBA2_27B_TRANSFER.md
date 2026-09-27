# VN97 G0/G1 — Mamba-2 2.7B direct intelligence transfer

This milestone replaces the old lossy P5B transplant idea for the new canonical
large-SSM path. The source is `state-spaces/mamba2-2.7b`, but Mamba is not a
runtime backend inside the Android application.

## Locked source contract

The G0 converter accepts only the official Mamba-2 2.7B architecture:

- `d_model = 2560`
- `n_layer = 64`
- `vocab_size = 50277`, padded to `50288`
- `d_intermediate = 0`
- Mamba2 selective SSD blocks only; no attention layers
- `d_state = 128`
- `d_conv = 4`
- `expand = 2`
- `headdim = 64`
- `ngroups = 1`
- `d_inner = 5120`
- `nheads = 80`
- tied embedding / LM head
- RMSNorm, residual-in-fp32, fused add+norm semantics

The resulting unique preserved core contains 2,702,599,680 parameters when the
tied LM head is counted once. This number is a shape-contract calculation, not
a claim that the repository currently contains or has trained those weights.

## G0 invariant: intelligence first

G0 is a tensor-value identity transfer:

    Mamba-2 2.7B checkpoint
              |
              | exact namespace conversion only
              v
    VN97 G0 checkpoint

Forbidden during G0:

- channel selection;
- layer compression;
- tokenizer conversion;
- embedding factorization;
- blending with another checkpoint;
- quantization / ternary snapping;
- random reinitialization of inherited tensors.

The converter fails closed if the source config, tensor names or tensor shapes
do not match the locked source contract.

Before G0 can become a production model, a later parity milestone must compare
the official source implementation against the VN97-native SSD implementation
on the same tokens, initial states and precision and verify hidden/state/logit
parity. Creating a G0 package alone does not prove that parity.

## VN97 additions

Every preserved SSM layer receives a parameter-efficient VN97 side path:

    native Mamba-2 SSD output
           |
           +--> Fast shadow state
           +--> Working shadow state
           +--> Slow shadow state
           |
           +--> State Highway
           |
           +--> optional VN97MEM1 retrieved vector
           |
           '--> preserved output + gated additions

`highway_alpha` and `memory_alpha` initialize to exact zero. Therefore a fresh
G0/G1 augmentation bank can update shadow states without changing the inherited
core output. Training the augmentation is a later G1 operation.

VN97MEM1 remains the one persistent memory system. Retrieval returns a vector
in the current VN97 hidden space; there is no second language model.

Recurrent reasoning is orchestration over the same VN97 model:

- FAST: one pass
- NORMAL: two passes
- DEEP: four passes by default
- ADAPTIVE: bounded by six passes by default

No reasoning backend or second model is introduced.

## Controlled evolution away from Mamba weights

The lineage is explicit:

1. **G0 DIRECT_TRANSFER** — exact inherited core, frozen.
2. **G1 AUGMENTED** — train Multi-Timescale / Highway / MEM1 integration while
   the inherited core remains frozen.
3. **G2 ADAPTIVE_CORE** — selectively unfreeze/adapt inherited core blocks.
4. **G3 NATIVE_BLOCKS** — incrementally replace inherited blocks with
   VN97-native SSM blocks under held-out regression gates.
5. **G4 NATIVE_PRETRAIN** — fresh VN97-native initialization and pretraining.
   Only this stage may claim Mamba-weight independence.

G0-G3 remain historically Mamba-derived even when the runtime no longer needs
Mamba code. The project must not rewrite that provenance.

## Promotion authority

Self-evolution may create/train/evaluate candidates, but it does not gain a
new activation bypass. Promotion continues to reuse the existing M17 chain:

    M17A candidate review
      -> M17B held-out evaluation
      -> M17C explicit human-approved promotion
      -> transactional activation / rollback evidence

A newer candidate is never promoted merely because it exists.

## ONNX/mobile boundary

The current R2 ONNX production executor is retained. It must not be pointed at
a G0 checkpoint until a VN97-native Mamba-2 SSD implementation and ONNX lowering
prove source/VN97/ORT state+logit parity.

The eventual graph contract must expose explicit:

- Mamba-2 convolution state;
- Mamba-2 SSD state `[heads, headdim, d_state]`;
- VN97 Fast/Working/Slow shadow states;
- step and chunk execution views over the same weights.

No Samsung Galaxy S21 FE performance claim is made until these real graphs and
real transferred weights are benchmarked on the physical device.
