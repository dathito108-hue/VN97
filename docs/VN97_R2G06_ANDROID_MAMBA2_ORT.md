# VN97 R2-G0.6 — Android Mamba-2 ORT executor

G0.6 moves the Android execution contract from the older VN97-R2 three-input
step/chunk layout to the canonical Mamba-2 G0.5 recurrent graph.

## Runtime contract

One ONNX graph is used for both prompt prefill and recurrent decode:

    input_ids[1,max_chunk]
    valid_length[1]
    conv_state[64,1,5376,4]
    ssm_state[64,1,80,64,128]

Outputs:

    logits[1,max_chunk,vocab]
    next_conv_state
    next_ssm_state

valid_length=1 is decode. Larger valid lengths are prompt chunks. The Android
executor pads only the unused token suffix and commits recurrent state only
after a successful finite-logit run.

## Runtime descriptor

VN97M2G06RUNTIME1 is compiled from a verified G0.5 manifest and binds the G0.5
manifest ID, G0.3 capsule/source identities, every ONNX/external-data file hash,
exact 2.7B geometry, the four-input/three-output graph contract, single-weight
graph semantics, no quantization, no Mamba runtime dependency, and
production_activation_authorized=false.

The Android loader re-hashes every graph/external-data file before session
creation.

## Provider execution

The new executor reuses the existing provider session factory and fail-closed
accelerator configuration: QNN where explicitly available on Qualcomm, NNAPI,
XNNPACK, and CPU as the explicit final fallback. No hidden CPU fallback is
enabled inside accelerator sessions.

G0.6 uses the conservative adaptive scheduler only for provider order and
thread count. Device-measured E3/E4 profiling must be repeated for the new
Mamba-2 graph before production authorization.

## Memory boundary

The dense float32 recurrent state contains 43,319,296 elements per batch,
approximately 173 MB per state slot. The executor currently uses double
buffering so failed provider attempts cannot corrupt committed state.

This is a correctness-first contract, not a claim that dense 2.7B execution is
mobile-optimal. State precision/compression changes belong after real parity.

## Production boundary

G0.6 does not switch the canonical cognition bridge automatically.

Required next gates are: real G0.3 capsule materialization; real G0.5 graph
export; source → VN97 → ORT parity; G0.6 Android provider profiling/autotuning
on the real graph; S21 FE RAM/latency/thermal qualification; and only then bind
the assistant bridge to the new runtime.

The old F1 executor remains existing compatibility code during migration, but
the G0.6 Mamba-2 executor has no fallback into it.
