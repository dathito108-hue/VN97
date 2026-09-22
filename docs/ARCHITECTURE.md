# VN97 Architecture Contract v0

VN97 is a mobile-first sovereign assistant architecture. The language model is one subsystem,
not the definition of AGI by itself.

## Core retained from the prototypes

- recurrent selective state-space dynamics instead of global softmax attention;
- constant-size per-layer recurrent state for token generation;
- input-dependent B, C and timestep selection;
- gated residual blocks with RMSNorm;
- tied input embedding / output head;
- ternary projection weights as the target execution representation;
- tile-aware physical weight layout as a backend optimization boundary;
- modality identity as an explicit signal rather than silently mixing sources.

## Corrections locked into the canonical design

1. Per-output-channel ternary scale replaces one scale for an entire matrix.
2. Bounded positive timestep prevents extreme state updates.
3. Negative diagonal A preserves stable decay dynamics.
4. Exact diagonal zero-order-hold input factor replaces the rough dt times B approximation.
5. Multi-timescale log-spaced dynamics improve memory diversity at initialization.
6. Full-sequence execution must match token-by-token recurrent execution while carrying state.
7. Model recurrent state is separated from long-term episodic and semantic memory.
8. "1.58-bit" is not used as a storage claim: the deployable v1 format stores fixed-width
   2-bit symbols, while log2(3) is only the ternary information lower bound.
9. Tile geometry is selected by a backend/device profile. Python loops are not treated as NPU
   acceleration and no universal 16x16/32x32 SRAM geometry is assumed.
10. Raw bytes may be a lossless transport/fallback representation, but audio and vision require
    modality-specific frontends before shared semantic reasoning. Modality tags remain useful
    after those frontends.
11. Conditional compute must be implemented with an exportable/vectorized routing contract;
    Python data-dependent branches are not part of the production graph.

## System layers

    Modality frontends / tokenizer / byte fallback
                    |
                    v
          VN97 recurrent language core
                    |
                    v
        Reasoning + planning controller
                    ^
                    |
       Working / episodic / semantic memory
                    ^
                    |
        Tool + internet + device action fabric
                    |
                    v
        Authority / approval / audit boundary
                    |
                    v
       Android native runtime + 3D assistant shell

## Non-negotiable mobile constraints

- generation memory must not grow linearly with context length;
- production ternary weights must be physically packed, not merely simulated as FP tensors;
- optimized kernels must consume packed weights without full-matrix dequantization;
- production recurrence must use a native fused/scan implementation, not a Python token loop;
- long-term exact recall must live in explicit memory/retrieval instead of overloading recurrent state;
- external side effects must pass through an explicit authority boundary;
- Android background continuity must respect operating-system scheduling and lifecycle limits.

## Roadmap

### M0 - Reference recurrent intelligence core - complete
Established the numerical contract, stable SSM dynamics, ternary training path and recurrent invariant tests.

### M1 - Native packed ternary execution - in progress
M1A defines `VN97T2`: per-channel scales, fixed 2-bit symbols, configurable tile-major layout,
versioned serialization, Python equivalence tests and a portable C++ packed matvec baseline.
M1B will add ARM64 vectorized kernels and runtime dispatch; GPU/NPU delegates may follow where
they can preserve the same operator semantics.

### M2 - Parallel training / fused recurrence
Replace the Python sequential training loop with scan/fused implementations while preserving exact recurrent inference semantics. Add exportable conditional-compute routing only after correctness is measurable.

### M3 - Mobile tokenizer, modality adapters and embedding compression
Design a compact text tokenizer and compressed/tied embedding representation. Keep byte-level
encoding as a lossless fallback/transport path. Audio and vision enter through efficient
modality-specific frontends and then receive explicit modality identity in the shared space.

### M4 - Sovereign memory
Add bounded working memory plus persistent episodic and semantic memory with retrieval, provenance and retention policy.

### M5 - Reasoning and planning controller
Add deliberate reasoning, task decomposition, verification, interruption/resume and uncertainty-aware execution.

### M6 - Tool and authority fabric
Internet, files, apps and device actions through typed capabilities, explicit permissions, approvals and auditable receipts.

### M7 - Android native runtime
JNI/NDK runtime, hardware-aware scheduling, checkpoint/recovery, background continuation and battery/thermal governance.

### M8 - Interactive 3D assistant
Low-latency avatar shell, speech/vision hooks and continuity with the sovereign cognition state.

### M9 - Capability acquisition
Controlled import/adaptation pipeline that converts compatible learned capability into VN97-native packages without silently mutating trusted runtime code.
