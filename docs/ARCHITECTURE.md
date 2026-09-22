# VN97 Architecture Contract v0

VN97 is a mobile-first sovereign assistant architecture. The language model is one subsystem,
not the definition of AGI by itself.

## Core retained from the prototype

- recurrent selective state-space dynamics instead of global softmax attention;
- constant-size per-layer recurrent state for token generation;
- input-dependent B, C and timestep selection;
- gated residual blocks with RMSNorm;
- tied input embedding / output head;
- ternary projection weights as the target execution representation.

## Core improvements in M0

1. Per-output-channel ternary scale replaces one scale for an entire matrix.
2. Bounded positive timestep prevents extreme state updates.
3. Negative diagonal A preserves stable decay dynamics.
4. Exact diagonal zero-order-hold input factor replaces the rough dt times B approximation.
5. Multi-timescale log-spaced dynamics improve memory diversity at initialization.
6. Canonical recurrent invariant: processing a sequence at once must match processing it token by token while carrying state.
7. The model state is explicitly separated from long-term episodic and semantic memory.

## System layers

    Tokenizer / compact vocabulary
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
- production recurrence must use a native fused/scan implementation, not a Python token loop;
- long-term exact recall must live in explicit memory/retrieval instead of overloading recurrent state;
- external side effects must pass through an explicit authority boundary;
- Android background continuity must respect operating-system scheduling and lifecycle limits.

## Roadmap

### M0 - Reference recurrent intelligence core
Current milestone. Establish the numerical contract, stable SSM dynamics, ternary training path and invariant tests.

### M1 - Native packed ternary execution
Define a VN97 packed ternary format, per-channel scales, reference pack/unpack, CPU SIMD kernels and correctness tests against M0.

### M2 - Parallel training / fused recurrence
Replace the Python sequential training loop with scan/fused implementations while preserving exact recurrent inference semantics.

### M3 - Mobile tokenizer and embedding compression
Design a compact tokenizer, vocabulary strategy and compressed/tied embedding representation suitable for device RAM budgets.

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
