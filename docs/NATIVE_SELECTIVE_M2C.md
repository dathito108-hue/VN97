# VN97 M2C native fused selective ZOH kernel

M2C moves token-dependent selective SSM discretization into the native deployment runtime while
preserving the M0/M2A numerical contract.

## Compact input contract

For batch B, sequence L, model width D and recurrent width N:

- signal: [B,L,D]
- dt_logits: [B,L,D]
- input_b: [B,L,N]
- readout_c: [B,L,N]
- gate: [B,L,D]
- cached stable A: [D,N]
- caller-owned state_io: [B,D,N]
- output before out-projection: [B,L,D]

A is prepared once per model/layer from canonical -exp(a_log) and must be finite and strictly
negative. Invalid A or invalid timestep bounds fail before recurrent state is mutated.

## Fused semantics

For each token and model channel:

    dt = clamp(softplus(dt_logits), dt_min, dt_max)
    z = A * dt
    decay = exp(z)
    zoh = expm1(z) / A
    drive = zoh * input_b * signal
    state = decay * state + drive
    y = sum(state * readout_c)
    output = y * silu(gate)

This is the same exact diagonal zero-order-hold discretization used by the PyTorch reference.
There is no dt * B approximation.

## Why M2C exists

M2B deliberately accepts expanded decay and drive so it can serve as a small native recurrence
oracle. Those tensors each have [B,L,D,N] elements. M2C computes them inside the state loop and
never materializes either tensor.

The avoided transient storage is:

    2 * B * L * D * N * sizeof(float)

For example, with B=1, L=4096, D=4096 and N=16, the two expanded FP32 tensors alone would
contain 536,870,912 values, or 2 GiB. M2C avoids that sequence-expanded storage.

## Backends

- scalar: portable C++17 correctness oracle;
- arm64-neon: vectorizes state update and C readout across d_state, while preserving scalar
  transcendental semantics for exp/expm1;
- auto: resolves to ARM64 NEON only when compiled for an available ARM64 target.

Explicit unavailable backends fail closed.

## Android boundary

C-linkage entry points:

- vn97_selective_prefill_f32
- vn97_selective_step_f32

are intentionally shaped for later JNI/NDK integration. M7 will own Android lifecycle,
threading, device profiling, thermal/battery policy and hardware validation; those concerns do
not change M2C operator semantics.

## Verification invariant

Native tests require:

- M2C scalar == canonical PyTorch exact-ZOH + affine scan;
- M2C scalar == M2B recurrent execution fed with explicitly materialized decay/drive;
- M2C prefill == repeated M2C token-step;
- ARM64 NEON == scalar when the backend is available;
- invalid A/timestep bounds do not mutate state.
