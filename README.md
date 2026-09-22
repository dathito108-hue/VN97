# VN97

VN97 is a mobile-first sovereign intelligence project built around a recurrent selective
state-space core and a physically packed ternary native runtime.

## Completed foundations

- M0: stable selective SSM reference with constant recurrent state;
- M1: physical VN97T2 2-bit ternary weights, scalar kernel, ARM64 NEON backend and dispatch;
- M2A: associative parallel affine scan for full-sequence training/reference execution;
- M2B: native fused recurrent state update/readout for prefill and token-step;
- M2C: native fused selective ZOH dynamics that avoids materializing expanded decay/drive.

## M2 execution contract

The canonical recurrence remains:

    h_t = decay_t * h_(t-1) + drive_t

M2A keeps the differentiable PyTorch path and associative scan oracle. M2B provides the
portable native recurrence primitive. M2C moves the token-dependent discretization into native
execution as well:

    dt = clamp(softplus(dt_logits), dt_min, dt_max)
    z = A * dt
    decay = exp(z)
    zoh = expm1(z) / A
    drive = zoh * B * signal

A [D,N] is prepared/cached once from canonical -exp(a_log). Native M2C consumes compact
signal/dt/B/C/gate/A tensors and updates the recurrent state in place, so deployment no longer
needs the two [B,L,D,N] intermediate tensors used by the reference formulation.

## Native execution

- VN97T2: four ternary symbols per byte;
- per-output-channel scales and configurable tile-major packing;
- portable scalar + ARM64 NEON packed matvec;
- portable scalar + ARM64 NEON recurrent/selective kernels;
- explicit auto / scalar / arm64-neon dispatch with fail-closed unavailable backends;
- C-linkage entry points suitable for later Android NDK/JNI integration.

See:

- docs/ARCHITECTURE.md
- docs/PACKED_TERNARY_V1.md
- docs/ARM64_NEON_BACKEND.md
- docs/PARALLEL_SCAN.md
- docs/NATIVE_SELECTIVE_M2C.md

## Local verification

    python -m pip install -e '.[dev]'
    pytest

    cmake -S native -B native/build
    cmake --build native/build
    ctest --test-dir native/build --output-on-failure

Production-only native build:

    cmake -S native -B native/build-prod -DVN97_BUILD_TESTS=OFF
    cmake --build native/build-prod
