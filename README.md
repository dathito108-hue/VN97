# VN97

VN97 is a mobile-first sovereign intelligence project built around a recurrent selective
state-space core and a physically packed ternary native runtime.

The project starts from the strongest ideas in the original prototypes while correcting the
parts that would block real mobile deployment: global-scale fake ternary quantization,
approximate SSM discretization, unbounded timestep dynamics, Python-only recurrence,
hardware claims that are not expressed as a real backend contract, and the assumption that a
language model alone is an AGI system.

## M0 - recurrent intelligence reference

The reference core provides:

- selective SSM recurrence with constant-size state;
- input-dependent B/C/dt;
- stable negative diagonal dynamics;
- exact diagonal zero-order-hold input factor;
- per-channel ternary STE projections;
- RMSNorm, gating and residual blocks;
- tied embedding/language head;
- stateful autoregressive generation;
- invariant tests for full-sequence vs token-by-token execution.

## M1 - native packed ternary execution

VN97 now provides:

- `VN97T2` physical 2-bit ternary storage (four weights per byte);
- per-output-channel FP32 scales;
- configurable tile-major packing rather than a hard-coded NPU tile assumption;
- versioned serialization with reserved-code validation;
- exact Python pack/unpack and linear-reference equivalence tests;
- a portable C++17 scalar packed matvec baseline;
- an ARM64 NEON packed matvec backend for vectorized mobile CPU execution;
- explicit `auto / scalar / arm64-neon` runtime dispatch with fail-closed unavailable backends;
- native tests that preserve the scalar kernel as the numerical contract.

`log2(3) ~= 1.585` bits is the information lower bound of a ternary alphabet. VN97T2 uses a
fixed-width 2-bit representation because it is simple, random-access friendly and practical
for native SIMD/kernel work.

See `docs/ARCHITECTURE.md`, `docs/PACKED_TERNARY_V1.md` and
`docs/ARM64_NEON_BACKEND.md` for the canonical contracts.

## Local verification

Python reference:

    python -m pip install -e '.[dev]'
    pytest

Native packed kernels:

    cmake -S native -B native/build
    cmake --build native/build
    ctest --test-dir native/build --output-on-failure

M2 is the next architecture milestone: move the selective SSM recurrence from the Python token
loop into fused/scan execution while preserving the M0 recurrent semantics.
