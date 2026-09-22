# VN97

VN97 is a mobile-first sovereign intelligence project built around a recurrent selective
state-space core and a physically packed ternary native runtime.

## Completed foundations

- M0: stable selective SSM reference with constant recurrent state;
- M1: physical `VN97T2` 2-bit ternary weights, scalar kernel, ARM64 NEON backend and dispatch;
- M2A: associative parallel affine scan for full-sequence SSM execution.

### M2A scan improvement

The recurrent equation `h_t = a_t*h_(t-1) + b_t` is an associative affine transform. VN97
now vectorizes B/C/dt over the complete sequence and composes state transitions with iterative
doubling. A 4096-token full-sequence pass therefore uses 12 Python scan rounds rather than
4096 Python token iterations while preserving the token-by-token recurrent result.

This is a reference parallel path, not the final mobile kernel: it performs O(L log L) tensor
work. M2B will fuse the scan/recurrent execution natively to reduce intermediate work and
allocations.

## Native packed execution

- `VN97T2`: four ternary symbols per byte;
- per-output-channel scales;
- configurable tile-major packing;
- portable scalar packed matvec;
- ARM64 NEON packed matvec;
- explicit `auto / scalar / arm64-neon` dispatch.

See:

- `docs/ARCHITECTURE.md`
- `docs/PACKED_TERNARY_V1.md`
- `docs/ARM64_NEON_BACKEND.md`
- `docs/PARALLEL_SCAN.md`

## Local verification

    python -m pip install -e '.[dev]'
    pytest

    cmake -S native -B native/build
    cmake --build native/build
    ctest --test-dir native/build --output-on-failure
