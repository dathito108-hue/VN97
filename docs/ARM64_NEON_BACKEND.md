# VN97 ARM64 NEON packed-ternary backend

M1B adds the first architecture-specific execution backend for the `VN97T2` packed format.
It is a CPU ARM64 Advanced SIMD (NEON) backend, not an NPU backend.

## Dispatch contract

`PackedTernaryMatVecF32()` uses `PackedTernaryBackend::kAuto`.

- ARM64 builds that include the NEON source resolve `auto` to `arm64-neon`.
- Other builds resolve `auto` to `scalar`.
- `scalar` is always available and is the correctness fallback.
- explicitly requesting `arm64-neon` on a build that does not contain it returns
  `kBackendUnavailable`; the runtime does not silently pretend the request succeeded.

ARM64 mobile platforms implement Advanced SIMD as part of the architecture, so VN97 treats a
successfully compiled ARM64 backend as available without a separate per-device feature probe.
Future optional ISA extensions must use their own feature detection before dispatch.

## Kernel strategy

The backend consumes `VN97T2` directly. For each tile row it:

1. locates the row segment inside tile-major packed storage;
2. decodes groups of 16 ternary symbols from four packed bytes;
3. widens the `{-1, 0, +1}` signs into NEON vectors;
4. accumulates 16 input values using vector multiply/FMA;
5. handles non-vector tails through the scalar packed-code reader;
6. applies the per-output-channel scale and optional bias.

Typical 16/32-column tile geometries naturally enter the vector path, while unusual or
unaligned geometries remain correct through the scalar tail/fallback path. The serialized
format is unchanged from M1A.

## Correctness rule

The scalar packed kernel is canonical for operator semantics. Optimized backends must match it
within floating-point accumulation tolerance. Native tests compare explicit scalar and NEON
execution when the ARM64 backend is available and verify fail-closed behavior otherwise.
