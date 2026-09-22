# M7F — Hardware-Aware Tiled Ternary Execution

M7F is the production realization of the hardware-aware tiling introduced by the second VN97
prototype. It upgrades the first prototype's canonical Selective-SSM model directly; it is not a
new model or an alternate intelligence backend.

## Why this block exists

VN97T2 has always been physically tile-major, with the Python packer defaulting to 16×16 tiles and
supporting bounded alternative geometries such as 32×32. Before M7F, the scalar kernel still walked
logical rows/columns and recomputed tile addressing for every element. The ARM64 implementation
understood tile columns but still completed one whole output row before moving to the next.

That left part of the Code-2 idea unrealized: input/SRAM/cache locality was represented in the file
format but not fully reflected in execution order.

## Tile-row execution

M7F schedules matvec as:

`tile row → input tile column → output rows inside tile → symbols inside input tile`.

For a 16×16 VN97T2 matrix tile this means:

1. access one contiguous block of 16 input F32 values;
2. reuse that block for up to 16 output rows;
3. accumulate those rows in a bounded local array;
4. advance to the next input tile;
5. after the row tile is complete, apply each row's VN97T2 scale and optional bias.

This preserves the exact logical dot-product column order for each row while improving temporal
locality of the input tile and sequential locality of packed symbols.

## Execution plan

`PlanPackedTernaryMatVecF32()` returns `PackedTernaryExecutionPlan` containing:

- resolved scalar or ARM64 NEON backend;
- matrix rows/columns;
- physical row and column block sizes;
- effective vector width;
- bounded accumulator-float count;
- physical tile count;
- estimated tile working-set bytes.

For the standard 16×16 layout, scalar working-set accounting is 64 bytes of packed ternary symbols
+ 64 bytes of F32 input + 64 bytes of row accumulators = roughly 192 bytes for the active tile
working set, excluding already-resident scales/output.

## Bounded scratch

VN97T2 format v1 already limits tile dimensions to 256 in Python. M7F enforces the same constraint
in the native parser. The public matvec path therefore uses a fixed `std::array<float,256>`
accumulator: at most 1 KiB, no heap allocation, deterministic lifetime.

## ARM64 NEON

The ARM64 backend follows the same tile-row schedule. When a row span is byte-aligned, it consumes
16 ternary symbols (4 packed bytes) per vector dot and multiplies them against 16 contiguous F32
inputs using NEON. Partial or unaligned spans fall back to the exact scalar range implementation.

Backend AUTO behavior remains unchanged: verified ARM64 NEON where compiled/available, otherwise
scalar.

## Relationship to M7D

M7D calls `PackedTernaryMatVecF32WithBackend()` for every in/dt/B/C/out projection. M7F changes
that implementation in place to plan and run the tiled kernel. Therefore the native language
executor receives the hardware-aware upgrade automatically without a second inference graph,
weight conversion, model identity or checkpoint format.

## NPU boundary

M7F intentionally does not label the CPU/NEON implementation as an NPU. Android devices expose
heterogeneous vendor accelerators and there is no single portable raw-NPU instruction interface
equivalent to ARM64 NEON.

Future NNAPI/vendor-specific adapters can use `PackedTernaryExecutionPlan` and VN97T2 tile
geometry as the scheduling contract. They must produce the same mathematical result and retain
scalar/NEON fallbacks. No vendor backend is allowed to redefine VN97 model math.

## Verification

The M7F regression surface verifies:

- native format-v1 tile bounds match Python (maximum 256);
- execution-plan geometry and working-set calculation;
- 17×19 matrix execution over 16×16 physical tiles, exercising both row and column partial tiles;
- dense reference equivalence with per-row scales and bias;
- plan/matrix mismatch rejection;
- ARM64 NEON equivalence to scalar when that backend is available.

An isolated x86-64 scalar build has passed C++17 with `-Wall -Wextra -Werror -pedantic` plus
AddressSanitizer and UndefinedBehaviorSanitizer. ARM64 NEON remains covered by the repository's
architecture-specific test path and must be exercised on an ARM64 runner/device for performance
claims.
