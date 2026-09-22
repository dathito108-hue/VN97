# VN97

VN97 is a new mobile-first sovereign intelligence project built around a recurrent selective
state-space core and a future packed ternary native runtime.

The project starts from the strongest ideas in the original prototype while correcting the
parts that would block real mobile deployment: global-scale fake ternary quantization,
approximate SSM discretization, unbounded timestep dynamics, Python-only recurrence and the
assumption that a language model alone is an AGI system.

## M0 status

The initial reference core provides:

- selective SSM recurrence with constant-size state;
- input-dependent B/C/dt;
- stable negative diagonal dynamics;
- exact diagonal zero-order-hold input factor;
- per-channel ternary STE projections;
- RMSNorm, gating and residual blocks;
- tied embedding/language head;
- stateful autoregressive generation;
- invariant tests for full-sequence vs token-by-token execution;
- an explicit roadmap toward packed native ternary kernels, sovereign memory, reasoning,
  tool/device authority, Android continuity and an interactive 3D assistant.

See docs/ARCHITECTURE.md for the canonical architecture contract.

## Local smoke test

    python -m pip install -e '.[dev]'
    pytest

This Python implementation is the numerical reference, not the final mobile runtime. In
production, ternary weights must be physically packed and recurrence must move to native
fused/scan kernels while matching these tests.
