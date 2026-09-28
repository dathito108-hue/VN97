# VN97 native architecture A1 boundary

VN97 now owns a concrete mobile execution-planning boundary rather than letting
the imported source architecture define scheduling. `VN97NativeExecutionPlanner`
produces exact, bounded segments for one identity-bound recurrent graph and its
carried state:

- `THROUGHPUT` preserves the previous largest-valid-chunk behavior;
- `RESPONSIVE_PREFILL` starts with up to eight tokens and doubles toward the
  graph maximum for flows that explicitly prefer earlier first work;
- every token is covered exactly once, without gaps, overlaps or state resets;
- only the supported 8/16/32-token graph contracts are accepted;
- the planner never selects another model or changes weights.

The Android ONNX executor now consumes this VN97-owned plan. Its default remains
`THROUGHPUT`, so this change does not claim an unmeasured speedup on the Samsung
S21 FE. A caller may select `RESPONSIVE_PREFILL`, but promotion of that policy
requires physical-device latency, memory and thermal evidence.

## Honest model identity

The surviving G0.3 payload is still an imported-equivalence capsule derived from
the pinned `state-spaces/mamba2-2.7b` release. A VN97 logical namespace and a
VN97-owned scheduler do not make those inherited weights natively trained VN97
weights. Source revision and hashes remain in the provenance chain.

The path toward a genuinely distinct VN97 model is therefore:

1. fully verify the recovered G0.3 payload and logical tensor inventory;
2. reproduce an identity-bound ONNX graph and numerical parity on the same
   weights;
3. introduce VN97-specific adapted-core changes behind held-out behavior and
   mobile-resource gates;
4. train/promote a native VN97 checkpoint with its own architecture and training
   evidence;
5. retain the same Android planner, authority, memory and tool contracts where
   compatible.

The full-capsule workflow hashes the 5.4 GB weight file and loads the logical
tensor inventory from artifact `10947911523`. Passing it proves artifact identity
and parseability only. It does not prove ONNX parity, production intelligence,
mobile performance, native VN97 training or activation authority.
