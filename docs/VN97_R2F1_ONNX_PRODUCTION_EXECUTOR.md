# VN97-R2F1 ONNX Production Executor

## Purpose

R2-F1 turns the validated R2-E execution contracts into one Android production
executor.

The production chain is:

```
validated VN97-R2 checkpoint
-> R2-E2 explicit-state ONNX bundle
-> R2-E3 device measurements
-> R2-E4 device tuning
-> R2-F1 portable runtime descriptor
-> VN97OrtProductionExecutor
```

F1 does not add another model or inference backend. ONNX Runtime remains the
single mobile inference engine for this path.

## Portable runtime descriptor

Schema:

```
VN97R2F1RUNTIME1
```

Default filename:

```
runtime.vn97ort1.json
```

The descriptor intentionally contains only portable JSON primitives whose
canonical representation is stable across Python and Kotlin.

It binds:

- E2 bundle ID;
- architecture fingerprint;
- fast/deep profile;
- E4 tuning ID;
- active layer count;
- vocabulary size;
- recurrent-state dimensions;
- batch size = 1;
- exact E2 input/output names;
- graph filename/kind/sequence length;
- graph SHA-256 and byte size;
- supported chunk sizes;
- float32 state / int64 token contract;
- same-weights semantics;
- no-quantization lock.

Identity:

```
SHA256("VN97R2F1RUNTIME1\0" + canonical_json(body))
```

## Build

After E2 and E4 evidence exists:

```bash
vn97-r2-onnx-runtime build \
  --bundle-dir /data/vn97-onnx \
  --e4-profile /data/s21fe-e4.json
```

This writes:

```
/data/vn97-onnx/runtime.vn97ort1.json
```

Verification:

```bash
vn97-r2-onnx-runtime verify \
  --package /data/vn97-onnx/runtime.vn97ort1.json \
  --bundle-id <E2_BUNDLE_ID> \
  --tuning-id <E4_TUNING_ID>
```

## Android package loading

`OrtProductionPackage.load(...)` verifies before any production session opens:

- descriptor SHA identity;
- batch=1 mobile contract;
- state/token dtypes;
- graph input/output contract;
- graph filename safety;
- no path traversal;
- no graph symlinks;
- exact graph byte length;
- exact graph SHA-256;
- E4 bundle binding;
- E4 tuning ID;
- architecture/profile binding;
- current device/OS identity;
- exact graph coverage between F1 and E4.

A copied tuning profile from another device or another E2 bundle is rejected.

## File-path session loading

Production sessions use ONNX files by path rather than first loading a complete
model into a Java `ByteArray`.

This avoids a second full Java-heap copy of a large model before ORT session
creation.

E3 profiling can keep its byte-array API for small/controlled measurement
fixtures. F1 production uses the path API.

## SessionOptions lifetime

ONNX Runtime requires `OrtSession.SessionOptions` to remain alive while the
session that uses it is alive.

F1 therefore changes `OrtSessionHandle` to own:

- `OrtSession`;
- its exact `SessionOptions`;
- the provider identity.

Close order is:

```
OrtSession.close()
SessionOptions.close()
```

This also corrects the lifecycle for the earlier E3 profiler path.

## Bounded session cache

Multiple ONNX graphs and providers can each require a distinct ORT session.

Caching all combinations is unsafe on a mobile device because each session may
materialize substantial model/provider memory.

F1 uses a bounded access-order cache.

Default:

```
maxCachedSessions = 1
```

The oldest session is closed when the bound is exceeded.

A later real-device E5 campaign may justify a larger bound, but the production
default does not assume that two full sessions fit comfortably on the S21 FE.

## Explicit recurrent state

E2 recurrent state is:

```
conv_state:
  [active_layers, 1, d_inner, d_conv_state]

ssm_state:
  [active_layers, 1, d_inner, d_state]
```

F1 allocates two direct state slots:

```
current state
next state
```

Each slot has reusable direct `FloatBuffer` storage and reusable ONNX tensors.

ORT outputs are pinned directly into the next slot.

After a successful invocation:

```
current <-> next
```

The state is therefore carried without copying the complete recurrent state
through Java arrays on every token.

## Pinned logits

Each fixed graph shape owns reusable direct buffers for:

- int64 input IDs;
- float32 logits.

The output tensors are pinned into those buffers.

For a chunk graph, F1 returns the logits of the last token in the chunk for
continued generation.

No state transition is committed until:

- ORT run succeeds;
- final-token logits are finite.

If a provider fails, the current state remains unchanged and the next provider
can retry the same graph/input against the same recurrent state.

## Sequential + variable parallel execution

Production APIs:

```
prefill(...)
step(...)
```

`prefill` sends the whole prompt workload through E4. E4 selects a variable
sequence of fixed E2 chunk graphs and residual recurrent step graphs.

`step` forces the realtime recurrent path.

Thus the user-selected architecture remains variable:

```
sequential when dependency/realtime requires it
hybrid chunking for ordinary work
larger measured chunks when the device can sustain them
```

All paths carry the same explicit state and the same VN97 weights semantics.

## Provider fallback

F1 does not ask one ORT session to hide the provider fallback result.

It attempts the E4 provider chain explicitly.

For every graph invocation:

1. request the measured preferred provider;
2. create/reuse its exact session;
3. run with pinned inputs/outputs;
4. on recoverable provider/session failure:
   - invalidate that cached session;
   - report failure to E4 quarantine;
   - retry the same graph and same current state on the next provider;
5. on success:
   - report provider success;
   - commit next recurrent state.

CPU remains the final provider in every E4 chain.

## Failure semantics

F1 does not treat fatal JVM errors such as OOM as a normal provider miss.

Normal ORT/session/runtime exceptions can drive provider fallback.

Sequence-position accounting is checked before a provider attempt so arithmetic
failure cannot occur after state has already been committed.

## Latency feedback

Successful invocation latency updates a bounded moving latency estimate.

The next E4 decision receives:

- current thermal status;
- current available memory;
- moving latency.

This closes the loop between measured E4 policy and actual production runtime
conditions.

## E5 hardening integration

`VN97OrtProductionExecutor` implements `OrtHardeningExecutor`.

Therefore E5 can benchmark the exact same executor used in production.

For the E5 cold phase, `resetState()`:

- zeros both recurrent-state slots;
- clears the bounded session cache;
- resets E4 control state.

This makes the cold phase include session creation instead of accidentally
benchmarking a warm cached session.

E5 control logic still remains separate from physical thermal/OOM forcing.

## Memory truth

Pinned state avoids recurrent-state copies, but it does not make an unvalidated
dense 1B float32 checkpoint magically fit the S21 FE.

Before later quantization/mobile lowering, the dense checkpoint remains the
quality oracle.

Real S21 FE memory, latency and thermal claims require the actual E3/E4/E5
campaign on a validated checkpoint.

## Chat and cognition

F1 provides the production token executor required by chat/cognition:

- prompt prefill;
- recurrent token step;
- sequence position;
- production provider feedback.

R2-F2 will bind the approved VN97TK1 tokenizer/sampler and existing
chat/cognition interfaces to this executor.

That bridge must not use the legacy model weights as a hidden second inference
backend.

## Next milestone

R2-F2 — R2 Chat/Cognition Bridge:

- bind exact tokenizer identity to the R2 runtime package;
- encode canonical chat/cognition prompts;
- use F1 `prefill/step` for generation;
- reuse the existing M6/planner/memory contracts;
- prove that legacy native model inference is not invoked on the R2 path;
- keep explicit fast/deep state-reset semantics.
