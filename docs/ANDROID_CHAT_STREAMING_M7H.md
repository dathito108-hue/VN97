# M7H — End-to-End Android Chat Streaming

M7H exposes the M7G native generation path as one canonical Android chat surface without
introducing another model, inference backend, planner, memory engine, or authority path.

Canonical direct-chat path:

`user turn → VN97CHAT1 → VN97TK1 → trusted activated VN97MI1 → native prefill → recurrent step → native sampler → UTF-8 delta stream`

The underlying architecture remains:

`Code 1 → Code 2 hardware/mobile-aware upgrade → VN97 production`.

## Stateful turn semantics

VN97's recurrent state is the live inference history. M7H therefore does **not** replay an entire
chat transcript on every message.

- A fresh runtime sequence may include the bounded system prompt and the first user turn.
- Subsequent turns frame only the new user message.
- VN97RUN2 already preserves recurrent state, sequence position, and exact activated-model
  identity, so a restored session continues from the same native context.
- An unbound runtime with a nonzero sequence position is rejected.
- A model-bound runtime is rejected before generation if its 32-byte model identity differs from
  the activated model.

This avoids quadratic transcript replay and duplicate context on mobile.

## VN97CHAT1 envelope

The chat-facing envelope is deterministic bounded UTF-8:

```text
VN97CHAT1
{"messages":[{"role":"system","content":"..."},{"role":"user","content":"..."}],"next_role":"assistant"}
```

System content is emitted only for a fresh sequence. JSON strings are escaped canonically and user
content is data rather than protocol syntax.

VN97CHAT1 is **not** a replacement for M5C `VN97COG1`. Direct chat is the response surface.
Reasoning/planning, retrieval, verification, external tools, approvals and capability activation
retain their existing M5/M6/M9 contracts.

## Streaming controller

`NativeChatController` binds one `NativeRuntimeOwner` and one trusted
`NativeActivatedModel`.

- exactly one active generation per controller;
- synchronous streaming delta callback, suitable for a host worker/executor;
- `cancelActive()` is thread-safe and cooperative;
- cancellation is observed between committed tokens;
- a token is committed into RuntimeSession before its delta is surfaced, preserving checkpoint
  consistency;
- no coroutine, UI toolkit, networking or third-party AI dependency is introduced.

The controller intentionally does not create an Android Activity or a second application layer.
Existing/future Compose/View surfaces and the M8 avatar can consume the same `NativeChatDelta`
stream.

## Bounds

`NativeChatLimits` bounds:

- system prompt UTF-8 bytes;
- one user-turn UTF-8 bytes;
- complete canonical turn prompt bytes;
- streamed response UTF-8 bytes.

The response limit is cooperative: the UTF-8 chunk produced by the already-committed token that
crosses the boundary is still surfaced, then generation stops. M7G's max-new-token bound remains
the hard token ceiling.

## Stop semantics

A turn reports one of:

- `EOS`;
- `TOKEN_LIMIT`;
- `CANCELLED`;
- `RESPONSE_LIMIT`.

The result carries sequence start/end and derives how many prompt tokens were consumed. This lets
the host correlate visible chat turns with the exact recurrent-runtime position.

## Continuity

M7H does not fork checkpoint formats. Hosts continue to use
`NativeRuntimeOwner.suspendAndPersist()`, which writes the existing VN97RUN1/VN97RUN2 format.
For a model-bound chat session this is VN97RUN2.

The visible UI transcript remains host presentation data; the authoritative inference continuity
is the model-bound native recurrent state. A later transcript/presentation persistence layer must
not replay already-committed turns into that state.

## Verification surface

M7H adds a host regression for:

- deterministic VN97CHAT1 framing;
- JSON escaping;
- system prompt only on a fresh sequence;
- UTF-8 byte bounds;
- cancellation state;
- sequence/token accounting validation.

The existing M7B lifecycle and M7G sampler/streaming host tests remain in the same gate.
