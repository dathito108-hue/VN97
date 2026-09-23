# M14D — Concurrent Multi-Touch Game Control

M14D extends the governed M14 game agent with one additional typed capability:

`device.game.multitouch`

It does not add a second planner, model, memory store, input backend, or authority path.

## Purpose

M14A provided exact-package governed tap/swipe/back actions. M14B enforced a strict
fresh-frame closed loop, and M14C added durable VN97MEM1 episode learning.

M14D allows one governed game action to contain several concurrent pointer strokes.
This enables patterns such as:

- movement swipe + skill tap;
- camera drag + movement touch;
- overlapping button taps;
- short bounded multi-finger combinations.

The key invariant remains:

`fresh frame -> one M6 action -> fresh post-action frame -> verification`

A multi-touch action is still exactly one M6 action, one lease consumption and one
receipt. Its internal strokes may overlap in time, but the game agent does not execute
multiple independent external actions from one frame.

## Canonical payload

The payload is canonical JSON:

```json
{"strokes":[{"duration_ms":600,"end_x_bps":2600,"end_y_bps":7600,"start_ms":0,"start_x_bps":1800,"start_y_bps":8200},{"duration_ms":80,"end_x_bps":8600,"end_y_bps":7200,"start_ms":120,"start_x_bps":8600,"start_y_bps":7200}]}
```

Each stroke represents one Android accessibility `StrokeDescription`.

Coordinates remain resolution-independent basis points in the inclusive range
0..10000.

## Bounds

The pure `VN97GameMultiTouchContract` enforces:

- 2..4 strokes;
- payload <= 1024 UTF-8 bytes;
- each stroke duration 20..3000 ms;
- each stroke start time 0..3000 ms;
- every stroke must finish by 3000 ms;
- start times are nondecreasing;
- all coordinates are within 0..10000;
- at least two strokes must overlap in time;
- exact canonical field ordering and no extra fields/whitespace.

The Android accessibility service re-validates the decoded strokes before dispatch.

## Authority and execution

The capability uses the same M6 path as the existing game actions:

1. the user explicitly authorizes an exact Android package through M14A;
2. the game accessibility service must be connected;
3. the authorized package must currently be foreground;
4. M6 must hold the exact-package `device.game.multitouch` grant;
5. the short one-use game action lease is consumed;
6. the Android adapter checks the game authorization again;
7. one `GestureDescription` is built with all bounded strokes;
8. one dispatch is awaited;
9. one durable M6 receipt is returned;
10. M14B waits for a strictly newer frame and verifies the outcome.

No accessibility view-tree scraping is introduced. Visual perception remains the
user-approved MediaProjection path.

## Planner behavior

`VN97GAME2` now knows that multi-touch is available and is told to use it only when
concurrent control is required. The capability schema is exposed through the same
typed capability surface as tap/swipe/back.

Visual content and recalled M14C strategy remain untrusted evidence, never authority.

## Memory

M14C's game episode memory accepts `device.game.multitouch` only after:

- the M6 receipt reports success;
- a fresh post-action frame has been acquired;
- the same VN97 verification path has evaluated the result.

Therefore learned strategy can retain whether a concurrent move/skill combination
worked without weakening the fresh-frame invariant.

## Regression contract

`M14DGameMultiTouchContractTest.kt` covers:

- valid movement + skill overlap;
- the four-stroke maximum;
- rejecting one-stroke pseudo-multitouch;
- rejecting more than four strokes;
- coordinate bounds;
- total duration bounds;
- ordered start times;
- mandatory temporal overlap;
- canonical JSON formatting and field order.

APK compilation remains the integration gate for Android `GestureDescription`, M6
registry wiring, adapter wiring and the game-agent service.

## Scope limitation

M14D supports concurrent strokes inside one bounded Android gesture. It does not yet
claim an indefinitely persistent pointer held across multiple fresh-frame decisions.
If a particular game requires touch continuity across separate Android gesture
dispatches, that would require a later continuation-stroke mechanism with its own
strict lifecycle and cancellation rules.
