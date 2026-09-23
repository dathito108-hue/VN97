# M7K — Canonical Android Planner

M7K ports the deterministic M5A planning state machine to the Android runtime surface so M7J typed cognition can drive the same bounded plan lifecycle on-device.

The architecture remains:

`Code 1 → Code 2 hardware/mobile-aware upgrade → VN97 production`

This is a platform port of the canonical M5 planner contract, not a competing planner or another model/backend.

## Cross-platform plan identity

M5A defines a plan ID as SHA-256 over compact sorted-key JSON containing:

- goal;
- immutable reasoning budget;
- immutable ordered plan-step specs.

M7K uses the shared strict JSON serializer and locks float formatting to Python `json.dumps(..., allow_nan=False)` behavior at the scientific-notation boundaries used by canonical plan JSON.

Regression fixtures include:

- `0.0` / `-0.0`;
- `1e-4 → 0.0001`;
- `1e-5 → 1e-05`;
- `1e15 → 1000000000000000.0`;
- `1e16 → 1e+16`.

The fixture plan `goal=answer` with one verified REASON step at confidence 0.75 has the same Python/Kotlin identity:

`55e15d8cfd131608bbb6f97a9e6b565799c7d2eeec3c63dcf82a564b5f87f2ef`

## Planner lifecycle

`NativePlanController` preserves the M5A operational semantics:

- dense ordered step IDs;
- dependencies may reference only earlier steps;
- at most one active internal/external step;
- deterministic first-ready-step selection;
- transition budget;
- per-step retry budget;
- memory-query count/hit bounds;
- verification wait/retry/failure;
- external wait/result handoff;
- pause resets an in-flight internal step before durable output;
- resume returns to READY and recomputes status;
- cancel marks unfinished steps cancelled;
- terminal plans fail closed against further mutation.

Step kinds are the same M5/M7J kinds:

`REASON | RETRIEVE | VERIFY | RESPOND | EXTERNAL`.

## M7J bridge

M7K consumes `NativePlanStepSpec` directly from M7J. Therefore a plan produced by the strict typed cognition adapter can be instantiated without translating into a second schema.

M7K intentionally does not perform model inference itself. M7I/M7J remain responsible for cognition generation and strict parsing.

## Authority boundary

An EXTERNAL step entering `WAITING_EXTERNAL` is only a planner state transition. It does not execute a capability.

External execution still requires:

`typed external intent → trusted capability binding → M6 authority / approval → tool execution → recordExternalResult(...)`

M7K cannot mint approval, bypass M6, activate capabilities, or treat model output as authority.

## Deliberate next boundary

M7K does not yet add an Android binary planner checkpoint. The Python canonical format remains `VN97PLN1`.

The next continuity milestone should implement byte-compatible `VN97PLN1` encode/decode and atomic Android storage, then bind it with the existing VN97RUN2 runtime checkpoint so cognition + recurrent inference can recover together after process death/reboot.

## Verification

The M7K host regression covers:

- Python/Kotlin plan-ID parity;
- canonical float parity;
- dependency ordering;
- verification retry;
- external waiting/result;
- pause/resume;
- memory-query budget;
- transition-budget exhaustion.

The local isolated gate runs with `kotlinc -Werror` and prints `M7K_PLANNER_PASS`.
