# M8A — Interactive 3D Avatar Foundation

M8A establishes VN97's visual assistant boundary. It is deliberately independent from the
reasoning, tool and authority subsystems.

## Module

`android/avatar` is a permission-free Android library using OpenGL ES 3.0 directly. It does not
depend on Unity, Filament or another 3D engine.

The initial avatar is procedural so the runtime can be tested without external assets. One cube
mesh is transformed into torso, head, eyes, mouth and arms. This keeps draw-state small and
provides a stable shell for later rig/asset work.

## Typed visual state

`AvatarCommand` contains:

- monotonic `sourceSequence`;
- assistant mode;
- bounded gesture;
- speaking level;
- blink amount;
- normalized gaze X/Y;
- energy.

Every floating-point input must be finite. Unit-range fields are restricted to 0..1 and gaze to
-1..1.

`AvatarStateBridge` serializes publishers, rejects stale/non-increasing source sequence values
and atomically publishes one immutable `AvatarFrameState` snapshot.

The GL thread only reads snapshots. It never calls cognition or mutates planner/tool state.

## Motion and rendering

`AvatarMotionFilter` applies bounded exponential smoothing and caps individual frame delta at
100 ms so a resumed/late frame cannot create an extreme interpolation jump.

The procedural renderer supports:

- idle breathing;
- gaze/head tracking;
- blink compression;
- speaking-driven mouth opening;
- thinking motion;
- nod/shake/wave gestures;
- mode-specific visual accents.

Shader compile/link failures fail fast instead of rendering an undefined program.

## Adaptive frame pacing

`AvatarFramePolicy` sets:

- SLEEPING: 5 FPS;
- IDLE: 15 FPS;
- THINKING / ERROR: 30 FPS;
- LISTENING / SPEAKING / WAITING_APPROVAL / EXECUTING: 60 FPS.

`VN97AvatarView` runs `GLSurfaceView.RENDERMODE_WHEN_DIRTY` and a Choreographer callback that
requests frames at the current mode's interval. View/host pause stops the frame loop.

## Interaction boundary

The view recognizes tap, long-press and drag-end and reports normalized coordinates through
`AvatarInteractionListener`.

These events are not authority. The avatar cannot:

- approve an M6 request;
- mint policy, token or lease state;
- launch an app;
- write clipboard;
- access network or arbitrary files.

A host may translate an interaction into cognition input, after which any external effect must
still pass M6C intent/approval and M6A policy/lease/audit.

## Android integration repair

Before M8A, `android/settings.gradle.kts` contained a literal `\n` between the runtime and
platform module includes. M8A replaces it with valid separate include statements and adds
`:avatar`.

## Verification

The exact pure avatar state/filter/frame-policy snapshot is compiled and executed with
`kotlinc -Werror` by `android/avatar/host-test/run.sh`.

Additional local verification compiled the complete avatar Kotlin source, including
`AvatarRenderer` and `VN97AvatarView`, against Android API-compatible stubs with
`kotlinc -Werror`. This checks Kotlin type/syntax boundaries without claiming a device GPU
instrumentation pass.

An Android device/emulator GPU test remains required to validate actual GLES driver behavior.
