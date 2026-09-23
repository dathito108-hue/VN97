# M10A — Sovereign Android App Bootstrap

M10A turns the existing VN97 Android libraries into an installable application module without
inventing a fallback model or bypassing the production trust path.

## Module

The Android build now contains:

- `:runtime` — canonical VN97 native/runtime bridge;
- `:platform` — M6/M7 authority, assistant and continuity integration;
- `:avatar` — M8 sovereign OpenGL ES 3.0 assistant presentation;
- `:app` — installable launcher shell.

The app uses application ID `ai.vn97.app`, minSdk 26 and targetSdk 37.

## Fail-closed model bootstrap

M10A does not bundle a model and does not use a Transformer/LLaMA/cloud fallback.

The initial app phase is `MODEL_REQUIRED`. Chat input is disabled until a trusted
M9-activated VN97 model is attached by the next production-bootstrap milestone.

The Activity therefore never fabricates an answer when no canonical model exists.

## UI shell

`VN97MainActivity` is dependency-light and uses Android framework views:

- sovereign M8 `VN97AvatarView`;
- bounded status/transcript area;
- message input;
- send control.

Avatar interactions remain presentation events only and do not grant M6 authority.

## Production platform initialization

`VN97Application` owns one lazy `AndroidPlatformRuntime` instance rooted in application
context. This establishes the existing M6/M7 Android platform surface without constructing a
parallel model/session.

The platform library manifest still contributes the non-exported persisted continuation
JobService and RECEIVE_BOOT_COMPLETED permission.

## Security defaults

The app manifest:

- disables Android backup;
- disallows cleartext network traffic;
- declares OpenGL ES 3.0 because the sovereign avatar renderer requires it;
- requests no microphone, camera, location or broad storage permissions in M10A.

## Verification

A pure host regression checks the app lifecycle reducer and single-flight input invariants:

`M10A_APP_STATE_PASS`

Full APK assembly and device launch remain Android build/device gates and are not inferred from
the host reducer test.

## Next boundary

M10B should connect an M9-activated VN97 model artifact to the app lifecycle:

`M9 trusted activation -> NativeActivatedModel -> M7 production memory-backed assistant -> M10A UI`

Only after that binding should the Send control become READY and execute real VN97 turns.
