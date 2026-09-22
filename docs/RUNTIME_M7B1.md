# M7B1 — Android JNI Runtime Bridge

M7B1 connects Android/Kotlin to the M7A native runtime without introducing a second execution or
authority architecture.

## Android build baseline

The Android project is an isolated library build under `android/`:

- Android Gradle Plugin 9.4.1;
- compile SDK 37;
- min SDK 26;
- NDK 28.2.13676358;
- Java 17;
- AGP built-in Kotlin support;
- CMake 3.22.1 interface.

The runtime module links `libvn97_jni.so` to the canonical `vn97_runtime` target. Native tests
are disabled for the Android library sub-build because they remain owned by the top-level native
CTest flow.

The module manifest is empty: M7B1 requests no Android permission.

## JNI contract

`NativeRuntimeBindings` exposes narrow JNI methods for:

- create / restore / destroy;
- activate / suspend / resume;
- sequence advance;
- typed runtime info;
- exact-size recurrent-state read/write;
- VN97RUN1 checkpoint size/write;
- parent-directory fsync for trusted runtime checkpoint persistence.

JNI validates required output array sizes and signed JVM inputs before entering the M7A C ABI.
The M7A `RuntimeStatus` values remain authoritative; JNI does not invent a parallel lifecycle
state machine.

M7A uses unsigned 64-bit handles. Kotlin/JVM uses signed `Long`, so JNI rejects and destroys a
new native handle if it exceeds `Long.MAX_VALUE`. No wrapped negative handle can become visible
to managed code.

## Kotlin session ownership

`NativeRuntimeSession` wraps exactly one native handle. All managed operations are serialized
against `close()`, preventing a Kotlin-level close/use race. This is additive to M7A native
lifetime safety: once a JNI call has resolved its handle, the native registry holds a shared
session reference for that operation.

Native failures map to `NativeRuntimeStatus` and raise `NativeRuntimeException` with the exact
operation and status.

The managed layer exposes no raw native state pointer. State and checkpoint data cross the
reference boundary as bounded JVM arrays.

## Atomic VN97RUN1 persistence

`AtomicCheckpointStore` is intentionally limited to trusted application-internal runtime state.
It is not a general filesystem capability and is not reachable from cognition.

The store:

- creates/uses one trusted root directory;
- rejects a symlink root;
- rejects multi-component checkpoint filenames;
- rejects a symlink checkpoint target;
- bounds checkpoint size;
- writes a same-directory temporary file;
- flushes and fsyncs file data;
- requires an atomic replace operation;
- fsyncs the parent directory through JNI;
- deletes leftover temporary files on failure.

If the filesystem does not support atomic replacement, persistence fails closed rather than
falling back to a partial-copy scheme.

## Lifecycle owner

`NativeRuntimeOwner` composes one `NativeRuntimeSession` with one
`AtomicCheckpointStore`.

On startup:

1. if no checkpoint exists, create a new M7A session;
2. if a checkpoint exists, restore it;
3. require the restored runtime config to exactly match the requested config;
4. never silently discard or overwrite a corrupt/mismatched checkpoint.

On persistence:

1. require ACTIVE or already SUSPENDED;
2. suspend if needed;
3. export VN97RUN1 from the safe SUSPENDED boundary;
4. atomically persist the checkpoint;
5. remain SUSPENDED.

Cold restore therefore never resumes computation automatically.

## M6 authority preservation

M7B1 contains no Android network permission, app-launch action, clipboard mutation, accessibility
control or arbitrary Intent execution.

Internal VN97RUN1 persistence is trusted runtime lifecycle state. External side effects still
follow the completed path:

`WAITING_EXTERNAL → M6C intent/approval → M6A policy/lease/audit → M6B handler`

M7B2 may supply Android implementations for M6B interfaces, but those implementations must remain
behind this existing M6 path.

## Host regression

`android/runtime/host-test/run.sh` provides an emulator-free regression path. It compiles
`runtime.cpp + recurrent.cpp + packed_ternary.cpp + vn97_jni.cpp` into a host JNI shared library,
compiles the Kotlin runtime wrappers with warnings as errors, then executes:

- runtime create;
- exact state import;
- activate/advance/suspend;
- JNI VN97RUN1 checkpoint;
- restore and exact state/position validation;
- atomic checkpoint save/load;
- NativeRuntimeOwner restore;
- resume/advance/suspend/persist;
- second cold restore;
- checkpoint deletion.

A successful run prints `M7B_JNI_KOTLIN_INTEGRATION_PASS`.

The host regression validates JNI/Kotlin ownership and persistence semantics. Android framework,
Keystore, permission and OS-lifecycle behavior remains M7B2/M7C work.


## M7D compatibility note

M7D preserves VN97RUN1 restore support for unbound/legacy sessions and adds VN97RUN2 for
model-bound recurrent state. AtomicCheckpointStore remains byte-opaque; its default size ceiling
is extended from the original 512 MiB + 64-byte RUN1 header to 512 MiB + 100 bytes so the same
store can persist RUN2 without changing its atomic/no-follow/fsync guarantees.
