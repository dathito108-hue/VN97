# M7A — Native Runtime Session ABI

M7A establishes a stable native execution-session boundary for the future Android/JNI layer.
It does not create a second cognition or authority architecture.

## ABI boundary

The exported runtime API uses nonzero `uint64_t` handles. Callers never receive a raw
`RuntimeSession*`. Internally the handle registry stores `shared_ptr<RuntimeSession>`; each
operation acquires its own strong reference before releasing the registry lock.

This means:

- a stale/destroyed handle fails with `kInvalidHandle`;
- destroy removes future access immediately;
- a concurrent in-flight operation may complete safely without use-after-free;
- JNI does not need to encode native pointers into Java/Kotlin objects.

The public header is intended for C++/JNI/NDK compilation. Its `extern "C"` functions have
stable C linkage, but the header also exposes the C++ `RuntimeSession` contract and is not
claimed to be a pure-C source header.

## Runtime configuration

A session is created with:

- layer count;
- batch size;
- `d_model`;
- `d_state`;
- requested recurrent backend;
- requested packed-ternary backend.

The recurrent-state element count is
`layers × batch × d_model × d_state`. Every multiplication is overflow checked. The reference
runtime limits state storage to 512 MiB.

Backend IDs preserve existing native dispatch semantics:

- 0 — AUTO;
- 1 — scalar;
- 2 — ARM64 NEON.

AUTO resolves through the existing VN97 backend resolver. Explicit unsupported backends fail
closed. Checkpoint restore re-resolves the requested profile on the current device.

## Lifecycle

The session lifecycle is deliberately small:

`CREATED → ACTIVE ↔ SUSPENDED`

Rules:

- only CREATED may activate;
- only ACTIVE may suspend;
- only SUSPENDED may resume;
- token/sequence advance is valid only while ACTIVE;
- host state replacement is rejected while ACTIVE;
- checkpoint export is valid only while SUSPENDED;
- a restored session always starts SUSPENDED.

Restore therefore cannot cause implicit computation after process recreation.

## Recurrent state

The session owns one contiguous F32 state buffer with the exact configured shape. ABI read/write
operations require the exact element count. Imported values must all be finite; NaN and infinity
fail closed.

The ABI exposes copy operations, not an internal state pointer. Future JNI direct-buffer or
zero-copy optimizations must preserve the same ownership/lifecycle rules before replacing this
reference boundary.

## VN97RUN1 checkpoint

VN97RUN1 uses a fixed 64-byte little-endian header followed by F32 state bits.

Header fields include:

- magic `VN97RUN1`;
- version and header size;
- layers, batch, d_model, d_state;
- requested recurrent and packed backend IDs;
- required checkpoint lifecycle marker (SUSPENDED);
- sequence position;
- exact state element count;
- payload CRC32;
- header CRC32.

Restore validates magic/version/header size, header CRC, backend IDs, lifecycle marker, shape,
state count, exact total blob length, payload CRC and finite state values before publishing a
runtime handle.

CRC32 protects against corruption/torn or altered data. It is not a cryptographic author
signature, encryption mechanism, capability grant or approval token. Higher-level Android
storage may add authenticated encryption without changing VN97RUN1 runtime semantics.

## Concurrency

Each session serializes lifecycle/state/checkpoint operations with a private mutex. The global
handle registry has a separate mutex and only retains/releases session ownership; it is not held
during session operations.

This separation avoids raw-pointer lifetime races and avoids holding a global registry lock
during potentially larger state copies/checkpoint serialization.

## M6 boundary preservation

M7A has no file/network/app/device side-effect entry point. It does not receive `PolicyGrant`,
`ApprovalToken`, `CapabilityLease` or capability handlers.

Android execution added later must continue to route external effects through:

`WAITING_EXTERNAL → M6C intent/approval → M6A authority/lease/audit → M6B handler`

The runtime session is compute/lifecycle state, not authority.

## Verification performed for M7A

The exact M7A header/source/test snapshot was compiled locally with:

- C++17;
- `-Wall -Wextra -Werror -pedantic`;
- AddressSanitizer;
- UndefinedBehaviorSanitizer.

The targeted runtime test covers:

- state-shape/count construction;
- AUTO backend resolution;
- exact state import;
- wrong-size and non-finite rejection;
- lifecycle restrictions;
- sequence advance;
- safe checkpoint sizing/export;
- output-too-small behavior;
- handle destruction and stale-handle rejection;
- checkpoint restore and exact state/position recovery;
- header and payload corruption detection;
- explicit unavailable backend rejection;
- oversized/overflowed runtime configuration rejection.

This isolated native gate passed. Full-repository native CMake/CTest still depends on the active
repository checkout/CI environment.
