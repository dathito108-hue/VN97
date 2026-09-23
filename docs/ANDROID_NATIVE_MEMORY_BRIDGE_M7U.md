# M7U — Android Native VN97MEM1 Memory Bridge

M7U connects the already-canonical M4 native VN97MEM1 memory engine to the Android M7N/M7T cognition path. It does not create a Kotlin memory engine, a second retrieval implementation, or a parallel model/backend.

Locked architecture remains:

`Code 1 → Code 2 hardware/mobile-aware upgrade → VN97 production`.

The production retrieval path becomes:

`VN97 activated model → M7I/M7J query embedding → M7N NativeMemoryRetriever → M7U JNI bridge → native MemoryStore/VN97MEM1 → bounded evidence records → canonical planner`.

## Native engine remains authoritative

The actual journal, index, retrieval ranking, append validation, torn-tail recovery, exclusive writer lock and compaction semantics remain in the existing native M4 implementation:

- `native/src/memory.cpp`
- `native/src/memory_index.cpp`
- `native/src/memory_store.cpp`
- `native/src/memory_crypto.cpp`

M7U only exposes that engine to Android/JVM code through `vn97_memory_jni.cpp` and `NativeMemoryStore`.

## Opaque JNI handles

Kotlin never receives a raw `MemoryStore*` pointer. The JNI bridge owns `MemoryStore` instances in a synchronized opaque-handle registry and returns a positive signed `Long` handle.

Stale/unknown handles fail closed as `INVALID_PARAMETER`. All JNI operations hold the registry lock for the complete native store operation, so a store cannot be closed while an operation is using it.

## Kotlin contract

`NativeMemoryStore` implements `NativeMemoryRetriever`, so it can be passed directly to M7N cognition and M7T assistant turns.

It also exposes the native store lifecycle required for sovereign memory maintenance:

- `create(...)`;
- `open(...)` with explicit torn-tail recovery choice;
- `append(...)` for episodic/semantic records;
- `retrieve(...)` through the M4 ranking engine;
- `compact(...)` through native retention policy;
- `stats()`;
- `close()`.

The bridge enforces JVM/mobile bounds before JNI:

- positive, exact vector dimension;
- finite append/query vectors;
- finite importance in `[0,1]`;
- bounded `topK <= 1024`;
- retrieval weights must be representable by native `float`;
- non-negative timestamps/parent IDs;
- native unsigned IDs/sizes must fit signed JVM ranges.

Record source/content bytes are copied out of the native mmap and decoded with strict UTF-8 before becoming `NativeMemoryContextItem` values.

## Evidence identity

M7U does not invent evidence IDs. `recordId` values are the stable native VN97MEM1 record IDs already used by M4/M5. M7N therefore continues to attach only retrieved, trusted memory evidence IDs to planner results.

## Android production storage

`AndroidPlatformRuntime.openOrCreateProductionMemory(...)` uses `Context.noBackupFilesDir` and a single validated file-name component. Existing symlink/non-regular targets fail closed. A newly created store uses the activated VN97 model `dModel` as the VN97MEM1 vector dimension; an existing store must match that exact dimension before it can be used.

The store lifetime stays explicit and host-owned (`AutoCloseable`) because native VN97MEM1 deliberately holds an exclusive writer lock while open.

## Native build integration

The Android runtime JNI library now links `vn97_memory` and compiles `vn97_memory_jni.cpp`. The host runtime gate also links the canonical native memory sources so the same Kotlin/JNI path can be exercised on the host.

## Verification

An isolated M7U bridge gate was executed locally before upload with:

- C++17 `-Wall -Wextra -Werror -pedantic` for the exact JNI bridge;
- Kotlin `-Werror` for `NativeMemoryStore`;
- an ABI-compatible in-process MemoryStore fixture to exercise the JNI/Kotlin boundary.

The gate printed:

`M7U_NATIVE_MEMORY_BRIDGE_PASS`

The repository host regression is wired to compile the same bridge against the real canonical M4 native memory sources and covers:

- create/open/reopen lifecycle;
- exclusive store lock;
- durable append and stable record IDs;
- exact vector-dimension rejection;
- semantic kind filtering;
- source/content UTF-8 round trip;
- importance ranking;
- bounded top-k rejection;
- native compaction and post-compaction reopen;
- use-after-close failure.

M7U does not change the VN97MEM1 file format or M4 retrieval math.
