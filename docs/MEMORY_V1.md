# VN97MEM1 sovereign memory journal

M4A defines the first persistent VN97-native memory format and the reference memory policy
layer. The recurrent model state remains separate from long-term memory.

## Memory tiers

### Working memory

WorkingMemory is an in-process bounded FIFO buffer. It has independent item-count and UTF-8
byte budgets. When either budget is exceeded, the oldest items are evicted first.

Working memory is intentionally not the persistence mechanism. Items that deserve long-term
retention are converted into episodic or semantic journal records.

### Episodic memory

Episodic records represent events and observations. They can contain a retrieval vector, source
provenance, importance, timestamp and an optional parent link.

### Semantic memory

Semantic records represent distilled facts or derived knowledge. A semantic record can point to
the earlier record from which it was derived, allowing provenance ancestry to survive retention
compaction.

## VN97MEM1 file header

All integers are little-endian.

    offset 0   magic[8] = "VN97MEM1"
    offset 8   version u32 = 1
    offset 12  vector_dim u32

The journal vector dimension is fixed for one file. A record either has no vector or exactly
vector_dim float32 values.

## Record framing

Every append is one frame:

    body_size u32
    crc32(body) u32
    body[body_size]

A valid record body has a fixed 76-byte header followed by variable fields:

    record_id      u64
    timestamp_ns   u64
    kind           u8    (1 episodic, 2 semantic)
    flags          u8    (0 in v1)
    reserved       u16   (0 in v1)
    importance     f32   [0,1]
    source_len     u32
    content_len    u32
    vector_count   u32   (0 or vector_dim)
    parent_id      u64   (0 or an existing earlier record)
    content_sha256 u8[32]
    source         UTF-8 bytes
    content        UTF-8 bytes
    vector         float32[vector_count]

Record IDs are strictly increasing. Compaction preserves the highest ID so a later append cannot
reuse an ID.

## Integrity and recovery

The reference reader validates:

- file magic/version/vector dimension;
- frame bounds;
- CRC32 for each complete body;
- strict record-ID ordering;
- record kind, flags and field lengths;
- valid UTF-8 source/content;
- finite importance/vector values;
- parent_id points to an existing earlier record;
- SHA-256 of content matches content_sha256.

A torn final frame can be recovered only when recovery is explicitly requested. Recovery
truncates the file to the final completely validated frame. A CRC mismatch or malformed
complete record fails closed and is not treated as a recoverable torn tail.

CRC32 and SHA-256 here provide corruption detection and content identity. They are not a
cryptographic author signature or authorization proof.

## Append durability

MemoryJournal appends one framed record, flushes it and uses fsync by default. The reference
implementation is a single-writer contract. Android multi-process locking belongs to the native
runtime integration milestone.

## Retrieval

M4A provides deterministic exact retrieval over records that contain vectors. The semantic term
is cosine similarity. Optional recency and importance terms can be combined:

    score =
        semantic_weight * cosine
      + recency_weight * half_life_recency
      + importance_weight * importance

Results use deterministic tie-breaking by score, timestamp and record ID. Kind filtering can
restrict retrieval to episodic or semantic memory.

M4A deliberately does not require a third-party vector database. Larger approximate indexes can
be introduced later without changing VN97MEM1 record semantics.

## Retention and compaction

RetentionPolicy supports:

- max_records;
- max_age_ns;
- min_importance.

Compaction writes a new journal to a temporary file, fsyncs it and replaces the original
atomically. The newest/highest record ID is always retained to prevent ID reuse. Parent ancestry
of retained derived records is preserved even if doing so exceeds the target max_records.

## Native validation boundary

libvn97_memory.a provides a zero-copy C++17 scanner and C ABI:

- ScanMemoryJournal
- vn97_memory_scan

The native scanner validates framing, CRC32, shape, UTF-8, finite values, ID ordering and parent
existence. The Python reference additionally recomputes content SHA-256. Native append,
compaction and retrieval are the next M4 deployment step.

## Scope

M4A establishes the durable format, correctness oracle, retrieval semantics and retention
policy. It does not claim that a particular embedding model is already trained; vectors are
supplied by VN97 cognition/capability layers while the memory store remains model-agnostic.
