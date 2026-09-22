# M9A — VN97CAP1 Capability Package + Verified Staging

M9A establishes the first controlled boundary for importing learned capability data into VN97.

It solves two separate problems:

1. deterministically validating what bytes a capability package contains;
2. durably staging those exact bytes without activating them.

## Binary container

VN97CAP1 uses a 96-byte little-endian header followed by a fixed 64-byte entry table and
contiguous section payloads.

The header contains:

- magic `VN97CAP1`;
- version 1;
- header size 96;
- zero flags/reserved fields;
- section count (2..64);
- entry size 64;
- exact table/payload offsets;
- exact total package size;
- manifest section index, fixed to zero;
- SHA-256 over the complete table + payload;
- CRC32 over the protected header prefix.

Entry zero must be the manifest. Every later entry is data. Each entry binds section type,
exact offset, exact size and SHA-256. Sections are contiguous; trailing/hidden bytes are rejected.

The reference package limit is 512 MiB and the manifest limit is 64 KiB.

## Canonical manifest

The manifest must be strict canonical UTF-8 JSON with exactly:

- `schema = VN97CAP1`;
- capability ID;
- positive capability version;
- capability kind;
- source provenance;
- ordered data-section metadata.

Supported initial kinds are weights, tokenizer, memory, avatar, multimodal, knowledge and
composite.

Source provenance contains origin, source SHA-256 and license metadata.

Each data section binds:

- 1-based section index;
- unique role;
- format identifier;
- exact byte size;
- exact SHA-256.

The parser recalculates those values against the actual binary sections.

## Data-only rule

M9A packages are not executable extension bundles.

Section roles including code, executable, plugin, script, native-library and shared-library are
rejected. A package may contain learned weights, tokenizer/memory/avatar data, metadata or other
bounded data formats, but M9A provides no dynamic code-loading API.

This keeps capability acquisition separate from trusted runtime mutation.

## Staging

`CapabilityStager` accepts only a complete successfully parsed package.

Its destination root must already exist, be absolute, be a directory and not be a symlink.
Staging then:

1. acquires an exclusive advisory lock on the root directory;
2. derives the destination name from full-package SHA-256;
3. verifies an existing digest path if present and returns it idempotently;
4. otherwise writes a same-directory random temporary file with O_EXCL/O_NOFOLLOW;
5. fsyncs file contents;
6. publishes with a create-only hard link;
7. removes the temporary name;
8. fsyncs the directory.

The resulting filename is `<sha256>.vn97cap1`.

No package is activated by this operation.

## Trust boundary

M9A integrity is not publisher authentication.

- CRC32 detects corruption.
- SHA-256 binds exact bytes.
- source SHA-256/provenance fields record claimed source identity.
- none of those prove who authorized or published a package.

M9B will add explicit signature/trust policy and compatibility/adaptation decisions.

Likewise, downloading or copying a package from an external source remains an M6-authorized
external side effect. The M9A stager does not access network or arbitrary external files.

## Verification

The isolated M9A regression suite covers:

- build/parse round trip;
- manifest-to-section role/format/size/SHA binding;
- whole-package content identity;
- payload tamper rejection;
- header tamper rejection;
- executable-role rejection;
- content-addressed idempotent staging;
- strict source hash and capability-ID validation.

The exact M9A core snapshot passed 6/6 pytest tests plus Python bytecode compilation.
