# M10D — Android VN97CAP1 / VN97SIG1 Staging

M10D ports the M9A/M9B package-envelope validation boundary into the Android runtime without
pretending that staging is activation.

Canonical provisioning remains:

`VN97CAP1 -> VN97SIG1 binding -> content-addressed stage -> future M9 trust verification -> VN97INV1 -> NativeActivatedModel`

M10D stops at the stage boundary. It never writes VN97INV1 and therefore cannot make M10B load a
model by itself.

## VN97CAP1 parser

The Android parser matches the existing M9A wire format:

- exact `VN97CAP1` magic/version/header;
- 96-byte header and 64-byte section entries;
- maximum 64 sections and 512 MiB package size;
- header CRC32;
- SHA-256 over table + payload;
- contiguous section layout with no hidden/trailing bytes;
- per-section SHA-256;
- strict canonical UTF-8 JSON manifest;
- exact manifest/source/section keys;
- unsigned 32-bit capability version;
- supported capability kinds only;
- unique section roles;
- executable/plugin/native-library/script roles rejected.

The parser records package offsets but does not execute or load any section.

## VN97SIG1 parser

The Android runtime now also parses the existing M9B detached envelope:

- schema `VN97SIG1`;
- algorithm `ed25519`;
- canonical publisher key ID;
- exact package SHA-256;
- capability ID/version;
- exactly 64 signature bytes encoded as lowercase hex;
- canonical domain-separated signing message
  `VN97CAP1-SIGNATURE-V1\0 + canonical claims`.

M10D validates envelope shape and package identity binding only. It does **not** treat a syntactically
valid signature as trusted.

## Content-addressed staging

`VN97CapabilityStager` writes into one app-private/non-symlink root:

- package: `<package_sha256>.vn97cap1`;
- signature: `<package_sha256>.<key_id>.vn97sig1`.

Writes are bounded, fsynced, committed with same-directory hard links so an existing digest target
is never silently replaced, then re-opened/reparsed before success. Directory fsync closes the
durability boundary.

Repeated staging of the exact same bytes is idempotent. Existing mismatched digest targets fail
closed.

## Trust boundary

M10D deliberately does not:

- verify Ed25519;
- install publisher keys;
- create/update VN97INV1;
- extract/activate VN97MI1;
- create M6 grants;
- execute package-provided code.

Those are later boundaries. Until trust verification and transactional activation exist on Android,
M10B remains MODEL_REQUIRED.

## Verification

Isolated host gate:

`M10D_ANDROID_CAPABILITY_STAGING_PASS`

The gate covers valid package/envelope parsing, idempotent content-addressed staging, package
tamper rejection, signature/package identity mismatch rejection, noncanonical signature rejection,
forbidden executable-role rejection, and confirms that staging creates no VN97INV1.
