# M9B — Publisher Trust + Typed Compatibility Planning

M9B consumes only packages already staged by M9A. It adds publisher authentication and a
deterministic compatibility decision, but deliberately stops before activation.

## VN97SIG1

VN97SIG1 is a detached canonical UTF-8 JSON envelope with exactly:

- schema `VN97SIG1`;
- algorithm `ed25519`;
- publisher `key_id`;
- full VN97CAP1 `package_sha256`;
- capability ID;
- capability version;
- 64-byte Ed25519 signature encoded as lowercase hex.

The signed message is domain-separated with `VN97CAP1-SIGNATURE-V1\0` and canonical claims.
Changing package digest, capability identity/version, key ID or algorithm invalidates the trust
binding.

## Staged-byte revalidation

Trust is never granted from the in-memory `StagedCapability` metadata alone.

`verify_staged_capability()`:

1. requires the expected content-addressed filename;
2. opens the configured stage root and package with no-follow semantics;
3. requires a bounded regular file;
4. reads the exact current bytes;
5. reparses the bytes through the M9A VN97CAP1 validator;
6. requires package SHA-256 and manifest identity to match the staged object;
7. parses the canonical signature envelope;
8. applies trust-store scope/revocation/version policy;
9. verifies the detached signature.

Only then is `VerifiedCapability` returned.

## Trust store

Each `TrustedPublisherKey` contains a 32-byte Ed25519 public key and explicit scope:

- canonical key ID;
- one or more segment-aware capability prefixes;
- allowed capability kinds;
- minimum/maximum capability versions;
- revocation flag.

A prefix `vision` permits `vision` and `vision.*` but not `visionevil.*`.

The built-in `Ed25519Verifier` lazily imports the crypto backend. Deployments using that adapter
must provide it; absence raises `TrustBackendUnavailable` and fails closed. Tests may supply a
separate verifier implementation through the narrow `SignatureVerifier` protocol.

## Compatibility profile

Trust and compatibility are separate decisions.

`CompatibilityProfile` declares:

- profile ID;
- runtime API version;
- supported capability kinds;
- accepted capability version range;
- one unique `FormatRule` per section role.

The profile has a deterministic SHA-256 fingerprint. The fingerprint is carried into the
compatibility plan so a plan cannot later be reused under silently changed rules with the same
human-readable profile ID.

## Adaptation planning

For each manifest section:

- if its format is accepted directly, the section is `DIRECT`;
- otherwise exactly one `AdapterSpec` must match role + input format and produce an accepted
  output format;
- zero matches fail as incompatible;
- multiple matches fail as ambiguous;
- lossy adapters fail by default and require explicit `allow_lossy=True`.

M9B does not execute adapters. It only produces the typed plan.

## Plan binding

`CompatibilityPlan` records:

- package SHA-256;
- capability ID/version;
- publisher key ID;
- signature-envelope SHA-256;
- profile ID;
- profile SHA-256;
- runtime API version;
- direct/adaptation disposition;
- exact per-section source/target format and adapter ID.

M9C must treat this as an input to transactional revalidation, not as an activation token.

## Authority boundary

M9B does not:

- download packages;
- write outside the existing stage root;
- register an M6 capability handler;
- change M6 policy, approval or lease state;
- load executable/native code;
- mutate live model/runtime weights;
- activate or roll back a capability.

External acquisition remains an M6-authorized side effect. Transactional activation/inventory is
M9C.

## Verification

The M9B regression covers eight groups:

1. signature verification plus staged-byte revalidation;
2. canonical envelope and package binding;
3. trust scope, namespace boundary, revocation and version limits;
4. deterministic direct/adapt planning bound to trust/profile fingerprints;
5. lossy adaptation denied by default;
6. ambiguous and profile-incompatible paths fail closed;
7. bounded/duplicate trust and profile inputs;
8. real Ed25519 adapter verification when the optional crypto backend is available.

The isolated core/test snapshot passes all eight tests and Python bytecode compilation.
