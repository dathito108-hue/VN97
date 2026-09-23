# M10E — Android Publisher Trust + Ed25519 Verification

M10E ports the M9B publisher-authentication boundary into the Android runtime while retaining
VN97's Android API 26 floor.

Android's platform `Signature("Ed25519")` support starts at API 33, so M10E does not raise
`minSdk` and does not depend on a newer-device-only provider. Publisher verification uses a
strict provisioning-only Ed25519 verifier implemented with `BigInteger` and SHA-512.

## Trust store

`VN97TrustedPublisherKey` carries:

- canonical publisher key ID;
- exact 32-byte Ed25519 public key;
- one or more segment-aware capability prefixes;
- allowed capability kinds;
- minimum/maximum unsigned-32 capability version;
- revocation state.

Prefix `model` permits `model` and `model.*`, but not `modelevil.*`.

The trust store is non-empty and rejects duplicate key IDs. Public-key bytes are defensively
copied.

## Strict Ed25519

`VN97Ed25519Verifier` implements RFC 8032 verification without Android's API-33 EdDSA classes:

- canonical compressed-point decoding;
- `S < L` scalar bound;
- public-key identity rejection;
- prime-order subgroup validation;
- SHA-512 challenge reduction;
- extended Edwards-coordinate addition/doubling;
- exact `[S]B = R + [k]A` verification.

This code is used only during capability provisioning; it is not an inference hot path.

## Staged-byte revalidation

`VN97CapabilityTrustVerifier.verify(...)` does not trust M10D in-memory metadata by itself.

It requires an explicit stage root, then:

1. rejects unsafe/non-directory roots;
2. requires exact content-addressed VN97CAP1/VN97SIG1 filenames;
3. rejects symlink/non-regular staged files;
4. reopens and reparses the exact current VN97CAP1 bytes;
5. requires package SHA/manifest identity to still match the staged object;
6. rereads/reparses VN97SIG1;
7. requires package/capability/version binding;
8. applies publisher scope, kind, version and revocation policy;
9. verifies the Ed25519 signature over the canonical domain-separated VN97SIG1 claims.

Only then is `VN97VerifiedCapability` returned.

## Still not activation

M10E still does not write VN97INV1, extract VN97MI1, change M6 policy, or load a model. The next
boundary must perform compatibility planning and transactional activation from a
`VN97VerifiedCapability`.

## Verification

The isolated host gate passes:

`M10E_PUBLISHER_TRUST_PASS`

It covers RFC 8032 vectors 1 and 2, signature tampering, oversized scalar rejection, identity-key
rejection, namespace scoping, allowed kind/version enforcement, revocation, duplicate trust keys,
and public-key defensive copying.

The host run also compiles `CapabilityTrust.kt` against the real M10D package/signature/staging
types under `-Werror`.
