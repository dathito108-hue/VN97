# M16C — Governed Remote Capability Artifact Acquisition

M16C adds a remote retrieval path for signed VN97 knowledge packages while
preserving the M6 deny-by-default authority boundary and the M16A/M16B explicit
publisher-trust boundary.

## Separation of authority

Three events remain deliberately separate:

1. **M6-approved download** — obtains exact bytes and a durable action receipt.
2. **Review signed knowledge** — validates VN97CAP1/VN97SIG1/publisher identity
   and displays provenance.
3. **Trust & Acquire** — explicitly enrolls the publisher and writes bounded
   semantic evidence into the existing VN97MEM1.

A successful download never means the publisher is trusted, the package is
activated, or its content has entered memory.

## M6 capability

M16C adds one typed production capability:

`capability.artifact.fetch`

Contract:

- exact required scope: `url`;
- canonical HTTPS URL only;
- payload exactly `{}`;
- approval required;
- one-use lease;
- maximum lease lifetime 60 seconds.

The grant is created only for the exact canonical URL the user prepared.

The capability is intentionally excluded from the normal assistant external
capability catalog. It is available only through the user-driven M16C fetch
coordinator, so ordinary cognition cannot silently invent a remote capability
download.

## Explicit approval flow

The Capability Acquisition screen now supports:

```text
exact HTTPS URL
  -> Prepare M6 fetch approval
  -> show exact URL + request digest
  -> Approve & Fetch / Reject
  -> durable M6 receipt
  -> app-private content-addressed artifact
  -> select VN97SIG1 + publisher key
  -> Review signed knowledge
  -> Trust & Acquire
```

Rejecting the approval performs no network fetch.

A process-lifetime pending approval contains no downloaded bytes and no trust
mutation. The downloaded artifact itself is durable and content-addressed after
a successful approved fetch.

## Network confinement

`VN97PinnedHttpsCapabilityTransport` uses a narrow HTTP/1.1 GET transport:

- HTTPS only;
- port 443 only;
- no URL credentials;
- no fragments;
- no redirects because only status 200 is accepted;
- no proxy path: VN97 opens the TCP socket directly;
- DNS is resolved before connection;
- **every** resolved address must be globally routable;
- loopback, private, link-local, multicast, CGNAT, documentation and other
  reserved ranges are rejected;
- one already-validated IP is selected and the socket connects to that exact
  address;
- TLS is layered over that connected socket;
- original hostname is retained for certificate endpoint identification;
- original hostname is used for TLS SNI when it is a DNS name;
- response compression is rejected;
- only identity or bounded chunked transfer framing is accepted;
- response headers and header count are bounded;
- chunk extensions and trailers are rejected;
- default body limit is 16 MiB;
- connect/read timeouts are bounded.

This avoids a second DNS lookup between address validation and TCP connection.

## Artifact validation and storage

Downloaded bytes must parse as a valid `VN97CAP1` package under the existing
binary integrity rules.

M16C additionally requires the untrusted manifest to declare:

- `kind=knowledge`;
- capability namespace `knowledge` or `knowledge.*`.

That manifest information is still untrusted metadata until M16A verifies the
independent VN97SIG1 and publisher public key.

The package is stored under app-private no-backup storage as:

`<package_sha256>.vn97cap1`

Publication is create-only/content-addressed, file contents are fsynced, the
directory is fsynced, and an existing digest target is parsed and revalidated
before reuse.

## M6 audit and replay

The remote coordinator constructs a synthetic one-step canonical
`NativePlanController` whose only step is the exact external fetch.

Execution then uses the existing:

- `M6CapabilityDescriptor`;
- exact scope digest;
- `M6PolicyGrant`;
- Android approval HMAC;
- one-use lease;
- `M6ExternalExecutionFabric`;
- durable `M6DurableActionAudit`.

The action receipt contains the deterministic artifact result:

`VN97ARTFETCH1|sha256|bytes|capability_id|version|kind`

The content-addressed file is the authoritative local byte identity; the receipt
does not create trust or activation authority.

## Android handoff

After an approved fetch, the Capability Acquisition screen records the returned
package SHA-256 and treats the app-private file as the package source.

The user must still independently select:

- VN97SIG1;
- publisher Ed25519 public key.

The existing M16A review path then reopens the fetched artifact by SHA-256 and
performs all normal staging, signature verification, VN97KN1 validation and
Review -> Trust & Acquire checks.

## No hidden expansion

M16C does not add:

- arbitrary web browsing;
- arbitrary file download;
- redirects;
- HTTP;
- private-network access;
- automatic signature discovery;
- automatic publisher enrollment;
- automatic acquisition after download;
- executable plugins/scripts/native libraries;
- dynamic M6 handler registration by imported data;
- model replacement;
- a second planner/model/memory engine.

## Regression coverage

The isolated M16C host contract verifies:

- canonical HTTPS URL normalization;
- cleartext/credential/fragment/non-443 rejection;
- private/reserved address rejection;
- public-address acceptance;
- bounded transport policy propagation;
- valid remote knowledge VN97CAP1 parsing;
- content-addressed persistence and reread;
- malformed package rejection;
- non-knowledge package rejection;
- deterministic artifact receipt serialization.

APK compilation remains the integration gate for:

- the new M6 production descriptor/handler;
- Android app-private artifact store;
- synthetic one-step M6 coordinator;
- Android approval handoff;
- Activity approval controls;
- handoff back into the existing M16A review path.

## Next

M16D should add a **bounded acquisition proposal layer**: canonical VN97 may
identify that missing knowledge would help a user goal and prepare a candidate
source request, but user approval, exact M6 URL scope, publisher review and
Trust & Acquire remain mandatory. Acquired data must remain evidence-only.
