# M6B — Production Capability Pack

M6B implements concrete external capabilities on top of the M6A authority contract. It does not
change planner/cognition semantics and it does not create a second execution path.

## Registered capabilities

The pack can register:

- `file.read` with scope `root + path` and empty payload;
- `file.write` with scope `root + path` and payload `text` plus optional
  `expected_sha256`;
- `web.fetch` with exact `url` scope and empty payload;
- `app.launch` with exact `package` scope;
- `device.clipboard.write` with `channel=system` and bounded `text` payload.

All descriptors require M6A approval by default. Exact policy grants and leases remain mandatory.

## Confined files

`ConfinedFileStore` receives trusted root IDs from the host runtime. Model/cognition requests
select a root ID and a canonical relative POSIX path; they never supply an unrestricted host
filesystem path.

On Linux/Android:

- the root must be an existing non-symlink absolute directory;
- every parent component is opened relative to the previous directory FD;
- `O_NOFOLLOW` and `O_DIRECTORY` prevent symlink traversal;
- the final read/write target uses no-follow semantics;
- reads accept only regular files, enforce per-root byte budgets and strict UTF-8;
- create mode is selected by omitting `expected_sha256` and cannot overwrite an existing file;
- replacement requires the exact lowercase SHA-256 of the current bounded file;
- VN97 writers coordinate with an exclusive directory lock;
- content is written to a 0600 temporary file, fsynced, then atomically published;
- the parent directory is fsynced after publication.

The SHA precondition protects against stale VN97 writes. It is not a general multi-process
transaction primitive for hostile writers outside the trusted app sandbox.

## Sovereign HTTPS fetch

`SovereignHttpsFetcher` is intentionally narrow:

- HTTPS only;
- port 443 only;
- GET only;
- no URL credentials;
- no fragments;
- no redirect following;
- no proxy tunnel or custom authorization header surface;
- bounded response bytes;
- successful 2xx responses only;
- text or JSON content only;
- UTF-8/ASCII decoding only.

Before transport, the host is resolved. Every returned address must be a globally routable IP;
loopback, private, link-local, multicast, unspecified and reserved/non-global targets fail
closed. One validated address is pinned for the socket connection, while the original hostname
is retained for TLS SNI and certificate verification. This prevents a second DNS lookup from
silently changing the validated destination.

The response returned to the planner is canonical JSON containing status, URL, media type,
strict text body, SHA-256 and byte count.

## App/device adapter

`AppDeviceAdapter` is a platform boundary rather than an Android implementation. M6B exposes
only:

- exact Android-style package launch;
- bounded system clipboard text write.

Package names are ASCII and structurally validated. There is no arbitrary shell command,
implicit intent URI, accessibility action or unrestricted device-control primitive in M6B.

M7 will supply the concrete Android implementation and OS permission/lifecycle integration while
preserving the same M6A authority path.

## Assembly

`register_m6b_capabilities(registry, CapabilityPackConfig(...))` registers only the
implementations supplied by the host. The registry is then sealed as required by M6A before an
`ExternalExecutionFabric` is constructed.

Capability handlers receive only an already authorized `AuthorizedAction`. Cognition does not
receive handler objects, registry mutation, policy grants, approval secrets or lease state.

## Replay and audit

M6B relies on the M6A execution ordering:

1. validate waiting planner step and typed capability request;
2. enforce policy, approval and lease;
3. execute the M6B handler;
4. persist a successful immutable receipt;
5. deliver the result to `PlanController.record_external_result()`.

If the process dies after the external effect but before the planner checkpoint advances, the
same exact request digest replays the M6A success receipt and does not invoke the M6B handler a
second time.

## Local verification scope

M6B tests cover file confinement, traversal/symlink rejection, create-no-overwrite, bounded
compare-and-swap replacement, HTTPS scheme/credential/private-IP rejection, redirect/binary
rejection, validated-IP transport pinning, exact typed registry schemas, Android package
validation, clipboard bounds and registry sealing.

The repository integration regression additionally binds `file.write` through the canonical
M6A gate/audit and M5 `WAITING_EXTERNAL` planner lifecycle, including exact-success replay.
