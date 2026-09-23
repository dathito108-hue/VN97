# M16B — Android Capability Acquisition Review Surface

M16B exposes the M16A signed, data-only knowledge acquisition path to the
installable Android app without widening model, tool, network, or device
authority.

## User flow

The VN97 main screen now has a dedicated **Capability acquisition** entry point.

It opens a private, non-exported, scrollable
`VN97CapabilityAcquisitionActivity` with three Android document pickers:

1. VN97CAP1 package;
2. VN97SIG1 signature envelope;
3. independently supplied Ed25519 publisher public key.

The user then performs two distinct actions:

```text
Select artifacts
  -> Review signed knowledge
  -> inspect exact identity/provenance/hashes
  -> Trust & Acquire
  -> existing VN97MEM1
```

Review and acquisition are intentionally separate.

## Review surface

Review shows:

- capability ID and version;
- record count;
- package SHA-256;
- signature-envelope SHA-256;
- knowledge payload SHA-256;
- publisher key ID;
- publisher public-key SHA-256 fingerprint;
- whether that exact publisher was already trusted;
- source origin;
- source license;
- accepted scope: signed VN97KN1 knowledge data only.

The screen explicitly states:

- review does not mutate VN97MEM1;
- imported data is evidence only;
- no executable code/plugin/script/native library is accepted;
- no M6 tool/device authority is created.

## Binding to canonical VN97 resources

M16B does not open a second model or a second memory database.

`VN97AppAssistant` owns the active production model and
`VN97ProductionAssistantResources`. M16B creates the M16A acquisition session
from exactly:

- that same active `NativeActivatedModel`;
- that same open canonical `VN97MEM1`;
- that same VN97 cognition embedding engine.

The acquisition session is invalidated whenever the assistant resources are
closed/reloaded.

Review/acquisition fail closed while:

- an assistant turn is active; or
- an M6 approval is pending.

This prevents capability import from racing the canonical foreground cognition
lifecycle.

## Android document boundary

M16B uses `ACTION_OPEN_DOCUMENT` with read-only URI grants. It requests a
persistable read permission when supported by the selected document provider.

The app does not request broad external-storage permission.

The app-side adapter bounds:

- VN97SIG1 to 16 KiB;
- publisher key document to 256 bytes;
- publisher key format to either exactly 32 raw Ed25519 bytes or exactly 64
  lowercase hexadecimal characters.

The VN97CAP1 package itself is streamed into the existing M9/M10
content-addressed stager, which applies the existing package-size, section,
integrity and data-only rules.

## Explicit Trust & Acquire

Only **Trust & Acquire** calls the M16A acquisition phase.

That phase still performs all authoritative checks:

- enroll exact knowledge-only publisher trust;
- re-read and re-verify the staged VN97CAP1/VN97SIG1;
- re-parse VN97KN1;
- compare against the exact reviewed payload;
- resume/create VN97KAL1;
- embed with the same activated VN97;
- append semantic `VN97KNMEM1` records into the same VN97MEM1;
- reconcile a crash-left pending append without blind duplication.

The Android screen cannot bypass those runtime checks.

## No autonomous download yet

M16B deliberately does not add a URL field or direct network downloader.

A user may select a file that already exists on the device through Android's
document provider, but M16B itself does not fetch remote capability packages.

Future governed remote acquisition must go through an existing M6-authorized
external-effect path, then hand the resulting exact bytes into the same M16A
review/trust boundary. Network access must never imply publisher trust.

## Scope

M16B adds only a user-facing acquisition surface. It does not:

- create executable plugins;
- dynamically register M6 handlers;
- run scripts or native libraries from VN97CAP1;
- replace the VN97 model;
- add a parallel planner;
- add a second memory engine;
- grant new device/app/network permissions;
- autonomously trust a publisher.

## Validation

APK compilation is the primary integration gate because M16B is an Android UI
and document-provider milestone.

The underlying trust, data parsing, crash recovery and VN97MEM1 idempotency
remain covered by the M16A host contract.

## Next

M16C should add **governed remote capability artifact acquisition** through the
existing M6 authority/audit path, with exact URL/source scope, bounded download,
content digest receipt, and mandatory handoff into the same explicit
Review -> Trust & Acquire boundary. Remote retrieval must not activate or trust
anything by itself.
