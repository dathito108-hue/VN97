# M19M — Clean-Device Turnkey Acceptance

M19M is the final physical acceptance layer after M19L materializes a signed
`VN97-production.apk` and immutable `VN97FINAL1`.

It answers a different question from M19J:

- M19J proves model latency/memory/thermal/energy evidence on physical phones;
- M19M proves the final signed turnkey APK installs cleanly, bootstraps its bundled
  intelligence, exposes no developer provisioning controls, enables the floating
  assistant under user-equivalent permissions, persists an autonomous goal, and
  recovers across a real device reboot.

## Command

```text
vn97-turnkey-accept \
  --final-receipt /release/VN97FINAL1 \
  --release-dir /release/final \
  --repository-root /work/VN97 \
  --serial <physical-adb-serial> \
  --output /release/device.vn97accept1
```

By default M19M uninstalls the acceptance copy after success. Use
`--keep-installed` only when the accepted APK should remain on the test phone.

## Exact release binding

Before ADB work M19M:

- parses canonical VN97FINAL1;
- requires VN97-production.apk byte count and SHA-256 to match VN97FINAL1;
- parses VN97APK1;
- requires VN97APK1 SHA to match VN97FINAL1;
- requires APK application/version/signer identities in VN97APK1 to match VN97FINAL1;
- obtains the exact embedded bootstrap VN97CAP1 SHA from VN97APK1;
- requires the acceptance checkout HEAD to equal VN97FINAL1.repository_commit;
- requires a clean tracked worktree;
- hash-binds the running M19M/materialization/attestation source to that checkout.

Therefore M19M cannot silently accept a different local APK or source revision.

M19M also requires the M19L release directory to remain the exact four-file canonical
bundle:

```text
VN97-production.apk
bootstrap-release.vn97bootrel6.json
production-readiness.vn97ready1
release-attestation.vn97apk1
```

The release directory itself must be a real non-symlink directory. M19M re-hashes the
VN97READY1 and VN97BOOTREL6 files against VN97FINAL1 and requires VN97APK1 to match the
same release-candidate, bootstrap-report and embedded VN97REL1 identities. Extra,
missing, symlinked or mutated release entries fail before any phone is touched.

When `--output` is supplied, the VN97ACCEPT1 sidecar must stay outside that four-file
release directory. An existing output path is rejected before ADB work, so physical
acceptance is not repeated only to discover an unusable receipt destination at the end.

## Clean-device rule

M19M requires `ai.vn97.app` to be absent before install.

It refuses to overwrite or clear an existing user's VN97 installation.

It also rejects Android QEMU/emulator indicators. The target must be a physical
ADB device.

## No helper/test APK

M19M installs only the exact production APK being accepted.

There is no companion acceptance APK, external model package, side-loaded runtime or
parallel inference backend.

## Permission setup

For deterministic acceptance the ADB shell grants the same permissions/preferences a
user may grant through Android UI:

- RECORD_AUDIO;
- POST_NOTIFICATIONS on Android 13+;
- SYSTEM_ALERT_WINDOW through the Android app-op.

These are product permissions/preferences, not extra software dependencies.

Camera, screen capture, game accessibility and other task-specific permissions remain
user-driven capabilities and are not required for the base turnkey acceptance.

## First-launch acceptance

M19M starts `VN97MainActivity` and waits for the Android UI hierarchy to expose:

- `VN97 native model ready.` or later READY state;
- `Microphone voice enabled`.

It also requires developer-only turnkey-hidden controls to remain absent:

- `IMPORT VN97 MODEL`;
- `Collect VN97 mobile evidence`.

This proves the production APK does not require manual model import to become READY.

## DUMP-protected internal probe

Some acceptance facts are app-private and cannot be read safely from a release APK
using `run-as`.

M19M therefore adds one explicit BroadcastReceiver:

`VN97TurnkeyAcceptanceReceiver`

The receiver:

- has no intent-filter;
- is exported only behind platform permission `android.permission.DUMP`;
- is intended for ADB shell/system acceptance tooling;
- rejects non-turnkey builds;
- accepts only an expected bootstrap package SHA and a bounded hexadecimal nonce;
- does not accept an arbitrary goal or tool request.

Ordinary third-party applications do not hold the platform DUMP permission.

## PRE_REBOOT self-test

After first-launch READY, the protected probe requires:

- `BuildConfig.VN97_TURNKEY_REQUIRED == true`;
- mobile recovery activation ready;
- active inventory package SHA equals the exact VN97CAP1 SHA in VN97APK1;
- the native assistant can open the activated model;
- overlay permission is effective;
- microphone permission is granted;
- notification permission is granted where required;
- floating assistant preference is enabled.

It then creates exactly one hard-coded no-tool autonomous goal containing the M19M
nonce:

```text
M19M acceptance persistence marker <nonce>.
Complete internally only.
Do not use tools, network, device actions, files, trading, game controls,
or any external capability.
```

The goal still goes through the existing canonical autonomous planner/store/scheduler.
M6 authority remains in force if a model unexpectedly proposes an external action.

## Floating assistant proof

After the protected probe enables the floating assistant, the host requires
`dumpsys activity services` to show a live `VN97FloatingAssistantService`.

No acceptance-only overlay service is introduced.

## Real reboot proof

M19M reads `/proc/sys/kernel/random/boot_id`, executes a real `adb reboot`, waits for
`sys.boot_completed=1`, and requires a different kernel boot-id.

Thus process restart or activity recreation cannot be mistaken for reboot acceptance.

## Post-reboot proof

Before manually reopening the activity, M19M requires the existing boot receiver to
restore `VN97FloatingAssistantService` from the persisted user preference.

It then relaunches VN97 and requires the UI to reach READY again.

The DUMP-protected POST_REBOOT probe requires:

- exact same active bundled package SHA;
- assistant opens successfully;
- overlay/microphone/notification permission state remains valid;
- floating preference remains enabled;
- the autonomous goal containing the exact nonce still exists in sovereign storage.

The autonomous goal may be scheduled, waiting, completed, or another canonical state;
the acceptance requirement is durable identity/persistence across reboot rather than a
particular model-generated final response.

## Voice boundary

M19M proves microphone permission and VN97 voice-ingress UI readiness.

It does not claim that a real spoken utterance was transcribed, because repository/ADB
automation has no trustworthy human audio source. Live speech quality remains a
separate physical interaction test when desired.

## VN97ACCEPT1

Only after all checks pass does M19M emit immutable canonical `VN97ACCEPT1`.

It binds:

- VN97FINAL1 SHA-256;
- repository commit;
- VN97APK1 SHA-256;
- exact embedded bootstrap VN97CAP1 SHA-256;
- final APK SHA-256;
- application/version identity;
- physical device manufacturer/model/API/ABI profile;
- hashed pre/post kernel boot IDs proving a distinct reboot;
- PRE_REBOOT and POST_REBOOT self-test SHA-256 identities;
- hashed autonomous nonce;
- complete ordered acceptance check set;
- cleanup state (`UNINSTALLED` or `KEPT_INSTALLED`).

ADB serial numbers are deliberately not persisted in VN97ACCEPT1.

## Cleanup

Default acceptance uninstalls `ai.vn97.app` after success/failure cleanup so the
dedicated test phone returns to a clean app state.

If a successful acceptance cannot perform requested cleanup, no VN97ACCEPT1 is emitted.

## CI boundary

Repository CI cannot create VN97ACCEPT1 because CI does not own a physical rebootable
Android phone or a real VN97FINAL1 production release.

CI can validate:

- Android receiver compilation and manifest permission;
- receiver source security contract;
- strict VN97ACCEPT1/VN97SELFTEST1 parsing;
- exact four-file M19L release-bundle binding and tamper rejection;
- VN97ACCEPT1 sidecar placement outside the release bundle;
- complete host orchestration with fake ADB;
- clean install ordering;
- emulator rejection before install;
- permission/app-op flow;
- UI READY parsing;
- protected broadcast flow;
- service check;
- distinct reboot observation;
- post-reboot flow;
- cleanup.

A real VN97ACCEPT1 exists only after the command runs on physical hardware.

## Architecture boundary

M19M adds no model, planner, memory engine, trainer, inference backend, benchmark
runtime, model import requirement or APK builder.

The final architecture remains:

```text
Code 1
  -> Code 2 hardware/mobile-aware
  -> one VN97 production model
  -> VN97FINAL1 materialized turnkey APK
  -> VN97ACCEPT1 physical clean-device acceptance
```