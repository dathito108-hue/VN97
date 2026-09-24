# M19J — Physical Device Evidence Campaign Orchestrator

M19J automates the real-phone evidence step between VN97PRODCAMP1 and M19F intake.
It does not simulate or manufacture device evidence.

## Canonical flow

```text
VN97RUN1
  -> verify
  -> M19I preflight
  -> train
  -> VN97CAMP2 + VN97PRODCAMP1
  -> M19J physical-device evidence campaign
  -> VN97MOBEVID1 files
  -> M19F intake
  -> VN97INTAKE1 + VN97RC1
  -> VN97READY1
  -> signed production APK
```

## Command

```text
vn97-device-evidence-campaign \
  --manifest /production/vn97/production-run.vn97run1 \
  --workspace-root /production/vn97 \
  --repository-root /work/VN97 \
  --serial <physical-adb-serial-1> \
  --serial <physical-adb-serial-2>
```

Every target phone must be explicit. M19J never selects all connected devices automatically.

## Host prerequisites

- M19H-bound Python/Torch environment;
- the VN97 signing extra with cryptography, used only for an ephemeral evidence key;
- JDK, Android SDK and Gradle compatible with the repository;
- adb;
- a clean VN97 checkout whose HEAD equals VN97RUN1.repository_commit.

## Source and APK binding

M19J requires the bound Git commit and a clean tracked worktree. It verifies its own
production/evidence source files against that checkout, then builds the debug harness
from the same checkout:

```text
gradle -p android --no-daemon --stacktrace :app:assembleDebug
```

The canonical CLI does not accept an arbitrary prebuilt APK.

## Exact production model identity

M19J opens VN97CAMP2 and VN97PRODCAMP1 through the existing M19F inspectors, rebuilds
the VN97MI1 using the exact checkpoint, tokenizer and deployment tile geometry, and
requires:

```text
rebuilt VN97MI1 SHA-256 == VN97PRODCAMP1 model_image_sha256
```

before any phone is touched.

## Evidence-only signed provisioning

M19J never enables unsigned/raw model activation. It creates an ephemeral Ed25519 key
in host memory and signs a temporary VN97CAP1 containing the exact production VN97MI1.
The debug app reviews and activates that package through the existing
VN97ModelImageProvisioningSession. The ephemeral private key is never written to the
repository or device.

## Clean app-state isolation

M19J refuses any target where ai.vn97.app is already installed. For each serial it:

1. requires ADB state device;
2. rejects Android emulator indicators;
3. installs the bound debug APK;
4. writes model.vn97cap1, model.vn97sig1 and publisher.ed25519 into app-private staging via run-as;
5. triggers the explicit M19J action on VN97MainActivity;
6. waits for canonical VN97MOBEVID1;
7. parses and gates the evidence on the host;
8. uninstalls ai.vn97.app in finally.

Uninstall removes the temporary inventory and ephemeral publisher trust. A successful
benchmark followed by failed uninstall is treated as campaign failure.

## Android action boundary

M19J adds no exported component. It reuses VN97MainActivity, already the launcher
activity. The explicit action is:

```text
ai.vn97.app.action.M19J_COLLECT_MOBILE_EVIDENCE
```

and is disabled whenever BuildConfig.VN97_TURNKEY_REQUIRED is true. Therefore the
turnkey/release APK does not expose the M19J developer evidence path.

## Existing collector only

The action reuses:

```text
VN97AppProvisioner
  -> VN97ModelImageProvisioningSession
  -> NativeActivatedInventoryModelLoader
  -> VN97AppAssistant.collectMobileEvidence()
  -> VN97OnDeviceEvidenceCollector
```

No second benchmark runtime is added. The output remains exactly VN97MOBEVID1 with
text prefill, decode/token, speech prefill, PSS, thermal, energy counter where
available, device profile and exact model-image SHA-256.

## Benchmark parameters

Defaults:

```text
warmup runs   = 1
measured runs = max(5, VN97RUN1 min_device_runs)
decode tokens = 16
speech frames = 8
timeout       = 300 seconds/device
```

Optional flags are --warmup-runs, --measured-runs, --decode-tokens, --speech-frames
and --timeout-seconds. Measured runs may not be lower than the M19F policy.

## Existing M19F policy is reused

M19J derives VN97DeviceEvidenceCriteria from VN97RUN1 intake.options and immediately
applies require_device_evidence(). Wrong model identity, insufficient runs, latency,
PSS, thermal, speech or energy-policy failures stop the campaign before publication.
M19F intake repeats the gate later; M19J is fail-early orchestration, not a replacement.

## Physical-device and profile rules

M19J checks ro.kernel.qemu and ro.boot.qemu and rejects QEMU/emulator targets. After
all requested phones pass, the collection must also satisfy the M19F distinct profile
count using manufacturer, model, sdk_int and abi.

ADB serial numbers are not persisted in VN97MOBEVID1 and are not release evidence.

## Atomic workspace publication

M19H creates an empty device-evidence directory. M19J requires it to remain empty.
All phones are first collected into a separate staging directory. Only after every
record passes and the distinct-profile requirement is met does M19J replace the empty
evidence slot. Evidence filenames use the evidence digest prefix, not device serial.

A failed campaign leaves the production evidence slot unpopulated; the final directory
replacement has a rollback handle for rename failure.

## Manual M11C fallback

The existing developer UI button Collect VN97 mobile evidence remains available for
individual diagnostics. M19J is the production path because it additionally binds the
VN97RUN1 commit, freshly built debug APK, VN97PRODCAMP1 model identity, M19F policy,
multiple explicit physical phones, atomic publication and harness cleanup.

## Honest boundary

Repository CI cannot create real VN97MOBEVID1 because it has no user-owned physical
phones. CI can only validate Android compilation and host orchestration behavior with
a fake ADB executor. Real evidence exists only after this command runs against real
connected Android hardware.

## Architecture boundary

M19J introduces no model, planner, memory engine, inference backend or alternate
release path. It only automates the existing physical evidence machinery required by
M19F.