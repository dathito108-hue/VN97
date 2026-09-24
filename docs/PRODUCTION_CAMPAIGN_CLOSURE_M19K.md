# M19K — Production Campaign Closure and Release Handoff

M19K turns the M19 production chain into one resumable coordinator without replacing
any canonical trainer, evidence collector, intake gate or release-readiness gate.

## Canonical command

```text
vn97-production-close \
  --manifest /production/vn97/production-run.vn97run1 \
  --workspace-root /production/vn97 \
  --repository-root /work/VN97
```

The command inspects the exact existing artifact state and advances only the missing
canonical steps that are safe to run automatically.

## State machine

`VN97CLOSE1` reports one of:

- `BLOCKED_ENVIRONMENT` — current Python/Torch/platform does not match VN97RUN1;
- `NEEDS_TRAINING` — language/production campaign artifacts are incomplete;
- `NEEDS_PHYSICAL_EVIDENCE` — VN97PRODCAMP1 exists but real VN97MOBEVID1 is absent;
- `NEEDS_INTAKE` — physical evidence exists but M19F intake has not been materialized;
- `NEEDS_RELEASE_INPUTS` — fresh held-out release validation/metadata/output inputs are missing;
- `NEEDS_SIGNING` — remaining M19E blockers are signing credentials/keystore;
- `READINESS_BLOCKED` — M19E reports another real readiness/toolchain blocker;
- `READY_TO_RELEASE` — canonical VN97READY1 is READY.

Exit code is `0` only for `READY_TO_RELEASE`. Expected dependency states return `2`.

## Resume behavior

M19K never deletes production outputs and never blindly reruns completed work.

If neither language nor production output exists it invokes:

```text
vn97-production-run --stage train
```

If language exists but production does not, it invokes the canonical production stage
only. Existing outputs are re-opened through the M19F inspectors and must verify.

An impossible partial state such as production-without-language or intake whose hashes
do not match current campaigns/evidence is a hard failure, not a resume opportunity.

## Physical evidence handoff

After VN97PRODCAMP1, M19K checks the canonical `device-evidence/` slot.

If it is empty and no `--serial` is supplied, M19K returns
`NEEDS_PHYSICAL_EVIDENCE` without pretending the campaign is complete.

When explicit physical serials are supplied, M19K calls the existing M19J coordinator.
All M19J guarantees remain in force: exact Git/source/APK/model binding, emulator
rejection, clean app-state isolation, signed ephemeral provisioning, real Android
collector measurements, M19F policy checks, atomic evidence publication and uninstall.

## Intake handoff

Once physical evidence exists, M19K invokes:

```text
vn97-production-run --stage intake
```

only if intake does not already exist.

Existing/new intake is re-opened and required to bind exactly to:

- current VN97CAMP2 report SHA;
- current VN97PRODCAMP1 report SHA;
- sorted current VN97MOBEVID1 SHA set;
- VN97RC1 manifest SHA;
- exact production model-image SHA.

## Release readiness

After VN97INTAKE1 + VN97RC1, M19K evaluates the existing M19E
`evaluate_production_readiness()` gate and therefore emits the same canonical
`VN97READY1` semantics used by `vn97-production-release`.

Optional M19K release-readiness inputs include:

```text
--private-key
--key-id
--capability-version
--source-origin
--source-license
--validation-input   # repeatable fresh held-out language validation
--speech-validation-input
--vision-validation-input
--max-validation-loss
--max-speech-validation-loss
--max-vision-validation-loss
--release-output-dir
--apksigner
--aapt
```

M19K intentionally does not substitute VN97RUN1 training/validation/release datasets for
these fresh release-readiness inputs. Missing values remain visible M19E blockers.

Android release-signing variables also remain external environment secrets:

```text
VN97_RELEASE_KEYSTORE
VN97_RELEASE_STORE_PASSWORD
VN97_RELEASE_KEY_ALIAS
VN97_RELEASE_KEY_PASSWORD
```

No secret is written into VN97RUN1 or VN97CLOSE1.

## Inspect-only mode

```text
vn97-production-close ... --inspect-only
```

does not launch training, M19J or intake. It validates existing artifacts and reports
the next phase. If the chain already reaches VN97RC1, M19E readiness evaluation still
runs because it is a preflight/status operation rather than model training.

## Canonical status report

`--closure-report <path>` atomically writes/updates canonical `VN97CLOSE1`.

It binds:

- VN97RUN1 manifest SHA-256;
- exact repository commit;
- environment-ready bit;
- VN97CAMP2 SHA;
- VN97PRODCAMP1 SHA;
- sorted VN97MOBEVID1 evidence SHA set;
- VN97INTAKE1 SHA;
- VN97RC1 manifest SHA;
- VN97READY1 SHA/status/blocker codes;
- current closure phase.

`--readiness-report <path>` atomically writes the current canonical VN97READY1 after
intake exists. Both persisted reports are parsed back and compared after write.

## Source binding

M19K first reuses the M19G repository verifier, requiring the checkout HEAD to equal
`VN97RUN1.repository_commit` and the tracked worktree to be clean. It additionally
hash-binds the running `production_closure.py` to that checkout before orchestration.

Every delegated stage retains its own canonical source/input checks.

## Honest external boundaries

M19K does not make physical devices or signing credentials appear from CI.

A normal first invocation after training can legitimately stop at:

```text
NEEDS_PHYSICAL_EVIDENCE
```

and after M19J/M19F can legitimately stop at:

```text
NEEDS_RELEASE_INPUTS
NEEDS_SIGNING
```

These are explicit production dependencies, not failed AGI/model implementation.

## Architecture boundary

M19K introduces no model, trainer, planner, memory system, benchmark runtime or
alternate release path. It is orchestration over the already locked chain:

```text
Code 1
  -> Code 2 hardware/mobile-aware
  -> one VN97 production model
  -> M19G/M19I training
  -> M19J real device evidence
  -> M19F intake
  -> M19E VN97READY1
```

## M19L final materialization

When M19K reaches `READY_TO_RELEASE`, the canonical next step is:

```text
vn97-production-materialize ...
```

M19L re-runs M19K in inspect-only mode with the same release inputs, delegates the actual signed APK build to `vn97-production-release`, independently re-verifies VN97READY1/VN97APK1/VN97BOOTREL6/APK/signature/package identities, and emits immutable VN97FINAL1 only after successful materialization.
