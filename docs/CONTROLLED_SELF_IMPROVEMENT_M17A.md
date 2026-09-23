# M17A — Controlled Self-Improvement Candidate Foundation

M17A begins Controlled Self-Improvement without adding a second model, a dynamic
code loader, or an autonomous update bypass.

The existing canonical deployment path remains authoritative:

```text
VN97CK1
  -> VN97MI1
  -> signed VN97CAP1 + VN97SIG1
  -> publisher trust
  -> compatibility plan
  -> transactional VN97INV1 activation
```

M17A adds a durable candidate layer **before** activation.

## Candidate definition

A controlled improvement candidate is still exactly one signed canonical
`model.language` package reviewed by the existing production provisioner.

The candidate is bound to the exact active baseline from VN97INV1:

- baseline activation ID;
- baseline VN97MI1 artifact SHA-256;
- baseline package SHA-256;
- baseline capability version.

It also binds the reviewed candidate:

- candidate package SHA-256;
- candidate capability version;
- publisher key ID;
- publisher public-key SHA-256;
- compatibility-plan SHA-256;
- signed source origin/license;
- bounded human improvement objective.

Candidate identity is deterministic SHA-256 over canonical
`VN97IMPIDENT1` JSON.

## Strict upgrade rule

M17A accepts only:

`candidate_version > baseline_version`

Same-version replacement and downgrade are rejected before a candidate record is
created.

The candidate package must differ from the baseline package.

These rules are stricter than generic provisioning because a self-improvement
candidate is not a manual rollback/recovery mechanism.

## VN97IMP1 ledger

Reviewed candidates are persisted app-private as:

`<candidate_id>.vn97imp1`

VN97IMP1 uses:

- strict UTF-8;
- canonical strict JSON;
- SHA-256 payload header;
- bounded fields;
- non-symlink storage;
- file fsync;
- atomic move;
- directory fsync;
- immediate post-write verification.

M17A states are intentionally minimal:

- `REVIEWED`;
- `REJECTED`.

M17A cannot mark a candidate evaluated, eligible, or promoted. Those transitions
do not exist yet.

One candidate package may have at most one active REVIEWED improvement identity.
A conflicting objective for the same reviewed package fails closed.

## Durable no-bypass gate

This is the key M17A invariant.

If a package SHA-256 has a durable REVIEWED VN97IMP1 record, the ordinary
`VN97AppProvisioner.activateReviewed()` path refuses to activate it.

The gate consults the durable ledger by candidate package SHA-256, not an
in-memory flag. Therefore:

- process death does not remove the gate;
- app restart does not remove the gate;
- re-reviewing the same package through normal provisioning does not bypass the
  gate.

Only later M17 promotion code may receive a dedicated controlled activation path
after evaluation gates pass.

Rejecting the candidate transitions VN97IMP1 to REJECTED and clears the pending
provisioning review.

## Android surface

The VN97 main screen adds **Controlled self-improvement**.

The private non-exported activity requires:

1. improvement objective;
2. candidate VN97CAP1;
3. VN97SIG1;
4. independent Ed25519 publisher key.

Review displays:

- candidate ID;
- baseline version and artifact SHA-256;
- candidate version and package SHA-256;
- publisher identity;
- compatibility plan SHA-256;
- state.

There is deliberately no Promote/Activate button in M17A.

The only terminal UI action in this milestone is **Reject candidate**.

## Authority boundary

M17A does not:

- train weights on Android;
- change active model weights;
- activate a reviewed candidate;
- modify source code;
- load package-provided executable code;
- create M6 grants;
- change the planner;
- change VN97MEM1;
- add another AI/model backend;
- let VN97 cognition decide its own promotion.

Existing signing, publisher trust, compatibility, native VN97MI1 validation,
transactional activation and rollback remain unchanged.

## Regression coverage

The isolated M17A host contract verifies:

- deterministic candidate identity;
- strict version upgrade requirement;
- baseline/candidate package separation;
- reviewed save/load;
- idempotent same candidate review;
- one active reviewed identity per package;
- durable package lookup used by no-bypass logic;
- explicit rejection;
- immutable rejection reason;
- ledger tamper rejection.

APK compilation validates the production integration:

- canonical VN97INV1 baseline lookup;
- provisioning review binding;
- durable normal-activation block;
- private Android review UI.

## Next

M17B — Held-Out Candidate Evaluation.

M17B should open the candidate through a non-active evaluation path, compare it
against the exact bound baseline on fixed held-out tasks and mobile resource
limits, then write a signed/durable evaluation result. Evaluation must not make
the candidate active.

Only a later M17 promotion milestone may use the existing transactional
activation coordinator after all gates and explicit user approval pass.
