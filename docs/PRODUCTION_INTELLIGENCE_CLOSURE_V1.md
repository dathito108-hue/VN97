# VN97 Production Intelligence Closure v1

This document freezes the next VN97 program phase after M19M.

The objective is no longer to expand the architecture. The objective is to turn the
existing canonical VN97 stack into a real, installable, release-qualified mobile
assistant with measured production intelligence.

Canonical architecture remains:

`Code 1 -> Code 2 hardware/mobile-aware -> VN97 production`

There is still one VN97 model, one canonical planner/cognition path, one VN97MEM1
memory system and one M6 authority fabric.

## Release target

VN97 1.0 is considered usable only when one release APK can be installed on a clean
Android device and can automatically activate its bundled signed VN97 model without
requiring Python, Termux, a manual model import, a second AI backend or another VN97
runtime.

The first usable 1.0 release is language-first. It must preserve the already-built
assistant, planner, memory, authority, Android lifecycle and floating-assistant
architecture. Speech/vision production weights may be promoted in later 1.x releases
if they are not yet release-qualified; missing multimodal quality must not block a
working native language release.

A production closure is complete only after the real release chain yields:

`VN97MOBEVID1 -> VN97FINAL1 -> VN97ACCEPT1`

from physical-device evidence and a clean-device acceptance run.

## P0 — Architecture and release freeze

From this point until VN97 1.0 acceptance:

- no Transformer/LLaMA/cloud inference fallback;
- no second model, planner, memory engine or authority system;
- no feature milestone may bypass production-intelligence work;
- allowed changes are training/data quality, evaluation, mobile performance,
  correctness, security, release hardening and defects discovered during those
  activities.

The production intelligence profile ID is:

`vn97-production-intelligence-v1`

The final model geometry is selected by the existing VN97 campaign and sealed-release
gates. It is not manually declared a winner before measurement.

The first CPU pilot keeps the existing conservative trainer defaults:

- d_model 128;
- 4 layers;
- d_state 8;
- sequence length 256;
- batch size 4;
- one training epoch by default.

That pilot is a pipeline proof, not a production-quality claim.

## P1 — Production Corpus v1

External/public datasets may be used only after local acquisition and explicit
provenance/license review. Availability on the internet is not sufficient evidence of
training rights.

The corpus workflow is:

`local source files
-> VN97CORPUSDEF1
-> explicit license_approved=true per source
-> strict JSONL parsing
-> exact-record fingerprinting
-> cross-split leakage rejection
-> deterministic within-split exact deduplication
-> canonical training/validation/release JSONL
-> VN97CORPUS1 manifest`

`vn97-corpus-seal` is the canonical sealing command.

Every source definition binds:

- stable source ID;
- source origin;
- declared license;
- explicit local license approval;
- dataset mode;
- target split;
- local source path.

A split may use text or chat mode, but all sources feeding one split must use the same
mode. Exact duplicates inside one split are deterministically collapsed. An exact
record that appears in two different splits is a hard failure.

The output set is:

- `training.jsonl`;
- `validation.jsonl`;
- `release.jsonl`;
- `corpus.vn97corpus1.json`.

VN97CORPUS1 binds each source's original SHA-256/byte count, input/kept/duplicate
record counts, the canonical split hashes and a deterministic manifest identity.

The sealed release split is evaluation-only. It must not be used for tokenizer
learning, candidate ranking or iterative model selection.

## P2 — CPU pilot

Run the smallest canonical VN97 training path first.

The pilot must prove:

`corpus -> VN97TK1 -> gradient updates -> decreasing/finite training loss
-> VN97CK1 -> reload -> inference -> native/export compatibility`.

Failure here is fixed in data/trainer/runtime before any rented GPU campaign is
started.

## P3 — Language production campaign

Only after P2 passes, run a bounded multi-candidate `vn97-campaign` on rented GPU
compute.

Candidates remain the same VN97 architecture and vary only bounded geometry/training
configuration. Admission is governed by:

- parameter budget;
- VN97MI1 mobile-image budget;
- recurrent-state budget;
- validation loss/accuracy;
- deterministic candidate ranking;
- sealed release holdout.

A release-holdout failure terminates the campaign. It never causes fallback to the
second-best validation candidate.

## P4 — Intelligence gate

The selected winner is evaluated against production tasks representing the actual
assistant:

- Vietnamese and English instruction following;
- bounded reasoning and planning;
- memory retrieval/use;
- strict structured cognition;
- typed external-intent generation;
- replanning after bounded failure;
- authority-boundary behavior.

Weak intelligence is addressed primarily through corpus/training improvements rather
than adding a second reasoning architecture.

## P5 — Native production package

The exact accepted language winner follows the existing canonical path:

`VN97CK1 + VN97TK1
-> VN97MI1
-> signed VN97CAP1 + VN97SIG1
-> trusted activation`.

The resulting model identity is immutable throughout mobile evidence and final release.

## P6 — First usable turnkey APK

The existing M10S/M19 path packages the signed production model directly into the
release APK.

The expected user path is:

`install APK -> launch -> verify bundled publisher/package -> transactional activation
-> open VN97MEM1 -> native VN97 assistant ready`.

No manual model provisioning is part of the release UX.

## P7 — Physical production acceptance

A real Android device is required.

The device campaign measures the exact model image and records mobile latency,
memory/thermal observations and other available M19 evidence. The final signed APK is
then installed onto a clean device and taken through the M19M install/launch/reboot/
recovery/self-test lifecycle.

VN97 1.0 is not declared production-complete until a real VN97ACCEPT1 exists.

## P8 — Multimodal 1.x upgrades

After the language-first APK is usable, train and qualify the already-defined speech
and visual adapters against the same VN97 recurrent core.

No Whisper, external STT, VLM or second inference backend is introduced.

Each upgraded model is treated as a new exact candidate and must pass the applicable
quality/mobile/release gates before promotion.

## P9 — Field capability improvement

Game-agent quality, long-horizon autonomy, paper-trading reasoning and M16 knowledge
acquisition are improved using field evidence and the existing M17 controlled
self-improvement path.

M17 promotion remains evidence-gated and rollback-capable. Field learning is not
permission to rewrite trusted source code or bypass M6 authority.

## Completion priorities

Until the first usable APK exists, priority order is:

1. corpus quality and provenance;
2. CPU pilot correctness;
3. language training and evaluation;
4. mobile footprint/latency;
5. signed turnkey packaging;
6. physical acceptance;
7. multimodal and specialty capability quality.

Feature expansion is lower priority than all seven items above.
