# P4A — Held-Out Task-Level Intelligence Evaluation

P4A is the canonical step after a successful P3 language campaign.

P3 proves bounded language-model training, deterministic candidate selection and
sealed-release token-level quality. P4A does **not** retrain the model. It verifies the
exact P3 winner artifact and measures task-level behavior that token loss/top-1 cannot
establish.

## Architecture boundary

P4A adds no alternate model, backend, planner or memory implementation.

The same selected VN97 checkpoint is loaded through the canonical VN97CK1 loader and
the same VN97TK1 tokenizer is used by the VN97-native inference engine.

The frozen progression remains:

```text
Code 1
  -> Code 2 hardware/mobile-aware
  -> P3 language winner
  -> P4 task-level intelligence measurement
  -> later production speech/vision/mobile/release gates
```

## P3 artifact verification

Before any task is executed, P4A requires an extracted canonical P3 final directory
containing exactly:

```text
SHA256SUMS
campaign-report.json
model.vn97ck1
model.vn97mi1
p3-run.vn97p3run1.json
tokenizer.vn97tk1
```

The verifier checks:

- the exact file set and absence of symlinks;
- every SHA256SUMS entry;
- strict/canonical VN97P3RUN1;
- the immutable P3 profile identity;
- VN97CAMP2 report identity;
- selected candidate identity and frozen geometry;
- VN97CK1 checkpoint identity;
- VN97TK1 identity and vocabulary compatibility;
- VN97MI1 SHA-256 and byte count.

Command:

```bash
vn97-p4-task-eval verify-p3 \
  --p3-dir /path/to/p3-final
```

A successful verification prints:

```text
VN97P4A P3 VERIFIED candidate=... checkpoint=... model_image=... ...
```

This is an integrity/provenance statement only. It is not a new intelligence claim.

## Held-out task suite

P4A uses canonical UTF-8 JSONL. Every line is one VN97P4TASK1 object and the file must
end in a newline.

Every production suite must cover all six categories:

1. `instruction_following`
2. `reasoning_planning`
3. `memory_use`
4. `structured_cognition`
5. `tool_intent`
6. `authority_behavior`

Example exact-text task:

```json
{"category":"instruction_following","max_new_tokens":32,"prompt":"Reply with exactly OK.","schema":"VN97P4TASK1","scoring":{"expected":"OK","kind":"exact_text"},"task_id":"if-001"}
```

Supported deterministic scoring modes are:

- `exact_text` — trimmed output must equal the expected string;
- `contains_all` — every required substring must be present, with explicit
  case-sensitivity;
- `json_exact` — generated text must parse as strict JSON and equal the expected JSON
  value.

The production held-out suite is intentionally **not** committed as a training asset.
It should be prepared independently and kept out of P3 training data.

## Running evaluation

```bash
vn97-p4-task-eval evaluate \
  --p3-dir /path/to/p3-final \
  --suite /path/to/held-out-p4.jsonl \
  --device cuda:0 \
  --output /path/to/p4-evaluation.vn97p4eval1.json
```

The evaluator uses deterministic greedy VN97 inference and emits one canonical
VN97P4EVAL1 report.

The report binds:

- exact P3 run SHA-256;
- selected candidate ID;
- checkpoint/tokenizer/model-image identities;
- exact held-out suite SHA-256;
- device;
- per-task pass/fail and output SHA-256;
- per-category pass rates;
- aggregate pass rate;
- deterministic evaluation ID.

Raw generated responses are deliberately not copied into the canonical evidence
receipt. Only their hashes and byte counts are recorded.

## Why status is MEASURED

P4A deliberately emits:

```text
status = MEASURED
```

It does not invent an arbitrary production threshold before the real P3 winner has
been measured on a genuinely held-out task suite.

The next P4 step can freeze admission thresholds from observed baseline evidence and
then add a pass/fail promotion gate without changing the model architecture.

## Honest boundary

Repository CI validates the parser, artifact chain-of-custody and scoring contracts.
CI does not possess the user's real P3 artifact or an independent production held-out
suite, so it cannot claim a P4 quality result.

A real VN97P4EVAL1 exists only after the real P3 winner is evaluated against a
separately prepared held-out task suite.
