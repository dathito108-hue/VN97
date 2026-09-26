# P4E-A — Generation Failure Diagnostic

P4E-A follows a measured P4D result where token-level validation and P3
retention improved, but strict unseen generation remained at 0/180.

P4E-A performs **no training** and changes no model weights. Its purpose is to
expose the real generated answer text so the next repair is based on evidence
rather than another blind fine-tune.

## Input

P4E-A accepts a complete P4D artifact, including rejected measured candidates:

```text
SHA256SUMS
model.vn97ck1
model.vn97mi1
p4d-report.json
tokenizer.vn97tk1
```

The verifier checks the exact file set, SHA-256 identities, canonical
`VN97P4D1` report, checkpoint identity, tokenizer identity, model-image
identity, and checkpoint/tokenizer vocabulary compatibility.

Accepted P4D statuses are `ELIGIBLE`, `REJECTED_VALIDATION`,
`REJECTED_RETENTION`, and `REJECTED_GENERALIZATION`.

## Diagnostic set

The default diagnostic deterministically selects three unseen validation
prompts from each of the six P4D categories. An optional existing P4 development
suite may also be supplied.

For every sampled task, P4E-A prints `PROMPT_JSON`, `EXPECTED_JSON`,
`GENERATED_JSON`, `PASS`, and `FAILURE`. JSON quoting is deliberate so
newlines, role-marker leakage, and extra leading/trailing text remain visible.

Failure labels distinguish empty output, role-marker leakage, extra text,
repetition, malformed/mismatched structured JSON, plain text mismatch, and
generation exceptions.

## Kaggle command

```bash
bash tools/kaggle_p4e_diagnose.sh \
  /kaggle/working/p4d-final \
  /kaggle/input/datasets/duongtuan1/vn97-p4c-resume/held-out-p4.jsonl \
  /kaggle/working/VN97-P4E-A-diagnostic.json
```

P4E-A never promotes a checkpoint and never changes the P4D status. The
diagnostic evidence is used to define P4E-B only after the dominant generation
failure mode is known.
