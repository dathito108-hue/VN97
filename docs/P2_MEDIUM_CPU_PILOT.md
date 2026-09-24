# P2 — Medium CPU-First Pilot

P2 replaces the earlier tiny VN97 pilot with one fixed medium-sized candidate.

The purpose is still proof before expensive production training, but the candidate is
large enough to expose realistic optimizer, checkpoint, packed-export and mobile-state
behavior.

## Fixed candidate identity

The canonical candidate manifest is:

`configs/p2-medium-pilot.vn97campdef1.json`

It binds:

- d_model: 192
- n_layers: 6
- d_state: 16
- factorized tied embedding rank: 96
- learning rate: 3e-4
- seed: 97

This remains the same VN97 Code-1 -> Code-2 architecture.

For batch-1, recurrent state is:

`6 * 192 * 16 * 4 = 73,728 bytes`.

The pilot does not claim production intelligence.

## Pilot data size

Do not feed the entire future production corpus into P2.

Prepare a licensed/provenance-reviewed representative subset that still preserves
three physically separate splits.

Recommended ceiling:

- training: at most 2,048 windows at sequence length 256;
- validation: at most 256 windows;
- sealed release: at most 256 windows;
- up to 4,096 learned tokenizer tokens.

At a full 256 tokens/window, the training ceiling represents about 524k token
positions per epoch. Two epochs therefore exercise roughly one million token
positions before padding/masking differences.

The subset should contain both Vietnamese and English assistant-style examples plus
reasoning/planning/structured-output samples. It must not reuse validation or release
records in training.

## Canonical run

Given a sealed P1 corpus directory, use the fixed P2 runner:

```bash
vn97-medium-pilot \
  --corpus-dir production-corpus-pilot \
  --output-dir p2-medium-output \
  --device cpu
```

The runner first verifies the exact VN97CORPUS1 output set, recomputes the corpus
manifest identity, checks every split hash/byte/record count, generates the fixed
medium candidate manifest internally, and then invokes the existing
`vn97-campaign` path with the locked P2 settings.

On a rented GPU, the same command changes only the explicit device:

```bash
vn97-medium-pilot \
  --corpus-dir production-corpus-pilot \
  --output-dir p2-medium-output \
  --device cuda
```

The loose loss ceilings are integrity/pipeline guards, not production-quality
thresholds. P3/P4 set real quality gates after pilot behavior is measured.

## Required P2 evidence

A successful P2 run must leave:

- `tokenizer.vn97tk1`;
- `model.vn97ck1`;
- `campaign-report.json` with schema `VN97CAMP2`;
- `model.vn97mi1` exported from the exact reloaded checkpoint;
- `pilot.vn97pilot1.json` with schema `VN97PILOT1`;
- finite training/evaluation losses;
- nonzero supervised target-token/step counts;
- a selected checkpoint whose SHA-256 survives canonical reload;
- successful sealed-release evaluation of that exact validation-selected candidate.

VN97PILOT1 binds the corpus manifest identity and SHA-256, campaign-report SHA-256,
fixed candidate/profile identities, checkpoint/tokenizer identities, model-image
identity/size and the explicit compute device used for the run.

## CPU-first, not CPU-only

P2 is deliberately bounded so a normal CPU can attempt it without committing to a
production GPU campaign.

If the available machine is impractically slow, the exact same candidate manifest,
corpus identities and campaign settings may run on a short rented GPU session by
changing only `--device`. Moving compute does not change the candidate identity,
dataset identities or acceptance requirements.

No production candidate campaign begins until P2 passes.
