# P2 — Public Corpus Intake v1

This step prepares the first real P2 medium-pilot corpus from reviewed public datasets
without teaching VN97 another assistant's identity or weakening train/holdout
separation.

## Canonical pilot sources

P2 uses three bounded sources:

1. `hoanghai2110/vietnamese-dataset`
   - role: Vietnamese conversational/instruction coverage
   - declared license: Apache-2.0
   - adapter: canonical `messages` JSONL
   - deterministic cap: 800 accepted records

2. `databricks/databricks-dolly-15k`
   - role: English instruction-following coverage
   - declared license: CC-BY-SA-3.0
   - adapter: instruction/context/response -> VN97 chat
   - deterministic cap: 600 accepted records

3. `openai/gsm8k` training split
   - role: multi-step arithmetic/reasoning coverage
   - declared license: MIT
   - adapter: question/answer -> VN97 chat
   - deterministic cap: 600 accepted records

These declarations do not automatically approve legal use. The intake command refuses
to run unless every exact source ID is explicitly acknowledged with
`--approve-source-license`.

The raw input SHA-256 and byte count are bound into VN97P2SRC1 so a later run cannot
silently substitute different source bytes under the same source name.

## Identity contamination filter

The Vietnamese dialogue source contains some examples that identify the assistant as
another product. P2 rejects conversations containing known external assistant/product
identity markers such as HyperMamba, Vimind, ChatGPT, Claude, Gemini or Copilot.

This is deliberately narrow. It is not a general content/safety classifier.

## Deterministic split assignment

P2 converts records to canonical VN97 chat first, then assigns the canonical
conversation to:

- 90% training
- 5% validation
- 5% sealed release

using domain-separated SHA-256.

The split hash does **not** include source ID. Therefore the same conversation mirrored
by two public datasets always receives the same split and cannot cross from training
into validation/release merely because the mirror has a different source name.

Within each source, duplicate canonical conversations are removed before the fixed
record cap is applied.

The resulting P1 `vn97-corpus-seal` pass performs an independent exact cross-split
leakage check again.

## Intake command

Download/review the three source files locally, then run:

```bash
vn97-p2-corpus-intake \
  --vi-dialogue hypermamba_data.jsonl \
  --dolly databricks-dolly-15k.jsonl \
  --gsm8k train.jsonl \
  --output-dir p2-intake \
  --approve-source-license vi-dialogue-hoanghai2110 \
  --approve-source-license databricks-dolly-15k \
  --approve-source-license openai-gsm8k-train
```

The command requires three physically distinct regular non-symlink raw files and
strict UTF-8 JSONL.

It emits per-source/per-split canonical chat JSONL plus:

- `corpus-definition.vn97corpusdef1.json`
- `p2-source-summary.vn97p2src1.json`

VN97P2SRC1 binds raw file SHA-256/size, accepted/rejected counts, adapter/license/origin
metadata, split counts and the exact VN97CORPUSDEF1 SHA-256.

Every aggregate split must contain at least 16 records.

## Seal into VN97CORPUS1

The intake directory is not yet the training authority.

Seal its generated definition:

```bash
vn97-corpus-seal \
  --definition p2-intake/corpus-definition.vn97corpusdef1.json \
  --output-dir production-corpus-pilot
```

Only `production-corpus-pilot` is accepted by the canonical P2 medium runner.

## Run the medium pilot

```bash
vn97-medium-pilot \
  --corpus-dir production-corpus-pilot \
  --output-dir p2-medium-output \
  --device cpu
```

A successful run must end in VN97PILOT1 and an exact VN97MI1 export before P3 GPU
candidate training begins.
