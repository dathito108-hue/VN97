# M10M — Native VN97 Language Training Pipeline

M10M adds the first canonical training path that produces real VN97 deployment weights without
introducing another model architecture or inference backend.

## Data modes

The `vn97-train` CLI accepts local UTF-8 JSONL only.

Text mode record:

`{"text":"..."}`

Chat SFT record:

`{"messages":[{"role":"system|user|assistant","content":"..."}, ...]}`

Inputs are bounded by byte count and example count. No internet/download connector exists in the
trainer.

## Tokenizer

The trainer learns deterministic byte-BPE using the existing `learn_byte_bpe` implementation and
writes canonical `tokenizer.vn97tk1`.

The resulting tokenizer vocabulary becomes the exact `VN97Config.vocab_size`.

## Supervision

Causal text training supervises all content/eos targets after BOS + TEXT controls.

Chat SFT masks system/user tokens and supervises assistant role marker + assistant content + trailing
newline, plus EOS when the final turn is assistant. This keeps prompt/user text in context without
training the model to imitate user turns.

Long examples are converted into deterministic fixed-length causal windows with optional overlap.
Padding labels use ignore index -100.

## Optimization

`train_vn97_language(...)` trains the existing `VN97LanguageCore` directly with:

- AdamW;
- cross-entropy next-token loss;
- deterministic seed/shuffle;
- gradient clipping;
- finite loss/gradient checks;
- CPU or CUDA device selection.

No Transformer/LLaMA teacher, backend or hidden inference dependency is used.

## Outputs

The CLI writes:

- `model.vn97ck1` — M10L non-pickle deployment checkpoint;
- `tokenizer.vn97tk1` — canonical tokenizer;
- `training-report.json` — canonical VN97TRAIN1 provenance/stats.

The report binds dataset SHA-256, tokenizer SHA-256, checkpoint SHA-256, model geometry, window
count, target-token count and loss statistics.

The checkpoint can then flow directly through M10L -> M10K -> M10J.

## Example

`vn97-train --input assistant.jsonl --format chat --output-dir out --d-model 256 --layers 6 --d-state 16 --sequence-length 256 --batch-size 4 --epochs 3 --device auto`

This command trains only on the supplied local dataset. Model quality depends on dataset quality,
quantity, optimization budget and model capacity; the pipeline does not claim AGI from a small
dataset.

## Verification boundary

Unit regressions cover SFT masking, deterministic windows, actual gradient updates on a tiny native
VN97 model and VN97CK1 export/load.
