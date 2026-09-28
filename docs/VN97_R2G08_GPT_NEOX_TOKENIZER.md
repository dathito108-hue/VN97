# VN97 R2-G0.8 — GPT-NeoX tokenizer identity and Android runtime

G0.8 closes a correctness gap between the inherited Mamba-2 weights and the
Android text path.

The Mamba-2 2.7B source declares tokenizer lineage
EleutherAI/gpt-neox-20b at the pinned tokenizer revision. VN97TK1 is not assumed
to be token-ID compatible with that tokenizer.

## Why VN97TK1 cannot be substituted

VN97TK1 is a byte-lossless longest-prefix package. GPT-NeoX uses GPT-2 style
byte-level BPE with ranked pair merges and a Unicode-aware pretokenizer.

Both can round-trip text, but round-trip ability alone does not imply identical
token IDs. The inherited embedding/output rows are indexed by GPT-NeoX token
IDs, so any tokenizer substitution would change model semantics.

G0.8 therefore keeps a dedicated GPT-NeoX tokenizer runtime for the transferred
Mamba-2 intelligence.

## Exact text path

The reference and Android implementations use:

1. canonical GPT-2 byte-to-Unicode bijection;
2. GPT-NeoX pretokenization pattern;
3. add_prefix_space=false;
4. ranked BPE pair merging from merges.txt;
5. exact vocab.json token IDs;
6. literal end-of-text special-token preservation.

The pinned metadata uses the same end-of-text token as BOS, EOS and UNK. The
runtime reads those IDs from the source vocabulary rather than inventing IDs.

## G0.8 descriptor

Schema:

    VN97M2G08TOK1

The descriptor binds:

- G0.3 capsule ID;
- pinned tokenizer model and revision;
- SHA-256 and size of all five tokenizer assets;
- tokenizer algorithm and pretokenizer pattern;
- BOS/EOS/UNK text and IDs;
- exact tokenizer token-ID space;
- runtime logits width;
- invalid padded-logit range;
- merge count;
- same_token_ids_required=true;
- source_runtime_required=false;
- production_activation_authorized=false.

Android re-hashes every tokenizer asset before loading.

## Padded logits

The Mamba-2 model contract uses source vocab_size 50,277 while the embedding
matrix is padded to 50,288 rows.

Those final 11 rows are not valid tokenizer IDs.

G0.8 therefore records:

    token_id_space = 50277
    runtime_logits_size = 50288
    invalid padded IDs = [50277, 50288)

The Android tokenizer exposes maskInvalidPaddedLogits(). Any future sampler
bound to this runtime must apply this mask before selecting a token.

A generated padded ID is treated as an error, not decoded as an invented token.

## Native Android implementation

VN97GptNeoXTokenizer is self-contained and uses only packaged assets plus the
Android/Java runtime.

It does not require Transformers, Hugging Face tokenizers, Python, Mamba,
Triton, CUDA or another model backend.

The pretokenizer uses the same Unicode category classes and whitespace
lookahead semantics as GPT-NeoX. BPE merge ranks are loaded directly from the
pinned merges.txt.

A bounded BPE cache avoids unbounded long-running memory growth.

## Host reference

src/vn97/r2/gpt_neox_tokenizer.py provides the auditable host reference and
G0.8 descriptor builder.

Commands are exposed through:

    vn97-r2-mamba2-g08

A real G0.8 package is built from the tokenizer assets already sealed inside a
verified G0.3 capsule, so tokenizer identity stays attached to the transferred
intelligence lineage.

## Production boundary

G0.8 does not switch the cognition bridge yet.

Before activation:

1. materialize the real G0.3 capsule;
2. build the real G0.8 tokenizer descriptor;
3. compare VN97 token IDs against the pinned official tokenizer on a broad
   multilingual, whitespace, punctuation, Unicode and special-token probe set;
4. pass source -> VN97 -> ORT model parity;
5. pass G0.7 physical S21 FE provider profiling;
6. bind runtime + tokenizer + tuning identities together in the activation
   bridge.

No approximate tokenizer conversion is an acceptable replacement for token-ID
parity.
