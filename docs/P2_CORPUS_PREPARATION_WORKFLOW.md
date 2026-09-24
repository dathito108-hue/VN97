# P2 — GitHub Corpus Preparation Workflow

VN97 now has a manual GitHub Actions workflow that prepares the complete sealed P2
medium-pilot corpus without running model training.

Workflow:

`.github/workflows/p2-corpus-preparation.yml`

It is **manual-only** through `workflow_dispatch`. A push or pull request never
downloads the public datasets automatically.

## License acknowledgement gate

The workflow requires the exact dispatch input:

`I_APPROVE_THE_THREE_P2_SOURCE_LICENSES`

This acknowledgement is intentionally separate from ordinary repository changes. The
workflow does not infer license approval from a commit, pull request, or chat message.

Before dispatch, review the exact pinned licenses for:

- `vi-dialogue-hoanghai2110` — Apache-2.0;
- `databricks-dolly-15k` — CC-BY-SA-3.0;
- `openai-gsm8k-train` — MIT.

## Workflow chain

One successful run executes:

`VN97P2FETCHDEF1
-> pinned HTTPS fetch
-> VN97P2FETCH1
-> deterministic public-data intake
-> VN97P2SRC1 + VN97CORPUSDEF1
-> VN97CORPUS1 seal
-> full provenance-chain verification
-> VN97P2BUNDLE1
-> GitHub artifact`

No model training runs in this workflow.

## Torch-free preparation

Corpus preparation uses:

`tools/p2_corpus_tool.py`

This lightweight launcher deliberately bypasses the eager top-level VN97 training
runtime and loads only the pure-Python corpus modules required by P2.

Therefore corpus preparation does not install PyTorch and does not consume GPU
resources. GitHub Actions time is used only for:

- checkout;
- Python setup;
- bounded public-source download;
- deterministic conversion/dedup/split;
- hashing/sealing;
- artifact upload.

The normal installed CLI entry points remain available for local/production hosts.

## VN97P2BUNDLE1

Before artifact upload, `vn97-p2-corpus-bundle` verifies the entire preparation
chain.

It checks:

- VN97P2FETCH1 canonical JSON and receipt identity;
- fetched raw SHA-256 identities against VN97P2SRC1;
- exact VN97CORPUSDEF1 SHA-256 bound by VN97P2SRC1;
- VN97CORPUS1 manifest identity;
- every definition source against the exact intake source bytes sealed into
  VN97CORPUS1;
- exact training/validation/release byte counts, record counts and SHA-256 values;
- repository commit identity.

The resulting `VN97P2BUNDLE1` binds:

- repository commit;
- source-fetch receipt SHA-256;
- source-summary SHA-256;
- corpus-definition SHA-256;
- VN97CORPUS1 manifest identity and SHA-256;
- exact split SHA-256/bytes/records;
- deterministic bundle identity.

## Artifact layout

The workflow uploads:

`VN97-P2-Corpus-<repository-commit>`

with this logical layout:

```text
corpus/
  corpus.vn97corpus1.json
  training.jsonl
  validation.jsonl
  release.jsonl

evidence/
  source-fetch.vn97p2fetch1.json
  p2-source-summary.vn97p2src1.json
  corpus-definition.vn97corpusdef1.json
  p2-corpus-bundle.vn97p2bundle1.json

SHA256SUMS
```

Raw downloaded source files are intentionally not uploaded in the artifact. Their exact
download identities remain bound by VN97P2FETCH1 and VN97P2SRC1.

Artifact retention is seven days. Once downloaded for a real P2 training run, its
VN97P2BUNDLE1 and VN97CORPUS1 identities should be retained with the training
evidence.

## Next step after the artifact

The artifact's `corpus/` directory is the input to:

```bash
vn97-medium-pilot \
  --corpus-dir corpus \
  --output-dir p2-medium-output \
  --device cpu
```

If CPU runtime is impractical, the same sealed corpus and fixed candidate are used
with `--device cuda` on a short rented GPU session.

No P3 production candidate campaign begins until the resulting P2 run produces a valid
VN97PILOT1.
