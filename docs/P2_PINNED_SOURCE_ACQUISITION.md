# P2 — Pinned Public Source Acquisition

This stage closes the gap between public dataset references and the local raw JSONL
files consumed by `vn97-p2-corpus-intake`.

The source lock is:

`configs/p2-source-lock.vn97fetchdef1.json`

It binds exact upstream revisions rather than mutable branch heads.

## Pinned sources

### Vietnamese dialogue

Repository:

`hoanghai2110/vietnamese-dataset`

Pinned revision:

`321a852edc7e9d36e8456fb7d7df583b520645a9`

File:

`hypermamba_data.jsonl`

Declared license:

`Apache-2.0`

The pinned revision is authoritative. The fetch receipt records the exact downloaded
SHA-256 because this small regular Git object does not expose an independent content
hash in the dataset lock.

Expected JSONL record count: 2,417.

### Databricks Dolly 15k

Repository:

`databricks/databricks-dolly-15k`

Pinned revision:

`d72c16e4644a463b9c678c71d9440befd4594556`

File:

`databricks-dolly-15k.jsonl`

Declared license:

`CC-BY-SA-3.0`

Expected source identity:

- bytes: 13,085,339
- SHA-256: `2df9083338b4abd6bceb5635764dab5d833b393b55759dffb0959b6fcbf794ec`
- records: 15,011

### GSM8K training data

Repository:

`openai/grade-school-math`

Pinned revision:

`3101c7d5072418e28b9008a6636bde82a006892c`

File:

`grade_school_math/data/train.jsonl`

Declared license:

`MIT`

Expected source identity:

- bytes: 4,166,206
- SHA-256: `17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465`
- records: 7,473

## Fetch command

After reviewing the exact three licenses, fetch the pinned sources with:

```bash
vn97-p2-source-fetch \
  --output-dir p2-raw \
  --approve-source-license vi-dialogue-hoanghai2110 \
  --approve-source-license databricks-dolly-15k \
  --approve-source-license openai-gsm8k-train
```

The command refuses to run unless the approval set exactly matches the locked source
set.

It permits HTTPS only and restricts initial/redirect hosts to the pinned GitHub /
Hugging Face delivery domains. Compressed HTTP responses are rejected so byte
identities are stable.

The downloader streams each source through a strict byte bound and verifies:

- non-empty response;
- expected byte count when independently known;
- expected SHA-256 when independently known;
- strict UTF-8 JSONL;
- every non-empty line is one JSON object;
- exact expected record count.

Outputs are create-only.

A successful fetch emits the three raw files plus:

`source-fetch.vn97p2fetch1.json`

VN97P2FETCH1 binds:

- source-lock SHA-256;
- source ID;
- pinned revision;
- output filename;
- exact downloaded bytes;
- exact JSONL record count;
- exact downloaded SHA-256;
- final HTTPS delivery URL;
- deterministic receipt identity.

## Next chain

The raw fetch directory is not training input.

The canonical chain remains:

`VN97P2FETCH1
-> vn97-p2-corpus-intake
-> VN97P2SRC1 + VN97CORPUSDEF1
-> vn97-corpus-seal
-> VN97CORPUS1
-> vn97-medium-pilot
-> VN97PILOT1`

No mutable upstream dataset branch is accepted as a P2 production-intelligence input.
