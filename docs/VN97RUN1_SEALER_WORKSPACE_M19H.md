# M19H — VN97RUN1 Sealer / Production Workspace Bootstrap

M19H removes the manual hash/path authoring step from M19G.

It adds one command:

```text
vn97-production-seal
```

with two modes:

```text
bootstrap
seal
```

M19H does not add a new model, trainer, selector, evidence format, release gate,
or parallel run-manifest format. The only production run manifest remains
`VN97RUN1`.

## Canonical production flow

```text
vn97-production-seal bootstrap
  -> populate real datasets / campaign definition / speech corpus / policies
vn97-production-seal seal
  -> production-run.vn97run1
vn97-production-run --stage verify
vn97-production-run --stage preflight
vn97-production-run --stage train
  -> exact VN97PRODCAMP1 model image
  -> collect real VN97MOBEVID1 on physical phones
vn97-production-run --stage intake
  -> VN97INTAKE1 + VN97RC1
vn97-production-release --preflight-only
  -> VN97READY1
  -> signed production release
```

## Bootstrap mode

```text
vn97-production-seal bootstrap \
  --workspace-root /production/vn97
```

The target must either not exist or be an empty real directory.

Bootstrap creates only a skeleton:

```text
config/
data/
  train/
  validation/
  release/
speech/
  train/
  validation/
  release/
device-evidence/
out/
PRODUCTION_WORKSPACE.md
```

It also creates example-only files:

```text
config/campaign.json.example
config/language-options.json.example
config/speech-options.json.example
config/intake-options.json.example
```

These examples intentionally contain invalid placeholder production values.

Bootstrap does not create:

- training data;
- validation data;
- sealed holdouts;
- speech audio;
- physical-phone evidence;
- a VN97RUN1 manifest;
- a production candidate;
- an APK.

Therefore a fresh bootstrap directory cannot accidentally pass the sealer.

## Required real workspace inputs

Before `seal`, the workspace must contain:

```text
config/campaign.json
config/language-options.json
config/speech-options.json
config/intake-options.json

data/train/**
data/validation/**
data/release/**

speech/train/manifest.jsonl
speech/validation/manifest.jsonl
speech/release/manifest.jsonl
```

Each speech manifest references its real audio files by canonical relative path.

The evidence directory may still be empty at seal time because physical evidence
is produced only after the exact production model image exists.

The output parent `out/` exists, but these must not exist before execution:

```text
out/language
out/production
out/intake
```

## Automatic language corpus discovery

M19H recursively scans:

- `data/train`;
- `data/validation`;
- `data/release`.

Only regular non-symlink `.jsonl` and `.txt` files are accepted.

For every discovered file the sealer records in VN97RUN1:

- canonical workspace-relative POSIX path;
- byte count;
- SHA-256.

Files are sorted by canonical path.

Unknown files in a language split are rejected rather than silently ignored.
This avoids accidentally sealing editor backups, binary dumps, temporary
artifacts, or stale corpora.

M19G later re-verifies the exact bytes and physical train/validation/release
file separation.

## Automatic campaign-definition binding

M19H expects:

```text
config/campaign.json
```

The file is read through a no-follow descriptor, byte bounded and SHA-bound into
VN97RUN1.

After manifest construction, M19H runs the existing M19G verifier. The
`VN97CAMPDEF1` schema, candidate bounds, duplicate-candidate rules and
candidate parameter fields are therefore checked before the manifest is
published.

## Automatic speech corpus discovery

M19H expects:

```text
speech/train/manifest.jsonl
speech/validation/manifest.jsonl
speech/release/manifest.jsonl
```

Each record must still be exactly:

```json
{"audio":"relative.wav","text":"transcript"}
```

For every split the sealer:

1. parses the JSONL using strict UTF-8 JSON;
2. rejects duplicate JSON keys;
3. rejects absolute, parent-escaping, non-canonical or symlink audio paths;
4. resolves the referenced files inside the production workspace;
5. records the manifest byte count/SHA-256;
6. records every unique referenced audio file byte count/SHA-256;
7. sorts the bound audio set by canonical workspace-relative path.

M19G then confirms that the bound audio set exactly equals the speech manifest
references and that train/validation/release audio paths do not overlap.

The sealer does not duplicate M11C PCM validation. Actual speech training still
owns WAV format, duration and semantic split validation.

## Production option files

M19H reads three strict JSON objects:

```text
config/language-options.json
config/speech-options.json
config/intake-options.json
```

They contain only values that become the existing M19G stage option maps.

Example minimal language options:

```json
{
  "device": "cuda:0",
  "max_parameters": 100000000,
  "max_validation_loss": 3.0
}
```

Example minimal speech options:

```json
{
  "device": "cuda:0",
  "max_speech_validation_loss": 3.0
}
```

Example intake policy:

```json
{
  "min_device_runs": 5,
  "min_distinct_device_profiles": 2
}
```

The source option files do not need canonical JSON formatting. Their parsed
values are normalized through the existing strict VN97RUN1 option contract.

Unknown option names, invalid types, `device=auto`, invalid thresholds and
unsafe policy values are rejected by the same M19G dataclass/parser contract.

## Environment auto-binding

`seal` probes rather than asks the user to type:

- Python version;
- installed Torch package version;
- platform system;
- platform machine architecture.

If Torch is not installed, sealing fails. A production execution environment
cannot be honestly sealed without the runtime package that M19G will later
require.

## Git auto-binding

`seal` probes the supplied VN97 repository:

```text
--repository-root /work/VN97
```

It requires:

- a real directory;
- canonical VN97 source layout;
- resolvable lowercase 40-hex Git HEAD;
- clean tracked worktree.

That commit is stored directly in VN97RUN1.

M19G later requires execution from exactly that commit and also binds its own
runner source bytes.

## Seal mode

```text
vn97-production-seal seal \
  --workspace-root /production/vn97 \
  --repository-root /work/VN97
```

The output is fixed:

```text
/production/vn97/production-run.vn97run1
```

There is no output-path override.

This keeps one obvious canonical manifest per workspace and prevents scripts
from accidentally sealing different copies under unrelated paths.

Before writing, M19H:

1. scans and hashes every real input;
2. constructs `VN97ProductionRunManifest`;
3. applies the same option/path/cross-split checks as direct M19G construction;
4. calls `verify_production_run_inputs()` on the in-memory manifest;
5. refuses publication if any bound file changed or any M19G contract fails.

The file is then written through a same-parent temporary file + `fsync` +
atomic replace.

The target must not already exist.

After writing, the CLI reloads the exact bytes through the strict
`load_production_run_manifest()` parser and requires the parsed object to equal
the verified in-memory manifest.

M19I then provides the next zero-training-compute gate:

```text
vn97-production-run --stage preflight
```

Running `--stage train` also invokes that same preflight automatically before
any language or speech training subprocess.

## Manifest immutability

M19H never overwrites an existing:

```text
production-run.vn97run1
```

To change data, candidate search space, policy, runtime or Git commit, use a new
production workspace or deliberately remove/archive the old manifest before
creating a new campaign.

This prevents a production run identity from silently changing in place.

## File-system safety

M19H uses no-follow file opens for hashed inputs and rejects:

- symlinked input files;
- symlinked workspace-critical directories;
- files outside the workspace;
- unsupported files in language split directories;
- speech references escaping the workspace;
- duplicate/invalid JSON keys;
- files that change while being read or hashed;
- an existing manifest target.

The final M19G verifier is still authoritative for the complete VN97RUN1
contract.

## What remains human/real-world work

M19H automates manifest authoring, not production intelligence creation.

Real inputs are still required:

- governed training corpus;
- validation corpus;
- sealed release corpus;
- real VN97CAMPDEF1 candidate search space;
- speech train/validation/sealed-release corpora;
- actual compute for M10P/M10R/M11C;
- physical phones for VN97MOBEVID1 capture;
- publisher key and Android release signing material.

No example file generated by `bootstrap` counts as production evidence or
production intelligence.

## Architecture boundary

The canonical architecture remains:

```text
Code 1
  -> Code 2 hardware/mobile-aware upgrade
  -> one VN97 production model
```

M19H is only an authoring/sealing layer for M19G. It introduces no
Transformer/LLaMA/cloud backend, alternate model, second planner, second memory
engine, second trainer or alternate release chain.
