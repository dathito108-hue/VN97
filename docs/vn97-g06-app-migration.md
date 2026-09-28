# G06-only app migration

User instruction 2026-09-29: remove R2-SSM1 and the fast/slow alternative before
evolution or training. Historical source/checkpoints remain in Git history at
3441d7754d92763dccae0b7024d94605c4d8adbb; no user memory files are deleted.

## Active route

App → VN97G06Model → VN97G06CognitionInference → existing G09 tokenizer bridge
→ G06 ONNX executor. Chat, game reasoning, paper trading, knowledge tasks,
autonomous planning/replanning and retrieval use this same lineage. The R2/F1
executor and fast/slow executor are removed. Each live G06 deployment owns one
serialized cognition engine shared by its task handles, with one cached provider
session. Closing a temporary task handle does not allocate another model or
close the engine used by the foreground assistant. The historical Python `vn97.r2`
namespace retains Mamba2/G06 modules, not the R2-SSM1 implementation.

The app no longer loads a second MI1 model to obtain tokenizer/metadata.
G06Model validates G06 runtime, G08 tokenizer, G07 measured device tuning, G09
binding and G10 promotion. The deployment/promotion identity binds planner
continuation. Existing offline MI1 package/evaluation utilities and shared
native memory/planner infrastructure are not an app inference fallback.
Legacy CAP/MI1 activation into app inference is rejected. Its import controls
and the obsolete self-improvement entry are removed from the main dashboard.

## Bundle and trust

Signed APK assets `vn97-g06/` contain:

- `runtime/runtime.vn97m2g06.json`, declared recurrent graph/external data;
- `tokenizer/tokenizer.vn97m2g08.json`, `vocab.json`, `merges.txt`;
- `tuning.vn97m2g07.json`, `binding.vn97m2g09.json`;
- `promotion.vn97m2g10.json`.

The APK distributor is the bootstrap trust root. Hashes alone are not publisher
authentication. No arbitrary ZIP or historical CAP is treated as a trusted G06
deployment. Within one process the installer avoids recopying immutable APK assets when
reopening a model; the model loader still verifies package bytes. A new process
or APK gets a fresh installer. Before staging, it compares the installed file tree
and all five deployment descriptors against signed APK assets, then runs full
installed deployment validation (including graph/weight/tokenizer hashes). An
unchanged valid installation needs no model payload copy, even in a new process;
a local identity marker alone cannot authorize this shortcut. Changed descriptors
or damaged payloads take the staging/repair path. Temporary staging and old backup
directories are reclaimed after successful validation. The installer validates staging before replacement and preserves
the previous installation on copy/layout/semantic validation failure. The 8 GiB
copy ceiling is not evidence that an APK of that size can be distributed or
installed; full-size G06 model delivery and physical qualification remain gates.
No production model assets or invented promotion receipts are added here.

## Planner continuation, not hidden-state persistence

G06 cognition operations already reconstruct a bounded request prompt and reset
their recurrent state per operation. Persist the planner, results, approvals and
deployment identity; do not prefill a legacy model merely to seed continuity.
`VN97PLN3` is an explicit 104-byte identity/lifecycle carrier reused by the
existing composite planner store. It cannot infer, advance tokens or accept
state writes. Restore enforces zero sequence position, fixed scalar geometry,
zero reserved payload and a nonzero 32-byte identity. No G06 hidden-state bytes
are represented by it. Suspended in-operation token generation is not resumed;
continuation remains at planner operation boundaries with existing M6 rules.
Old deployment identities are rejected rather than silently migrated/replayed.

Retrieval uses G08 token features. VN97MEM1 and write journals live in a tokenizer
identity namespace, preserving old stores without mixing incompatible vectors.
These deterministic features are retrieval support, not a learned embedding
model or proof of semantic retrieval quality.

## Evidence and remaining work

Native tests exercise identity continuation round-trip, lifecycle behavior,
tamper rejection and refusal of legacy inference/state writes. Source gates
check removed implementations, retained Python imports/CLI targets and app
routes. Android CI builds the APK and runs app/platform JVM and host contracts.
Device evidence now measures actual G06 prefill/decode, without the old native
speech probe; its schema identifies a G06 deployment.

The developer evidence intent no longer stages or activates CAP/MI1 files. It
opens only the already installed, fully validated G06 deployment and refuses to
measure when that deployment is absent. The main activity also no longer
constructs hidden legacy model-picker controls; G06 delivery remains the signed
APK/bootstrap responsibility until a separately authenticated streaming
distribution format is specified.

Compilation and these tests are not real G06 language quality or S21 FE latency
evidence. Need a qualified full deployment to test chat/tool decisions and real
process death/reboot on-device. No training is started by this migration. The
user's requested order remains cleanup → demonstrated working app → evolution
loop → controlled training, with the latter stages pending this qualification.
