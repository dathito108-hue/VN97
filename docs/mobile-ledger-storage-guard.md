# Service ledger storage repair

The activity previously used `runCatching(...).getOrNull() ?: emptyLedger()`
when opening/parsing storage. An I/O error or malformed JSON could therefore be
presented as zero orders and overwritten by the next mutation. Its separate
read/write calls also allowed independently recreated Activity workers to lose
updates against the same file.

All order, export and payment mutations now use a shared repository transaction
(read, decode, mutate, encode, AtomicFile commit) under one process-wide lock.
Only confirmed absence of the base, `.bak` and `.new` files produces a fresh
ledger; unknown filesystem state or a dangling link is not considered absence.
Parse/schema/limit/I/O errors stop the transaction before mutation or writing.
The screen reports unavailable totals rather than fabricating a zero balance.
There is no automatic reset or deletion of unreadable financial records.

The existing schema and Android AtomicFile commit/rollback adapter are retained.
Seven JVM regressions exercise the real codec and repository with real JSON:
missing-file initialization, evidence round-trip, malformed/oversize/unknown-schema
reads, I/O failure, failed mutation/commit, concurrent Activity-like repositories,
and base/backup/pending/symlink presence. Commit-failure tests inject an I/O adapter;
they do not establish physical Android power-loss behavior. The lock serializes
instances in one app process, not multiple OS processes.

No provider settlement verification is added. Manual receipts stay unverified.
No training or model change is part of this patch. Device testing and workbench
export restoration across Activity/process recreation remain separate follow-up
items; this patch addresses ledger preservation rather than claiming full AGI.
