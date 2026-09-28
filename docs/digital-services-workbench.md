# Digital services workbench

Open **Dịch vụ số** from the VN97 main screen. This is a local workbench for producing
deliverables for potential service jobs; it is not proof of paid demand or autonomous revenue.

Current services:
- Clean UTF-8 CSV: trim fields, remove empty and exact duplicate rows, validate column counts,
  preserve quoted commas/newlines, and protect formula-like cells with a leading apostrophe.
  Negative numbers are also protected; inspect the report before using the result.
- Format plain text: preserve wording and indentation, normalize line endings, trim trailing
  whitespace and collapse repeated empty lines. This does not translate or verify facts.
- Create a static, responsive HTML catalog from a CSV header and rows. Customer-provided
  content is escaped. The page has no scripts, external resources or automatic publishing.

Enter a local job name and optional estimated price/cost in whole VND. The displayed margin
is an estimate, never a receipt or verified income. Data stays on the device until the user
chooses a destination in Android's document picker. The latest successfully generated job
can be restored with its source in app-private no-backup storage. A separate bounded ledger
keeps metadata for up to 500 orders and 2,000 payment records; it never exports the source or
internal quote. A failed transformation preserves the previously saved job. Unsubmitted edits
are not persisted across activity recreation.

Export creates a ZIP with the deliverable and a transformation report containing input/output
SHA-256 hashes. It excludes the original input, internal job name and price/cost fields.
Hashes identify bytes, not quality, customer acceptance or payment. Review the preview and
full output before delivering to a customer. Deleting the local job does not delete exported files.
If export is interrupted, discard the incomplete destination and export again.

Limits: 128 KiB UTF-8 input, 64 CSV columns, 5,000 physical parser rows including the header,
and 1 MiB output. CSV uses commas and strict quotes; malformed data is rejected rather than
silently repaired. The renderer adds a size guard while constructing repeated catalog markup.
Preview is capped at 8,000 characters. Transformations and I/O run off the UI thread.

These utilities use no additional model/backend and do not change VN97's single-model
architecture. They are deterministic tools, not evidence of AGI or successful model inference.
Each successful ZIP write marks its order exported. Payment reconciliation is provider-neutral
and deduplicates on `provider + account fingerprint + external event ID`: exact retries are
no-ops and conflicting reuse is rejected. Verified net proceeds deduct fees, refunds and
realized costs. The UI can record a receipt claim from any channel, but always marks it
`MANUAL_UNVERIFIED` and excludes it from verified revenue. A `PROVIDER_API_VERIFIED` record
requires an account fingerprint and authenticated-response evidence digest from a future
provider adapter. No authenticated settlement was available for this change.

Next integration work is model-assisted writing/translation only when the activated VN97
model is available, customer acquisition channels, and authenticated provider adapters.
Exness remains an optional separate channel. No outbound marketing, marketplace posting,
payment request or financial trade is performed here.

Validation: host regression covers Vietnamese text, quoted/multiline CSV, duplicates,
formula protection, invalid structure, HTML injection, input/output bounds, cost arithmetic,
order/export state, duplicate payment events and verified/unverified net separation.
Android compilation runs in the existing revenue workflow. Physical-device UI tests,
rotation/export recovery and performance measurements remain outstanding.
