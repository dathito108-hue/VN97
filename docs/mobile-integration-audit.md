# VN97 integration audit — 2026-09-28

Scope: main 8818894965487c5fa009bc483defaad852b9eee2 and this patch.
This is source-level evidence plus CI; it is not a physical-device or AGI evaluation.

| Flow | Observed state | Remaining work |
| --- | --- | --- |
| APK to usable intelligence | `assets/vn97-bootstrap` contains only a README; release requires signed model, signature and publisher key. Debug supports developer provisioning. | Supply and verify the actual compatible model/runtime package, then test fresh installation. An APK alone does not establish intelligence availability. |
| Foreground chat continuation | Previously, the bounded driver could return YIELDED while the canonical session remained active. The app discarded its resumable result and the UI entered ERROR. | This patch retains the result, exposes a bounded Continue operation on the same turn and restores the yielded UI when a new activity attaches to the live process. Device tests remain required. |
| Speech input inference | `runVoiceTurn` explicitly throws because the R2 identity-bound ONNX audio graph is not installed. The availability query incorrectly relied on legacy audio-projection metadata. | This patch stops advertising production speech availability. Implement and validate the real audio graph/adapter; do not substitute a second model/service silently. |
| Vision and visual verification | `perceiveVision` and `verifyVisualOutcome` explicitly reject execution without the R2 multimodal package. Legacy vision metadata could still enable controls. | This patch aligns the availability query with the gated implementation. Implement graph binding and perception-to-action validation before claiming camera/screen intelligence. |
| Digital services | Main-screen tools generate CSV/text/HTML deliverables locally, with preview and explicit ZIP export. They are deterministic utilities. | Add typed planner integration and customer-workflow execution. Multiple-order and revenue-ledger work is already in separate PR #294; avoid overlapping it. |
| Revenue | Quotes and paper results are not verified receipts. Read-only Exness work is separate PR #291. | Customer acquisition, accepted delivery and authenticated settlement reconciliation remain required across multiple channels. No real revenue is demonstrated by this audit. |
| Background work | Separate autonomous scheduling/continuation code exists; foreground active turns block handoff and resource release. | Exercise process death, reboot, approval restoration, thermal/resource pressure and actual model availability on Android. A schedule is not proof of uninterrupted cognition. |
| Mobile speed | Earlier FP16 storage/copy and bundled-install repairs are merged. | Benchmark this exact APK/model on the S21 FE. No new physical RAM, token/s, latency or thermal measurements were collected in this audit. |

## Continuation behavior

- The initial result counts against the per-call advance limit. A yielded result is retained
  with its original user text and canonical turn object, not rebuilt with `startTurn`.
- New chat input remains disabled while yielded. Continue rechecks the interactive resource
  policy and runs at most the configured advance count. A denied resource check leaves the
  turn available for a later attempt.
- Approval-required results remain separate and require the existing M6 resolution path.
  Continue cannot resolve or approve them.
- Completion clears the yielded slot and appends the final answer once. Repeated yield
  does not append fabricated answers or duplicate user messages.
- This patch covers live-process foreground continuation and Activity recreation, not
  durable restoration of a foreground conversation after process death. The autonomous
  subsystem has a separate checkpoint/recovery path.

Regression coverage: reducer transitions, disabled new-turn submission while yielded,
same-turn bounded advancement, no advance across terminal/approval boundaries, and
single final transcript insertion. CI also compiles the actual Android activity/assistant.
Real-model continuation, Activity lifecycle and device interaction tests remain outstanding.

Next priorities: provision the production model; validate chat end to end; connect service
tools to the planner and verified payment flow; implement missing speech/vision adapters;
then measure and harden the full mobile lifecycle. Preserve the single VN97 model and M6.

## PAUSED follow-up

The canonical session now has an explicit `resumePausedTurn` entry point. It verifies
the active turn identity, absence of pending M6 approval, non-negative time and the
original memory binding before changing the paused planner. It invokes the existing
controller's resume operation and continues that plan; it does not build a new one.
The app retains both YIELDED and PAUSED results, distinguishes them in the UI, and
requires the same resource-limited Continue action in main chat and the floating panel.
The bounded driver never automatically resumes a PAUSED boundary.

Android JVM regression tests exercise the real session/planner/coordinator with scripted
inference. They cover completion after a pause without repeated memory retrieval,
wrong-turn/wrong-memory/invalid-time rejection, completed-turn rejection and an M6
approval flow where resume is rejected before the normal approve/execute path. These
are lifecycle tests; they do not validate model intelligence or physical-device behavior.
