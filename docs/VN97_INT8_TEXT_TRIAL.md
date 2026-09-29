# Android INT8 text trial

This debug-only trial uses the existing pinned G06 INT8 candidate. It performs text continuation, not instruction-tuned chat, tool execution, or production model activation.

## Installation

1. Install the verified debug APK from this change's Android CI artifact.
2. On the first dashboard page, press **Nạp mô hình INT8 (.zip)** to open the system file picker directly. Use **Mở mô hình INT8 đã nạp** to return to the trial. The existing performance-evidence route also remains available.
3. Import `VN97-G06-INT8-UNQUALIFIED-36512587857` ZIP (3.01 GB). ZIP files may contain the three files at root or together under a wrapper directory. Alternatively press **Hoặc nhập 3 tệp mô hình đã giải nén** and select `candidate.onnx`, `candidate.onnx.data`, and `quantization.json` together. Existing imports remain usable.
4. Import `VN97-G08-TOKENIZER-AUDIT-36504664417` ZIP (about 1.16 MB):
   https://github.com/dathito108-hue/VN97/actions/runs/36504664417/artifacts/11006213380
   Alternatively select all six extracted files in `g06-candidate/tokenizer`.
5. Enter a short passage and press **Thử viết tiếp bằng INT8**.

The tokenizer is bound to ID `27ce0a2f005befaa98ec4ee05d5d83a71aac1bcc5e4dd00032e33a17615ca575` and its asset digests are checked. The INT8 graph/data hashes are rechecked before each run. Artifact links have retention limits.

## Execution

- Maximum 128 prompt tokens, 16 generated tokens; inputs over the limit are rejected, never silently truncated.
- Prompt chunks of 8, then one-token greedy decoding with carried FP16 convolution/SSM state. Each new trial starts from zero state.
- Vocabulary IDs 50277–50287 are excluded. EOS stops without appearing in output.
- All logits and recurrent states must remain finite.
- Stop is cooperative between operations. An ongoing native session load/inference may finish before stopping. Back closes the trial process immediately.
- JSON checkpoints are written atomically. Interrupted native execution is not a successful trial; `execution_passed` is only true after completion.
- Reports include generated text, model/tokenizer identity, device, ORT version, load time, inference time and sampled PSS. Treat copied reports as potentially private.

The chunk-8 graph still executes padded work for single-token decode. This change makes text behavior testable; it does not claim a speedup or quality qualification. Physical S21 FE results are still required.

## Import diagnostics

The screen separately shows model and tokenizer readiness. Run buttons remain disabled until their packages have been committed. Import displays copied MB and the SHA-256 verification stage. Non-ZIP inputs, wrong-package entries, and missing filenames produce distinct messages; arbitrary single model files must use the three-file selector. Staging is removed on failure and only promoted after pinned payload hashes pass. No inference qualification criteria are relaxed.

## Interrupted second-run investigation

Device reports show one completed inference followed by process loss, in both synthetic and text trials. Post-run PSS is about 3.7 GB; this alone does not prove an out-of-memory kill.

Trial sessions now disable ORT memory-pattern optimization and CPU arena retention. This may trade speed for lower retained/peak allocations and requires device measurement. Model bytes, precision, thread count and qualification gates are unchanged.

Checkpoints are emitted before session load and each inference, with phase and pre-run PSS. Saved receipts include process PID/time. On reopening, an unfinished report from a different process is displayed as interrupted and augmented with matching Android process-exit information where available (Android 11+). Native crash, low-memory kill, ANR and signal are reported distinctly. No missing reason is inferred to be OOM and no run restarts automatically. Older receipts without a PID cannot be matched reliably.
