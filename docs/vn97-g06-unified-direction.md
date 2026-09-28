# VN97 unified Mamba2/G0.6 evolution
Authoritative user correction: 2026-09-29 02:03 Asia/Saigon.

## One foundation
Develop the existing Mamba2/G03 weights → G05 SSD graphs → G06 ONNX Runtime lineage directly into one multipurpose PC/mobile model. Preserve tokenizer/weight provenance, one planner, VN97MEM1 and M6. PC and Android are execution targets, not separate intelligence backends. Training remains paused.

This supersedes the alternative fast/slow roadmap in vn97-native-architecture-path.md and vn97-portable-adaptive-core.md. Park research/vn97_native (VN97ASC-LM1 / VN97ASCORT1), preserve historical files/checkpoints, and do not continue its development/export/promotion. PR309 small-model parity is not G06 evidence. This correction does not itself activate G06 in canonical chat.

## Existing foundations verified in code
- mamba2_parallel_onnx.py: one SSD graph with maximum chunks 8/16/32; valid_length=1 decode and larger valid lengths prefill, carrying convolution and SSD states.
- Mamba2OrtRuntimePackage.kt: VN97M2G06RUNTIME1 binds source/capsule/graph identity and fixed 2.7B geometry; state supports FP16/FP32.
- Mamba2OrtExecutionBuffers.kt: reusable token buffers and final-position logits reads.
- mamba2_augmentation.py: zero-impact-initialized extensions around preserved Mamba2 layers already exist. They are not proof of trained gains or G06 ONNX integration; do not enable nonzero effects without evaluation.

## Ordered implementation
1. Establish the exact G06 baseline: verify preserved capsule/full weight hashes, graph inventory, tokenizer, source-to-ORT parity and canonical bridge. Reuse available artifacts before expensive recovery. Record actual blockers and outputs. No retraining.
2. Improve context around this core: bounded exact-prefix recurrent snapshots bound to model/runtime/tokenizer, token-prefix hash, position, dtype and state layout. Invalidate on edits or incompatibility; never combine arbitrary SSD states. Retrieve bounded original VN97MEM1 records with provenance, as data rather than authority. Corrections supersede stale records; summaries are lossy supplements, not replacements for original evidence.

Implementation checkpoint: the G06 executor exposes caller-budgeted capture/restore for exact-prefix recurrent state. `VN97M2PREFIXBIND1` binds runtime, selected graph hash, tokenizer model, vocabulary, dtype and both state layouts; `VN97M2PREFIXTOKENS1` hashes every token in order. A bounded LRU refuses oversized entries and returns only exact binding+prefix hits. Nothing is cached implicitly, and the feature is not wired into canonical chat because G06 source/ORT parity and production activation are still outstanding. State snapshots are large (about 86.6 MB at FP16 or 173.3 MB at FP32 for the SSD state alone), so device policy must be based on measured memory and latency before enabling them on S21 FE.

VN97MEM1 correction checkpoint: original semantic evidence remains append-only and directly addressable. A correction is a new, vector-bearing semantic record whose parent is the corrected semantic record and whose source uses the reserved `VN97COR1:` envelope through the explicit Kotlin correction API. Retrieval excludes superseded semantic parents before bounded ranking, including correction chains, while ordinary episodic/semantic parent-child lineage does not imply correction. Returned planner context now includes timestamp, kind, parent and superseded-record provenance. The envelope is an internal behavior contract, not cryptographic proof or execution authority; M6 remains unchanged. Native retrieval still scans the durable index, so this change bounds returned context rather than claiming bounded scan cost.
3. Reduce execution cost: measure prefill separately from decode. Inspect padded work at valid_length=1, copies, state allocations and session reuse first. Any specialized decode graph must derive from the SAME weights and pass cross-graph state/logit parity. Select chunks, threads and providers using actual latency, memory and thermal evidence. Quantization needs separate numerical and task-quality gates.
4. Budget reasoning on the same model: bounded direct passes for simple tasks; additional model/tool verification for harder tasks. Repeating the same input into live state is not deeper reasoning. Retain resumable planner and M6 semantics. Evaluate success and time-to-correct-answer, not reasoning token count.
5. Qualify PC and S21 FE using matched prompts, weight/tokenizer identities and quality protocols. Measure first-token latency, prefill/decode tokens per second, p50/p95 latency, peak PSS/RSS and sustained thermals. CPU is the baseline; GPU/NPU requires actual supported-provider evidence.

## Context and quality gates
Test distant facts, distractors, corrected facts, exact identifiers, multi-turn tasks and abstention. Compare baseline, retrieval, and exact-prefix cache independently; cache should preserve outputs within numerical tolerance while reducing repeated prefill, not claim new knowledge. Freeze evaluation criteria before interpreting results.

Learned SSD/memory-gate/geometry changes are later in-place evolution of this lineage, requiring training authorization and evaluation. Longer chunks do not mean larger retained context, and extra passes do not prove better reasoning. Bounded recurrent state cannot provide unlimited exact recall.

This document changes direction, not runtime behavior. It claims no completed context/speed improvement, mobile measurement, production activation or AGI. Promotion requires exact artifacts, quality/device evidence and rollback. Existing service work continues; architecture progress is not verified revenue.
