# P4E-N — Sequence-Ranking Reasoning Repair

P4E-L and P4E-M together show that the remaining arithmetic failure is not a
chat-prefix bug or a decoding-search problem:

- prefix equivalence: 30/30;
- current greedy reasoning: 1/30;
- raw greedy reasoning: 1/30;
- numeric beam widths 4, 8, and 16: all 1/30;
- beam rescues: 0/30;
- yet the correct complete answer is among the compact likelihood top-5 on
  22/30 tasks.

P4E-N therefore changes the **training objective**, not the VN97 architecture.

## Objective

Each arithmetic training item contains:

- the prompt;
- the correct numeric answer;
- one deterministic hard negative such as an off-by-one result, operand
  confusion, or wrong-operation result.

For each batch VN97 optimizes three terms:

1. completion cross-entropy on the correct answer;
2. a sequence-level ranking loss that requires the correct answer's mean
   log-likelihood to exceed the hard negative by a margin;
3. preservation cross-entropy over numeric-copy, tool/authority/cognition
   anchors, and deterministic P3 replay.

The default learning rate is only `2e-6` because P4E-J showed that aggressive
arithmetic updates cause interference.

## Held-out safety

Frozen P4 reasoning expressions are excluded from ranking training. If a dev
suite is provided, parsed reasoning expressions and exact dev prompts are also
excluded. The 60-task numeric-copy validation suite remains held out.

## Default preservation gates

Promotion requires:

- reasoning >= 5/30;
- numeric copy >= 52/60 and no more than 3 below the parent;
- overall canonical score no more than 2 below the parent;
- tool intent no more than 1 below the parent;
- authority behavior no more than 1 below the parent;
- instruction/memory/structured cognition no more than 2 below the parent;
- bounded token-validation and P3-retention regressions;
- dev no more than 1 below the parent when supplied.

The parent is the **selected safe P4E-K checkpoint**, not P4E-J.

## Kaggle

Fresh:

```bash
bash tools/kaggle_p4e_sequence_ranking_repair.sh fresh \
  /kaggle/working/p4e-k-final \
  /kaggle/input/datasets/duongtuan1/vn97-p4c-resume/p3-corpus \
  /kaggle/input/datasets/duongtuan1/vn97-p4c-resume/held-out-p4.jsonl
```

Resume:

```bash
bash tools/kaggle_p4e_sequence_ranking_repair.sh resume \
  /kaggle/working/p4e-k-final \
  /kaggle/input/datasets/duongtuan1/vn97-p4c-resume/p3-corpus \
  /kaggle/input/datasets/duongtuan1/vn97-p4c-resume/held-out-p4.jsonl
```

The trainer writes an optimizer/model checkpoint every 50 steps to
`/kaggle/working/p4e-n-work/state.p4en.pt`.


## Tesla T4 memory-safe execution

The P4E-N v2 trainer is specifically hardened for 16 GB-class GPUs:

- ranking batch defaults to 2;
- preservation batch defaults to 2;
- preservation CE is backpropagated and released before reasoning scoring;
- hard-negative sequence scores are detached/no-gradient;
- only the correct-answer path retains the ranking graph;
- Kaggle launcher enables PyTorch expandable CUDA allocator segments.

This preserves the sequence-ranking objective while avoiding the three
simultaneous recurrent graphs that caused the original Tesla T4 OOM.
