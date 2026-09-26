# P4E-B — Chat Response Boundary Recovery

P4E-A measured 21/30 failures as textual role-marker leakage. Historical
P3/P4C training supervised the literal assistant marker, while P4D only stopped
supervising that marker for new completion-aligned records. The legacy replay
distribution can therefore still emit another assistant boundary before the
intended payload.

P4E-B performs no training and changes no weights.

It measures a deterministic recovery rule:

1. generate the raw VN97 response exactly as P4D did;
2. if one or more textual `<|assistant|>` markers were emitted, use the text
   after the last assistant marker as the response payload;
3. truncate at a subsequent textual role marker;
4. score RAW and RECOVERED independently on the same tasks.

The generic `generate_text` behavior remains unchanged in P4E-B. Promotion of
boundary recovery into production inference is deferred until this measurement
shows a real gain with zero regressions.

## Kaggle

```bash
bash tools/kaggle_p4e_boundary_eval.sh \
  /kaggle/working/p4d-final \
  /kaggle/input/datasets/duongtuan1/vn97-p4c-resume/held-out-p4.jsonl \
  /kaggle/working/VN97-P4E-B-boundary.json
```

The final line reports `raw`, `recovered`, `gains`, and `regressions`.
