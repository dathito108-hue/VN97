# VN97 R2-F4 — Background / Continuity Hardening

R2-F4 keeps the existing VN97CNT1 planner, M6 approval and JobScheduler
continuity model, and adds an R2 production identity guard.

Each autonomous job persists VN97R2F4CONT1 with the F2 binding ID, F1 runtime
ID, validated checkpoint SHA-256 and exact VN97TK1 artifact SHA-256. Every
process-death/reboot wake and foreground WAITING_APPROVAL restore must match
the currently installed R2 identity before cognition or side effects resume.

ORT recurrent tensors are deliberately not replayed across process death.
Each cognition operation reconstructs its prompt from canonical planner/memory
state and starts a fresh explicit recurrent state. Existing action receipts,
continuation epochs and M6 deny-by-default authority remain authoritative for
non-replay of external side effects.
